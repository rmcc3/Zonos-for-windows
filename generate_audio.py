import torch
import torchaudio
import os
import sys
import json
import traceback
import math
import multiprocessing as mp

# --- Zonos Imports ---
# These might be needed globally for type hints or initial checks
try:
    from zonos.model import Zonos
    from zonos.conditioning import make_cond_dict
    from zonos.utils import DEFAULT_DEVICE as default_device_name # Get the default name ('cuda:0', 'cpu')
except ImportError as e:
    print(f"Error importing Zonos modules: {e}", file=sys.stderr)
    print("Ensure you are running this script within the activated Zonos virtual environment (.venv/Scripts/activate)", file=sys.stderr)
    sys.exit(1)

# --- Constants ---
DEFAULT_LANGUAGE = "en-us"
# VRAM requirement per worker (for the main model + embedding + overhead)
REQUIRED_VRAM_PER_WORKER_GB = 4.0
REQUIRED_VRAM_BYTES_PER_WORKER = int(REQUIRED_VRAM_PER_WORKER_GB * (1024**3))
MODEL_NAME = "Zyphra/Zonos-v0.1-transformer"
SENTINEL = None # Signal for workers to terminate

# --- Utility Functions ---
def get_available_vram():
    """Gets available VRAM in bytes on the default CUDA device."""
    if torch.cuda.is_available():
        try:
            device_index = torch.cuda.current_device()
            free_mem, _ = torch.cuda.mem_get_info(device_index)
            buffer = 100 * 1024 * 1024 # Slightly larger buffer for persistent workers
            return max(0, free_mem - buffer)
        except Exception as e:
            print(f"Warning: Could not get VRAM info: {e}", file=sys.stderr)
            return 0
    return 0

# --- Speaker Embedding Loader (Used by Worker) ---
# Renamed slightly to avoid conflict if generate_single_audio is kept below
def load_speaker_embedding_for_worker(model_instance, speaker_path, script_dir, device):
    """Loads speaker embedding within a worker process."""
    abs_speaker_path = os.path.join(script_dir, speaker_path)
    # print(f"[Worker {os.getpid()}] Attempting to load speaker: {abs_speaker_path}", file=sys.stderr) # Verbose
    if not os.path.exists(abs_speaker_path):
        print(f"[Worker {os.getpid()}] Error: Speaker audio file not found at {abs_speaker_path}", file=sys.stderr)
        return None
    try:
        wav, sampling_rate = torchaudio.load(abs_speaker_path)
        wav = wav.to(device)
        speaker = model_instance.make_speaker_embedding(wav, sampling_rate)
        # print(f"[Worker {os.getpid()}] Speaker embedding loaded successfully.", file=sys.stderr) # Verbose
        return speaker
    except Exception as e:
        print(f"[Worker {os.getpid()}] Error loading speaker embedding {speaker_path}: {e}", file=sys.stderr)
        # traceback.print_exc(file=sys.stderr) # Can be too verbose
        return None

# --- Single Generation Function (Used by Worker) ---
def generate_audio_segment(model_instance, text, output_path, speaker_embedding, request_id):
    """Generates audio for a single request using the pre-loaded model and speaker embedding."""
    output_abs_path = os.path.abspath(output_path)
    try:
        # print(f"[Worker {os.getpid()}] Generating audio for request '{request_id}'...", file=sys.stderr) # Verbose
        # Re-import make_cond_dict if needed (might be safer within the function)
        from zonos.conditioning import make_cond_dict
        cond_dict = make_cond_dict(text=text, speaker=speaker_embedding, language=DEFAULT_LANGUAGE)
        conditioning = model_instance.prepare_conditioning(cond_dict)
        codes = model_instance.generate(conditioning, disable_torch_compile=True)
        codes = codes.to(model_instance.autoencoder.dac.device)
        wavs = model_instance.autoencoder.decode(codes).cpu()

        output_dir = os.path.dirname(output_abs_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        torchaudio.save(output_abs_path, wavs[0], model_instance.autoencoder.sampling_rate)
        # print(f"[Worker {os.getpid()}] Audio saved to {output_abs_path} for request '{request_id}'", file=sys.stderr) # Verbose
        return {"request_id": request_id, "output_path": output_abs_path, "success": True, "error": None}

    except Exception as e:
        print(f"[Worker {os.getpid()}] Error during audio generation for request '{request_id}': {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {"request_id": request_id, "output_path": None, "success": False, "error": str(e)}

# --- Persistent Worker Process Function ---
def persistent_worker(task_queue, result_queue, script_dir):
    """
    Worker process function. Loads model once, then processes tasks from queue.
    Manages its own speaker embedding cache (currently just the last used one).
    """
    worker_pid = os.getpid()
    print(f"[Worker {worker_pid}] Initializing...", file=sys.stderr)

    # Determine device for this worker
    import torch # Ensure torch is imported in the worker process
    device = torch.device(default_device_name if torch.cuda.is_available() else "cpu")
    print(f"[Worker {worker_pid}] Using device: {device}", file=sys.stderr)

    # --- Load Model Instance ONCE ---
    worker_model = None
    try:
        # Re-import Zonos locally within the worker function's scope
        from zonos.model import Zonos
        print(f"[Worker {worker_pid}] Loading model '{MODEL_NAME}' on {device}...", file=sys.stderr)
        worker_model = Zonos.from_pretrained(MODEL_NAME, device=device)
        print(f"[Worker {worker_pid}] Model loaded.", file=sys.stderr)
    except Exception as model_load_error:
        print(f"[Worker {worker_pid}] FATAL: Failed to load Zonos model: {model_load_error}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # Cannot proceed without model, maybe put an error on result queue? Difficult.
        # For now, just exit the worker. Main process will hang waiting for results.
        return # Exit worker

    # --- Worker State ---
    current_speaker_path = None
    current_speaker_embedding = None

    # --- Task Processing Loop ---
    while True:
        try:
            task_data = task_queue.get() # Blocking wait for a task

            if task_data is SENTINEL:
                print(f"[Worker {worker_pid}] Received sentinel. Exiting.", file=sys.stderr)
                break # Exit loop and terminate worker

            # Extract task details
            text = task_data["text"]
            output_path = task_data["output_path"]
            required_speaker_path = task_data["speaker_path"]
            request_id = task_data["request_id"]

            # print(f"[Worker {worker_pid}] Received task: {request_id} (Speaker: {required_speaker_path})", file=sys.stderr) # Verbose

            # --- Manage Speaker Embedding ---
            if required_speaker_path != current_speaker_path:
                print(f"[Worker {worker_pid}] Speaker change required for request '{request_id}'. Current: '{current_speaker_path}', Required: '{required_speaker_path}'.", file=sys.stderr)
                # Clear previous embedding (optional, helps if memory is tight)
                if current_speaker_embedding is not None:
                    del current_speaker_embedding
                    if torch.cuda.is_available(): torch.cuda.empty_cache() # Try to free VRAM
                    # print(f"[Worker {worker_pid}] Cleared previous speaker embedding.", file=sys.stderr) # Verbose
                current_speaker_embedding = load_speaker_embedding_for_worker(worker_model, required_speaker_path, script_dir, device)
                current_speaker_path = required_speaker_path # Update even if loading failed

                if current_speaker_embedding is None:
                    print(f"[Worker {worker_pid}] Failed to load speaker '{required_speaker_path}' for request '{request_id}'.", file=sys.stderr)
                    result_queue.put({"request_id": request_id, "output_path": None, "success": False, "error": f"Failed to load speaker embedding: {required_speaker_path}"})
                    continue # Skip to next task
                else:
                     print(f"[Worker {worker_pid}] Loaded speaker '{required_speaker_path}'.", file=sys.stderr)

            # --- Generate Audio ---
            if current_speaker_embedding is not None: # Should always be true unless load failed above
                result = generate_audio_segment(
                    worker_model,
                    text,
                    output_path,
                    current_speaker_embedding,
                    request_id
                )
                result_queue.put(result)
            # else case handled above by putting error result and continuing

        except Exception as loop_error:
            # Catch unexpected errors in the loop/queue handling
            print(f"[Worker {worker_pid}] UNEXPECTED ERROR in processing loop: {loop_error}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            # Try to report error for the current task if possible, otherwise might lose track
            request_id_for_error = task_data.get("request_id", "unknown") if isinstance(task_data, dict) else "unknown"
            result_queue.put({"request_id": request_id_for_error, "output_path": None, "success": False, "error": f"Unexpected worker error: {loop_error}"})
            # Continue processing next task if possible

    # --- Worker Cleanup ---
    print(f"[Worker {worker_pid}] Cleaning up...", file=sys.stderr)
    if worker_model is not None:
        del worker_model
    if current_speaker_embedding is not None:
        del current_speaker_embedding
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"[Worker {worker_pid}] Finished.", file=sys.stderr)


# --- Main Execution ---
if __name__ == "__main__":
    # Set start method for multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        print("Warning: Could not set multiprocessing start method to 'spawn'. Using default.", file=sys.stderr)

    # --- Initialization ---
    print(f"Main process started. Persistent workers will load model '{MODEL_NAME}'.", file=sys.stderr)
    print(f"VRAM Requirement per worker: {REQUIRED_VRAM_PER_WORKER_GB:.1f} GB ({REQUIRED_VRAM_BYTES_PER_WORKER} bytes)", file=sys.stderr)

    script_dir = os.path.dirname(os.path.realpath(__file__))
    all_requests = [] # Flat list to hold all normalized requests
    total_lines_read = 0
    total_segments_parsed = 0
    total_segments_succeeded = 0
    results_map = {} # To store results by request_id

    # --- Calculate Number of Workers ---
    available_vram = get_available_vram()
    print(f"Available VRAM check: {available_vram / (1024**3):.2f} GB", file=sys.stderr)

    if available_vram < REQUIRED_VRAM_BYTES_PER_WORKER:
         print(f"FATAL: Insufficient VRAM ({available_vram / (1024**3):.2f} GB) for even one worker ({REQUIRED_VRAM_PER_WORKER_GB:.1f} GB required). Exiting.", file=sys.stderr)
         sys.exit(1)

    num_workers = math.floor(available_vram / REQUIRED_VRAM_BYTES_PER_WORKER)
    num_workers = max(1, num_workers) # Ensure at least one worker
    print(f"Determined number of persistent workers: {num_workers}", file=sys.stderr)

    # --- Read, Parse, Normalize All Input ---
    print("Reading JSON lines from stdin (supports single 'text' or list 'texts')...", file=sys.stderr)
    for line_num, line in enumerate(sys.stdin, 1):
        total_lines_read += 1
        line = line.strip()
        if not line: continue
        try:
            data = json.loads(line)
            speaker_path = data.get("speaker_path")
            if not speaker_path or not isinstance(speaker_path, str): raise ValueError("Missing or invalid 'speaker_path'")

            if "texts" in data and "output_paths" in data:
                texts, output_paths = data["texts"], data["output_paths"]
                if not isinstance(texts, list) or not isinstance(output_paths, list): raise ValueError("'texts' or 'output_paths' is not a list")
                if len(texts) != len(output_paths): raise ValueError("Mismatched lengths for 'texts' and 'output_paths'")
                if len(texts) == 0: print(f"Warning: Empty 'texts' list on line {line_num}. Skipping.", file=sys.stderr); continue

                for i, (text, output_path) in enumerate(zip(texts, output_paths)):
                    if not isinstance(text, str) or not isinstance(output_path, str): raise ValueError(f"Invalid item type at index {i}")
                    all_requests.append({"text": text, "output_path": output_path, "speaker_path": speaker_path, "request_id": f"line{line_num}_seg{i}"})
                    total_segments_parsed += 1
                # print(f"Parsed {len(texts)} segments from line {line_num}.", file=sys.stderr) # Verbose

            elif "text" in data and "output_path" in data:
                text, output_path = data["text"], data["output_path"]
                if not isinstance(text, str) or not isinstance(output_path, str): raise ValueError("Invalid type for 'text' or 'output_path'")
                all_requests.append({"text": text, "output_path": output_path, "speaker_path": speaker_path, "request_id": f"line{line_num}"})
                total_segments_parsed += 1
                # print(f"Parsed 1 segment from line {line_num}.", file=sys.stderr) # Verbose

            else: raise ValueError("Expected ('texts'/'output_paths') or ('text'/'output_path')")

        except json.JSONDecodeError: print(f"Error: Invalid JSON on line {line_num}: {line}", file=sys.stderr)
        except Exception as read_error: print(f"Error parsing line {line_num}: {read_error}", file=sys.stderr); traceback.print_exc(file=sys.stderr)

    print(f"\nFinished reading stdin. Total lines read: {total_lines_read}. Total segments parsed: {total_segments_parsed}.", file=sys.stderr)

    if not all_requests:
        print("No valid requests found to process. Exiting.", file=sys.stderr)
        sys.exit(0)

    # --- Create Queues and Start Workers ---
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    workers = []

    print(f"Starting {num_workers} worker processes...", file=sys.stderr)
    for _ in range(num_workers):
        process = mp.Process(target=persistent_worker, args=(task_queue, result_queue, script_dir))
        workers.append(process)
        process.start()

    # --- Dispatch Tasks ---
    print(f"Dispatching {len(all_requests)} tasks to workers...", file=sys.stderr)
    for request_data in all_requests:
        task_queue.put(request_data)

    # --- Signal Workers to Stop ---
    print("All tasks dispatched. Sending stop signals to workers...", file=sys.stderr)
    for _ in range(num_workers):
        task_queue.put(SENTINEL)

    # --- Collect Results ---
    print(f"Collecting {len(all_requests)} results...", file=sys.stderr)
    for _ in range(len(all_requests)):
        try:
            # Use timeout to prevent hanging indefinitely if a worker dies unexpectedly
            result = result_queue.get(timeout=300) # 5 min timeout per result
            request_id = result.get("request_id", "unknown")
            results_map[request_id] = result # Store result
            if result and result.get("success"):
                total_segments_succeeded += 1
                # Print successful path to stdout for the calling process
                print(result["output_path"], flush=True)
            elif result:
                 print(f"Error reported for request {request_id}: {result.get('error', 'Unknown worker error')}", file=sys.stderr)
            else:
                 print(f"Warning: Received unexpected item from result queue: {result}", file=sys.stderr)

        except mp.queues.Empty:
             print("Error: Result queue timed out. A worker might have crashed.", file=sys.stderr)
             # How to handle? Mark remaining as failed? For now, just break collection.
             break
        except Exception as collect_err:
             print(f"Error collecting result: {collect_err}", file=sys.stderr)


    # --- Wait for Workers to Finish ---
    print("Waiting for workers to terminate...", file=sys.stderr)
    for process in workers:
        process.join(timeout=30) # Give workers time to exit cleanly
        if process.is_alive():
             print(f"Warning: Worker process {process.pid} did not terminate gracefully. Forcing termination.", file=sys.stderr)
             process.terminate() # Force kill if stuck

    print(f"\n--- Processing Complete ---", file=sys.stderr)
    print(f"Total lines read from input: {total_lines_read}", file=sys.stderr)
    print(f"Total segments parsed from input: {total_segments_parsed}", file=sys.stderr)
    # Could iterate through results_map for more detailed error summary if needed
    print(f"Total audio files successfully generated: {total_segments_succeeded}", file=sys.stderr)
    sys.exit(0) # Exit cleanly
