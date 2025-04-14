import torch
import torchaudio
import os
import sys
import json
import traceback

try:
    from zonos.model import Zonos
    from zonos.conditioning import make_cond_dict
    from zonos.utils import DEFAULT_DEVICE as device
except ImportError as e:
    print(f"Error importing Zonos modules: {e}", file=sys.stderr)
    print("Ensure you are running this script within the activated Zonos virtual environment (.venv/Scripts/activate)", file=sys.stderr)
    sys.exit(1)

# Default language - can be made configurable later via JSON input if needed
DEFAULT_LANGUAGE = "en-us"

def load_speaker_embedding(model, speaker_path, script_dir):
    """Loads or reuses speaker embedding."""
    abs_speaker_path = os.path.join(script_dir, speaker_path)
    print(f"Loading speaker embedding from: {abs_speaker_path}", file=sys.stderr)
    if not os.path.exists(abs_speaker_path):
        print(f"Error: Speaker audio file not found at {abs_speaker_path}", file=sys.stderr)
        return None

    try:
        wav, sampling_rate = torchaudio.load(abs_speaker_path)
        wav = wav.to(device)
        speaker = model.make_speaker_embedding(wav, sampling_rate)
        print("Speaker embedding created.", file=sys.stderr)
        return speaker
    except Exception as e:
        print(f"Error loading speaker embedding: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None


def generate_single_audio(model, text, output_path, speaker_embedding, language=DEFAULT_LANGUAGE):
    """Generates audio for a single text input using pre-loaded model and speaker."""
    try:
        print(f"Preparing conditioning for text: '{text[:50]}...'", file=sys.stderr)
        cond_dict = make_cond_dict(text=text, speaker=speaker_embedding, language=language)
        conditioning = model.prepare_conditioning(cond_dict)
        print("Conditioning prepared.", file=sys.stderr)

        print("Generating audio codes...", file=sys.stderr)
        codes = model.generate(conditioning, disable_torch_compile=True) # Keep compile disabled for now
        print("Audio codes generated.", file=sys.stderr)

        print("Decoding audio...", file=sys.stderr)
        codes = codes.to(model.autoencoder.dac.device)
        wavs = model.autoencoder.decode(codes).cpu()
        print("Audio decoded.", file=sys.stderr)

        output_abs_path = os.path.abspath(output_path)
        output_dir = os.path.dirname(output_abs_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}", file=sys.stderr)

        print(f"Saving audio to: {output_abs_path}", file=sys.stderr)
        torchaudio.save(output_abs_path, wavs[0], model.autoencoder.sampling_rate)
        print(f"Audio saved successfully to {output_abs_path}", file=sys.stderr)

        # Print the final path to stdout for the calling process
        print(output_abs_path, flush=True) # Ensure output is flushed immediately
        return True

    except Exception as e:
        print(f"Error during single audio generation for text '{text[:50]}...': {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return False

if __name__ == "__main__":
    # --- Initialization ---
    print(f"Initializing Zonos model (transformer)...", file=sys.stderr)
    try:
        # Ensure the model path is correct relative to this script if needed,
        # but from_pretrained should handle it if run within the venv.
        model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
        print(f"Model loaded on device: {device}", file=sys.stderr)
    except Exception as model_load_error:
        print(f"FATAL: Failed to load Zonos model: {model_load_error}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    script_dir = os.path.dirname(os.path.realpath(__file__))
    current_speaker_path = None
    current_speaker_embedding = None
    lines_processed = 0
    lines_succeeded = 0

    print("Ready to process JSON lines from stdin...", file=sys.stderr)

    # --- Processing Loop ---
    for line in sys.stdin:
        lines_processed += 1
        line = line.strip()
        if not line:
            continue # Skip empty lines

        print(f"\n--- Processing line {lines_processed} ---", file=sys.stderr)
        try:
            data = json.loads(line)
            text = data.get("text")
            output_path = data.get("output_path")
            speaker_path = data.get("speaker_path") # Relative path expected

            if not all([text, output_path, speaker_path]):
                print(f"Error: Missing required field in JSON input: {line}", file=sys.stderr)
                continue

            print(f"Request: Text='{text[:50]}...', Output='{output_path}', Speaker='{speaker_path}'", file=sys.stderr)

            # --- Speaker Embedding Management ---
            if speaker_path != current_speaker_path:
                print(f"Speaker changed from '{current_speaker_path}' to '{speaker_path}'. Loading new embedding.", file=sys.stderr)
                current_speaker_embedding = load_speaker_embedding(model, speaker_path, script_dir)
                if current_speaker_embedding is None:
                    print(f"Error: Failed to load speaker embedding for {speaker_path}. Skipping this line.", file=sys.stderr)
                    current_speaker_path = None # Reset so it tries again next time
                    continue
                current_speaker_path = speaker_path
            else:
                print(f"Reusing speaker embedding for '{speaker_path}'.", file=sys.stderr)
                if current_speaker_embedding is None:
                     print(f"Error: Tried to reuse speaker embedding for {speaker_path}, but it failed to load previously. Skipping.", file=sys.stderr)
                     continue # Skip if the embedding failed previously

            # --- Generate Audio for this line ---
            success = generate_single_audio(model, text, output_path, current_speaker_embedding)
            if success:
                lines_succeeded += 1

        except json.JSONDecodeError:
            print(f"Error: Invalid JSON received: {line}", file=sys.stderr)
        except Exception as loop_error:
            # Catch any other unexpected errors during the loop processing for a single line
            print(f"Error processing line {lines_processed}: {loop_error}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    print(f"\n--- Processing Complete ---", file=sys.stderr)
    print(f"Total lines processed: {lines_processed}", file=sys.stderr)
    print(f"Total lines succeeded: {lines_succeeded}", file=sys.stderr)
    sys.exit(0) # Exit cleanly after processing all input
