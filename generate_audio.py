import torch
import torchaudio
import argparse
import os
import sys
import base64

# Add the project root to sys.path if necessary, although running via venv should handle imports
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from zonos.model import Zonos
    from zonos.conditioning import make_cond_dict
    from zonos.utils import DEFAULT_DEVICE as device
except ImportError as e:
    print(f"Error importing Zonos modules: {e}", file=sys.stderr)
    print("Ensure you are running this script within the activated Zonos virtual environment (.venv/Scripts/activate)", file=sys.stderr)
    sys.exit(1)


# Default language - can be made configurable later
DEFAULT_LANGUAGE = "en-us"

def generate_audio(text, output_path, speaker_path, language=DEFAULT_LANGUAGE):
    """Generates audio from text using Zonos and saves it to output_path."""
    try:
        print(f"Initializing Zonos model (transformer)...", file=sys.stderr)
        # Ensure the model path is correct relative to this script if needed,
        # but from_pretrained should handle it if run within the venv.
        model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
        print(f"Model loaded on device: {device}", file=sys.stderr)

        # Determine absolute path for speaker audio relative to this script
        script_dir = os.path.dirname(os.path.realpath(__file__))
        # Use the provided speaker_path argument
        abs_speaker_path = os.path.join(script_dir, speaker_path)

        print(f"Loading speaker embedding from: {abs_speaker_path}", file=sys.stderr)
        if not os.path.exists(abs_speaker_path):
            print(f"Error: Speaker audio file not found at {abs_speaker_path}", file=sys.stderr)
            return False

        wav, sampling_rate = torchaudio.load(abs_speaker_path)
        # Ensure wav is on the correct device
        wav = wav.to(device)
        speaker = model.make_speaker_embedding(wav, sampling_rate)
        print("Speaker embedding created.", file=sys.stderr)

        print(f"Preparing conditioning for text: '{text}'", file=sys.stderr)
        cond_dict = make_cond_dict(text=text, speaker=speaker, language=language)
        conditioning = model.prepare_conditioning(cond_dict)
        print("Conditioning prepared.", file=sys.stderr)

        print("Generating audio codes...", file=sys.stderr)
        # Disable torch.compile to potentially bypass the dataclass TypeError
        codes = model.generate(conditioning, disable_torch_compile=True)
        print("Audio codes generated.", file=sys.stderr)

        print("Decoding audio...", file=sys.stderr)
        # Ensure codes are on the correct device before decoding (use internal dac model's device)
        codes = codes.to(model.autoencoder.dac.device)
        wavs = model.autoencoder.decode(codes).cpu()
        print("Audio decoded.", file=sys.stderr)

        # Ensure output directory exists
        output_abs_path = os.path.abspath(output_path)
        output_dir = os.path.dirname(output_abs_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}", file=sys.stderr)

        print(f"Saving audio to: {output_abs_path}", file=sys.stderr)
        torchaudio.save(output_abs_path, wavs[0], model.autoencoder.sampling_rate)
        print(f"Audio saved successfully to {output_abs_path}", file=sys.stderr)
        # Print the final path to stdout for the calling process
        print(output_abs_path)
        return True

    except Exception as e:
        print(f"Error during audio generation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio from text using Zonos TTS.")
    parser.add_argument("text_base64", type=str, help="The Base64 encoded input text to synthesize.")
    parser.add_argument("output_path", type=str, help="The path to save the generated WAV file.")
    parser.add_argument("--speaker_path", type=str, required=True, help="Path to the speaker reference audio file (relative to script directory).")
    # Optional arguments for future enhancement
    # parser.add_argument("--lang", type=str, default=DEFAULT_LANGUAGE, help="Language code (e.g., en-us, ja, zh, fr, de).")

    args = parser.parse_args()

    # Decode the Base64 input text
    try:
        decoded_text = base64.b64decode(args.text_base64).decode('utf-8')
    except Exception as decode_error:
        print(f"Error decoding Base64 input: {decode_error}", file=sys.stderr)
        sys.exit(1)

    print(f"Starting audio generation for decoded text: '{decoded_text[:100]}...'", file=sys.stderr) # Log truncated decoded text
    print(f"Requested output file path: {args.output_path}", file=sys.stderr)
    print(f"Using speaker voice: {args.speaker_path}", file=sys.stderr) # Log speaker path

    # Call the generation function with the decoded text and speaker path
    success = generate_audio(decoded_text, args.output_path, speaker_path=args.speaker_path)

    if success:
        print("Audio generation completed successfully.", file=sys.stderr)
        sys.exit(0)
    else:
        print("Audio generation failed.", file=sys.stderr)
        sys.exit(1)
