import sys
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000  # required by the model — do not change

def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    output_path = sys.argv[2] if len(sys.argv) > 2 else "test_audio.wav"

    print(f"Recording {duration}s of audio at {SAMPLE_RATE}Hz mono...")
    print("Speak now.")

    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()

    sf.write(output_path, audio, SAMPLE_RATE)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()