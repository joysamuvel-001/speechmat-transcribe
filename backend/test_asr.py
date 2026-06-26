
import os
import sys

MODEL_NAME = "nvidia/nemotron-speech-streaming-en-0.6b"
AUDIO_PATH = sys.argv[1] if len(sys.argv) > 1 else "test_audio.wav"

# If you've already downloaded the .nemo file locally (recommended -- avoids
# re-downloading on every run), point this at it, e.g.:
#   LOCAL_MODEL_PATH = "./model_cache/nemotron-speech-streaming-en-0.6b.nemo"
# Leave as None to fetch from Hugging Face Hub via from_pretrained() instead.
LOCAL_MODEL_PATH = "./model_cache/nemotron-speech-streaming-en-0.6b.nemo"

# Optional HF login -- only matters for gated models, harmless to include now.
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    from huggingface_hub import login
    login(token=hf_token)

if not os.path.exists(AUDIO_PATH):
    raise FileNotFoundError(
        f"Couldn't find '{AUDIO_PATH}'. Run record_audio.py first, "
        "or pass a path to an existing 16kHz mono WAV file."
    )

print("Importing NeMo (first import can take a little while)...")
import torch
import nemo.collections.asr as nemo_asr

if LOCAL_MODEL_PATH:
    if not os.path.exists(LOCAL_MODEL_PATH):
        raise FileNotFoundError(f"LOCAL_MODEL_PATH is set but '{LOCAL_MODEL_PATH}' doesn't exist.")
    print(f"Loading model from local file: {LOCAL_MODEL_PATH} ...")
    asr_model = nemo_asr.models.ASRModel.restore_from(LOCAL_MODEL_PATH)
else:
    print(f"Loading {MODEL_NAME} from Hugging Face Hub (first run downloads ~2.5GB)...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
asr_model.eval()

if torch.cuda.is_available():
    asr_model = asr_model.cuda()
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected -- running on CPU (will be noticeably slower).")

print(f"Transcribing {AUDIO_PATH} ...")
with torch.no_grad():
    output = asr_model.transcribe([AUDIO_PATH])

# NeMo's transcribe() return shape has changed across versions -- handle both.
result = output[0]
text = result.text if hasattr(result, "text") else result

print("\n--- TRANSCRIPT ---")
print(text if text else "(empty -- no speech detected, or input format issue)")