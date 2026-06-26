
import os
import torch
from nemo.collections.asr.models import SortformerEncLabelModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DIAR_MODEL_NAME      = "nvidia/diar_sortformer_4spk-v1"
DEFAULT_LOCAL_PATH   = "./model_cache/diar_sortformer_4spk-v1.nemo"
DIAR_LOCAL_MODEL_PATH = os.environ.get("DIAR_LOCAL_MODEL_PATH") or DEFAULT_LOCAL_PATH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Model load (once at startup)
# ---------------------------------------------------------------------------

print("[diarization] Loading Sortformer model…")
if os.path.exists(DIAR_LOCAL_MODEL_PATH):
    print(f"[diarization] Restoring from local checkpoint: {DIAR_LOCAL_MODEL_PATH}")
    _diar_model = SortformerEncLabelModel.restore_from(
        restore_path=DIAR_LOCAL_MODEL_PATH,
        map_location=torch.device(DEVICE),
        strict=False,
    )
else:
    print(f"[diarization] Local checkpoint not found — downloading {DIAR_MODEL_NAME}")
    print("[diarization] Requires HuggingFace login: run `huggingface-cli login` if this fails.")
    _diar_model = SortformerEncLabelModel.from_pretrained(DIAR_MODEL_NAME)

_diar_model.eval()
if DEVICE == "cuda":
    _diar_model = _diar_model.cuda()
    print("[diarization] Running on GPU.")
else:
    print("[diarization] Running on CPU (slower).")

print("[diarization] Model ready.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_diarization(wav_path: str) -> list:
    """
    Run Sortformer on a single WAV file.
    Returns the raw segment list for that file (list of strings).
    """
    raw_outputs = _diar_model.diarize(audio=[wav_path], batch_size=1)
    return raw_outputs[0] if raw_outputs else []