"""
transcription/asr.py
---------------------
Loads the Nemotron ASR model once at import time and exposes
transcribe_segments() for per-turn inference.

Environment overrides:
    LOCAL_MODEL_PATH   — path to a local .nemo checkpoint (ASR)
"""

import os
from typing import List, Tuple

import torch
import numpy as np
from scipy.io import wavfile

import nemo.collections.asr as nemo_asr

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "nvidia/nemotron-speech-streaming-en-0.6b"
DEFAULT_LOCAL_PATH = "./model_cache/nemotron-speech-streaming-en-0.6b.nemo"
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH") or DEFAULT_LOCAL_PATH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Model load (once at startup)
# ---------------------------------------------------------------------------

print("[asr] Loading ASR model…")
if os.path.exists(LOCAL_MODEL_PATH):
    print(f"[asr] Restoring from local checkpoint: {LOCAL_MODEL_PATH}")
    _asr_model = nemo_asr.models.ASRModel.restore_from(LOCAL_MODEL_PATH)
else:
    print(f"[asr] Local checkpoint not found — downloading {MODEL_NAME}")
    _asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)

_asr_model.eval()

if DEVICE == "cuda":
    _asr_model = _asr_model.cuda()
    print(f"[asr] Running on GPU: {torch.cuda.get_device_name(0)}")
else:
    print("[asr] No GPU detected — running on CPU (slower).")

# Widen attention context for best accuracy — no latency constraint since
# we transcribe pre-sliced segments offline, not a live mic stream.
try:
    _asr_model.encoder.set_default_att_context_size([70, 13])  # highest accuracy from this model's trained look-aheads
    print("[asr] att_context_size set to [70, 13] (full context)")
except AttributeError:
    print("[asr] encoder has no att_context_size — skipping context widen")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

Segment = Tuple[float, float, str]  # (start, end, role)


def transcribe_segments(
    wav_path: str,
    labeled_segments: List[Segment],
    tmp_dir: str,
) -> List[dict]:
    """
    Slice wav_path into one clip per labeled segment, run the ASR model
    on all clips in a single batched call, and return a list of turn dicts:

        [{"speaker": "Doctor", "text": "...", "start": 0.0, "end": 3.2}, ...]

    Turns where the ASR produces no text (silence / non-speech) are skipped.
    """
    if not labeled_segments:
        return []

    sample_rate, audio_np = wavfile.read(wav_path)  # int16, mono, 16kHz

    # Write each turn to a temporary WAV file
    segment_paths: List[str] = []
    PAD_SECONDS = 0.2  # small buffer to avoid clipping words at segment edges
    for i, seg in enumerate(labeled_segments):
        start_s = max(0, int((seg["start"] - PAD_SECONDS) * sample_rate))
        end_s   = min(len(audio_np), int((seg["end"] + PAD_SECONDS) * sample_rate))
        clip    = audio_np[start_s:end_s]
        path    = os.path.join(tmp_dir, f"seg_{i}.wav")
        wavfile.write(path, sample_rate, clip)
        segment_paths.append(path)

    # Batched ASR inference
    with torch.no_grad():
        outputs = _asr_model.transcribe(segment_paths)

    conversation: List[dict] = []
    for seg, output in zip(labeled_segments, outputs):
        text = output.text if hasattr(output, "text") else output
        text = (text or "").strip()
        if not text:
            continue
        conversation.append({
            "speaker":     seg["speaker"],       # TitaNet-identified name (or "Unknown")
            "diarized_as": seg.get("diarized_as"),
            "similarity":  seg.get("similarity"),
            "text":        text,
            "start":       round(seg["start"], 2),
            "end":         round(seg["end"], 2),
        })
    return conversation