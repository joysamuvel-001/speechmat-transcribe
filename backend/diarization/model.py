"""
diarization/model.py
---------------------
Speaker diarization using pyannote/speaker-diarization-3.1.
Replaces Sortformer. Same output format — server.py unchanged.

Requires:
    pip install pyannote.audio
    HF_TOKEN in .env (gated model — accept at huggingface.co/pyannote/speaker-diarization-3.1)
"""

import os
import logging
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
from scipy.io import wavfile
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("medtranscribe.diarization")

# ── torchaudio compatibility shims (same as your other project) ───────────────
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda *args, **kwargs: None

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

if not hasattr(torchaudio, "AudioMetaData"):
    @dataclass
    class _AudioMetaDataShim:
        sample_rate: int = 0
        num_frames: int = 0
        num_channels: int = 0
        bits_per_sample: int = 0
        encoding: str = ""
    torchaudio.AudioMetaData = _AudioMetaDataShim

# ── speechbrain LazyModule patch (same as your other project) ─────────────────
try:
    from speechbrain.utils.importutils import LazyModule
    _orig = LazyModule.__getattr__
    def _patched(self, attr):
        try:
            return _orig(self, attr)
        except ImportError as e:
            raise AttributeError(str(e)) from e
    LazyModule.__getattr__ = _patched
except Exception as e:
    logger.warning("Could not patch speechbrain LazyModule: %s", e)

from pyannote.audio import Pipeline

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    if not HF_TOKEN:
        raise EnvironmentError(
            "HF_TOKEN is required for pyannote/speaker-diarization-3.1.\n"
            "Add HF_TOKEN=your_token to your .env file.\n"
            "Accept the model license at: huggingface.co/pyannote/speaker-diarization-3.1"
        )

    print(f"[diarization] Loading pyannote/speaker-diarization-3.1 on {DEVICE}...")

    # pyannote checkpoint predates PyTorch 2.6 weights_only=True default
    _orig_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig_load(*args, **kwargs)

    torch.load = _patched_load
    try:
        _pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN   # not token= — that's the 4.0 API
        )
    finally:
        torch.load = _orig_load

    _pipeline.to(torch.device(DEVICE))

    # ── Tuning: reduce false speaker-splits on consultation audio ──────────
    # _pipeline.instantiate({
    #     "segmentation": {
    #         "min_duration_off": 0.6,   # merge pauses shorter than this instead of splitting
    #     },
    #     "clustering": {
    #         "threshold": 0.75,         # stricter than default 0.7 — fewer spurious new speakers
    #     },
    # })
    # ─────────────────────────────────────────────────────────────────────

    print(f"[diarization] pyannote ready on {DEVICE}.")
    return _pipeline


def run_diarization(wav_path: str) -> list:
    """
    Run pyannote diarization on a 16kHz mono WAV file.
    Returns list of dicts matching what process_diarization() in speaker.py expects:
        [{"speaker": "SPEAKER_00", "start": 0.0, "end": 3.2}, ...]
    """
    pipeline = _get_pipeline()

    sample_rate, audio_np = wavfile.read(wav_path)

    # Convert int16 → float32 normalised [-1, 1]
    if audio_np.dtype == np.int16:
        audio_float = audio_np.astype(np.float32) / 32768.0
    else:
        audio_float = audio_np.astype(np.float32)

    # pyannote expects (1, time) tensor
    tensor = torch.from_numpy(audio_float).unsqueeze(0)
    audio_input = {"waveform": tensor, "sample_rate": sample_rate}

    diarization = pipeline(audio_input)

    # 3.3.2 returns the Annotation directly — no .speaker_diarization wrapper
    segments = []
    for turn, _, speaker_label in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker_label,
            "start":   round(turn.start, 3),
            "end":     round(turn.end,   3),
        })

    segments.sort(key=lambda s: s["start"])
    print(f"[diarization] pyannote found {len(segments)} speaker turns")
    return segments