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

# ── huggingface_hub use_auth_token patch (resolves newer hf_hub compatibility) ──
try:
    import huggingface_hub
    _orig_hf_hub_download = huggingface_hub.hf_hub_download

    def _patched_hf_hub_download(*args, **kwargs):
        if "use_auth_token" in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        return _orig_hf_hub_download(*args, **kwargs)

    huggingface_hub.hf_hub_download = _patched_hf_hub_download

    import huggingface_hub.file_download
    huggingface_hub.file_download.hf_hub_download = _patched_hf_hub_download
except Exception as e:
    logger.warning("Could not patch huggingface_hub hf_hub_download: %s", e)

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
            use_auth_token=HF_TOKEN
        )
    finally:
        torch.load = _orig_load

    _pipeline.to(torch.device(DEVICE))

    # ── Tuning: reduce false speaker-splits on consultation audio ──────────
    clustering_threshold = float(os.environ.get("PYANNOTE_CLUSTERING_THRESHOLD", "0.65"))
    min_duration_off = float(os.environ.get("PYANNOTE_MIN_DURATION_OFF", "0.097"))

    print(f"[diarization] Instantiating pyannote with clustering_threshold={clustering_threshold}, min_duration_off={min_duration_off}")
    _pipeline.instantiate({
        "segmentation": {
            "min_duration_off": min_duration_off,
        },
        "clustering": {
            "threshold": clustering_threshold,
        },
    })
    # ─────────────────────────────────────────────────────────────────────

    print(f"[diarization] pyannote ready on {DEVICE}.")
    return _pipeline


_vad_model = None
_vad_utils = None

def _load_silero_vad():
    global _vad_model, _vad_utils
    if _vad_model is not None:
        return _vad_model, _vad_utils
    
    print("[VAD] Loading Silero VAD model...")
    try:
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', 
            model='silero_vad', 
            force_reload=False, 
            trust_repo=True
        )
        _vad_model = model
        _vad_utils = utils
        print("[VAD] Silero VAD loaded successfully.")
    except Exception as e:
        print(f"[VAD] Error loading Silero VAD: {e}")
    return _vad_model, _vad_utils


def run_silero_vad(wav_path: str) -> list:
    """
    Run Silero VAD on a 16kHz mono WAV file.
    Returns a list of dicts: [{'start': float, 'end': float}] in seconds.
    """
    model, utils = _load_silero_vad()
    if model is None or utils is None:
        print("[VAD] Fallback: treating entire audio as speech.")
        try:
            import soundfile as sf
            audio, sr = sf.read(wav_path)
            duration = len(audio) / sr
            return [{"start": 0.0, "end": duration}]
        except Exception:
            return [{"start": 0.0, "end": 9999.0}]

    get_speech_timestamps, _, _, _, _ = utils
    
    try:
        wav, sr = torchaudio.load(wav_path)
        if sr != 16000:
            transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            wav = transform(wav)
            sr = 16000
        
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)
        
        with torch.no_grad():
            speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
            
        segments = []
        for ts in speech_timestamps:
            segments.append({
                "start": round(ts["start"] / 16000.0, 3),
                "end": round(ts["end"] / 16000.0, 3)
            })
        print(f"[VAD] Silero VAD found {len(segments)} speech regions.")
        return segments
    except Exception as e:
        print(f"[VAD] Silero VAD run failed: {e}. Fallback to entire audio.")
        try:
            import soundfile as sf
            audio, sr = sf.read(wav_path)
            return [{"start": 0.0, "end": len(audio) / sr}]
        except Exception:
            return [{"start": 0.0, "end": 9999.0}]


def intersect_with_vad(diar_segments: list, vad_segments: list) -> list:
    """
    Keep only portions of diarization segments that intersect with VAD speech regions.
    """
    filtered = []
    for d_seg in diar_segments:
        d_start = d_seg["start"]
        d_end = d_seg["end"]
        speaker = d_seg["speaker"]
        
        for v_seg in vad_segments:
            v_start = v_seg["start"]
            v_end = v_seg["end"]
            
            # Intersection
            overlap_start = max(d_start, v_start)
            overlap_end = min(d_end, v_end)
            
            if overlap_end - overlap_start > 0.05:  # Overlap greater than 50ms
                filtered.append({
                    "speaker": speaker,
                    "start": round(overlap_start, 3),
                    "end": round(overlap_end, 3)
                })
    return filtered


_sepformer = None

def _load_sepformer():
    global _sepformer
    if _sepformer is not None:
        return _sepformer
    try:
        from speechbrain.inference.separation import SepformerSeparation
        print("[SepFormer] Loading SepFormer wsj02mix...")
        os.makedirs("model_cache/sepformer", exist_ok=True)
        _sepformer = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-wsj02mix", 
            savedir="model_cache/sepformer"
        )
        print("[SepFormer] SepFormer loaded successfully.")
    except Exception as e:
        print(f"[SepFormer] Error loading SepFormer: {e}")
    return _sepformer


def separate_overlap(wav_path: str, start: float, end: float, tmp_dir: str) -> tuple:
    """
    Run SepFormer overlap source separation on a specific audio segment.
    Returns paths to the separated wav files (source_1.wav, source_2.wav).
    """
    model = _load_sepformer()
    if model is None:
        return None, None
    
    import soundfile as sf
    audio, sr = sf.read(wav_path)
    start_idx = int(start * sr)
    end_idx = int(end * sr)
    mix_clip = audio[start_idx:end_idx]
    
    mix_path = os.path.join(tmp_dir, f"mix_{start:.1f}_{end:.1f}.wav")
    sf.write(mix_path, mix_clip, sr)
    
    try:
        est_sources = model.separate_file(path=mix_path)
        src1_path = os.path.join(tmp_dir, f"sep_{start:.1f}_{end:.1f}_src1.wav")
        src2_path = os.path.join(tmp_dir, f"sep_{start:.1f}_{end:.1f}_src2.wav")
        
        sf.write(src1_path, est_sources[:, :, 0].detach().cpu().numpy().squeeze(), sr)
        sf.write(src2_path, est_sources[:, :, 1].detach().cpu().numpy().squeeze(), sr)
        
        return src1_path, src2_path
    except Exception as e:
        print(f"[SepFormer] Source separation failed: {e}")
        return None, None


def run_diarization(wav_path: str, num_speakers: int = None) -> list:
    """
    Run Silero VAD + pyannote diarization on a 16kHz mono WAV file.
    Returns list of dicts:
        [{"speaker": "SPEAKER_00", "start": 0.0, "end": 3.2}, ...]
    """
    # 1. Run Silero VAD to detect speech regions
    vad_segments = run_silero_vad(wav_path)

    # 2. Run pyannote diarization
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

    kwargs = {}
    if num_speakers is not None and num_speakers >= 1:
        kwargs["num_speakers"] = num_speakers
    diarization = pipeline(audio_input, **kwargs)

    raw_segments = []
    for turn, _, speaker_label in diarization.itertracks(yield_label=True):
        raw_segments.append({
            "speaker": speaker_label,
            "start":   round(turn.start, 3),
            "end":     round(turn.end,   3),
        })

    # 3. Speech region filtering
    filtered_segments = intersect_with_vad(raw_segments, vad_segments)
    filtered_segments.sort(key=lambda s: s["start"])
    
    print(f"[diarization] pyannote found {len(raw_segments)} segments, filtered to {len(filtered_segments)} using Silero VAD")
    return filtered_segments