"""
identification/titanet.py
--------------------------
TitaNet Large speaker embedding extractor.
Forces deterministic CPU inference so stored embeddings
match across server restarts.
"""

import numpy as np
import torch
import soundfile as sf
import os

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model

    # Force deterministic behavior — critical so embeddings are
    # consistent across server restarts on CPU
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("[titanet] Loading TitaNet Large...")
    import nemo.collections.asr as nemo_asr

    _model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
        "nvidia/speakerverification_en_titanet_large"
    )
    _model.eval()

    # Keep on CPU — consistent with rest of pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = _model.to(device)
    print(f"[titanet] Loaded on {device}")
    return _model


def get_embedding(wav_path: str) -> np.ndarray:
    """
    Extract a 192-dim L2-normalised TitaNet speaker embedding.
    Audio must be 16kHz mono WAV.
    """
    model = _load_model()

    audio, sr = sf.read(wav_path)

    if sr != 16000:
        raise ValueError(f"Expected 16kHz, got {sr}Hz")

    # Ensure float32 mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    # Minimum length guard — TitaNet needs at least 0.5s
    min_samples = int(0.5 * sr)
    if len(audio) < min_samples:
        audio = np.pad(audio, (0, min_samples - len(audio)))

    audio_tensor  = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    length_tensor = torch.tensor([audio_tensor.shape[1]], dtype=torch.long)

    device = next(model.parameters()).device
    audio_tensor  = audio_tensor.to(device)
    length_tensor = length_tensor.to(device)

    # Deterministic inference — no dropout, no randomness
    with torch.inference_mode():
        torch.manual_seed(42)
        _, emb = model.forward(
            input_signal=audio_tensor,
            input_signal_length=length_tensor,
        )

    emb_np = emb.squeeze().cpu().numpy().astype(np.float64)

    # L2 normalise — makes cosine similarity = dot product
    norm = np.linalg.norm(emb_np)
    if norm > 1e-9:
        emb_np = emb_np / norm

    return emb_np


def get_embedding_windowed(wav_path: str, window_sec: float = 3.0, hop_sec: float = 1.5) -> np.ndarray:
    """
    For segments longer than one window, extract embeddings from overlapping
    windows and average them (then re-normalise). More robust than a single
    embedding over a long, possibly non-uniform segment.
    """
    import soundfile as sf
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration = len(audio) / sr

    if duration <= window_sec:
        return get_embedding(wav_path)  # unchanged path for short clips

    embeddings = []
    start = 0.0
    while start < duration:
        end = min(start + window_sec, duration)
        if end - start < 1.0:  # trailing sliver too short, skip
            break
        chunk = audio[int(start * sr):int(end * sr)]
        tmp_path = wav_path + f".win_{start:.1f}.wav"
        sf.write(tmp_path, chunk, sr)
        try:
            embeddings.append(get_embedding(tmp_path))
        finally:
            os.remove(tmp_path)
        start += hop_sec

    if not embeddings:
        return get_embedding(wav_path)

    avg = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(avg)
    return avg / norm if norm > 1e-9 else avg