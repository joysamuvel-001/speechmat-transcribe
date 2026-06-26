
import numpy as np
import torch
import soundfile as sf

_model = None  # singleton


def _load_model():
    global _model
    if _model is None:
        print("[titanet] Loading TitaNet Large...")
        import nemo.collections.asr as nemo_asr
        _model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            "nvidia/speakerverification_en_titanet_large"
        )
        _model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = _model.to(device)
        print(f"[titanet] Loaded on {device}")
    return _model


def get_embedding(wav_path: str) -> np.ndarray:
    """
    Extract a 192-dim TitaNet speaker embedding from a 16kHz mono WAV file.
    Returns a normalized numpy array.
    """
    model = _load_model()

    # TitaNet expects 16kHz mono — your converter.py already ensures this
    audio, sr = sf.read(wav_path)
    if sr != 16000:
        raise ValueError(f"Expected 16kHz audio, got {sr}Hz. Run through converter first.")

    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    length_tensor = torch.tensor([audio_tensor.shape[1]], dtype=torch.long)

    device = next(model.parameters()).device
    audio_tensor = audio_tensor.to(device)
    length_tensor = length_tensor.to(device)

    with torch.no_grad():
        _, emb = model.forward(
            input_signal=audio_tensor,
            input_signal_length=length_tensor
        )

    emb_np = emb.squeeze().cpu().numpy()
    # L2 normalize for cosine similarity
    emb_np = emb_np / (np.linalg.norm(emb_np) + 1e-9)
    return emb_np