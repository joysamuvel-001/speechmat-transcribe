
import os
from typing import List, Tuple

import torch
import numpy as np
from scipy.io import wavfile

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "omi-health/omi-med-stt-v1"
DEFAULT_LOCAL_PATH = "./model_cache/omi-med-stt-v1.nemo"
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH") or DEFAULT_LOCAL_PATH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Lazy model load
# ---------------------------------------------------------------------------

_asr_model = None


def _load_model():
    global _asr_model
    if _asr_model is not None:
        return _asr_model

    print("[asr] Loading ASR model…")
    import nemo.collections.asr as nemo_asr

    if os.path.exists(LOCAL_MODEL_PATH):
        print(f"[asr] Restoring from local checkpoint: {LOCAL_MODEL_PATH}")
        model = nemo_asr.models.ASRModel.restore_from(LOCAL_MODEL_PATH)
    else:
        print(f"[asr] Local checkpoint not found — downloading {MODEL_NAME}")
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)

    model.eval()
    if DEVICE == "cuda":
        model = model.cuda()
        print(f"[asr] Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[asr] No GPU detected — running on CPU (slower).")

    _asr_model = model
    return _asr_model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

Segment = Tuple[float, float, str]  # (start, end, role)

MAX_CHUNK_SECONDS = 12.0
CHUNK_OVERLAP_SECONDS = 0.5  # small overlap so words at chunk boundaries aren't clipped
PAD_SECONDS = 0.2            # small buffer to avoid clipping words at segment edges


def _split_long_segment(seg: dict, max_duration: float = MAX_CHUNK_SECONDS):
    """Break a long segment into smaller (start, end) sub-chunks for transcription."""
    duration = seg["end"] - seg["start"]
    if duration <= max_duration:
        return [(seg["start"], seg["end"])]

    chunks = []
    cur_start = seg["start"]
    while cur_start < seg["end"]:
        cur_end = min(cur_start + max_duration, seg["end"])
        chunks.append((cur_start, cur_end))
        if cur_end >= seg["end"]:
            break
        cur_start = cur_end - CHUNK_OVERLAP_SECONDS
    return chunks


def _merge_chunk_texts(chunks_text: list) -> str:
    """
    Join per-chunk transcripts while removing word-level duplication caused
    by CHUNK_OVERLAP_SECONDS — the overlapping audio region gets transcribed
    by both adjacent chunks, so we trim the repeated words at the seam
    instead of naively concatenating (which produces stutters like
    "once daily. once daily.").
    """
    if not chunks_text:
        return ""

    merged_words = chunks_text[0].split()
    for nxt_text in chunks_text[1:]:
        nxt_words = nxt_text.split()
        max_check = min(len(merged_words), len(nxt_words), 8)  # overlap is short, no need to check far
        best_k = 0
        for k in range(max_check, 0, -1):
            tail = [w.lower().strip(".,!?") for w in merged_words[-k:]]
            head = [w.lower().strip(".,!?") for w in nxt_words[:k]]
            if tail == head:
                best_k = k
                break
        merged_words.extend(nxt_words[best_k:])

    return " ".join(merged_words)


def transcribe_segments(
    wav_path: str,
    labeled_segments: List[Segment],
    tmp_dir: str,
) -> List[dict]:
    """
    Slice wav_path into one or more clips per labeled segment (splitting
    long segments into smaller chunks), run the ASR model on all clips in
    a single batched call, and return a list of turn dicts:

        [{"speaker": "Doctor", "text": "...", "start": 0.0, "end": 3.2}, ...]

    Turns where the ASR produces no text (silence / non-speech) are skipped.
    """
    if not labeled_segments:
        return []

    sample_rate, audio_np = wavfile.read(wav_path)  # int16, mono, 16kHz

    # Build a flat list of (segment_index, path) for every chunk across
    # every segment, so long segments become multiple ASR calls internally
    # but still produce ONE merged text per original turn.
    chunk_plan = []
    for i, seg in enumerate(labeled_segments):
        for (c_start, c_end) in _split_long_segment(seg):
            start_s = max(0, int((c_start - PAD_SECONDS) * sample_rate))
            end_s   = min(len(audio_np), int((c_end + PAD_SECONDS) * sample_rate))
            clip    = audio_np[start_s:end_s]
            path    = os.path.join(tmp_dir, f"seg_{i}_{len(chunk_plan)}.wav")
            wavfile.write(path, sample_rate, clip)
            chunk_plan.append((i, path))

    segment_paths = [p for (_, p) in chunk_plan]

    model = _load_model()

    # Batched ASR inference
    with torch.no_grad():
        outputs = model.transcribe(segment_paths)

    # Re-assemble: join all chunk texts belonging to the same original segment
    texts_by_segment = {}
    for (seg_idx, _), output in zip(chunk_plan, outputs):
        text = output.text if hasattr(output, "text") else output
        text = (text or "").strip()
        texts_by_segment.setdefault(seg_idx, []).append(text)

    conversation: List[dict] = []
    for i, seg in enumerate(labeled_segments):
        chunk_texts = [t for t in texts_by_segment.get(i, []) if t]
        full_text = _merge_chunk_texts(chunk_texts).strip()
        if not full_text:
            continue
        conversation.append({
            "speaker":     seg["speaker"],       # TitaNet-identified name (or "Unknown")
            "diarized_as": seg.get("diarized_as"),
            "similarity":  seg.get("similarity"),
            "text":        full_text,
            "start":       round(seg["start"], 2),
            "end":         round(seg["end"], 2),
        })

    return conversation