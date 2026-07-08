import os
import shutil
import tempfile
import uuid
import torch
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from correction.medgemma import correct_transcript
from audio_utils.converter import convert_to_wav
from diarization.model import run_diarization
from diarization.speaker import process_diarization
from identification.titanet import get_embedding_windowed  # add this import
from transcription.asr import transcribe_segments
from identification.registry import (
    enroll_speaker, identify_speaker,
    list_enrolled, delete_speaker
)

app = FastAPI(title="MedTranscribe API", version="2.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ──────────────────────────────────────────────────────────────────

@app.post("/api/enroll")
async def enroll(audio: UploadFile = File(...), name: str = Form(...)):
    tmp_dir  = tempfile.mkdtemp()
    wav_path = f"{tmp_dir}/{uuid.uuid4()}.wav"
    try:
        convert_to_wav(await audio.read(), wav_path)
        result = enroll_speaker(name, wav_path)
        return result
    except Exception as exc:
        print(f"[enroll] error: {exc}")
        return JSONResponse(status_code=500, content={"error": str(exc)})
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/api/speakers")
async def get_speakers():
    return {"speakers": list_enrolled()}


@app.delete("/api/speakers/{name}")
async def remove_speaker(name: str):
    return delete_speaker(name)


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """
    Pipeline:
      1. Convert WebM → 16kHz mono WAV
      2. Sortformer diarization  →  raw SPEAKER_xx segments
      3. process_diarization     →  merge close same-speaker segments (no renaming)
      4. TitaNet identification  →  replace SPEAKER_xx with real name if enrolled
      5. merge_by_identity       →  merge consecutive same-person turns (with gap guard)
      6. Nemotron ASR            →  transcribe each turn
    """
    tmp_dir  = tempfile.mkdtemp()
    wav_path = f"{tmp_dir}/{uuid.uuid4()}.wav"
    try:
        audio_bytes = await audio.read()
        print(f"[server] received {len(audio_bytes)} bytes")

        convert_to_wav(audio_bytes, wav_path)

        raw_segments    = run_diarization(wav_path)
        labeled_segments = process_diarization(raw_segments)

        print(f"[server] {len(labeled_segments)} segments after diarization")

        labeled_segments = _identify_segments(wav_path, labeled_segments, tmp_dir)
        labeled_segments = _smooth_short_unknowns(labeled_segments)
        labeled_segments = merge_by_identity(labeled_segments)

        print(f"[server] {len(labeled_segments)} segments after identity merge")

        conversation = transcribe_segments(wav_path, labeled_segments, tmp_dir)

        # Correction is best-effort: a RunPod failure must never lose a
        # transcript that diarization + ASR already produced.
        correction_applied = False
        if conversation:
            print(f"[server] Sending {len(conversation)} turns to MedGemma (single request)")
            try:
                conversation = correct_transcript(conversation)["corrected_conversation"]
                correction_applied = True
                print(f"[server] MedGemma correction complete — {len(conversation)} turns returned")
            except Exception as corr_exc:
                print(f"[server] MedGemma correction failed ({corr_exc}) — returning uncorrected transcript")
        else:
            print("[server] No speech detected — skipping MedGemma correction")

        full_text = " ".join(t["text"] for t in conversation)

        return {
            "text":               full_text,
            "conversation":       conversation,
            "correction_applied": correction_applied,
        }

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(exc)})
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/api/health")
async def health():
    return {
        "status":      "ok",
        "gpu":         torch.cuda.is_available(),
        "device":      "cuda" if torch.cuda.is_available() else "cpu",
        "speakers":    list_enrolled(),
        "correction":  bool(os.environ.get("HF_API_TOKEN", "")),
    }


# ── Internal helpers ────────────────────────────────────────────────────────

def _identify_segments(wav_path: str, segments: list, tmp_dir: str) -> list:
    """
    Slice each segment, extract TitaNet embedding, match against enrolled speakers.

    Critical change: pass seg["speaker"] (which is "SPEAKER_00" etc.) as
    fallback_label so unenrolled speakers keep their unique Sortformer ID
    rather than all collapsing to "Unknown".

    Minimum segment length: 1.0s (was 1.5s — allows short turns to be identified)
    """
    audio, sr = sf.read(wav_path)

    for seg in segments:
        start    = seg.get("start", 0.0)
        end      = seg.get("end",   0.0)
        duration = end - start
        diarized_label = seg["speaker"]  # e.g. "SPEAKER_00"

        seg["diarized_as"] = diarized_label

        if duration < 1.2:
            # Too short — keep Sortformer label, mark as not identified
            seg["similarity"] = 0.0
            seg["identified"] = False
            continue

        # Slice audio
        start_s = int(start * sr)
        end_s   = int(end   * sr)
        seg_wav = f"{tmp_dir}/seg_{uuid.uuid4()}.wav"
        sf.write(seg_wav, audio[start_s:end_s], sr)

        # TitaNet match — pass Sortformer ID as fallback
        result = identify_speaker(seg_wav, fallback_label=diarized_label)

        seg["speaker"]    = result["name"]        # real name OR "SPEAKER_xx"
        seg["similarity"] = result.get("score", 0.0)
        seg["identified"] = result["name"] not in (diarized_label, "Unknown")

    return segments

def _smooth_short_unknowns(segments: list, max_short_duration: float = 2.5, max_gap: float = 2.0) -> list:
    if len(segments) < 2:
        return segments

    # ── Edge case: first segment ──
    first = segments[0]
    duration = first["end"] - first["start"]
    if not first.get("identified") and duration <= max_short_duration:
        nxt = segments[1]
        gap = nxt["start"] - first["end"]
        if nxt.get("identified") and gap <= max_gap:
            print(f"[server] Smoothing first segment '{first['speaker']}' ({duration:.1f}s) -> '{nxt['speaker']}' via next-only")
            first["speaker"]    = nxt["speaker"]
            first["identified"] = True
            first["smoothed"]   = True

    # ── Middle segments (unchanged) ──
    for i in range(1, len(segments) - 1):
        seg = segments[i]
        duration = seg["end"] - seg["start"]
        if seg.get("identified") or duration > max_short_duration:
            continue

        prev_seg, next_seg = segments[i - 1], segments[i + 1]
        if (prev_seg["speaker"] == next_seg["speaker"]
                and prev_seg.get("identified") and next_seg.get("identified")):
            gap_before = seg["start"] - prev_seg["end"]
            gap_after  = next_seg["start"] - seg["end"]
            if gap_before <= max_gap and gap_after <= max_gap:
                print(f"[server] Smoothing '{seg['speaker']}' ({duration:.1f}s) -> '{prev_seg['speaker']}' via neighbors")
                seg["speaker"]    = prev_seg["speaker"]
                seg["identified"] = True
                seg["smoothed"]   = True

    # ── Edge case: last segment ──
    last = segments[-1]
    duration = last["end"] - last["start"]
    if not last.get("identified") and duration <= max_short_duration:
        prev = segments[-2]
        gap = last["start"] - prev["end"]
        if prev.get("identified") and gap <= max_gap:
            print(f"[server] Smoothing last segment '{last['speaker']}' ({duration:.1f}s) -> '{prev['speaker']}' via prev-only")
            last["speaker"]    = prev["speaker"]
            last["identified"] = True
            last["smoothed"]   = True

    return segments


def merge_by_identity(segments: list, max_gap: float = 1.5) -> list:
    """
    Merge consecutive segments where the SAME identified speaker continues.

    Rules:
      1. Only merge if speaker names match exactly
      2. Only merge if the gap between segments is <= max_gap seconds
         (prevents merging across long pauses — different thoughts/questions)
      3. Never merge two SPEAKER_xx labels together unless they are identical
         (i.e. Sortformer already said they're the same person)
      4. Keep the highest similarity score when merging
    """
    if not segments:
        return []

    merged = [dict(segments[0])]

    for seg in segments[1:]:
        last = merged[-1]
        gap  = seg["start"] - last["end"]

        same_speaker = seg["speaker"] == last["speaker"]
        gap_ok       = gap <= max_gap

        if same_speaker and gap_ok:
            last["end"]        = max(last["end"], seg["end"])
            last["similarity"] = max(
                last.get("similarity", 0.0),
                seg.get("similarity", 0.0)
            )
        else:
            merged.append(dict(seg))

    return merged


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)