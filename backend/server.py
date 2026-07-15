import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import shutil
import socket
import tempfile
import uuid
import torch
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from audio_utils.converter import convert_to_wav
from identification.titanet import get_embedding_windowed 
from identification.registry import (
    enroll_speaker, identify_speaker,
    identify_cluster_embedding,
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
async def transcribe(
    audio: UploadFile = File(...),
    ground_truth_diarization: str = Form(None),
    num_speakers: int = Form(None),
    language: str = Form("en")
):
    """
    Pipeline:
      1. Convert WebM -> 16kHz mono WAV
      2. Run Speechmatics batch-wise transcription & diarization (using selected language)
      3. Run TitaNet speaker identification to map Speaker 1/2 to enrolled profiles
    """
    tmp_dir  = tempfile.mkdtemp()
    wav_path = f"{tmp_dir}/{uuid.uuid4()}.wav"
    try:
        audio_bytes = await audio.read()
        print(f"\n[server] === NEW TRANSCRIBE REQUEST RECEIVED ===")
        print(f"[server] Audio file payload size: {len(audio_bytes)} bytes")
        print(f"[server] Target language select: '{language}'")
        print(f"[server] Target num_speakers select: {num_speakers}")

        print(f"[server] Converting input audio bytes to 16kHz mono WAV format...")
        convert_to_wav(audio_bytes, wav_path)
        print(f"[server] Audio conversion complete. Saved temporary WAV to: {wav_path}")

        # --- Speechmatics Pipeline Integration (Pauses: PyAnnote, TitaNet, Omi Med STT, MedGemma) ---
        from speechmatics_pipeline import run_speechmatics_pipeline, parse_speechmatics_results
        print(f"[server] Delegating transcription & diarization to Speechmatics batch service...")
        speechmatics_res = await run_speechmatics_pipeline(wav_path, num_speakers=num_speakers, language=language)
        
        results = speechmatics_res.get("results", [])
        print(f"[server] Speechmatics returned raw response. Parsing {len(results)} elements into speaker turns...")
        conversation = parse_speechmatics_results(results)
        
        print(f"[server] Performing TitaNet speaker identification on parsed dialogue turns...")
        conversation = _identify_segments(wav_path, conversation, tmp_dir)
        print(f"[server] Post-processing complete. Total final turns: {len(conversation)}")
        
        correction_applied = False
        full_text = " ".join(t["text"] for t in conversation)
        print(f"[server] Returning final transcription payload to client. Character count: {len(full_text)}")
        print(f"[server] === TRANSCRIBE REQUEST PROCESS COMPLETED ===\n")

        # --- (Paused Old Pipeline) ---
        # if ground_truth_diarization:
        #     import json
        #     raw_segments = json.loads(ground_truth_diarization)
        #     labeled_segments = process_diarization(raw_segments)
        #     print(f"[server] Using ground-truth diarization with {len(labeled_segments)} segments")
        # else:
        #     raw_segments    = run_diarization(wav_path, num_speakers=num_speakers)
        #     labeled_segments = process_diarization(raw_segments)
        #
        # print(f"[server] {len(labeled_segments)} segments after diarization")
        #
        # labeled_segments = _identify_segments(wav_path, labeled_segments, tmp_dir)
        #
        # conversation = transcribe_segments(wav_path, labeled_segments, tmp_dir)
        #
        # # Correction is best-effort: a RunPod failure must never lose a
        # # transcript that diarization + ASR already produced.
        # correction_applied = False
        # if conversation:
        #     print(f"[server] Sending {len(conversation)} turns to MedGemma (single request)")
        #     try:
        #         conversation = correct_transcript(conversation)["corrected_conversation"]
        #         correction_applied = True
        #         print(f"[server] MedGemma correction complete — {len(conversation)} turns returned")
        #     except Exception as corr_exc:
        #         print(f"[server] MedGemma correction failed ({corr_exc}) — returning uncorrected transcript")
        # else:
        #     print("[server] No speech detected — skipping MedGemma correction")
        #
        # full_text = " ".join(t["text"] for t in conversation)

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
        "correction":  bool(os.environ.get("RUNPOD_API_KEY", "")),
    }


# ── Internal helpers ────────────────────────────────────────────────────────

def _identify_segments(wav_path: str, segments: list, tmp_dir: str) -> list:
    """
    Perform cluster-based speaker identification.
    
    1. Group all segments by their diarized cluster label (e.g. "SPEAKER_00").
    2. Extract TitaNet embeddings for segments in each cluster that are >= 1.2s.
       If a cluster has no segments >= 1.2s, use all available segments of that cluster.
    3. Average the embeddings to compute a single high-quality centroid embedding for the cluster.
    4. Match this cluster centroid against enrolled speakers.
    5. Propagate the identified speaker name (or raw label if unidentified) to all segments of that cluster.
    """
    if not segments:
        return []

    import numpy as np
    from identification.titanet import get_embedding_windowed as get_embedding

    audio, sr = sf.read(wav_path)

    # Step 1: Group segments by pyannote speaker label (diarized_as)
    clusters = {}
    for seg in segments:
        diarized_label = seg["speaker"]  # originally "SPEAKER_00", "SPEAKER_01", etc.
        seg["diarized_as"] = diarized_label
        clusters.setdefault(diarized_label, []).append(seg)

    # Step 2 & 3: For each cluster, extract embeddings and calculate the centroid
    cluster_identities = {} # maps diarized_label -> {"name": str, "score": float, "identified": bool}

    for diarized_label, cluster_segs in clusters.items():
        # Select segments for extracting speaker signature (prefer >= 1.2 seconds)
        signature_segs = [s for s in cluster_segs if (s["end"] - s["start"]) >= 1.2]
        if not signature_segs:
            # Fallback to all segments if they are all short
            signature_segs = cluster_segs

        embeddings = []
        for s in signature_segs:
            start_s = int(s["start"] * sr)
            end_s   = int(s["end"]   * sr)
            if end_s - start_s < int(0.1 * sr):  # Too short to slice safely, skip
                continue
            seg_wav = os.path.join(tmp_dir, f"sig_{uuid.uuid4()}.wav")
            try:
                sf.write(seg_wav, audio[start_s:end_s], sr)
                emb = get_embedding(seg_wav)
                embeddings.append(emb)
            except Exception as e:
                print(f"[server] Error getting signature embedding: {e}")
            finally:
                if os.path.exists(seg_wav):
                    os.remove(seg_wav)

        if not embeddings:
            # No embeddings could be extracted for this cluster
            cluster_identities[diarized_label] = {
                "name": diarized_label,
                "score": 0.0,
                "identified": False
            }
            continue

        # Compute centroid embedding
        centroid = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-9:
            centroid = centroid / norm

        # Step 4: Identify cluster centroid
        result = identify_cluster_embedding(centroid, fallback_label=diarized_label)
        identified = result["name"] not in (diarized_label, "Unknown")
        cluster_identities[diarized_label] = {
            "name": result["name"],
            "score": result.get("score", 0.0),
            "identified": identified
        }
        print(f"[server] Cluster '{diarized_label}' identified as '{result['name']}' (score: {result.get('score', 0.0):.3f}, identified: {identified})")

    # Conflict Resolution: Ensure each enrolled speaker is mapped to at most one cluster.
    assigned_groups = {}  # maps enrolled_name -> list of (diarized_label, score)
    for diarized_label, id_info in cluster_identities.items():
        if id_info["identified"]:
            assigned_groups.setdefault(id_info["name"], []).append((diarized_label, id_info["score"]))

    for enrolled_name, candidates in assigned_groups.items():
        if len(candidates) > 1:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_label = candidates[0][0]
            print(f"[server] Conflict for '{enrolled_name}': multiple clusters matched {candidates}. Keeping cluster '{best_label}'")
            for label, score in candidates[1:]:
                cluster_identities[label]["name"] = label
                cluster_identities[label]["identified"] = False
                cluster_identities[label]["score"] = 0.0

    # Step 5: Propagate cluster identities back to segments
    for seg in segments:
        diarized_label = seg["diarized_as"]
        identity = cluster_identities[diarized_label]
        seg["speaker"] = identity["name"]
        seg["similarity"] = identity["score"]
        seg["identified"] = identity["identified"]

    return segments





def _get_available_port(preferred_port: int = 8000) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("0.0.0.0", preferred_port))
            return preferred_port
        except OSError:
            pass

    for port in range(preferred_port + 1, preferred_port + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue

    raise RuntimeError(f"No available port found near {preferred_port}")


if __name__ == "__main__":
    import uvicorn

    port = _get_available_port(8000)
    print(f"[server] Starting on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)