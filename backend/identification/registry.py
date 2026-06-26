

import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from identification.titanet import get_embedding

ENROLLED_DIR = os.path.join(os.path.dirname(__file__), "..", "enrolled_speakers")

# Lower = more lenient matches. Start at 0.65, raise to 0.72 once you
# have 3+ enrollment samples per person for best accuracy.
THRESHOLD = 0.65


def enroll_speaker(name: str, wav_path: str) -> dict:
    """
    Enroll a speaker. Call multiple times with the same name to average
    embeddings — accuracy improves significantly after 3+ samples.
    """
    os.makedirs(ENROLLED_DIR, exist_ok=True)
    new_emb = get_embedding(wav_path)
    save_path = os.path.join(ENROLLED_DIR, f"{name}.npy")

    if os.path.exists(save_path):
        existing = np.load(save_path)
        # Weighted average — newer enrollments weighted slightly higher
        averaged = (existing * 0.45 + new_emb * 0.55)
        averaged = averaged / (np.linalg.norm(averaged) + 1e-9)
        np.save(save_path, averaged)
        print(f"[registry] Updated enrollment for '{name}'")
    else:
        np.save(save_path, new_emb)
        print(f"[registry] New enrollment: '{name}'")

    count = len(list_enrolled())
    return {"name": name, "status": "enrolled", "total_enrolled": count}


def identify_speaker(wav_path: str, fallback_label: str = "Unknown") -> dict:
    enrolled = _load_all_enrolled()

    # No enrolled speakers at all → always Unknown
    if not enrolled:
        return {
            "name": "Unknown",
            "score": 0.0,
            "reason": "no enrolled speakers"
        }

    try:
        query_emb = get_embedding(wav_path)
    except Exception as e:
        print(f"[registry] Embedding failed: {e}")
        return {"name": "Unknown", "score": 0.0, "reason": str(e)}

    query_emb  = query_emb.reshape(1, -1)
    best_name  = fallback_label   # SPEAKER_xx — keeps turn separation
    best_score = 0.0

    for name, ref_emb in enrolled.items():
        score = float(cosine_similarity(query_emb, ref_emb.reshape(1, -1))[0][0])
        if score > best_score:
            best_score = score
            best_name  = name

    if best_score < THRESHOLD:
        return {
            "name":   "Unknown",        # ← Unknown when enrolled exist but no match
            "score":  round(best_score, 3),
            "reason": "below threshold"
        }

    return {"name": best_name, "score": round(best_score, 3)}


def list_enrolled() -> list:
    os.makedirs(ENROLLED_DIR, exist_ok=True)
    return [f.replace(".npy", "") for f in os.listdir(ENROLLED_DIR) if f.endswith(".npy")]


def delete_speaker(name: str) -> dict:
    path = os.path.join(ENROLLED_DIR, f"{name}.npy")
    if os.path.exists(path):
        os.remove(path)
        return {"name": name, "status": "deleted"}
    return {"name": name, "status": "not_found"}


def _load_all_enrolled() -> dict:
    os.makedirs(ENROLLED_DIR, exist_ok=True)
    result = {}
    for f in os.listdir(ENROLLED_DIR):
        if f.endswith(".npy"):
            result[f.replace(".npy", "")] = np.load(
                os.path.join(ENROLLED_DIR, f)
            )
    return result