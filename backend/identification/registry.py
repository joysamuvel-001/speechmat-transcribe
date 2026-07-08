"""
identification/registry.py
---------------------------
Stores ALL enrollment embeddings separately (not averaged).
Scores a query against every stored embedding and takes the max.
This makes identification robust across server restarts because
one good enrollment session is enough to anchor the identity.
"""

import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from identification.titanet import get_embedding
from identification.titanet import get_embedding_windowed as get_embedding

ENROLLED_DIR = os.path.join(os.path.dirname(__file__), "..", "enrolled_speakers")
THRESHOLD    = 0.65


def _safe_name(name: str) -> str:
    clean = re.sub(r"[^\w\- ]", "", name or "").strip()
    if not clean:
        raise ValueError(f"Invalid speaker name: {name!r}")
    return clean


def enroll_speaker(name: str, wav_path: str) -> dict:
    """
    Save each enrollment as a separate .npy file:
        enrolled_speakers/<name>_0.npy
        enrolled_speakers/<name>_1.npy
        ...
    Multiple files = more robust matching across sessions.
    """
    name = _safe_name(name)
    os.makedirs(ENROLLED_DIR, exist_ok=True)

    new_emb = get_embedding(wav_path)

    # Find next available index for this speaker
    existing = _get_speaker_files(name)
    idx      = len(existing)
    save_path = os.path.join(ENROLLED_DIR, f"{name}_{idx}.npy")

    np.save(save_path, new_emb)
    total = idx + 1
    print(f"[registry] Enrolled '{name}' sample {total} → {save_path}")

    return {
        "name":            name,
        "status":          "enrolled",
        "sample_count":    total,
        "total_enrolled":  len(list_enrolled()),
    }


def identify_speaker(wav_path: str, fallback_label: str = "Unknown") -> dict:
    """
    Score query against EVERY stored embedding for every enrolled speaker.
    Takes the maximum score per speaker, then picks the best speaker.
    This way a single good enrollment session from any previous run
    can still produce a match.
    """
    all_embeddings = _load_all_enrolled()   # { name: [emb0, emb1, ...] }

    if not all_embeddings:
        return {"name": fallback_label, "score": 0.0, "reason": "no enrolled speakers"}

    try:
        query_emb = get_embedding(wav_path).reshape(1, -1)
    except Exception as e:
        print(f"[registry] Embedding failed: {e}")
        return {"name": fallback_label, "score": 0.0, "reason": str(e)}

    best_name  = fallback_label
    best_score = 0.0

    for name, emb_list in all_embeddings.items():
        # Score against all stored embeddings for this speaker
        for ref_emb in emb_list:
            score = float(
                cosine_similarity(query_emb, ref_emb.reshape(1, -1))[0][0]
            )
            if score > best_score:
                best_score = score
                best_name  = name

    print(f"[registry] best match: {best_name} (score={best_score:.3f}, threshold={THRESHOLD})")

    if best_score < THRESHOLD:
        return {
            "name":   fallback_label,
            "score":  round(best_score, 3),
            "reason": "below threshold",
        }

    return {"name": best_name, "score": round(best_score, 3)}


def list_enrolled() -> list:
    """Return unique speaker names (deduplicated across multiple sample files)."""
    os.makedirs(ENROLLED_DIR, exist_ok=True)
    names = set()
    for f in os.listdir(ENROLLED_DIR):
        if f.endswith(".npy"):
            # Strip trailing _0, _1, _2 suffix to get base name
            base = re.sub(r"_\d+$", "", f[:-4])
            names.add(base)
    return sorted(names)


def delete_speaker(name: str) -> dict:
    """Delete all enrollment files for a speaker."""
    name  = _safe_name(name)
    files = _get_speaker_files(name)
    if not files:
        return {"name": name, "status": "not_found"}
    for f in files:
        os.remove(f)
    print(f"[registry] Deleted {len(files)} enrollment(s) for '{name}'")
    return {"name": name, "status": "deleted", "files_removed": len(files)}


def _get_speaker_files(name: str) -> list:
    """Return all .npy file paths for a given speaker name."""
    os.makedirs(ENROLLED_DIR, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(name)}_\d+\.npy$")
    return sorted(
        os.path.join(ENROLLED_DIR, f)
        for f in os.listdir(ENROLLED_DIR)
        if pattern.match(f)
    )


def _load_all_enrolled() -> dict:
    """
    Returns { speaker_name: [emb_array_0, emb_array_1, ...] }
    Loads all individual enrollment files per speaker.
    """
    os.makedirs(ENROLLED_DIR, exist_ok=True)
    result = {}

    for f in os.listdir(ENROLLED_DIR):
        if not f.endswith(".npy"):
            continue
        base = re.sub(r"_\d+$", "", f[:-4])   # "joy_2" → "joy"
        path = os.path.join(ENROLLED_DIR, f)
        emb  = np.load(path).astype(np.float64)

        # Re-normalise on load in case file was saved before norm fix
        norm = np.linalg.norm(emb)
        if norm > 1e-9:
            emb = emb / norm

        if base not in result:
            result[base] = []
        result[base].append(emb)

    return result