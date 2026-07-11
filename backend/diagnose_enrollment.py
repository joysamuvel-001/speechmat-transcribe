"""
diagnose_enrollment.py
-----------------------
Run from backend/ directory:  python diagnose_enrollment.py

Checks two things using ONLY the existing .npy files (no model needed):

1. Within-speaker consistency: do booja_0, booja_1, booja_2 actually look
   like the same person? (Should be high, e.g. > 0.85)

2. Cross-speaker separation: is each speaker's centroid closer to their
   OWN other samples than to a different speaker's centroid?
   If a speaker's samples are closer to someone ELSE's centroid than to
   their own, that's the signature of a mislabeled enrollment (wrong
   name typed in for that recording).
"""

import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

ENROLLED_DIR = "enrolled_speakers"


def load_all():
    speakers = {}
    for f in sorted(os.listdir(ENROLLED_DIR)):
        if not f.endswith(".npy"):
            continue
        base = re.sub(r"_\d+$", "", f[:-4])
        emb = np.load(os.path.join(ENROLLED_DIR, f)).astype(np.float64)
        norm = np.linalg.norm(emb)
        if norm > 1e-9:
            emb = emb / norm
        speakers.setdefault(base, {})[f] = emb
    return speakers


def centroid(emb_dict):
    stacked = np.stack(list(emb_dict.values()))
    c = stacked.mean(axis=0)
    n = np.linalg.norm(c)
    return c / n if n > 1e-9 else c


def main():
    speakers = load_all()
    names = sorted(speakers.keys())
    print(f"Found speakers: {names}\n")

    centroids = {name: centroid(embs) for name, embs in speakers.items()}

    # 1. Within-speaker consistency
    print("=== Within-speaker sample consistency ===")
    for name, embs in speakers.items():
        files = list(embs.keys())
        vecs = list(embs.values())
        if len(vecs) < 2:
            print(f"{name}: only 1 sample, skipping")
            continue
        sims = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                s = cosine_similarity(vecs[i].reshape(1, -1), vecs[j].reshape(1, -1))[0][0]
                sims.append(s)
                print(f"  {files[i]} vs {files[j]}: {s:.3f}")
        print(f"  {name} avg pairwise similarity: {np.mean(sims):.3f}\n")

    # 2. Cross-speaker: is every sample closest to its OWN centroid?
    print("=== Per-sample: which centroid is it actually closest to? ===")
    mislabel_suspects = []
    for name, embs in speakers.items():
        for fname, emb in embs.items():
            scores = {
                other: float(cosine_similarity(emb.reshape(1, -1), centroids[other].reshape(1, -1))[0][0])
                for other in names
            }
            best = max(scores, key=scores.get)
            flag = "  <-- MISLABEL SUSPECT" if best != name else ""
            if best != name:
                mislabel_suspects.append(fname)
            print(f"  {fname} (labeled '{name}') closest to: {best} ({scores[best]:.3f}) | full: {scores}{flag}")

    print()
    if mislabel_suspects:
        print(f"⚠️  Possible mislabeled enrollment files: {mislabel_suspects}")
        print("These files are closer to a DIFFERENT speaker's centroid than to their own label.")
        print("Recommend: delete these specific files and re-enroll that sample.")
    else:
        print("✅ No mislabeling detected — every sample is closest to its own speaker's centroid.")
        print("If booja/snegha are still swapping during real transcription, it's likely")
        print("condition/channel drift between enrollment and test recordings, not mislabeling.")
        print("Try re-enrolling both fresh, immediately before your next test, same mic/setup.")


if __name__ == "__main__":
    main()