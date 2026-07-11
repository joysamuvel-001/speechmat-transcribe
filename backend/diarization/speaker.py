SAME_SPEAKER_MERGE_GAP = 1.2

def process_diarization(raw_segments: list) -> list:
    if not raw_segments:
        return []

    parsed = []
    for seg in raw_segments:
        # --- FIX: handle Sortformer strings, tuples, AND dicts ---
        if isinstance(seg, str):
            parts = seg.strip().split()
            if len(parts) < 3:
                continue
            try:
                start = float(parts[0])
                end   = float(parts[1])
            except ValueError:
                print(f"[diarization] skipping malformed segment: {seg!r}")
                continue
            label = parts[2]
        elif isinstance(seg, (tuple, list)):
            print(f"[diarization] DEBUG raw tuple/list segment: {seg!r}")
            start, end, label = float(seg[0]), float(seg[1]), seg[2]
        elif isinstance(seg, dict):
            label = seg.get("speaker") or seg.get("label") or "SPEAKER_00"
            start = float(seg.get("start", 0.0))
            end   = float(seg.get("end",   0.0))
        else:
            print(f"[diarization] unrecognized segment type: {type(seg)} -> {seg!r}")
            continue
        # ------------------------------------------------------------------

        if end <= start:
            continue

        parsed.append({"speaker": label, "start": start, "end": end})

    # --- FIX: sort chronologically before merging ---
    # Upstream diarization output is not guaranteed to be time-ordered
    # (e.g. it can be grouped by speaker). Merging on list-adjacency
    # instead of time-adjacency silently collapses turns from opposite
    # ends of the conversation into one another.
    parsed.sort(key=lambda s: s["start"])

    cleaned = []
    for seg in parsed:
        if cleaned and cleaned[-1]["speaker"] == seg["speaker"]:
            gap = seg["start"] - cleaned[-1]["end"]
            if gap < SAME_SPEAKER_MERGE_GAP:
                cleaned[-1]["end"] = max(cleaned[-1]["end"], seg["end"])
                continue

        cleaned.append({"speaker": seg["speaker"], "start": seg["start"], "end": seg["end"]})

    return cleaned