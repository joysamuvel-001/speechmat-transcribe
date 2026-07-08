
SAME_SPEAKER_MERGE_GAP = 1.2

def process_diarization(raw_segments: list) -> list:
    if not raw_segments:
        return []

    cleaned = []
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
        # ------------------------------------------------------------------

        if end <= start:
            continue

        if cleaned and cleaned[-1]["speaker"] == label:
            gap = start - cleaned[-1]["end"]
            if gap < SAME_SPEAKER_MERGE_GAP:
                cleaned[-1]["end"] = max(cleaned[-1]["end"], end)
                continue

        cleaned.append({"speaker": label, "start": start, "end": end})

    return cleaned