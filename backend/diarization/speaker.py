
def process_diarization(raw_segments: list) -> list:
    if not raw_segments:
        return []

    cleaned = []
    for seg in raw_segments:
        # --- FIX: handle raw Sortformer strings "start end SPEAKER_xx" ---
        if isinstance(seg, str):
            parts = seg.strip().split()
            if len(parts) < 3:
                continue
            start = float(parts[0])
            end   = float(parts[1])
            label = parts[2]
        else:
            # already a dict
            label = seg.get("speaker") or seg.get("label") or "SPEAKER_00"
            start = float(seg.get("start", 0.0))
            end   = float(seg.get("end",   0.0))
        # ------------------------------------------------------------------

        if end <= start:
            continue

        if cleaned and cleaned[-1]["speaker"] == label:
            gap = start - cleaned[-1]["end"]
            if gap < 0.4:
                cleaned[-1]["end"] = max(cleaned[-1]["end"], end)
                continue

        cleaned.append({"speaker": label, "start": start, "end": end})

    return cleaned