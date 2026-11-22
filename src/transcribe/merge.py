"""Rebuild one transcript from independently transcribed, overlapping chunks."""

from __future__ import annotations


def _midpoint(segment: dict) -> float | None:
    start, end = segment.get("start"), segment.get("end")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        return (start + end) / 2
    return None


def merge_chunks(chunk_results: list[dict], overlap: float) -> dict:
    """Shift timings to the original recording and drop text duplicated in overlaps.

    Adjacent chunks share `overlap` seconds. The cut is the middle of that shared window:
    a segment is kept from the chunk on whose side of the cut its midpoint falls. Speaker
    labels are only meaningful inside one chunk, so they are prefixed with the chunk index
    and never merged across chunks.
    """
    ordered = sorted(chunk_results, key=lambda item: item["index"])
    segments: list[dict] = []
    texts: list[str] = []
    for position, chunk in enumerate(ordered):
        result, offset = chunk["result"], chunk["offset"]
        lower = offset + overlap / 2 if position > 0 else float("-inf")
        upper = (ordered[position + 1]["offset"] + overlap / 2
                 if position + 1 < len(ordered) else float("inf"))
        chunk_segments = result.get("segments") or []
        if not chunk_segments:
            # No timings to reconcile with: keep the whole chunk text.
            if result.get("text", "").strip():
                texts.append(result["text"].strip())
            continue
        kept = []
        for segment in chunk_segments:
            middle = _midpoint(segment)
            if middle is not None and not lower <= offset + middle < upper:
                continue
            shifted = dict(segment)
            if middle is not None:
                shifted["start"] = round(offset + segment["start"], 3)
                shifted["end"] = round(offset + segment["end"], 3)
            if shifted.get("speaker_id"):
                shifted["speaker_id"] = f"c{chunk['index']:02d}:{shifted['speaker_id']}"
            kept.append(shifted)
        segments.extend(kept)
        text = "".join(str(segment.get("text", "")) for segment in kept).strip()
        if text:
            texts.append(text)
    first = ordered[0]["result"]
    seconds = [c["result"].get("usage", {}).get("prompt_audio_seconds") for c in ordered]
    return {
        "model": first.get("model"),
        "text": " ".join(texts),
        "language": first.get("language"),
        "segments": segments,
        "usage": {"prompt_audio_seconds": sum(s for s in seconds if isinstance(s, (int, float)))},
        "merged_from_chunks": [
            {"index": c["index"], "offset": c["offset"], "length": c["length"]} for c in ordered
        ],
    }
