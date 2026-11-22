"""Derived exports. Captions are segment-aligned; no word timings are invented."""

from __future__ import annotations

import math


def _valid(segment: dict) -> bool:
    start, end = segment.get("start"), segment.get("end")
    return (isinstance(start, (int, float)) and isinstance(end, (int, float))
            and math.isfinite(start) and math.isfinite(end) and 0 <= start <= end
            and bool(str(segment.get("text", "")).strip()))


def timestamp(seconds: float, separator: str) -> str:
    total = round(seconds * 1000)  # round once, then carry
    hours, rest = divmod(total, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _caption(segment: dict, speaker_names: dict[str, str]) -> str:
    text = str(segment["text"]).strip()
    speaker = segment.get("speaker_id")
    return f"{speaker_names.get(speaker, speaker)}: {text}" if speaker else text


def to_srt(segments: list[dict], speaker_names: dict[str, str] | None = None) -> str:
    names = speaker_names or {}
    cues = []
    for segment in filter(_valid, segments):
        cues.append(f"{len(cues) + 1}\n{timestamp(segment['start'], ',')} --> "
                    f"{timestamp(segment['end'], ',')}\n{_caption(segment, names)}\n")
    return "\n".join(cues)


def to_vtt(segments: list[dict], speaker_names: dict[str, str] | None = None) -> str:
    names = speaker_names or {}
    cues = ["WEBVTT\n"]
    for segment in filter(_valid, segments):
        text = (_caption(segment, names).replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;"))
        cues.append(f"{timestamp(segment['start'], '.')} --> "
                    f"{timestamp(segment['end'], '.')}\n{text}\n")
    return "\n".join(cues)


def summary_markdown(summary: dict) -> str:
    points = "\n".join(f"- {point}" for point in summary.get("key_points", []))
    return f"# {summary['title']}\n\n{summary['summary']}\n\n## Key points\n\n{points}\n"
