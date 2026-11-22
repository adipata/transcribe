"""Local audio helpers built on ffprobe/ffmpeg (both optional until chunking is needed)."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_duration(path: Path) -> float | None:
    """Decoded duration in seconds, or None when ffprobe is missing or cannot read the file."""
    if shutil.which("ffprobe") is None:
        return None
    done = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=False,
    )
    try:
        return float(done.stdout.strip())
    except ValueError:
        return None


def plan_chunks(duration: float, chunk_seconds: float, overlap: float) -> list[tuple[float, float]]:
    """(offset, length) windows. Each window extends `overlap` seconds into the next one."""
    if chunk_seconds <= 0 or overlap < 0 or overlap >= chunk_seconds:
        raise ValueError("chunk length must be positive and longer than the overlap")
    windows = []
    offset = 0.0
    while duration - offset > 1.0 or not windows:
        windows.append((offset, min(chunk_seconds + overlap, duration - offset)))
        offset += chunk_seconds
    return windows


def split_audio(path: Path, duration: float, chunk_seconds: float, overlap: float,
                out_dir: Path) -> list[dict]:
    """Cut `path` into fixed windows without re-encoding. Returns chunk records with offsets."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to split long recordings")
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    for index, (offset, length) in enumerate(plan_chunks(duration, chunk_seconds, overlap)):
        target = out_dir / f"chunk{index:03d}{path.suffix}"
        done = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", f"{offset:.3f}", "-t", f"{length:.3f}",
             "-i", str(path), "-vn", "-c:a", "copy", str(target)],
            capture_output=True, text=True, check=False,
        )
        if done.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg could not create chunk {index}: {done.stderr.strip()[:300]}")
        chunks.append({"index": index, "path": str(target), "offset": offset, "length": length})
    return chunks
