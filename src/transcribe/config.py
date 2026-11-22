"""Application settings. Limits here are application policy, not verified provider limits."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .voxtral_client import DEFAULT_MAX_BYTES, MODEL

# Dated planning estimate from the model card (22 Nov 2026), not a billing guarantee.
USD_PER_MINUTE = 0.0035

# The 53 languages documented for the Voxtral 3 file models (ISO 639-1).
LANGUAGES = frozenset(
    ["ar", "bn", "bg", "my", "ca", "zh", "cs", "da", "nl", "en", "et", "tl", "fi", "fr", "de", "el", "gu", "hi", "hu", "is", "id", "it", "ja", "kn", "km", "ko", "lv", "lt", "lb", "mk", "ms", "ml", "mt", "mr", "no", "or", "fa", "pl", "pt", "pa", "ro", "ru", "sr", "sk", "sl", "es", "sv", "ta", "te", "th", "tr", "uk", "vi"]
)


@dataclass(frozen=True)
class Settings:
    model: str = field(default_factory=lambda: os.environ.get("TRANSCRIBE_MODEL", MODEL))
    summary_model: str = field(
        default_factory=lambda: os.environ.get("TRANSCRIBE_SUMMARY_MODEL", "mistral-small-latest")
    )
    max_bytes: int = DEFAULT_MAX_BYTES
    read_timeout: float = 600.0
    # Used only when the provider rejects a recording as too long and no --chunk-minutes was given.
    fallback_chunk_minutes: float = 10.0
    chunk_overlap_seconds: float = 5.0
    max_attempts: int = 3
    max_retry_after_seconds: float = 60.0
    work_dir: Path = Path(".transcribe")
