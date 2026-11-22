"""LangChain integration: turn audio blobs / Voxtral results into Documents."""

from __future__ import annotations

from collections.abc import Iterator

from langchain_core.document_loaders import BaseBlobParser, Blob
from langchain_core.documents import Document

from .voxtral_client import VoxtralClient


def result_to_documents(result: dict, source: str) -> list[Document]:
    """One Document per segment when timings exist, otherwise one for the whole transcript."""
    base = {"source": source, "model": result.get("model"), "language": result.get("language")}
    segments = result.get("segments") or []
    if not segments:
        return [Document(page_content=result["text"], metadata=base)]
    return [
        Document(
            page_content=str(segment.get("text", "")).strip(),
            metadata={**base, "start": segment.get("start"), "end": segment.get("end"),
                      "speaker_id": segment.get("speaker_id")},
        )
        for segment in segments
    ]


def format_documents(documents: list[Document]) -> str:
    """Readable transcript for an LLM prompt: `[HH:MM:SS] speaker: text` where known."""
    lines = []
    for document in documents:
        prefix = ""
        start = document.metadata.get("start")
        if isinstance(start, (int, float)):
            whole = int(start)
            prefix += f"[{whole // 3600:02d}:{whole % 3600 // 60:02d}:{whole % 60:02d}] "
        if document.metadata.get("speaker_id"):
            prefix += f"{document.metadata['speaker_id']}: "
        lines.append(prefix + document.page_content)
    return "\n".join(lines)


class VoxtralTranscriptionParser(BaseBlobParser):
    """Blob parser so Voxtral plugs into LangChain loaders, e.g. `GenericLoader`.

    `options` are passed to `VoxtralClient.transcribe` (language, timestamps, diarize,
    context_bias, model, max_bytes).
    """

    def __init__(self, client: VoxtralClient, **options):
        self.client = client
        self.options = options

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        if blob.path is None:
            raise ValueError("VoxtralTranscriptionParser needs a blob backed by a local file")
        result = self.client.transcribe(path=blob.path, **self.options)
        yield from result_to_documents(result, source=str(blob.path))
