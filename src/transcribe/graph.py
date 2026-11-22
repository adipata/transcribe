"""LangGraph workflow: prepare -> transcribe (or split -> parallel chunks -> merge)
-> summarize -> write outputs."""

from __future__ import annotations

import json
import shutil
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, TypedDict
from urllib.parse import urlsplit, urlunsplit

from langchain_core.runnables import Runnable
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, Send

from . import audio, formats
from .config import USD_PER_MINUTE, Settings
from .merge import merge_chunks
from .parser import format_documents, result_to_documents
from .voxtral_client import APIError, TranscriptionError, VoxtralClient


def _append_or_reset(current: list | None, update: list | None) -> list:
    """Fan-in reducer. `None` clears results left on the thread by an earlier run."""
    return [] if update is None else (current or []) + update


class State(TypedDict, total=False):
    source: dict  # {"kind": "file" | "url" | "file_id", "value": str}
    options: dict  # language, timestamps, diarize, context_bias, stream, chunk_minutes, summary
    out_dir: str
    job_id: str
    sha256: str | None
    duration: float | None
    started_at: str
    needs_split: bool
    chunks: list[dict]
    chunk_results: Annotated[list[dict], _append_or_reset]
    result: dict | None
    summary: dict | None
    summary_error: str | None
    status: str
    completed_at: str
    outputs: list[str]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def stem_for(source: dict) -> str:
    if source["kind"] == "file":
        return Path(source["value"]).stem
    if source["kind"] == "url":
        return Path(urlsplit(source["value"]).path).stem or "audio"
    return source["value"]


def output_paths(out_dir: str, source: dict) -> dict[str, Path]:
    base = Path(out_dir) / stem_for(source)
    return {name: base.with_name(f"{base.name}.{suffix}") for name, suffix in {
        "transcript": "transcript.json", "job": "job.json", "text": "txt", "srt": "srt",
        "vtt": "vtt", "summary": "summary.md", "chunks": "chunks.json"}.items()}


def is_retryable(exc: Exception) -> bool:
    """Rate limits and transient server errors only. Timeouts may already have been billed."""
    return isinstance(exc, APIError) and (exc.status == 429 or exc.status >= 500)


def is_context_error(exc: APIError) -> bool:
    # The exact error body was not live-verified; this matches "context" in a 400 body.
    return exc.status == 400 and "context" in json.dumps(exc.body, default=str).lower()


def build_graph(settings: Settings, *, make_client: Callable[[], VoxtralClient],
                summarizer: Runnable | None = None, checkpointer=None,
                splitter: Callable[..., list[dict]] = audio.split_audio):
    def client_options(options: dict) -> dict:
        return {"model": settings.model, "language": options.get("language"),
                "timestamps": options.get("timestamps", False),
                "diarize": options.get("diarize", False),
                "context_bias": options.get("context_bias", []),
                "max_bytes": settings.max_bytes}

    def honour_retry_after(exc: APIError) -> None:
        """Sleep for a short numeric Retry-After before the retry policy re-runs the node."""
        try:
            wait = float(exc.retry_after) if exc.status == 429 and exc.retry_after else 0.0
        except ValueError:
            wait = 0.0
        if 0 < wait <= settings.max_retry_after_seconds:
            time.sleep(wait)

    def prepare(state: State) -> dict:
        source, options = state["source"], state["options"]
        existing = [str(p) for p in output_paths(state["out_dir"], source).values() if p.exists()]
        if existing:
            raise ValueError(f"Output already exists, choose another --out-dir: {existing[0]}")
        digest = duration = None
        if source["kind"] == "file":
            path = Path(source["value"])
            if not path.is_file():
                raise ValueError(f"Audio file not found: {path}")
            size = path.stat().st_size
            if size == 0 or size > settings.max_bytes:
                raise ValueError("Audio is empty or exceeds the application byte limit")
            digest, duration = audio.sha256_file(path), audio.probe_duration(path)
            if options.get("chunk_minutes") and duration is None:
                raise ValueError("--chunk-minutes needs ffprobe to read the audio duration")
        elif options.get("chunk_minutes"):
            raise ValueError("--chunk-minutes only applies to local files")
        return {"job_id": uuid.uuid4().hex, "sha256": digest, "duration": duration,
                "started_at": _now(), "status": "running", "needs_split": False, "chunks": [],
                "chunk_results": None, "result": None, "summary": None, "summary_error": None}

    def route_after_prepare(state: State) -> str:
        minutes = state["options"].get("chunk_minutes")
        if minutes and state["duration"] > minutes * 60:
            return "split_audio"
        return "transcribe"

    def transcribe(state: State) -> dict:
        source, options = state["source"], state["options"]
        key = {"file": "path", "url": "file_url", "file_id": "file_id"}[source["kind"]]
        kwargs = {key: source["value"], **client_options(options)}
        writer = get_stream_writer()
        try:
            with make_client() as client:
                if options.get("stream"):
                    writer({"event": "attempt"})  # a new attempt replaces any earlier preview

                    def preview(name: str, payload: dict) -> None:
                        if name == "transcription.text.delta":
                            writer({"event": "delta", "text": payload.get("text", "")})

                    result = client.transcribe_stream(on_event=preview, **kwargs)
                else:
                    result = client.transcribe(**kwargs)
        except APIError as exc:
            if is_context_error(exc):
                if source["kind"] == "file":
                    return {"needs_split": True}
                raise TranscriptionError(
                    "The recording is too long for one request. Splitting is only possible "
                    "for local files: download it and use --file.") from exc
            honour_retry_after(exc)
            raise
        return {"result": result}

    def route_after_transcribe(state: State) -> str:
        return "split_audio" if state.get("needs_split") else "summarize"

    def split_audio(state: State) -> dict:
        path, duration = Path(state["source"]["value"]), state.get("duration")
        if duration is None:
            raise TranscriptionError(
                "The recording is too long for one request and ffprobe is not available to split it")
        seconds = (state["options"].get("chunk_minutes") or settings.fallback_chunk_minutes) * 60
        while seconds >= duration and seconds / 2 > settings.chunk_overlap_seconds:
            seconds /= 2  # provider-rejected recording shorter than the default chunk length
        work = settings.work_dir / "chunks" / state["job_id"]
        return {"chunks": splitter(path, duration, seconds, settings.chunk_overlap_seconds, work)}

    def fan_out(state: State) -> list[Send]:
        return [Send("transcribe_chunk", {"chunk": chunk, "options": state["options"]})
                for chunk in state["chunks"]]

    def transcribe_chunk(payload: dict) -> dict:
        chunk = payload["chunk"]
        # Segment timings are always requested here: they are what overlap removal works from.
        kwargs = {**client_options(payload["options"]), "timestamps": True}
        try:
            with make_client() as client:
                result = client.transcribe(path=chunk["path"], **kwargs)
        except APIError as exc:
            honour_retry_after(exc)
            raise
        return {"chunk_results": [{**chunk, "result": result}]}

    def merge(state: State) -> dict:
        return {"result": merge_chunks(state["chunk_results"], settings.chunk_overlap_seconds)}

    def summarize(state: State) -> dict:
        result = state["result"]
        if not state["options"].get("summary") or summarizer is None or not result["text"].strip():
            return {}
        documents = result_to_documents(result, source=stem_for(state["source"]))
        try:
            summary = summarizer.invoke({"transcript": format_documents(documents)})
        except Exception as exc:  # noqa: BLE001 - transcript is already paid for; never lose it
            return {"summary_error": f"{type(exc).__name__}: {exc}"}
        return {"summary": summary.model_dump() if hasattr(summary, "model_dump") else dict(summary)}

    def write_outputs(state: State) -> dict:
        source, options, result = state["source"], state["options"], state["result"]
        paths = output_paths(state["out_dir"], source)
        Path(state["out_dir"]).mkdir(parents=True, exist_ok=True)
        written: list[str] = []

        def write(name: str, content: str) -> None:
            with paths[name].open("x", encoding="utf-8") as target:  # never overwrite
                target.write(content)
            written.append(str(paths[name]))

        def dump(value: object) -> str:
            return json.dumps(value, ensure_ascii=False, indent=2) + "\n"

        write("transcript", dump(result))
        if state.get("chunk_results"):
            write("chunks", dump(sorted(state["chunk_results"], key=lambda c: c["index"])))
        write("text", result["text"].strip() + "\n")
        if result.get("segments"):
            write("srt", formats.to_srt(result["segments"]))
            write("vtt", formats.to_vtt(result["segments"]))
        if state.get("summary"):
            write("summary", formats.summary_markdown(state["summary"]))

        completed = _now()
        seconds = (result.get("usage") or {}).get("prompt_audio_seconds")
        shown = source["value"]
        if source["kind"] == "url":  # drop the query string: it may carry a signature
            shown = urlunsplit(urlsplit(shown)._replace(query=""))
        elif source["kind"] == "file":
            shown = Path(shown).name
        write("job", dump({
            "job_id": state["job_id"], "status": "succeeded",
            "input": {"kind": source["kind"], "value": shown, "sha256": state.get("sha256"),
                      "duration_seconds": state.get("duration")},
            "requested_model": settings.model, "returned_model": result.get("model"),
            "options": options,
            "chunks": [{k: c[k] for k in ("index", "offset", "length")}
                       for c in state.get("chunks", [])],
            "submitted_at": state["started_at"], "completed_at": completed,
            "audio_seconds_billed": seconds,
            "estimated_cost_usd": (round(seconds / 60 * USD_PER_MINUTE, 4)
                                   if isinstance(seconds, (int, float)) else None),
            "summary_model": settings.summary_model if state.get("summary") else None,
            "summary_error": state.get("summary_error"),
            "outputs": [Path(p).name for p in written],
        }))
        shutil.rmtree(settings.work_dir / "chunks" / state["job_id"], ignore_errors=True)
        return {"status": "succeeded", "completed_at": completed, "outputs": written}

    retry = RetryPolicy(max_attempts=settings.max_attempts, retry_on=is_retryable)
    graph = StateGraph(State)
    graph.add_node("prepare", prepare)
    graph.add_node("transcribe", transcribe, retry_policy=retry)
    graph.add_node("split_audio", split_audio)
    graph.add_node("transcribe_chunk", transcribe_chunk, retry_policy=retry)
    graph.add_node("merge_chunks", merge)
    graph.add_node("summarize", summarize)
    graph.add_node("write_outputs", write_outputs)
    graph.add_edge(START, "prepare")
    graph.add_conditional_edges("prepare", route_after_prepare, ["transcribe", "split_audio"])
    graph.add_conditional_edges("transcribe", route_after_transcribe, ["split_audio", "summarize"])
    graph.add_conditional_edges("split_audio", fan_out, ["transcribe_chunk"])
    graph.add_edge("transcribe_chunk", "merge_chunks")
    graph.add_edge("merge_chunks", "summarize")
    graph.add_edge("summarize", "write_outputs")
    graph.add_edge("write_outputs", END)
    return graph.compile(checkpointer=checkpointer)
