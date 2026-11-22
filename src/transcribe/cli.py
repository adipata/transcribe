"""Command-line entry point. Progress goes to stderr, the result listing to stdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path

from .config import LANGUAGES, Settings
from .graph import build_graph
from .voxtral_client import DEFAULT_MAX_BYTES, TranscriptionError, VoxtralClient


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Transcribe recorded audio with Voxtral Mini Transcribe 3 "
                    "(LangChain + LangGraph example)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", type=Path, help="local audio file, e.g. meeting.mp3")
    source.add_argument("--url", help="HTTPS URL of an audio file reachable by Mistral")
    source.add_argument("--file-id", help="ID of audio already uploaded to Mistral")
    parser.add_argument("--language", help="two-letter ISO 639-1 code; omit to auto-detect")
    parser.add_argument("--timestamps", action="store_true",
                        help="segment timestamps (also writes .srt and .vtt)")
    parser.add_argument("--diarize", action="store_true", help="speaker labels (implies timestamps)")
    parser.add_argument("--bias", action="append", default=[], metavar="TERM",
                        help="vocabulary hint; repeat for several")
    parser.add_argument("--stream", action="store_true", help="show text as it is transcribed")
    parser.add_argument("--chunk-minutes", type=float,
                        help="split local files longer than this and transcribe chunks in parallel")
    parser.add_argument("--no-summary", action="store_true", help="skip the summary step")
    parser.add_argument("--out-dir", type=Path, default=Path("transcripts"))
    parser.add_argument("--model", help="transcription model ID (default: voxtral-mini-transcribe-3-0)")
    parser.add_argument("--summary-model", help="chat model for the summary")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--read-timeout", type=float, default=600.0)
    args = parser.parse_args(argv)
    if args.language and args.language not in LANGUAGES:
        parser.error(f"unsupported language code: {args.language}")
    if args.chunk_minutes is not None and args.chunk_minutes <= 0:
        parser.error("--chunk-minutes must be positive")
    return args


def main(argv: list[str] | None = None, *, make_client=None, summarizer=None,
         settings: Settings | None = None) -> int:
    args = parse_args(argv)
    settings = settings or Settings()
    settings = replace(settings, max_bytes=args.max_bytes, read_timeout=args.read_timeout,
                       **({"model": args.model} if args.model else {}),
                       **({"summary_model": args.summary_model} if args.summary_model else {}))
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if make_client is None:
        if not api_key.strip():
            print("Error: MISTRAL_API_KEY is not set", file=sys.stderr)
            return 1
        make_client = lambda: VoxtralClient(api_key, read_timeout=settings.read_timeout)
    if summarizer is None and not args.no_summary:
        from .chains import build_summary_chain
        summarizer = build_summary_chain(settings.summary_model)

    if args.file:
        source = {"kind": "file", "value": str(args.file)}
    elif args.url:
        source = {"kind": "url", "value": args.url}
    else:
        source = {"kind": "file_id", "value": args.file_id}
    options = {"language": args.language, "timestamps": args.timestamps or args.diarize,
               "diarize": args.diarize, "context_bias": args.bias, "stream": args.stream,
               "chunk_minutes": args.chunk_minutes, "summary": not args.no_summary}
    run_input = {"source": source, "options": options, "out_dir": str(args.out_dir)}

    # One checkpoint thread per (input, options, model): re-running the same command after a
    # failure resumes it, so chunks that already succeeded are not transcribed again.
    identity = json.dumps([run_input, settings.model], sort_keys=True)
    if args.file and args.file.is_file():
        stat = args.file.stat()
        identity += f"{stat.st_size}:{stat.st_mtime_ns}"
    config = {"configurable": {"thread_id": hashlib.sha256(identity.encode()).hexdigest()[:24]}}

    settings.work_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(settings.work_dir / "checkpoints.sqlite", check_same_thread=False)
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        graph = build_graph(settings, make_client=make_client, summarizer=summarizer,
                            checkpointer=SqliteSaver(connection))
        previous = graph.get_state(config)
        resume = bool(previous.next) and bool(previous.values.get("job_id"))
        if resume:
            print("Resuming the earlier unfinished run of this command", file=sys.stderr)
        previewing = False
        try:
            for mode, data in graph.stream(None if resume else run_input, config,
                                           stream_mode=["updates", "custom"]):
                if mode == "custom":
                    if data.get("event") == "delta":
                        print(data["text"], end="", file=sys.stderr, flush=True)
                        previewing = True
                    continue
                if previewing:
                    print(file=sys.stderr)
                    previewing = False
                for node, update in data.items():
                    detail = ""
                    if node == "split_audio":
                        detail = f" ({len(update['chunks'])} chunks)"
                    elif node == "transcribe" and (update or {}).get("needs_split"):
                        detail = " (too long for one request, splitting)"
                    print(f"[{node}]{detail}", file=sys.stderr)
        except (ValueError, OSError, RuntimeError) as exc:  # TranscriptionError is a RuntimeError
            if previewing:
                print(file=sys.stderr)
            print(f"Error: {exc}", file=sys.stderr)
            if isinstance(exc, TranscriptionError):
                print("No transcript was written. Re-run the same command to resume.",
                      file=sys.stderr)
            return 1
        final = graph.get_state(config).values
    finally:
        connection.close()

    for path in final["outputs"]:
        print(path)
    if final.get("summary_error"):
        print(f"Warning: transcript saved, but the summary failed: {final['summary_error']}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
