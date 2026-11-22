from __future__ import annotations

import json
import mimetypes
import re
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlsplit

import httpx

MODEL = "voxtral-mini-transcribe-3-0"
ENDPOINT = "https://api.mistral.ai/v1/audio/transcriptions"
# Application policy, NOT a verified Mistral audio-file limit.
DEFAULT_MAX_BYTES = 100 * 1024 * 1024


class TranscriptionError(RuntimeError):
    pass


class APIError(TranscriptionError):
    def __init__(self, status: int, body: object, retry_after: str | None):
        # Keep provider details accessible, but out of default logs.
        self.status = status
        self.body = body
        self.retry_after = retry_after
        super().__init__(f"Transcription request failed: HTTP {status}")


def final_result(value: object) -> dict:
    if not isinstance(value, dict) or not isinstance(value.get("text"), str):
        raise TranscriptionError("Expected a JSON object with a text string")
    segments = value.get("segments")
    if segments is not None and not isinstance(segments, list):
        raise TranscriptionError("Expected segments to be a list or null")
    if value.get("finish_reason") in {"length", "max_tokens"}:
        raise TranscriptionError("Provider reported a truncated transcript")
    return value  # Retain optional and future fields without rewriting them.


def check_status(response: httpx.Response) -> None:
    if response.is_success:
        return
    response.read()
    try:
        body = response.json()
    except ValueError:
        body = response.text[:2000]
    raise APIError(response.status_code, body, response.headers.get("Retry-After"))


def iter_sse(lines: Iterable[str]):
    """Parse SSE frames; HTTPX handles byte decoding and network chunk splits."""
    event_name = ""
    data: list[str] = []
    for line in lines:
        if line == "":
            if data:
                yield event_name, "\n".join(data)
            event_name, data = "", []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value
        elif field == "data":
            data.append(value)
        # id/retry fields do not establish resumability for this POST.
    # An unterminated frame at EOF is deliberately not accepted as complete.


@contextmanager
def request_parts(
    *,
    path: str | Path | None = None,
    file_url: str | None = None,
    file_id: str | None = None,
    model: str = MODEL,
    language: str | None = None,
    timestamps: bool = False,
    diarize: bool = False,
    context_bias: Iterable[str] = (),
    stream: bool = False,
    max_bytes: int = DEFAULT_MAX_BYTES,
):
    if sum(x is not None for x in (path, file_url, file_id)) != 1:
        raise ValueError("Supply exactly one of path, file_url, file_id")
    if not model.strip():
        raise ValueError("model must not be empty")
    if language is not None and not re.fullmatch(r"[a-z]{2}", language):
        raise ValueError("language must be a lowercase two-letter code")
    if isinstance(context_bias, (str, bytes)):
        raise ValueError("context_bias must be a collection of strings")
    terms = list(context_bias)
    if any(not isinstance(t, str) or not t.strip() for t in terms):
        raise ValueError("context_bias entries must be nonempty strings")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")

    fields = [("model", model), ("stream", str(stream).lower()),
              ("diarize", str(diarize).lower())]
    if language is not None:
        fields.append(("language", language))
    if timestamps or diarize:
        fields.append(("timestamp_granularities", "segment"))
    fields.extend(("context_bias", term) for term in terms)
    if file_url is not None:
        url = urlsplit(file_url)
        if (len(file_url) > 2083 or url.scheme != "https" or not url.hostname
                or url.username is not None or url.password is not None
                or url.fragment):
            raise ValueError("Use an HTTPS audio URL, <=2083 characters, without credentials or a fragment")
        fields.append(("file_url", file_url))
    if file_id is not None:
        if not file_id.strip():
            raise ValueError("file_id must not be empty")
        fields.append(("file_id", file_id))

    # filename=None creates ordinary multipart form fields, preserving repeats.
    parts = [(name, (None, value)) for name, value in fields]
    with ExitStack() as stack:
        if path is not None:
            audio_path = Path(path)
            if not audio_path.is_file():
                raise ValueError("Audio path must be an existing regular file")
            size = audio_path.stat().st_size
            if size == 0 or size > max_bytes:
                raise ValueError("Audio is empty or exceeds the application byte limit")
            audio = stack.enter_context(audio_path.open("rb"))
            mime = mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"
            parts.append(("file", (audio_path.name, audio, mime)))
        yield parts


class VoxtralClient:
    def __init__(self, api_key: str, *, read_timeout: float = 600.0,
                 transport: httpx.BaseTransport | None = None):
        if not api_key or not api_key.strip():
            raise ValueError("MISTRAL_API_KEY is missing")
        if read_timeout <= 0:
            raise ValueError("read_timeout must be positive")
        self._http = httpx.Client(
            headers={"Authorization": f"Bearer {api_key.strip()}"},
            timeout=httpx.Timeout(connect=15.0, read=read_timeout,
                                  write=120.0, pool=15.0),
            follow_redirects=False,
            transport=transport,
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._http.close()

    def transcribe(self, **options) -> dict:
        """One request. No automatic retries after ambiguous failures."""
        try:
            with request_parts(stream=False, **options) as parts:
                response = self._http.post(ENDPOINT, files=parts)
            check_status(response)
            try:
                value = response.json()
            except ValueError as exc:
                raise TranscriptionError("Expected JSON from transcription API") from exc
            return final_result(value)
        except httpx.RequestError as exc:
            raise TranscriptionError(
                "Network failure; request outcome may be unknown. No retry was made."
            ) from exc

    def transcribe_stream(
        self, *, on_event: Callable[[str, dict], None] | None = None, **options
    ) -> dict:
        """Return only a complete final result; intermediate events are previews."""
        try:
            with request_parts(stream=True, **options) as parts:
                with self._http.stream("POST", ENDPOINT, files=parts,
                                       headers={"Accept": "text/event-stream"}) as response:
                    check_status(response)
                    content_type = response.headers.get("Content-Type", "").split(";", 1)[0]
                    if content_type.strip().lower() != "text/event-stream":
                        raise TranscriptionError("Expected text/event-stream")
                    response.encoding = "utf-8"
                    for event, raw in iter_sse(response.iter_lines()):
                        if raw.strip() == "[DONE]":
                            break  # This alone is not the required final payload.
                        try:
                            payload = json.loads(raw)
                        except ValueError as exc:
                            raise TranscriptionError("Malformed SSE JSON") from exc
                        if not isinstance(payload, dict):
                            raise TranscriptionError("Expected an SSE JSON object")
                        name = event or payload.get("type", "")
                        if name in {"error", "transcription.error"} or payload.get("error"):
                            raise TranscriptionError("Provider returned a streaming error")
                        if name == "transcription.done":
                            return final_result(payload)
                        if on_event is not None:
                            on_event(name, payload)
        except httpx.RequestError as exc:
            raise TranscriptionError(
                "Stream interrupted; partial text is not a completed transcript."
            ) from exc
        raise TranscriptionError("Stream ended without transcription.done")
