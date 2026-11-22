import json
import shutil
import subprocess
from email import policy
from email.parser import BytesParser

import httpx
import pytest
from langchain_core.runnables import RunnableLambda

from transcribe.chains import Summary
from transcribe.config import Settings
from transcribe.voxtral_client import MODEL, VoxtralClient


def parse_request(request: httpx.Request) -> dict:
    """Multipart fields as {name: [values]}, plus the uploaded filename under 'filename'."""
    message = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.read())
    fields: dict[str, list] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        fields.setdefault(name, []).append(part.get_payload(decode=True))
        if part.get_filename():
            fields["filename"] = [part.get_filename()]
    return fields


def provider_result(text="Hello world.", segments=None, seconds=5):
    return {"model": MODEL, "text": text, "language": None, "segments": segments or [],
            "usage": {"prompt_audio_seconds": seconds}, "finish_reason": None}


class FakeAPI:
    """Records requests and answers them with `handler(fields, call_number)`."""

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[dict] = []

    def _handle(self, request: httpx.Request) -> httpx.Response:
        fields = parse_request(request)
        self.calls.append(fields)
        answer = self.handler(fields, len(self.calls))
        if isinstance(answer, httpx.Response):
            return answer
        return httpx.Response(200, json=answer)

    def make_client(self) -> VoxtralClient:
        return VoxtralClient("fake-test-key", transport=httpx.MockTransport(self._handle))


def sse(events: list[tuple[str, dict]]) -> httpx.Response:
    body = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)
    return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=body.encode())


@pytest.fixture
def settings(tmp_path):
    return Settings(work_dir=tmp_path / "work", max_retry_after_seconds=0)


@pytest.fixture
def summarizer():
    return RunnableLambda(lambda inputs: Summary(
        title="Test title", summary=f"{len(inputs['transcript'])} chars", key_points=["one"]))


@pytest.fixture
def fake_mp3(tmp_path):
    path = tmp_path / "meeting.mp3"
    path.write_bytes(b"synthetic-audio")  # transport fixture, not decodable audio
    return path


@pytest.fixture(scope="session")
def real_mp3(tmp_path_factory):
    """A 30-second tone, so ffprobe/ffmpeg splitting is exercised for real."""
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("ffmpeg/ffprobe not installed")
    path = tmp_path_factory.mktemp("audio") / "long.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=30",
                    "-c:a", "libmp3lame", "-b:a", "32k", str(path)], check=True)
    return path
