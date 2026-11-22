import json
import tempfile
import unittest
from email import policy
from email.parser import BytesParser
from pathlib import Path

import httpx

from transcribe.voxtral_client import APIError, MODEL, TranscriptionError, VoxtralClient


FINAL = {"model": MODEL, "text": "Bonjour, Léa.", "language": None,
         "segments": [], "usage": {"prompt_audio_seconds": 2},
         "finish_reason": None}


def fields(request):
    message = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode()
        + request.read()
    )
    return [(part.get_param("name", header="content-disposition"),
             part.get_payload(decode=True)) for part in message.iter_parts()]


class Chunks(httpx.SyncByteStream):
    def __init__(self, raw):
        self.raw = raw

    def __iter__(self):
        # Include boundaries inside multibyte UTF-8 characters and SSE frames.
        for offset in range(0, len(self.raw), 3):
            yield self.raw[offset:offset + 3]


def stream_response(text):
    return httpx.Response(200, headers={"Content-Type": "text/event-stream"},
                          stream=Chunks(text.encode("utf-8")))


class ClientTests(unittest.TestCase):
    def test_upload_wire_format(self):
        def handler(request):
            self.assertEqual(str(request.url), "https://api.mistral.ai/v1/audio/transcriptions")
            self.assertEqual(request.headers["Authorization"], "Bearer fake-test-key")
            parts = fields(request)
            self.assertIn(("model", MODEL.encode()), parts)
            self.assertIn(("file", b"synthetic-audio"), parts)
            self.assertIn(("diarize", b"true"), parts)
            self.assertIn(("timestamp_granularities", b"segment"), parts)
            self.assertIn(("language", b"fr"), parts)
            self.assertEqual([value for key, value in parts if key == "context_bias"],
                             [b"PKCS#11", b"Northwind Labs"])
            return httpx.Response(200, json=FINAL)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.wav"
            path.write_bytes(b"synthetic-audio")  # Transport fixture, not valid audio.
            with VoxtralClient("fake-test-key", transport=httpx.MockTransport(handler)) as client:
                self.assertEqual(client.transcribe(path=path, language="fr", diarize=True,
                                 context_bias=["PKCS#11", "Northwind Labs"]), FINAL)

    def test_remote_sources(self):
        for source in ({"file_url": "https://example.com/test.wav"}, {"file_id": "file-test"}):
            def handler(request):
                key, value = next(iter(source.items()))
                self.assertIn((key, value.encode()), fields(request))
                return httpx.Response(200, json=FINAL)
            with self.subTest(source=source), VoxtralClient("fake", transport=httpx.MockTransport(handler)) as client:
                self.assertEqual(client.transcribe(**source)["text"], FINAL["text"])

    def test_invalid_options_never_send(self):
        def handler(request):
            self.fail("Invalid input reached the transport")
        cases = [{}, {"file_id": "x", "file_url": "https://example.com/x.wav"},
                 {"file_id": ""}, {"file_id": "x", "language": "fr-FR"},
                 {"file_id": "x", "context_bias": "one string"},
                 {"file_id": "x", "context_bias": [""]},
                 {"file_url": "http://example.com/x.wav"}]
        with VoxtralClient("fake", transport=httpx.MockTransport(handler)) as client:
            for case in cases:
                with self.subTest(case=case), self.assertRaises(ValueError):
                    client.transcribe(**case)

    def test_empty_and_oversize_local_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.wav"
            with VoxtralClient("fake", transport=httpx.MockTransport(lambda r: self.fail("Unexpected request"))) as client:
                for content in (b"", b"1234"):
                    path.write_bytes(content)
                    with self.assertRaises(ValueError):
                        client.transcribe(path=path, max_bytes=3)

    def test_http_errors_are_not_retried(self):
        for status in (400, 401, 403, 413, 422, 429, 503):
            attempts = []
            def handler(request):
                attempts.append(1)
                return httpx.Response(status, headers={"Retry-After": "3"}, json={"message": "synthetic"})
            with self.subTest(status=status), VoxtralClient("fake", transport=httpx.MockTransport(handler)) as client:
                with self.assertRaises(APIError) as raised:
                    client.transcribe(file_id="test")
                self.assertEqual(raised.exception.status, status)
                self.assertEqual(raised.exception.retry_after, "3")
                self.assertEqual(len(attempts), 1)

    def test_stream_final_is_authoritative(self):
        raw = (': heartbeat\r\n\r\nevent: transcription.text.delta\r\n'
               'data: {"text":\r\ndata: "préliminaire"}\r\n\r\n'
               'event: transcription.done\r\ndata: '
               + json.dumps(FINAL, ensure_ascii=False) + '\r\n\r\n')
        previews = []
        with VoxtralClient("fake", transport=httpx.MockTransport(lambda r: stream_response(raw))) as client:
            result = client.transcribe_stream(file_id="test", on_event=lambda n, p: previews.append((n, p)))
        self.assertEqual(result, FINAL)
        self.assertEqual(previews, [("transcription.text.delta", {"text": "préliminaire"})])

    def test_stream_failures(self):
        cases = ['event: transcription.text.delta\ndata: {"text":"partial"}\n\n',
                 'data: [DONE]\n\n', 'event: error\ndata: {"error":{"message":"failed"}}\n\n',
                 'data: invalid-json\n\n',
                 'event: transcription.done\ndata: ' + json.dumps(FINAL)]
        for raw in cases:
            with self.subTest(raw=raw), VoxtralClient("fake", transport=httpx.MockTransport(lambda r: stream_response(raw))) as client:
                with self.assertRaises(TranscriptionError):
                    client.transcribe_stream(file_id="test")

    def test_invalid_success_payloads(self):
        for value in ({"text": 3}, {"text": "cut", "finish_reason": "length"},
                      {"text": "ok", "segments": "invalid"}):
            with self.subTest(value=value), VoxtralClient("fake", transport=httpx.MockTransport(lambda r: httpx.Response(200, json=value))) as client:
                with self.assertRaises(TranscriptionError):
                    client.transcribe(file_id="test")

    def test_network_failure_surfaces(self):
        def handler(request):
            raise httpx.ReadTimeout("synthetic timeout", request=request)
        with VoxtralClient("fake", transport=httpx.MockTransport(handler)) as client:
            with self.assertRaises(TranscriptionError):
                client.transcribe(file_id="test")


if __name__ == "__main__":
    unittest.main()
