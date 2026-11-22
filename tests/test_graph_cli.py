import json
from pathlib import Path

import httpx
import pytest
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from conftest import FakeAPI, provider_result, sse
from transcribe import cli
from transcribe.graph import build_graph
from transcribe.voxtral_client import APIError

SEGMENTS = [{"type": "transcription_segment", "text": " Hello world.", "start": 0.0, "end": 2.0,
             "speaker_id": "speaker_1"}]
CONTEXT_ERROR = httpx.Response(400, json={"message": "Input exceeds the model context length"})


def run_input(source, out_dir, **options):
    base = {"language": None, "timestamps": False, "diarize": False, "context_bias": [],
            "stream": False, "chunk_minutes": None, "summary": True}
    return {"source": source, "options": {**base, **options}, "out_dir": str(out_dir)}


def chunk_handler(fields, n):
    name = fields["filename"][0]
    return provider_result(name, [{"text": f" {name}", "start": 5.0, "end": 6.0, "speaker_id": None}], 12)


def test_single_request_writes_raw_result_and_exports(settings, summarizer, fake_mp3, tmp_path):
    raw = provider_result("Hello world.", SEGMENTS)
    raw["future_field"] = {"kept": True}
    api = FakeAPI(lambda fields, n: raw)
    graph = build_graph(settings, make_client=api.make_client, summarizer=summarizer)
    out = tmp_path / "out"
    final = graph.invoke(run_input({"kind": "file", "value": str(fake_mp3)}, out,
                                   diarize=True, timestamps=True, language="fr",
                                   context_bias=["PKCS#11", "Northwind Labs"]))
    assert final["status"] == "succeeded" and len(api.calls) == 1
    sent = api.calls[0]
    assert sent["model"] == [b"voxtral-mini-transcribe-3-0"] and sent["language"] == [b"fr"]
    assert sent["timestamp_granularities"] == [b"segment"] and sent["diarize"] == [b"true"]
    assert sent["context_bias"] == [b"PKCS#11", b"Northwind Labs"]
    assert json.loads((out / "meeting.transcript.json").read_text()) == raw  # unchanged
    assert (out / "meeting.txt").read_text() == "Hello world.\n"
    assert "speaker_1: Hello world." in (out / "meeting.srt").read_text()
    assert (out / "meeting.vtt").read_text().startswith("WEBVTT")
    assert (out / "meeting.summary.md").read_text().startswith("# Test title")
    job = json.loads((out / "meeting.job.json").read_text())
    assert job["status"] == "succeeded" and job["returned_model"] == raw["model"]
    assert len(job["input"]["sha256"]) == 64 and job["estimated_cost_usd"] == round(5 / 60 * 0.0035, 4)


@pytest.mark.parametrize("source,field", [
    ({"kind": "url", "value": "https://example.com/a/talk.wav?sig=secret"}, "file_url"),
    ({"kind": "file_id", "value": "file-123"}, "file_id")])
def test_remote_sources(settings, tmp_path, source, field):
    api = FakeAPI(lambda fields, n: provider_result())
    graph = build_graph(settings, make_client=api.make_client)
    final = graph.invoke(run_input(source, tmp_path, summary=False))
    assert api.calls[0][field] == [source["value"].encode()] and "file" not in api.calls[0]
    job = next(p for p in final["outputs"] if p.endswith(".job.json"))
    assert "secret" not in Path(job).read_text()
    assert not list(tmp_path.glob("*.srt")) and not list(tmp_path.glob("*.summary.md"))


def test_retries_rate_limit_then_succeeds(settings, fake_mp3, tmp_path):
    api = FakeAPI(lambda fields, n: httpx.Response(429, headers={"Retry-After": "1"}, json={})
                  if n < 3 else provider_result())
    graph = build_graph(settings, make_client=api.make_client)
    final = graph.invoke(run_input({"kind": "file", "value": str(fake_mp3)}, tmp_path / "o",
                                   summary=False))
    assert final["status"] == "succeeded" and len(api.calls) == 3


@pytest.mark.parametrize("response,calls", [
    (httpx.Response(401, json={"message": "bad key"}), 1),  # not retried
    (httpx.Response(503, json={}), 3)])                       # retried, then given up
def test_failures_write_nothing(settings, fake_mp3, tmp_path, response, calls):
    api = FakeAPI(lambda fields, n: response)
    graph = build_graph(settings, make_client=api.make_client)
    with pytest.raises(APIError):
        graph.invoke(run_input({"kind": "file", "value": str(fake_mp3)}, tmp_path / "o"))
    assert len(api.calls) == calls and not (tmp_path / "o").exists()


def test_refuses_to_overwrite_before_calling_the_api(settings, fake_mp3, tmp_path):
    (tmp_path / "meeting.txt").write_text("keep me")
    api = FakeAPI(lambda fields, n: provider_result())
    graph = build_graph(settings, make_client=api.make_client)
    with pytest.raises(ValueError, match="already exists"):
        graph.invoke(run_input({"kind": "file", "value": str(fake_mp3)}, tmp_path))
    assert api.calls == [] and (tmp_path / "meeting.txt").read_text() == "keep me"


def test_summary_failure_keeps_transcript(settings, fake_mp3, tmp_path):
    def boom(_):
        raise RuntimeError("chat model down")
    api = FakeAPI(lambda fields, n: provider_result())
    graph = build_graph(settings, make_client=api.make_client, summarizer=RunnableLambda(boom))
    final = graph.invoke(run_input({"kind": "file", "value": str(fake_mp3)}, tmp_path / "o"))
    assert final["status"] == "succeeded" and "chat model down" in final["summary_error"]
    assert (tmp_path / "o" / "meeting.txt").exists()
    assert not (tmp_path / "o" / "meeting.summary.md").exists()


def test_chunk_minutes_fans_out_and_merges(settings, summarizer, real_mp3, tmp_path):
    api = FakeAPI(chunk_handler)
    graph = build_graph(settings, make_client=api.make_client, summarizer=summarizer)
    out = tmp_path / "o"
    final = graph.invoke(run_input({"kind": "file", "value": str(real_mp3)}, out,
                                   chunk_minutes=0.2, diarize=False))
    assert sorted(c["filename"][0] for c in api.calls) == ["chunk000.mp3", "chunk001.mp3", "chunk002.mp3"]
    assert all(c["timestamp_granularities"] == [b"segment"] for c in api.calls)
    merged = json.loads((out / "long.transcript.json").read_text())
    assert merged["text"] == "chunk000.mp3 chunk001.mp3 chunk002.mp3"
    assert [s["start"] for s in merged["segments"]] == [5.0, 17.0, 29.0]
    assert len(json.loads((out / "long.chunks.json").read_text())) == 3
    assert json.loads((out / "long.job.json").read_text())["chunks"][1]["offset"] == 12.0
    assert not (settings.work_dir / "chunks" / final["job_id"]).exists()  # temp audio removed


def test_context_error_falls_back_to_chunks(settings, real_mp3, tmp_path):
    api = FakeAPI(lambda fields, n: CONTEXT_ERROR if fields["filename"][0] == "long.mp3"
                  else chunk_handler(fields, n))
    graph = build_graph(settings, make_client=api.make_client)
    final = graph.invoke(run_input({"kind": "file", "value": str(real_mp3)}, tmp_path / "o",
                                   summary=False))
    assert final["status"] == "succeeded" and len(final["chunks"]) >= 2
    assert len(api.calls) == 1 + len(final["chunks"])


def test_context_error_on_url_is_explained(settings, tmp_path):
    api = FakeAPI(lambda fields, n: CONTEXT_ERROR)
    graph = build_graph(settings, make_client=api.make_client)
    with pytest.raises(Exception, match="only possible for local files"):
        graph.invoke(run_input({"kind": "url", "value": "https://example.com/x.mp3"}, tmp_path))


def test_resume_only_repeats_the_failed_chunk(settings, real_mp3, tmp_path):
    broken = {"on": True}

    def handler(fields, n):
        if broken["on"] and fields["filename"][0] == "chunk001.mp3":
            return httpx.Response(401, json={"message": "expired key"})
        return chunk_handler(fields, n)

    api = FakeAPI(handler)
    graph = build_graph(settings, make_client=api.make_client, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t1"}}
    with pytest.raises(APIError):
        graph.invoke(run_input({"kind": "file", "value": str(real_mp3)}, tmp_path / "o",
                               chunk_minutes=0.2, summary=False), config)
    assert len(api.calls) == 3
    broken["on"] = False
    final = graph.invoke(None, config)
    assert [c["filename"][0] for c in api.calls[3:]] == ["chunk001.mp3"]
    assert final["result"]["text"] == "chunk000.mp3 chunk001.mp3 chunk002.mp3"


def test_cli_stream_previews_then_writes_final(settings, summarizer, fake_mp3, tmp_path, capsys):
    final = provider_result("Bonjour, Léa.")
    api = FakeAPI(lambda fields, n: sse([
        ("transcription.text.delta", {"text": "Bonjour, "}),
        ("transcription.text.delta", {"text": "Léa."}),
        ("transcription.done", final)]))
    out = tmp_path / "o"
    code = cli.main(["--file", str(fake_mp3), "--stream", "--out-dir", str(out)],
                    make_client=api.make_client, summarizer=summarizer, settings=settings)
    captured = capsys.readouterr()
    assert code == 0 and api.calls[0]["stream"] == [b"true"]
    assert "Bonjour, Léa.\n[transcribe]" in captured.err
    assert str(out / "meeting.transcript.json") in captured.out
    assert (out / "meeting.txt").read_text() == "Bonjour, Léa.\n"  # final only, no duplicated deltas


def test_cli_resumes_after_failure_and_rejects_bad_input(settings, real_mp3, tmp_path, capsys):
    broken = {"on": True}

    def handler(fields, n):
        if broken["on"] and fields["filename"][0] == "chunk002.mp3":
            return httpx.Response(403, json={})
        return chunk_handler(fields, n)

    api = FakeAPI(handler)
    argv = ["--file", str(real_mp3), "--chunk-minutes", "0.2", "--no-summary",
            "--out-dir", str(tmp_path / "o")]
    assert cli.main(argv, make_client=api.make_client, settings=settings) == 1
    assert "HTTP 403" in capsys.readouterr().err and not (tmp_path / "o").exists()
    broken["on"] = False
    assert cli.main(argv, make_client=api.make_client, settings=settings) == 0
    assert len(api.calls) == 4 and "Resuming" in capsys.readouterr().err
    # A finished run is not resumed; the same command now refuses to overwrite its outputs.
    assert cli.main(argv, make_client=api.make_client, settings=settings) == 1
    assert "already exists" in capsys.readouterr().err and len(api.calls) == 4
    with pytest.raises(SystemExit):
        cli.main(["--file", str(real_mp3), "--language", "xx"], make_client=api.make_client)
