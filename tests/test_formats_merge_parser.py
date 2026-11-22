from langchain_core.document_loaders import Blob

from conftest import FakeAPI, provider_result
from transcribe import formats
from transcribe.merge import merge_chunks
from transcribe.parser import VoxtralTranscriptionParser, format_documents


def seg(text, start, end, speaker=None):
    return {"type": "transcription_segment", "text": text, "start": start, "end": end,
            "speaker_id": speaker}


def test_timestamp_rounds_once_and_carries():
    assert formats.timestamp(59.9995, ",") == "00:01:00,000"
    assert formats.timestamp(3599.9996, ".") == "01:00:00.000"
    assert formats.timestamp(3725.25, ",") == "01:02:05,250"


def test_srt_and_vtt_skip_untimed_segments_and_escape():
    segments = [seg(" Hi <b> & bye", 0.0, 1.5, "speaker_1"), seg("lost", None, None),
                seg(" backwards", 5.0, 4.0), seg(" Next.", 1.5, 1.5)]
    assert formats.to_srt(segments) == (
        "1\n00:00:00,000 --> 00:00:01,500\nspeaker_1: Hi <b> & bye\n\n"
        "2\n00:00:01,500 --> 00:00:01,500\nNext.\n")
    vtt = formats.to_vtt(segments, {"speaker_1": "Léa"})
    assert vtt.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nLéa: Hi &lt;b&gt; &amp; bye\n")


def test_merge_shifts_offsets_dedupes_overlap_and_scopes_speakers():
    # 10 s chunks with 4 s overlap: chunk 0 covers 0-14, chunk 1 covers 10-20. Cut at 12 s.
    chunks = [
        {"index": 1, "offset": 10.0, "length": 10.0, "result": provider_result(
            "dup late", [seg(" dup", 0.0, 1.0, "speaker_1"), seg(" late", 3.0, 6.0, "speaker_1")], 10)},
        {"index": 0, "offset": 0.0, "length": 14.0, "result": provider_result(
            "early dup tail", [seg(" early", 1.0, 3.0, "speaker_1"), seg(" dup", 10.0, 11.0, "speaker_2"),
                               seg(" tail", 12.5, 13.5, "speaker_2")], 14)},
    ]
    merged = merge_chunks(chunks, overlap=4.0)
    assert merged["text"] == "early dup late"
    assert [(s["start"], s["end"], s["speaker_id"]) for s in merged["segments"]] == [
        (1.0, 3.0, "c00:speaker_1"), (10.0, 11.0, "c00:speaker_2"), (13.0, 16.0, "c01:speaker_1")]
    assert merged["usage"]["prompt_audio_seconds"] == 24
    assert chunks[0]["result"]["segments"][0]["start"] == 0.0  # raw chunk output untouched


def test_parser_yields_one_document_per_segment(fake_mp3):
    api = FakeAPI(lambda fields, n: provider_result(
        "A. B.", [seg(" A.", 0.0, 1.0, "speaker_1"), seg(" B.", 3661.0, 3662.0, "speaker_2")]))
    with api.make_client() as client:
        parser = VoxtralTranscriptionParser(client, diarize=True, language="en")
        documents = list(parser.lazy_parse(Blob.from_path(fake_mp3)))
    assert api.calls[0]["diarize"] == [b"true"] and api.calls[0]["language"] == [b"en"]
    assert [d.page_content for d in documents] == ["A.", "B."]
    assert documents[1].metadata["speaker_id"] == "speaker_2"
    assert format_documents(documents) == "[00:00:00] speaker_1: A.\n[01:01:01] speaker_2: B."
