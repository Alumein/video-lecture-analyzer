"""Tests for audio pipeline (transcriber + diarizer)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.audio.transcriber import (
    Transcriber,
    TranscriptSegment,
    WordTimestamp,
    _seconds_to_srt_time,
)
from src.audio.diarizer import Diarizer, SpeakerSegment


# ===================== WordTimestamp =====================

class TestWordTimestamp:
    def test_creation(self):
        w = WordTimestamp(word="hello", start=1.0, end=1.5, probability=0.95)
        assert w.word == "hello"
        assert w.start == 1.0
        assert w.end == 1.5
        assert w.probability == 0.95


# ===================== TranscriptSegment =====================

class TestTranscriptSegment:
    def test_creation_minimal(self):
        seg = TranscriptSegment(start=0.0, end=5.2, text="Hello world")
        assert seg.start == 0.0
        assert seg.end == 5.2
        assert seg.text == "Hello world"
        assert seg.language is None
        assert seg.confidence == 0.0
        assert seg.words == []

    def test_creation_with_words(self):
        words = [
            WordTimestamp("Hello", 0.0, 0.5, 0.99),
            WordTimestamp("world", 0.5, 1.0, 0.95),
        ]
        seg = TranscriptSegment(
            start=0.0, end=1.0, text="Hello world",
            language="en", confidence=-0.3, words=words,
        )
        assert len(seg.words) == 2
        assert seg.language == "en"


# ===================== Transcriber =====================

class TestTranscriber:
    def test_init_defaults(self):
        t = Transcriber()
        assert t.model_size == "large-v3"
        assert t.device == "cuda"
        assert t.compute_type == "float16"
        assert t.language is None
        assert t.beam_size == 5
        assert t.vad_filter is True
        assert t.word_timestamps is True
        assert t._model is None

    def test_init_cpu_config(self):
        t = Transcriber(
            model_size="small", device="cpu",
            compute_type="int8", language="tr",
        )
        assert t.model_size == "small"
        assert t.device == "cpu"
        assert t.compute_type == "int8"
        assert t.language == "tr"

    def test_get_text_at_timerange(self):
        """Test time-range text extraction without needing a model."""
        t = Transcriber()
        segments = [
            TranscriptSegment(start=0.0, end=3.0, text="First segment"),
            TranscriptSegment(start=3.0, end=6.0, text="Second segment"),
            TranscriptSegment(start=6.0, end=9.0, text="Third segment"),
            TranscriptSegment(start=9.0, end=12.0, text="Fourth segment"),
        ]

        # Exact range match
        text = t.get_text_at_timerange(segments, 3.0, 6.0)
        assert text == "Second segment"

        # Spanning two segments
        text = t.get_text_at_timerange(segments, 2.0, 7.0)
        assert "First segment" in text
        assert "Second segment" in text
        assert "Third segment" in text

        # No match
        text = t.get_text_at_timerange(segments, 20.0, 25.0)
        assert text == ""

    def test_get_text_at_timerange_with_words(self):
        """Test precise word-level extraction."""
        t = Transcriber()
        segments = [
            TranscriptSegment(
                start=0.0, end=3.0, text="Hello beautiful world",
                words=[
                    WordTimestamp("Hello", 0.0, 0.5, 0.9),
                    WordTimestamp("beautiful", 0.5, 1.2, 0.9),
                    WordTimestamp("world", 1.2, 1.8, 0.9),
                ],
            ),
        ]

        # Only get words in range 0.4-1.3
        text = t.get_text_at_timerange(segments, 0.4, 1.3)
        assert "Hello" in text
        assert "beautiful" in text
        assert "world" in text  # overlaps with range

    def test_segments_to_srt(self):
        t = Transcriber()
        segments = [
            TranscriptSegment(start=0.0, end=2.5, text="First line"),
            TranscriptSegment(start=3.0, end=5.0, text="Second line"),
        ]

        srt = t.segments_to_srt(segments)
        lines = srt.strip().split("\n")

        assert lines[0] == "1"
        assert lines[1] == "00:00:00,000 --> 00:00:02,500"
        assert lines[2] == "First line"
        assert lines[4] == "2"
        assert lines[5] == "00:00:03,000 --> 00:00:05,000"
        assert lines[6] == "Second line"

    def test_transcribe_file_not_found(self):
        t = Transcriber()
        try:
            t.transcribe("/nonexistent/audio.wav")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass


# ===================== SRT Time =====================

class TestSrtTime:
    def test_zero(self):
        assert _seconds_to_srt_time(0.0) == "00:00:00,000"

    def test_simple(self):
        assert _seconds_to_srt_time(65.5) == "00:01:05,500"

    def test_hours(self):
        assert _seconds_to_srt_time(3723.123) == "01:02:03,123"


# ===================== SpeakerSegment =====================

class TestSpeakerSegment:
    def test_creation(self):
        s = SpeakerSegment(start=0.0, end=5.0, speaker="SPEAKER_00")
        assert s.start == 0.0
        assert s.end == 5.0
        assert s.speaker == "SPEAKER_00"


# ===================== Diarizer =====================

class TestDiarizer:
    def test_init_defaults(self):
        d = Diarizer()
        assert d.min_speakers is None
        assert d.max_speakers is None
        assert d.device == "cuda"
        assert d._pipeline is None

    def test_init_custom(self):
        d = Diarizer(min_speakers=1, max_speakers=3, device="cpu")
        assert d.min_speakers == 1
        assert d.max_speakers == 3
        assert d.device == "cpu"

    def test_get_speaker_at_time(self):
        d = Diarizer()
        segments = [
            SpeakerSegment(0.0, 5.0, "SPEAKER_00"),
            SpeakerSegment(5.5, 10.0, "SPEAKER_01"),
            SpeakerSegment(10.0, 15.0, "SPEAKER_00"),
        ]

        assert d.get_speaker_at_time(segments, 3.0) == "SPEAKER_00"
        assert d.get_speaker_at_time(segments, 7.0) == "SPEAKER_01"
        assert d.get_speaker_at_time(segments, 12.0) == "SPEAKER_00"
        assert d.get_speaker_at_time(segments, 5.2) is None  # in gap

    def test_get_dominant_speaker(self):
        d = Diarizer()
        segments = [
            SpeakerSegment(0.0, 8.0, "SPEAKER_00"),   # 8s
            SpeakerSegment(3.0, 5.0, "SPEAKER_01"),    # 2s overlap in 0-10
            SpeakerSegment(8.5, 10.0, "SPEAKER_01"),   # 1.5s
        ]

        # SPEAKER_00 dominates 0-10 range
        dominant = d.get_dominant_speaker(segments, 0.0, 10.0)
        assert dominant == "SPEAKER_00"

        # SPEAKER_01 dominates 8-10 range
        dominant = d.get_dominant_speaker(segments, 8.0, 10.0)
        assert dominant == "SPEAKER_01"

        # Empty range
        dominant = d.get_dominant_speaker(segments, 20.0, 25.0)
        assert dominant is None

    def test_merge_short_segments(self):
        d = Diarizer()
        segments = [
            SpeakerSegment(0.0, 3.0, "SPEAKER_00"),
            SpeakerSegment(3.1, 6.0, "SPEAKER_00"),    # same speaker, small gap -> merge
            SpeakerSegment(6.5, 7.0, "SPEAKER_01"),     # different speaker
            SpeakerSegment(7.5, 7.6, "SPEAKER_01"),     # very short -> drop
            SpeakerSegment(8.0, 12.0, "SPEAKER_00"),
        ]

        merged = d.merge_short_segments(segments, min_duration=0.5)

        # First two should merge, the 0.1s one should be dropped
        assert len(merged) < len(segments)
        assert merged[0].speaker == "SPEAKER_00"
        assert merged[0].start == 0.0
        assert merged[0].end == 6.0  # merged

    def test_merge_empty(self):
        d = Diarizer()
        assert d.merge_short_segments([]) == []

    def test_diarize_file_not_found(self):
        d = Diarizer(hf_token="fake_token")
        try:
            d.diarize("/nonexistent/audio.wav")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass


def run_all_tests():
    """Run all tests without pytest."""
    test_classes = [
        TestWordTimestamp,
        TestTranscriptSegment,
        TestTranscriber,
        TestSrtTime,
        TestSpeakerSegment,
        TestDiarizer,
    ]

    total = 0
    passed = 0
    failed = 0

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]

        for method_name in methods:
            total += 1
            test_name = f"{cls.__name__}.{method_name}"
            try:
                getattr(instance, method_name)()
                print(f"  PASS: {test_name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL: {test_name} -> {e}")
                failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 50}")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
