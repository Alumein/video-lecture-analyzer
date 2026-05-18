"""Tests for NLP pipeline (gemini_client, topic_segmenter, summarizer)."""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.nlp.gemini_client import GeminiClient
from src.nlp.topic_segmenter import TopicSegmenter, TopicSegment
from src.nlp.summarizer import Summarizer


# ===================== Mock GeminiClient =====================

class MockGeminiClient(GeminiClient):
    """A mock client that returns predefined responses without calling the API."""

    def __init__(self, responses: list[str] | None = None):
        # Skip parent __init__ validation by setting attrs directly
        self.api_key = "mock-key"
        self.model = "mock-model"
        self.temperature = 0.3
        self.max_tokens = 4096
        self.max_retries = 1
        self.retry_delay = 0.0
        self._client = "mock"
        self._responses = responses or []
        self._call_index = 0
        self._last_prompt = None

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        self._last_prompt = prompt
        if self._call_index < len(self._responses):
            resp = self._responses[self._call_index]
            self._call_index += 1
            return resp
        return '{"error": "no more mock responses"}'


# ===================== GeminiClient._parse_json_response =====================

class TestParseJsonResponse:
    def test_plain_json(self):
        result = GeminiClient._parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_with_code_fences(self):
        raw = '```json\n{"key": "value"}\n```'
        result = GeminiClient._parse_json_response(raw)
        assert result == {"key": "value"}

    def test_json_with_plain_fences(self):
        raw = '```\n[1, 2, 3]\n```'
        result = GeminiClient._parse_json_response(raw)
        assert result == [1, 2, 3]

    def test_json_with_surrounding_text(self):
        raw = 'Here is the result:\n{"key": "value"}\nHope this helps!'
        result = GeminiClient._parse_json_response(raw)
        assert result == {"key": "value"}

    def test_json_array(self):
        raw = '[{"a": 1}, {"b": 2}]'
        result = GeminiClient._parse_json_response(raw)
        assert len(result) == 2

    def test_invalid_json_raises(self):
        try:
            GeminiClient._parse_json_response("not json at all")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_nested_json(self):
        raw = '{"segments": [{"topic": "Intro", "start": 0}]}'
        result = GeminiClient._parse_json_response(raw)
        assert "segments" in result

    def test_empty_object(self):
        result = GeminiClient._parse_json_response("{}")
        assert result == {}


class TestGeminiClientInit:
    def test_empty_api_key_raises(self):
        try:
            GeminiClient(api_key="")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_valid_init(self):
        # This only tests init, not API connection
        c = GeminiClient(api_key="test-key", model="gemini-2.0-flash")
        assert c.api_key == "test-key"
        assert c.model == "gemini-2.0-flash"
        assert c.temperature == 0.3
        assert c.max_retries == 3

    def test_custom_params(self):
        c = GeminiClient(
            api_key="key", model="gemini-pro",
            temperature=0.7, max_tokens=8192, max_retries=5,
        )
        assert c.temperature == 0.7
        assert c.max_tokens == 8192
        assert c.max_retries == 5


class TestMockGeminiGenerate:
    def test_generate_returns_response(self):
        client = MockGeminiClient(responses=["Hello from mock!"])
        result = client.generate("Say hello")
        assert result == "Hello from mock!"

    def test_generate_json(self):
        client = MockGeminiClient(responses=['{"answer": 42}'])
        result = client.generate_json("Return json")
        assert result == {"answer": 42}

    def test_generate_json_with_fences(self):
        client = MockGeminiClient(responses=['```json\n{"answer": 42}\n```'])
        result = client.generate_json("Return json")
        assert result == {"answer": 42}

    def test_multiple_calls(self):
        client = MockGeminiClient(responses=["first", "second"])
        assert client.generate("q1") == "first"
        assert client.generate("q2") == "second"


# ===================== TopicSegment =====================

class TestTopicSegment:
    def test_creation(self):
        seg = TopicSegment(
            topic="Intro to ML",
            start_time=0.0,
            end_time=300.0,
            summary="Overview of machine learning concepts.",
            key_points=["Supervised learning", "Unsupervised learning"],
        )
        assert seg.topic == "Intro to ML"
        assert seg.start_time == 0.0
        assert seg.end_time == 300.0
        assert len(seg.key_points) == 2

    def test_default_key_points(self):
        seg = TopicSegment(
            topic="Test", start_time=0, end_time=10, summary="Test"
        )
        assert seg.key_points == []


# ===================== TopicSegmenter =====================

class TestTopicSegmenter:
    def _make_mock_response(self):
        return json.dumps([
            {
                "topic": "Introduction",
                "start_time": 0.0,
                "end_time": 120.0,
                "summary": "Overview of the course.",
                "key_points": ["Course structure", "Grading policy"],
            },
            {
                "topic": "Neural Networks",
                "start_time": 120.0,
                "end_time": 360.0,
                "summary": "Fundamentals of neural networks.",
                "key_points": ["Perceptrons", "Backpropagation", "Activation functions"],
            },
        ])

    def test_segment_basic(self):
        client = MockGeminiClient(responses=[self._make_mock_response()])
        segmenter = TopicSegmenter(client=client, max_topics=10)

        segments = segmenter.segment(
            transcript="[00:00] Welcome to the course...\n[02:00] Neural networks are...",
            slide_timestamps=[0.0, 120.0],
        )

        assert len(segments) == 2
        assert segments[0].topic == "Introduction"
        assert segments[0].start_time == 0.0
        assert segments[1].topic == "Neural Networks"
        assert len(segments[1].key_points) == 3

    def test_segment_empty_transcript(self):
        client = MockGeminiClient()
        segmenter = TopicSegmenter(client=client)
        segments = segmenter.segment("", [])
        assert segments == []

    def test_segment_wrapped_in_dict(self):
        """Test when LLM wraps segments in a dict."""
        wrapped = json.dumps({
            "segments": [
                {"topic": "Test", "start_time": 0, "end_time": 60,
                 "summary": "A test.", "key_points": ["Point 1"]},
            ]
        })
        client = MockGeminiClient(responses=[wrapped])
        segmenter = TopicSegmenter(client=client)

        segments = segmenter.segment("Some transcript", [0.0])
        assert len(segments) == 1
        assert segments[0].topic == "Test"

    def test_segment_sorted_by_time(self):
        """Test that segments come back sorted even if LLM returns out of order."""
        out_of_order = json.dumps([
            {"topic": "Later", "start_time": 200, "end_time": 300,
             "summary": "Later.", "key_points": []},
            {"topic": "Earlier", "start_time": 0, "end_time": 100,
             "summary": "Earlier.", "key_points": []},
        ])
        client = MockGeminiClient(responses=[out_of_order])
        segmenter = TopicSegmenter(client=client)

        segments = segmenter.segment("text", [0.0])
        assert segments[0].topic == "Earlier"
        assert segments[1].topic == "Later"

    def test_truncate_transcript(self):
        segmenter = TopicSegmenter(client=MockGeminiClient(), max_topics=5)
        long_text = "x" * 50000
        truncated = segmenter._truncate_transcript(long_text, max_chars=1000)
        assert len(truncated) < len(long_text)
        assert "TRUNCATED" in truncated

    def test_truncate_short_text_unchanged(self):
        segmenter = TopicSegmenter(client=MockGeminiClient())
        short = "Hello world"
        assert segmenter._truncate_transcript(short) == short

    def test_parse_malformed_segment(self):
        """Test that malformed entries are skipped."""
        mixed = json.dumps([
            {"topic": "Good", "start_time": 0, "end_time": 60,
             "summary": "OK", "key_points": []},
            "this is not a dict",
            {"topic": "Also Good", "start_time": 60, "end_time": 120,
             "summary": "OK", "key_points": []},
        ])
        client = MockGeminiClient(responses=[mixed])
        segmenter = TopicSegmenter(client=client)
        segments = segmenter.segment("text", [0.0])
        assert len(segments) == 2


# ===================== Summarizer =====================

class TestSummarizer:
    def test_summarize(self):
        client = MockGeminiClient(
            responses=['{"summary": "This is a summary of the lecture."}']
        )
        s = Summarizer(client=client)
        result = s.summarize("A long lecture text about machine learning...")
        assert result == "This is a summary of the lecture."

    def test_summarize_empty(self):
        client = MockGeminiClient()
        s = Summarizer(client=client)
        assert s.summarize("") == ""
        assert s.summarize("   ") == ""

    def test_extract_key_points(self):
        client = MockGeminiClient(
            responses=['{"key_points": ["Point A", "Point B", "Point C"]}']
        )
        s = Summarizer(client=client)
        points = s.extract_key_points("Lecture text here")
        assert len(points) == 3
        assert points[0] == "Point A"

    def test_extract_key_points_empty(self):
        client = MockGeminiClient()
        s = Summarizer(client=client)
        assert s.extract_key_points("") == []

    def test_full_analysis(self):
        response = json.dumps({
            "summary": "Overview of AI.",
            "key_points": ["AI is broad", "ML is a subset"],
            "difficulty_level": "beginner",
            "main_concept": "Artificial Intelligence",
        })
        client = MockGeminiClient(responses=[response])
        s = Summarizer(client=client)

        result = s.full_analysis("Lecture about AI...")
        assert result["summary"] == "Overview of AI."
        assert len(result["key_points"]) == 2
        assert result["difficulty_level"] == "beginner"
        assert result["main_concept"] == "Artificial Intelligence"

    def test_full_analysis_empty(self):
        client = MockGeminiClient()
        s = Summarizer(client=client)
        result = s.full_analysis("")
        assert result["summary"] == ""
        assert result["key_points"] == []
        assert result["difficulty_level"] == "unknown"

    def test_batch_summarize(self):
        responses = [
            json.dumps({"summary": "Sum 1", "key_points": ["P1"],
                        "difficulty_level": "beginner", "main_concept": "C1"}),
            json.dumps({"summary": "Sum 2", "key_points": ["P2", "P3"],
                        "difficulty_level": "advanced", "main_concept": "C2"}),
        ]
        client = MockGeminiClient(responses=responses)
        s = Summarizer(client=client)

        segments = [
            {"text": "First segment text", "start": 0},
            {"text": "Second segment text", "start": 100},
        ]
        results = s.batch_summarize(segments)

        assert len(results) == 2
        assert results[0]["summary"] == "Sum 1"
        assert results[1]["summary"] == "Sum 2"
        assert results[0]["start"] == 0  # original data preserved
        assert len(results[1]["key_points"]) == 2


# ===================== Runner =====================

def run_all_tests():
    """Run all tests without pytest."""
    test_classes = [
        TestParseJsonResponse,
        TestGeminiClientInit,
        TestMockGeminiGenerate,
        TestTopicSegment,
        TestTopicSegmenter,
        TestSummarizer,
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
