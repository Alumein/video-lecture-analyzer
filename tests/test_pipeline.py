"""Tests for the main pipeline."""

from src.integration.pipeline import AnalysisResult, Pipeline
from src.utils.video_utils import validate_video_file


class TestValidateVideoFile:
    def test_nonexistent_file(self):
        assert validate_video_file("nonexistent.mp4") is False

    def test_unsupported_format(self):
        # Would need a real file to fully test
        assert validate_video_file("file.txt") is False


class TestAnalysisResult:
    def test_default_creation(self):
        result = AnalysisResult()
        assert result.metadata == {}
        assert result.segments == []

    def test_with_data(self):
        result = AnalysisResult(
            metadata={"source": "test.mp4"},
            segments=[{"id": 1, "topic": "Intro"}],
        )
        assert result.metadata["source"] == "test.mp4"
        assert len(result.segments) == 1
