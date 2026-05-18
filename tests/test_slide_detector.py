"""Tests for slide detection and frame extraction modules."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.visual.slide_detector import SlideDetector, SlideTransition
from src.visual.frame_extractor import FrameExtractor


def create_test_video(path: str, num_frames: int = 90, fps: int = 30, slides: int = 3):
    """Create a synthetic test video with distinct slide transitions.

    Generates a video where every (num_frames // slides) frames,
    the background color changes sharply, simulating a slide change.
    """
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    frames_per_slide = num_frames // slides

    colors = [
        (40, 40, 40),       # dark gray
        (200, 200, 200),    # light gray
        (40, 40, 180),      # blue
        (40, 180, 40),      # green
        (180, 40, 40),      # red
    ]

    for i in range(num_frames):
        slide_index = min(i // frames_per_slide, len(colors) - 1)
        color = colors[slide_index % len(colors)]
        frame = np.full((height, width, 3), color, dtype=np.uint8)

        # Add some text so it's not entirely uniform
        cv2.putText(
            frame,
            f"Slide {slide_index + 1}",
            (200, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            3,
        )

        writer.write(frame)

    writer.release()
    return path


# ===================== SlideTransition =====================

class TestSlideTransition:
    def test_creation(self):
        t = SlideTransition(timestamp=10.5, frame_index=315, confidence=0.95)
        assert t.timestamp == 10.5
        assert t.frame_index == 315
        assert t.confidence == 0.95
        assert t.image_path is None

    def test_with_image_path(self):
        t = SlideTransition(
            timestamp=5.0, frame_index=150, confidence=0.8, image_path="/tmp/slide.png"
        )
        assert t.image_path == "/tmp/slide.png"


# ===================== SlideDetector =====================

class TestSlideDetector:
    def test_init_default_params(self):
        d = SlideDetector()
        assert d.threshold == 30.0
        assert d.min_interval == 2.0
        assert d.resize_width == 960
        assert d.use_grayscale is True

    def test_init_custom_params(self):
        d = SlideDetector(threshold=50.0, min_interval=5.0, resize_width=640)
        assert d.threshold == 50.0
        assert d.min_interval == 5.0
        assert d.resize_width == 640

    def test_detect_on_synthetic_video(self):
        """Test that detector finds transitions in a synthetic video."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = str(Path(tmpdir) / "test.mp4")
            create_test_video(video_path, num_frames=90, fps=30, slides=3)

            detector = SlideDetector(threshold=15.0, min_interval=0.5)
            transitions = detector.detect(video_path)

            # Should detect at least the first frame + transitions
            assert len(transitions) >= 2
            # First transition should be at t=0
            assert transitions[0].timestamp == 0.0
            assert transitions[0].confidence == 1.0

    def test_detect_nonexistent_file(self):
        detector = SlideDetector()
        with pytest.raises(FileNotFoundError):
            detector.detect("/nonexistent/video.mp4")

    def test_extract_slide_images(self):
        """Test that slide images are saved correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = str(Path(tmpdir) / "test.mp4")
            create_test_video(video_path, num_frames=90, fps=30, slides=3)

            detector = SlideDetector(threshold=15.0, min_interval=0.5)
            transitions = detector.detect(video_path)

            slides_dir = str(Path(tmpdir) / "slides")
            saved = detector.extract_slide_images(
                video_path, transitions, slides_dir
            )

            assert len(saved) == len(transitions)
            for p in saved:
                assert Path(p).exists()
                assert Path(p).suffix == ".png"

    def test_get_diff_timeline(self):
        """Test diff timeline generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = str(Path(tmpdir) / "test.mp4")
            create_test_video(video_path, num_frames=30, fps=30, slides=2)

            detector = SlideDetector()
            timeline = detector.get_diff_timeline(video_path)

            assert len(timeline) == 30
            assert timeline[0]["frame_index"] == 0
            assert timeline[0]["diff_score"] == 0.0
            # Should have a spike at the slide transition
            scores = [t["diff_score"] for t in timeline]
            assert max(scores) > 0  # at least some difference exists


# ===================== FrameExtractor =====================

class TestFrameExtractor:
    def test_init_default(self):
        fe = FrameExtractor()
        assert fe.output_format == "png"
        assert fe.quality == 85

    def test_init_jpg(self):
        fe = FrameExtractor(output_format="jpg", quality=90)
        assert fe.output_format == "jpg"
        assert fe.quality == 90

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            FrameExtractor(output_format="bmp")

    def test_extract_at_timestamps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = str(Path(tmpdir) / "test.mp4")
            create_test_video(video_path, num_frames=90, fps=30, slides=3)

            fe = FrameExtractor(output_format="jpg")
            saved = fe.extract_at_timestamps(
                video_path,
                timestamps=[0.0, 1.0, 2.0],
                output_dir=str(Path(tmpdir) / "frames"),
            )

            assert len(saved) == 3
            for p in saved:
                assert Path(p).exists()
                assert p.endswith(".jpg")

    def test_extract_every_n_seconds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = str(Path(tmpdir) / "test.mp4")
            create_test_video(video_path, num_frames=90, fps=30, slides=3)

            fe = FrameExtractor()
            saved = fe.extract_every_n_seconds(
                video_path,
                interval=1.0,
                output_dir=str(Path(tmpdir) / "frames"),
            )

            # 90 frames at 30 fps = 3 seconds -> expect 3 frames (0s, 1s, 2s)
            assert len(saved) == 3

    def test_extract_nonexistent_file(self):
        fe = FrameExtractor()
        with pytest.raises(FileNotFoundError):
            fe.extract_at_timestamps("/nonexistent.mp4", [0.0], "/tmp")
