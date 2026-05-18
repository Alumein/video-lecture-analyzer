"""Visual pipeline - Slide detection and frame extraction."""

from src.visual.slide_detector import SlideDetector, SlideTransition
from src.visual.frame_extractor import FrameExtractor
from src.visual.slide_ocr import SlideOCR

__all__ = ["SlideDetector", "SlideTransition", "FrameExtractor", "SlideOCR"]
