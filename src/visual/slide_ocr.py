"""OCR module for extracting text from slide images.

Uses Tesseract OCR to read text from detected slide images.
The extracted text is added to topic segmentation input,
giving the LLM both spoken words and slide content.
"""

import cv2
import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SlideOCR:
    """Extracts text from slide images using Tesseract OCR.

    Preprocessing pipeline:
        1. Convert to grayscale
        2. Resize for better OCR accuracy
        3. Apply adaptive thresholding for clean text
        4. Run Tesseract OCR
    """

    def __init__(self, language: str = "eng", min_confidence: int = 40):
        """
        Args:
            language: Tesseract language code ("eng", "tur", etc.)
            min_confidence: Minimum word confidence to keep (0-100).
        """
        self.language = language
        self.min_confidence = min_confidence
        self._verified = False

    def _verify(self):
        """Check that pytesseract is available."""
        if self._verified:
            return
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self._verified = True
        except Exception:
            raise RuntimeError(
                "Tesseract OCR not found. To install:\n"
                "  Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
                "  Linux: sudo apt install tesseract-ocr\n"
                "  Python: pip install pytesseract"
            )

    def extract_text(self, image_path: str) -> str:
        """Extract text from a single slide image.

        Args:
            image_path: Path to the slide image file.

        Returns:
            Extracted text, cleaned and filtered.
        """
        self._verify()
        import pytesseract

        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Could not read image: {image_path}")
            return ""

        processed = self._preprocess(img)

        try:
            # Get word-level data with confidence scores
            data = pytesseract.image_to_data(
                processed,
                lang=self.language,
                output_type=pytesseract.Output.DICT,
            )

            # Filter by confidence and build text
            words = []
            for i, word in enumerate(data["text"]):
                conf = int(data["conf"][i])
                if conf >= self.min_confidence and word.strip():
                    words.append(word.strip())

            text = " ".join(words)
            return self._clean_text(text)

        except Exception as e:
            logger.warning(f"OCR failed for {image_path}: {e}")
            return ""

    def extract_from_slides(self, slide_dir: str) -> dict[str, str]:
        """Extract text from all slide images in parallel.

        Args:
            slide_dir: Path to directory containing slide images.

        Returns:
            Dict mapping filename -> extracted text.
        """
        from pathlib import Path
        from concurrent.futures import ThreadPoolExecutor

        slides_path = Path(slide_dir)
        if not slides_path.exists():
            return {}

        image_files = sorted(
            list(slides_path.glob("*.png")) + list(slides_path.glob("*.jpg"))
        )

        logger.info(f"Running OCR on {len(image_files)} slides...")

        results = {}

        def process_one(img_path):
            text = self.extract_text(str(img_path))
            return img_path.name, text

        with ThreadPoolExecutor(max_workers=4) as pool:
            for name, text in pool.map(process_one, image_files):
                if text:
                    results[name] = text

        logger.info(
            f"OCR complete: {len(results)}/{len(image_files)} slides had text"
        )
        return results

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR accuracy."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize if too small (OCR works better on larger images)
        h, w = gray.shape
        if w < 1000:
            scale = 1000 / w
            gray = cv2.resize(gray, (1000, int(h * scale)))

        # Adaptive thresholding for clean black/white text
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        return thresh

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean OCR output text."""
        import re

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove very short garbage strings
        words = text.split()
        words = [w for w in words if len(w) > 1 or w.isalpha()]

        return " ".join(words)
