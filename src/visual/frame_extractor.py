"""Key frame extraction from video at arbitrary timestamps."""

from pathlib import Path

import cv2

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class FrameExtractor:
    """Extracts and saves key frames from video at specified timestamps.

    Can be used independently of SlideDetector for extracting frames
    at any arbitrary timestamp (e.g., every N seconds, at specific events).
    """

    def __init__(self, output_format: str = "png", quality: int = 85):
        """
        Args:
            output_format: Image format - "png" (lossless) or "jpg" (smaller).
            quality: JPEG quality 1-100. Ignored for PNG.
        """
        self.output_format = output_format.lower().strip(".")
        self.quality = quality

        if self.output_format not in ("png", "jpg", "jpeg"):
            raise ValueError(f"Unsupported format: {output_format}. Use png or jpg.")

    def extract_at_timestamps(
        self,
        video_path: str,
        timestamps: list[float],
        output_dir: str,
        prefix: str = "frame",
    ) -> list[str]:
        """Extract frames at given timestamps and save to disk.

        Args:
            video_path: Path to video file.
            timestamps: List of timestamps in seconds.
            output_dir: Directory to save frames.
            prefix: Filename prefix for saved images.

        Returns:
            List of saved image file paths.
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        saved_paths: list[str] = []
        num_digits = len(str(len(timestamps)))

        for i, ts in enumerate(sorted(timestamps)):
            # Seek to target frame
            target_frame = int(ts * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Could not read frame at {ts:.2f}s (frame {target_frame})")
                continue

            # Build filename
            idx_str = str(i + 1).zfill(num_digits)
            ts_str = f"{int(ts // 60):02d}m{int(ts % 60):02d}s"
            filename = f"{prefix}_{idx_str}_{ts_str}.{self.output_format}"
            filepath = out_dir / filename

            # Save with appropriate params
            if self.output_format in ("jpg", "jpeg"):
                params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
            else:
                params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

            cv2.imwrite(str(filepath), frame, params)
            saved_paths.append(str(filepath))

        cap.release()
        logger.info(f"Extracted {len(saved_paths)}/{len(timestamps)} frames to {out_dir}")
        return saved_paths

    def extract_every_n_seconds(
        self,
        video_path: str,
        interval: float,
        output_dir: str,
        prefix: str = "frame",
    ) -> list[str]:
        """Extract frames at regular intervals throughout the video.

        Args:
            video_path: Path to video file.
            interval: Seconds between each extracted frame.
            output_dir: Directory to save frames.
            prefix: Filename prefix.

        Returns:
            List of saved image file paths.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()

        timestamps = []
        t = 0.0
        while t < duration:
            timestamps.append(t)
            t += interval

        logger.info(
            f"Extracting {len(timestamps)} frames "
            f"(every {interval}s from {duration:.1f}s video)"
        )

        return self.extract_at_timestamps(video_path, timestamps, output_dir, prefix)
