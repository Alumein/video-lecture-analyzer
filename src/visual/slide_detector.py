"""Slide transition detection - dual-method (pixel + edge) with edge-based dedup."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.utils.logger import setup_logger
from src.utils.video_utils import format_timestamp

logger = setup_logger(__name__)


@dataclass
class SlideTransition:
    """Represents a detected slide transition."""
    timestamp: float
    frame_index: int
    confidence: float
    image_path: str | None = None


class SlideDetector:
    """Detects slide transitions using pixel + edge difference.

    Uses BOTH pixel diff (for visual changes) and Canny edge diff
    (for text changes on same background) so transitions are caught
    even when only text changes.

    Duplicate detection uses edge fingerprints, not pHash, because
    pHash is dominated by background and misses text differences.
    """

    def __init__(
        self,
        threshold: float | None = None,
        min_interval: float = 2.0,
        resize_width: int = 480,
        duplicate_threshold: float = 0.90,
    ):
        self.threshold = threshold
        self.min_interval = min_interval
        self.resize_width = resize_width
        self.duplicate_threshold = duplicate_threshold

    def detect(self, video_path: str) -> list[SlideTransition]:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        logger.info(
            f"Analyzing video: {path.name} "
            f"({total_frames} frames, {fps:.1f} fps, {format_timestamp(duration)})"
        )

        # Pass 1: Collect pixel and edge diffs at sampled frames
        diff_scores, frames_data = self._collect_diffs(cap, fps, total_frames)

        # Determine pixel and edge thresholds
        pixel_thresh, edge_thresh = self._determine_thresholds(diff_scores)

        # Pass 2: Find transitions
        candidates = self._find_transitions(
            diff_scores, frames_data, pixel_thresh, edge_thresh
        )
        logger.info(
            f"Pass 1: {len(candidates)} transitions "
            f"(pixel_thr={pixel_thresh:.2f}, edge_thr={edge_thresh:.3f})"
        )

        # Pass 3: Remove duplicates using edge fingerprints
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        unique = self._remove_duplicates(cap, candidates, fps)
        logger.info(f"Pass 2: {len(unique)} unique slides")

        cap.release()
        return unique

    def _collect_diffs(self, cap, fps, total_frames):
        """Sample frames using sequential grab+read (no seeking).

        Returns list of (pixel_diff, edge_diff) tuples.
        """
        sample_interval = max(int(fps * 1.0), 1)

        diff_scores = []
        frames_data = []
        prev_gray = None
        prev_edges = None

        total_samples = total_frames // sample_interval
        progress = tqdm(total=total_samples, desc="Detecting slides", unit="sample")

        frame_index = 0
        while frame_index < total_frames:
            if frame_index % sample_interval == 0:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = self._preprocess(frame)
                edges = cv2.Canny(gray, 50, 150)
                timestamp = frame_index / fps

                if prev_gray is not None:
                    pixel_diff = float(np.mean(cv2.absdiff(prev_gray, gray)))
                    edge_xor = cv2.bitwise_xor(prev_edges, edges)
                    edge_total = max(
                        np.count_nonzero(prev_edges) + np.count_nonzero(edges), 1
                    )
                    edge_diff = np.count_nonzero(edge_xor) / edge_total

                    diff_scores.append((pixel_diff, edge_diff))
                    frames_data.append((frame_index, timestamp))
                else:
                    diff_scores.append((0.0, 0.0))
                    frames_data.append((frame_index, timestamp))

                prev_gray = gray
                prev_edges = edges
                progress.update(1)
            else:
                cap.grab()

            frame_index += 1

        progress.close()

        scores = np.array(diff_scores)
        pixel_scores = scores[:, 0]
        edge_scores = scores[:, 1]
        logger.info(
            f"Pixel diff: mean={pixel_scores.mean():.2f}, "
            f"std={pixel_scores.std():.2f}, max={pixel_scores.max():.2f}"
        )
        logger.info(
            f"Edge diff: mean={edge_scores.mean():.3f}, "
            f"std={edge_scores.std():.3f}, max={edge_scores.max():.3f}"
        )

        return diff_scores, frames_data

    def _determine_thresholds(self, diff_scores):
        """Determine pixel and edge thresholds from video statistics."""
        if self.threshold is not None:
            return (self.threshold, 0.50)

        scores = np.array(diff_scores)
        pixel_scores = scores[:, 0]
        edge_scores = scores[:, 1]

        pixel_thresh = max(pixel_scores.mean() + 1.8 * pixel_scores.std(), 1.5)
        edge_thresh = max(edge_scores.mean() + 1.5 * edge_scores.std(), 0.50)

        logger.info(
            f"Auto thresholds - pixel: {pixel_thresh:.2f}, edge: {edge_thresh:.3f}"
        )
        return (pixel_thresh, edge_thresh)

    def _find_transitions(self, diff_scores, frames_data, pixel_thresh, edge_thresh):
        """Find transitions where either pixel OR edge diff exceeds threshold."""
        candidates = []

        candidates.append(SlideTransition(
            timestamp=0.0, frame_index=0, confidence=1.0
        ))
        last_transition_time = 0.0

        for i, ((pixel_diff, edge_diff), (frame_idx, timestamp)) in enumerate(
            zip(diff_scores, frames_data)
        ):
            if i == 0:
                continue

            time_since_last = timestamp - last_transition_time
            pixel_trigger = pixel_diff >= pixel_thresh
            edge_trigger = edge_diff >= edge_thresh

            if (pixel_trigger or edge_trigger) and time_since_last >= self.min_interval:
                if pixel_trigger and edge_trigger:
                    confidence = 1.0
                elif pixel_trigger:
                    confidence = round(min(pixel_diff / (pixel_thresh * 2), 1.0), 3)
                else:
                    confidence = round(min(edge_diff / (edge_thresh * 2), 1.0), 3)

                candidates.append(SlideTransition(
                    timestamp=timestamp,
                    frame_index=frame_idx,
                    confidence=confidence,
                ))
                last_transition_time = timestamp

        return candidates

    def _remove_duplicates(self, cap, candidates, fps):
        """Remove near-identical slides using edge fingerprints.

        Edge fingerprints are MUCH better than pHash for slides because
        they capture text and structure, not background colors.
        """
        if len(candidates) <= 1:
            return candidates

        unique = []
        accepted_fingerprints = []

        for c in candidates:
            target = min(
                c.frame_index + max(int(fps) // 2, 5),
                int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1,
            )
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ret, frame = cap.read()
            if not ret:
                continue

            fp = self._compute_edge_fingerprint(frame)

            is_dup = False
            for af in accepted_fingerprints:
                xor = np.logical_xor(fp, af)
                total_edges = max(np.count_nonzero(fp) + np.count_nonzero(af), 1)
                diff_ratio = np.count_nonzero(xor) / total_edges
                similarity = 1.0 - diff_ratio
                if similarity > self.duplicate_threshold:
                    is_dup = True
                    break

            if not is_dup:
                unique.append(c)
                accepted_fingerprints.append(fp)

        return unique

    def _compute_edge_fingerprint(self, frame):
        """Edge-based fingerprint capturing text and structure."""
        h, w = frame.shape[:2]
        crop_w = int(w * 0.75)  # Ignore right webcam panel
        frame = frame[:, :crop_w]

        resized = cv2.resize(frame, (256, 144))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 50, 150)
        # Dilate so 1-px jitter between repeat captures still overlaps
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        return edges.flatten() > 0

    def _preprocess(self, frame):
        """Crop webcam, resize, grayscale, blur."""
        h, w = frame.shape[:2]
        crop_w = int(w * 0.75)
        frame = frame[:, :crop_w]

        h, w = frame.shape[:2]
        if w > self.resize_width:
            scale = self.resize_width / w
            frame = cv2.resize(frame, (self.resize_width, int(h * scale)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def extract_slide_images(self, video_path, transitions, output_dir, format="png"):
        """Extract slide images at transition points."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        saved = []
        digits = len(str(len(transitions)))

        for i, t in enumerate(transitions):
            target = min(
                t.frame_index + max(int(fps) // 2, 5),
                int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1,
            )
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ret, frame = cap.read()
            if not ret:
                continue

            fn = f"slide_{str(i+1).zfill(digits)}.{format}"
            fp = out_dir / fn
            cv2.imwrite(str(fp), frame)
            saved.append(str(fp))
            t.image_path = fn

        cap.release()
        logger.info(f"Saved {len(saved)} slide images to {out_dir}")
        return saved
