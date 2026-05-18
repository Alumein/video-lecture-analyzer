"""Merges visual, audio, and NLP outputs into a unified JSON payload.

Key design: Segments are based on TOPIC CHANGES in the transcript,
not on slide transitions. Slides are attached to segments as references.
"""

from src.utils.logger import setup_logger
from src.utils.video_utils import format_timestamp

logger = setup_logger(__name__)


class Merger:
    """Synchronizes and merges outputs from all pipelines.

    Segmentation priority:
        1. NLP topic segments (from Gemini) — best quality
        2. Topic detection from transcript (pause/keyword based) — fallback
        3. Slide transitions — last resort
    """

    def merge_by_topics(
        self,
        transcript_segments: list,
        speaker_segments: list | None,
        transitions: list,
        video_duration: float,
        min_segment_duration: float = 60.0,
    ) -> list[dict]:
        """Merge data using transcript-based topic detection.

        Groups transcript segments into larger topic-based segments
        by detecting significant pauses and topic shifts.

        Args:
            transcript_segments: TranscriptSegment objects from Whisper.
            speaker_segments: SpeakerSegment objects (optional).
            transitions: SlideTransition objects (for slide image references).
            video_duration: Total video duration in seconds.
            min_segment_duration: Minimum segment duration in seconds.
                                  Prevents too many tiny segments.

        Returns:
            List of merged segment dicts.
        """
        if not transcript_segments:
            # No transcript — fall back to slide-based segmentation
            return self._merge_by_slides(transitions, video_duration)

        # Detect topic boundaries from transcript pauses and content
        boundaries = self._detect_topic_boundaries(
            transcript_segments, min_segment_duration
        )

        # Build segments from boundaries
        segments = []
        for i, (start, end) in enumerate(boundaries):
            # Collect transcript text for this segment
            text = self._get_transcript_for_range(start, end, transcript_segments)

            # Find best matching slide for this segment
            slide_image = self._find_slide_for_range(start, end, transitions)

            # Find dominant speaker
            speaker = None
            if speaker_segments:
                speaker = self._get_dominant_speaker(start, end, speaker_segments)

            # Generate topic title from first sentence
            topic = self._extract_topic_title(text, i + 1)

            segments.append({
                "id": i + 1,
                "start_time": round(start, 3),
                "end_time": round(end, 3),
                "start_fmt": format_timestamp(start),
                "end_fmt": format_timestamp(end),
                "duration": round(end - start, 3),
                "slide_image": slide_image,
                "transcript": text,
                "speaker": speaker,
                "topic": topic,
                "summary": "",
                "key_points": [],
            })

        logger.info(f"Merged {len(segments)} topic-based segments")
        return segments

    def _detect_topic_boundaries(
        self,
        transcript_segments: list,
        min_duration: float,
    ) -> list[tuple[float, float]]:
        """Detect topic boundaries from transcript timing patterns.

        A new topic is detected when:
            - There's a significant pause (>2s gap between segments)
            - The segment has been going on for long enough (>min_duration)

        Args:
            transcript_segments: Sorted transcript segments with start/end times.
            min_duration: Minimum seconds per topic segment.

        Returns:
            List of (start, end) tuples for each topic segment.
        """
        if not transcript_segments:
            return []

        boundaries: list[tuple[float, float]] = []
        current_start = transcript_segments[0].start
        prev_end = transcript_segments[0].end

        for seg in transcript_segments[1:]:
            gap = seg.start - prev_end
            elapsed = prev_end - current_start

            # Detect topic boundary: significant pause AND minimum duration met
            if gap > 2.0 and elapsed >= min_duration:
                boundaries.append((current_start, prev_end))
                current_start = seg.start

            prev_end = seg.end

        # Add final segment
        if prev_end > current_start:
            boundaries.append((current_start, prev_end))

        # Merge very short trailing segments into the previous one
        merged = []
        for start, end in boundaries:
            if merged and (end - start) < min_duration * 0.5:
                # Too short — merge with previous
                prev_start, _ = merged[-1]
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))

        return merged

    def _merge_by_slides(self, transitions: list, video_duration: float) -> list[dict]:
        """Fallback: segment by slide transitions when no transcript is available."""
        if not transitions:
            return []

        segments = []
        for i, transition in enumerate(transitions):
            start = transition.timestamp
            end = transitions[i + 1].timestamp if i + 1 < len(transitions) else video_duration

            segments.append({
                "id": i + 1,
                "start_time": round(start, 3),
                "end_time": round(end, 3),
                "start_fmt": format_timestamp(start),
                "end_fmt": format_timestamp(end),
                "duration": round(end - start, 3),
                "slide_image": transition.image_path,
                "transcript": "",
                "speaker": None,
                "topic": f"Slide {i + 1}",
                "summary": "",
                "key_points": [],
            })

        logger.info(f"Merged {len(segments)} slide-based segments (no transcript)")
        return segments

    def enrich_with_nlp(self, segments: list[dict], topic_segments: list) -> list[dict]:
        """Add NLP results (topics, summaries, key points) to merged segments."""
        for seg in segments:
            best_match = self._find_matching_topic(
                seg["start_time"], seg["end_time"], topic_segments
            )
            if best_match:
                seg["topic"] = best_match.topic
                seg["summary"] = best_match.summary
                seg["key_points"] = best_match.key_points
        return segments

    def merge_by_llm_topics(
        self,
        topics: list,
        transcript_segments: list,
        speaker_segments: list | None,
        transitions: list,
        slide_texts: dict | None = None,
    ) -> list[dict]:
        """Build segments directly from LLM-determined topic boundaries.

        The LLM has already identified the topic boundaries, summaries,
        and key points. This method builds the final segments using
        those boundaries and fills in transcript text, slide images,
        and speaker info.

        Args:
            topics: TopicSegment objects from LLM (with start/end times).
            transcript_segments: TranscriptSegment objects for text.
            speaker_segments: SpeakerSegment objects (optional).
            transitions: SlideTransition objects for slide images.

        Returns:
            List of segment dicts with all data merged.
        """
        segments = []

        for i, topic in enumerate(topics):
            start = topic.start_time
            end = topic.end_time

            text = self._get_transcript_for_range(start, end, transcript_segments)
            slide_images = self._find_slides_for_range(start, end, transitions)
            speaker = None
            if speaker_segments:
                speaker = self._get_dominant_speaker(start, end, speaker_segments)

            # Get OCR text for slides in this range
            slide_ocr_texts = []
            if slide_texts:
                for t in transitions:
                    if t.image_path and t.image_path in slide_texts:
                        if start <= t.timestamp <= end:
                            slide_ocr_texts.append(slide_texts[t.image_path])

            segments.append({
                "id": i + 1,
                "start_time": round(start, 3),
                "end_time": round(end, 3),
                "start_fmt": format_timestamp(start),
                "end_fmt": format_timestamp(end),
                "duration": round(end - start, 3),
                "slide_image": slide_images[0] if slide_images else None,
                "slide_images": slide_images,
                "slide_text": " | ".join(slide_ocr_texts) if slide_ocr_texts else "",
                "transcript": text,
                "speaker": speaker,
                "topic": topic.topic,
                "summary": topic.summary,
                "key_points": topic.key_points,
            })

        logger.info(f"Built {len(segments)} segments from LLM topics")
        return segments

    def _get_transcript_for_range(self, start: float, end: float, segments: list) -> str:
        """Get concatenated transcript text for a time range."""
        parts = []
        for seg in segments:
            if seg.end <= start:
                continue
            if seg.start >= end:
                break
            parts.append(seg.text)
        return " ".join(parts).strip()

    def _find_slide_for_range(self, start: float, end: float, transitions: list) -> str | None:
        """Find the most relevant slide image for a time range."""
        best_slide = None
        for t in transitions:
            if t.timestamp <= end:
                best_slide = t.image_path
            if t.timestamp > start:
                break
        return best_slide

    def _find_slides_for_range(self, start: float, end: float, transitions: list) -> list[str]:
        """Find ALL slide images within a time range."""
        slides = []
        # Include the slide that was active when this range started
        last_before = None
        for t in transitions:
            if t.timestamp < start:
                last_before = t.image_path
            elif t.timestamp <= end:
                if last_before and last_before not in slides:
                    slides.append(last_before)
                    last_before = None
                if t.image_path and t.image_path not in slides:
                    slides.append(t.image_path)
            else:
                break
        # If no slides found in range, use the last one before range
        if not slides and last_before:
            slides.append(last_before)
        return slides

    def _get_dominant_speaker(self, start: float, end: float, segments: list) -> str | None:
        """Find speaker with most speaking time in range."""
        durations: dict[str, float] = {}
        for seg in segments:
            overlap_start = max(seg.start, start)
            overlap_end = min(seg.end, end)
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > 0:
                durations[seg.speaker] = durations.get(seg.speaker, 0) + overlap
        if not durations:
            return None
        return max(durations, key=durations.get)

    def _find_matching_topic(self, start: float, end: float, topics: list):
        """Find topic segment with most overlap."""
        best = None
        best_overlap = 0
        for topic in topics:
            o_start = max(start, topic.start_time)
            o_end = min(end, topic.end_time)
            overlap = max(0.0, o_end - o_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best = topic
        return best

    @staticmethod
    def _extract_topic_title(text: str, segment_id: int) -> str:
        """Extract a topic title from the first sentence of transcript text."""
        if not text:
            return f"Segment {segment_id}"

        # Take first sentence
        first = text
        for sep in [". ", "? ", "! ", "\n"]:
            idx = text.find(sep)
            if 0 < idx < 120:
                first = text[:idx]
                break

        if len(first) > 80:
            first = first[:77] + "..."

        title = first.strip().strip(",.;:").strip()
        return title if title else f"Segment {segment_id}"
