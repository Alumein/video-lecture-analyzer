"""Topic segmentation using LLM with chunked transcript processing."""

from dataclasses import dataclass, field

from src.utils.logger import setup_logger
from src.utils.video_utils import format_timestamp

logger = setup_logger(__name__)

SYSTEM_INSTRUCTION = """You are an expert academic lecture analyst with deep knowledge across all fields. 
Your job is to break down lecture transcripts into precise, specific sub-topics — not broad categories.
You write detailed, informative summaries that capture the actual content taught.
You always respond in valid JSON format only, with no additional text or markdown."""

SEGMENTATION_PROMPT = """Analyze this SECTION of a lecture transcript and identify SPECIFIC sub-topics.

THIS SECTION COVERS: {time_range_start} to {time_range_end} ({time_range_start_sec:.0f}s to {time_range_end_sec:.0f}s)

SLIDE TRANSITIONS IN THIS SECTION:
{slide_timestamps}

TRANSCRIPT:
{transcript}

CRITICAL INSTRUCTIONS:
1. TOPIC TITLES must be SPECIFIC and DESCRIPTIVE — not generic categories.
   BAD:  "Introduction", "Variables", "Examples", "Summary"
   GOOD: "Why Human Cognitive Capacity Creates Design Constraints"
   GOOD: "Fitts's Law: Predicting Movement Time to Targets"  
   GOOD: "The Gulf of Execution vs Gulf of Evaluation in User Interfaces"
   
2. Each topic title should clearly state WHAT SPECIFIC CONCEPT is being taught.

3. SUMMARIES must be RICH and DETAILED (5-8 sentences). Each summary MUST include:
   - The specific concept/theorem/algorithm being explained
   - The exact definition or formula mentioned by the professor
   - At least ONE concrete example given in the lecture
   - The reasoning or motivation (WHY this matters)
   - Connection to other concepts in the lecture
   AVOID generic statements like "the professor discusses X" or "this section covers Y".
   INSTEAD write what was actually taught, with specifics.
   
4. KEY POINTS should be specific facts, not vague statements.
   BAD:  "The professor discusses variables"
   GOOD: "Working memory can hold 7±2 items according to Miller's Law"

5. Create segments of 2-5 minutes each. Each new concept = new segment.
   IMPORTANT: Your segments MUST cover the ENTIRE time range from {time_range_start_sec:.0f}s to {time_range_end_sec:.0f}s with NO GAPS.
   The first segment's start_time should be {time_range_start_sec:.0f}s.
   The last segment's end_time should be {time_range_end_sec:.0f}s.
   Aim for 3-6 segments per section depending on content.

6. IMPORTANT: start_time and end_time must be in SECONDS (float) and must fall 
   within this section's range: {time_range_start_sec:.0f} to {time_range_end_sec:.0f}.

JSON FORMAT (respond with ONLY this JSON array, nothing else):
[
  {{
    "topic": "Specific Topic Title Including Key Concept Name",
    "start_time": {time_range_start_sec:.0f},
    "end_time": 300.0,
    "summary": "Rich 5-8 sentence summary. Start by stating what specific concept is taught. Include the exact definition or formula. Mention at least one concrete example used. Explain why this matters and how it connects to other concepts. Be specific - reference actual content, not vague generalities.",
    "key_points": [
      "Specific point with concrete detail",
      "Another point with actual concepts/numbers"
    ]
  }}
]"""


@dataclass
class TopicSegment:
    """A topic-based segment of the lecture."""
    topic: str
    start_time: float
    end_time: float
    summary: str
    key_points: list[str] = field(default_factory=list)


class TopicSegmenter:
    """Segments lecture transcripts into topic-based sections.

    For long transcripts, uses chunked processing:
        1. Split transcript into overlapping chunks (~8K chars each)
        2. Send each chunk to LLM for topic identification
        3. Merge results, removing duplicates at chunk boundaries
    """

    def __init__(self, client, max_topics: int = 25):
        self.client = client
        self.max_topics = max_topics

    def segment(
        self,
        transcript: str,
        slide_timestamps: list[float],
    ) -> list[TopicSegment]:
        """Segment full transcript into topics.

        Automatically handles long transcripts by chunking.
        Each chunk includes its time range so the LLM knows
        what part of the lecture it's analyzing.
        """
        if not transcript.strip():
            return []

        ts_str = ", ".join(
            f"{format_timestamp(t)} ({t:.1f}s)" for t in slide_timestamps
        )
        if not ts_str:
            ts_str = "No slide transitions detected"

        logger.info(
            f"Topic segmentation: {len(transcript)} chars, "
            f"{len(slide_timestamps)} slide transitions"
        )

        # Split into chunks with time range info
        chunks = self._split_transcript_with_times(transcript, max_chunk_chars=8000)
        logger.info(f"Processing {len(chunks)} transcript chunks")

        all_topics: list[TopicSegment] = []

        for i, (chunk_text, chunk_start, chunk_end) in enumerate(chunks):
            logger.info(
                f"Chunk {i+1}/{len(chunks)}: "
                f"{format_timestamp(chunk_start)}-{format_timestamp(chunk_end)} "
                f"({len(chunk_text)} chars)"
            )

            # Filter slide timestamps for this chunk's time range
            chunk_slides = [t for t in slide_timestamps if chunk_start <= t <= chunk_end]
            chunk_ts_str = ", ".join(
                f"{format_timestamp(t)} ({t:.1f}s)" for t in chunk_slides
            ) if chunk_slides else "None in this section"

            prompt = SEGMENTATION_PROMPT.format(
                slide_timestamps=chunk_ts_str,
                transcript=chunk_text,
                time_range_start=format_timestamp(chunk_start),
                time_range_end=format_timestamp(chunk_end),
                time_range_start_sec=chunk_start,
                time_range_end_sec=chunk_end,
            )

            try:
                raw = self.client.generate_json(
                    prompt=prompt,
                    system_instruction=SYSTEM_INSTRUCTION,
                )
                topics = self._parse_segments(raw)
                all_topics.extend(topics)
                logger.info(f"  -> {len(topics)} topics found in chunk {i+1}")
            except Exception as e:
                logger.warning(f"  -> Chunk {i+1} failed: {e}")

        # Merge overlapping topics from adjacent chunks
        merged = self._merge_chunk_topics(all_topics)

        # Fill gaps between segments to ensure full coverage
        merged = self._fill_coverage_gaps(merged)

        logger.info(f"Total: {len(merged)} topic segments after merging")
        return merged

    def _fill_coverage_gaps(self, topics: list[TopicSegment]) -> list[TopicSegment]:
        """Extend topics to fill gaps between adjacent segments.
        
        If there's a gap between topic N's end and topic N+1's start,
        extend topic N to fill that gap. This ensures the segments
        cover the entire lecture timeline.
        """
        if len(topics) <= 1:
            return topics

        topics = sorted(topics, key=lambda t: t.start_time)
        
        for i in range(len(topics) - 1):
            current = topics[i]
            next_t = topics[i + 1]
            gap = next_t.start_time - current.end_time
            if gap > 0:
                # Extend current to meet next topic
                topics[i] = TopicSegment(
                    topic=current.topic,
                    start_time=current.start_time,
                    end_time=next_t.start_time,
                    summary=current.summary,
                    key_points=current.key_points,
                )
        
        return topics

    def _split_transcript(self, transcript: str, max_chunk_chars: int = 10000) -> list[str]:
        """Split transcript into chunks at natural boundaries."""
        lines = transcript.strip().split("\n")
        if len(transcript) <= max_chunk_chars:
            return [transcript]
        chunks: list[str] = []
        current_lines: list[str] = []
        current_size = 0
        for line in lines:
            line_size = len(line) + 1
            if current_size + line_size > max_chunk_chars and current_lines:
                chunks.append("\n".join(current_lines))
                overlap = current_lines[-3:] if len(current_lines) >= 3 else current_lines[-1:]
                current_lines = list(overlap)
                current_size = sum(len(l) + 1 for l in current_lines)
            current_lines.append(line)
            current_size += line_size
        if current_lines:
            chunks.append("\n".join(current_lines))
        return chunks

    def _split_transcript_with_times(
        self, transcript: str, max_chunk_chars: int = 7000
    ) -> list[tuple[str, float, float]]:
        """Split transcript into chunks, each with its time range.

        Returns list of (chunk_text, start_seconds, end_seconds).
        Parses [HH:MM:SS] timestamps from the transcript lines.
        """
        import re

        lines = transcript.strip().split("\n")

        def parse_time_from_line(line: str) -> float | None:
            """Extract timestamp in seconds from a line like [01:23:45] ..."""
            m = re.match(r'\[(\d+):(\d+):(\d+)\]', line)
            if m:
                return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
            return None

        if len(transcript) <= max_chunk_chars:
            first_t = parse_time_from_line(lines[0]) or 0.0
            last_t = parse_time_from_line(lines[-1]) or first_t
            return [(transcript, first_t, last_t)]

        chunks: list[tuple[str, float, float]] = []
        current_lines: list[str] = []
        current_size = 0

        for line in lines:
            line_size = len(line) + 1

            if current_size + line_size > max_chunk_chars and current_lines:
                # Save current chunk with time range
                chunk_text = "\n".join(current_lines)
                start_t = parse_time_from_line(current_lines[0]) or 0.0
                end_t = parse_time_from_line(current_lines[-1]) or start_t
                chunks.append((chunk_text, start_t, end_t))

                # Overlap
                overlap = current_lines[-3:] if len(current_lines) >= 3 else current_lines[-1:]
                current_lines = list(overlap)
                current_size = sum(len(l) + 1 for l in current_lines)

            current_lines.append(line)
            current_size += line_size

        # Last chunk
        if current_lines:
            chunk_text = "\n".join(current_lines)
            start_t = parse_time_from_line(current_lines[0]) or 0.0
            end_t = parse_time_from_line(current_lines[-1]) or start_t
            chunks.append((chunk_text, start_t, end_t))

        return chunks

    def _merge_chunk_topics(self, topics: list[TopicSegment]) -> list[TopicSegment]:
        """Merge topics from different chunks, removing overlaps.

        If two topics from adjacent chunks have overlapping time ranges,
        they likely represent the same topic split at a chunk boundary.
        Merge them into one.
        """
        if len(topics) <= 1:
            return topics

        # Sort by start time
        topics.sort(key=lambda t: t.start_time)

        merged: list[TopicSegment] = [topics[0]]

        for topic in topics[1:]:
            prev = merged[-1]

            # Check for significant time overlap (>30% of shorter segment)
            overlap_start = max(prev.start_time, topic.start_time)
            overlap_end = min(prev.end_time, topic.end_time)
            overlap = max(0, overlap_end - overlap_start)
            shorter_duration = min(
                prev.end_time - prev.start_time,
                topic.end_time - topic.start_time,
            )

            if shorter_duration > 0 and overlap / shorter_duration > 0.3:
                # Merge: extend previous topic to cover both
                merged[-1] = TopicSegment(
                    topic=prev.topic,  # keep first topic's name
                    start_time=min(prev.start_time, topic.start_time),
                    end_time=max(prev.end_time, topic.end_time),
                    summary=prev.summary,
                    key_points=list(set(prev.key_points + topic.key_points))[:5],
                )
            else:
                # No significant overlap — keep as separate topic
                # Adjust boundary: previous topic ends where next begins
                if topic.start_time < prev.end_time:
                    merged[-1] = TopicSegment(
                        topic=prev.topic,
                        start_time=prev.start_time,
                        end_time=topic.start_time,
                        summary=prev.summary,
                        key_points=prev.key_points,
                    )
                merged.append(topic)

        return merged

    def _parse_segments(self, raw) -> list[TopicSegment]:
        """Parse JSON response into TopicSegment objects."""
        if isinstance(raw, dict):
            for key in ("segments", "topics", "data", "results"):
                if key in raw and isinstance(raw[key], list):
                    raw = raw[key]
                    break
            else:
                raw = [raw]

        segments = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                seg = TopicSegment(
                    topic=str(item.get("topic", "Unknown")),
                    start_time=float(item.get("start_time", 0)),
                    end_time=float(item.get("end_time", 0)),
                    summary=str(item.get("summary", "")),
                    key_points=[str(p) for p in item.get("key_points", [])],
                )
                # Skip invalid segments
                if seg.end_time > seg.start_time:
                    segments.append(seg)
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping malformed segment: {e}")

        segments.sort(key=lambda s: s.start_time)
        return segments
