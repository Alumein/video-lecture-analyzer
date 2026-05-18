"""Speech-to-text transcription using Faster-Whisper."""

from dataclasses import dataclass, field
from pathlib import Path

from src.utils.logger import setup_logger
from src.utils.video_utils import format_timestamp

logger = setup_logger(__name__)


@dataclass
class WordTimestamp:
    """A single word with its timestamp."""

    word: str
    start: float
    end: float
    probability: float


@dataclass
class TranscriptSegment:
    """A segment of transcribed speech."""

    start: float
    end: float
    text: str
    language: str | None = None
    confidence: float = 0.0
    words: list[WordTimestamp] = field(default_factory=list)


class Transcriber:
    """Transcribes audio using Faster-Whisper for local speech-to-text.

    Faster-Whisper is a CTranslate2-based reimplementation of Whisper
    that runs up to 4x faster with the same accuracy.

    Features:
        - Segment-level and word-level timestamps
        - Language auto-detection or forced language
        - VAD (Voice Activity Detection) filtering to skip silence
        - Lazy model loading (loaded on first transcribe call)

    Model sizes and approximate VRAM usage:
        - tiny:    ~1 GB   (fastest, least accurate)
        - base:    ~1 GB
        - small:   ~2 GB
        - medium:  ~5 GB
        - large-v3: ~10 GB (slowest, most accurate)

    For CPU usage, set device="cpu" and compute_type="int8".
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        word_timestamps: bool = True,
    ):
        """
        Args:
            model_size: Whisper model to use (tiny/base/small/medium/large-v3).
            device: Compute device - "cuda" for GPU, "cpu" for CPU.
            compute_type: Precision - "float16" for GPU, "int8" for CPU.
            language: Force language code (e.g. "tr", "en"). None = auto-detect.
            beam_size: Beam search width. Higher = better quality, slower.
            vad_filter: Use Silero VAD to skip silent parts.
            word_timestamps: Enable word-level timestamps.
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.word_timestamps = word_timestamps
        self._model = None

    def _load_model(self):
        """Lazy-load the Whisper model on first use."""
        if self._model is not None:
            return

        logger.info(
            f"Loading Whisper model: {self.model_size} "
            f"(device={self.device}, compute_type={self.compute_type})"
        )

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper is not installed. "
                "Install it with: pip install faster-whisper"
            )

        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

        logger.info(f"Whisper model loaded: {self.model_size}")

    def transcribe(self, audio_path: str) -> list[TranscriptSegment]:
        """Transcribe an audio file to text with timestamps.

        Args:
            audio_path: Path to audio file (16kHz mono WAV recommended).

        Returns:
            List of TranscriptSegment with timestamps, text, and optionally words.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._load_model()

        logger.info(f"Transcribing: {path.name}")

        # Run transcription
        segments_iter, info = self._model.transcribe(
            str(path),
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            word_timestamps=self.word_timestamps,
        )

        detected_language = info.language
        language_prob = info.language_probability
        logger.info(
            f"Detected language: {detected_language} "
            f"(probability: {language_prob:.2%})"
        )

        # Collect segments
        result: list[TranscriptSegment] = []
        total_duration = 0.0

        for segment in segments_iter:
            # Build word timestamps if available
            words = []
            if self.word_timestamps and segment.words:
                words = [
                    WordTimestamp(
                        word=w.word.strip(),
                        start=round(w.start, 3),
                        end=round(w.end, 3),
                        probability=round(w.probability, 3),
                    )
                    for w in segment.words
                    if w.word.strip()
                ]

            text = segment.text.strip()
            if not text:
                continue

            ts = TranscriptSegment(
                start=round(segment.start, 3),
                end=round(segment.end, 3),
                text=text,
                language=detected_language,
                confidence=round(segment.avg_logprob, 4) if segment.avg_logprob else 0.0,
                words=words,
            )
            result.append(ts)
            total_duration = max(total_duration, segment.end)

        logger.info(
            f"Transcription complete: {len(result)} segments, "
            f"duration={format_timestamp(total_duration)}, "
            f"language={detected_language}"
        )

        return result

    def transcribe_with_full_text(self, audio_path: str) -> tuple[str, list[TranscriptSegment]]:
        """Transcribe and return both full text and segments.

        Convenience method that returns the concatenated full transcript
        along with the detailed segments.

        Args:
            audio_path: Path to audio file.

        Returns:
            Tuple of (full_text, segments).
        """
        segments = self.transcribe(audio_path)
        full_text = " ".join(seg.text for seg in segments)
        return full_text, segments

    def get_text_at_timerange(
        self, segments: list[TranscriptSegment], start: float, end: float
    ) -> str:
        """Extract transcript text for a specific time range.

        Useful for getting the text that corresponds to a specific slide.

        Args:
            segments: Previously transcribed segments.
            start: Start time in seconds.
            end: End time in seconds.

        Returns:
            Concatenated text within the time range.
        """
        parts = []
        for seg in segments:
            # Check for overlap with the requested range
            if seg.end <= start:
                continue
            if seg.start >= end:
                break

            # If word timestamps are available, be more precise
            if seg.words:
                word_texts = [
                    w.word for w in seg.words
                    if w.start < end and w.end > start
                ]
                if word_texts:
                    parts.append(" ".join(word_texts))
            else:
                parts.append(seg.text)

        return " ".join(parts).strip()

    def segments_to_srt(self, segments: list[TranscriptSegment]) -> str:
        """Convert transcript segments to SRT subtitle format.

        Args:
            segments: Transcribed segments.

        Returns:
            SRT formatted string.
        """
        lines = []
        for i, seg in enumerate(segments, 1):
            start_srt = _seconds_to_srt_time(seg.start)
            end_srt = _seconds_to_srt_time(seg.end)
            lines.append(f"{i}")
            lines.append(f"{start_srt} --> {end_srt}")
            lines.append(seg.text)
            lines.append("")

        return "\n".join(lines)


def _seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
