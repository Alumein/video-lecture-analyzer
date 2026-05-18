"""Speaker diarization using pyannote.audio."""

import os
from dataclasses import dataclass
from pathlib import Path

from src.utils.logger import setup_logger
from src.utils.video_utils import format_timestamp

logger = setup_logger(__name__)


@dataclass
class SpeakerSegment:
    """A segment attributed to a specific speaker."""

    start: float   # seconds
    end: float     # seconds
    speaker: str   # e.g. "SPEAKER_00"


class Diarizer:
    """Identifies and separates different speakers in audio.

    Uses pyannote.audio's pre-trained speaker diarization pipeline
    to determine who is speaking at each point in the audio.

    Prerequisites:
        - Accept pyannote's conditions at https://huggingface.co/pyannote/speaker-diarization-3.1
        - Set HF_TOKEN environment variable with your HuggingFace token
        - Or pass hf_token explicitly in constructor

    The pipeline outputs speaker segments like:
        SPEAKER_00: 0.0s - 5.2s
        SPEAKER_01: 5.5s - 12.3s
        SPEAKER_00: 12.5s - 18.0s
    """

    def __init__(
        self,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        device: str = "cuda",
        hf_token: str | None = None,
        model: str = "pyannote/speaker-diarization-3.1",
    ):
        """
        Args:
            min_speakers: Minimum expected speakers. None = auto.
            max_speakers: Maximum expected speakers. None = auto.
            device: Compute device - "cuda" or "cpu".
            hf_token: HuggingFace access token. Falls back to HF_TOKEN env var.
            model: pyannote model identifier.
        """
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.device = device
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.model = model
        self._pipeline = None

    def _load_pipeline(self):
        """Lazy-load the diarization pipeline on first use."""
        if self._pipeline is not None:
            return

        if not self.hf_token:
            raise ValueError(
                "HuggingFace token required for pyannote.audio. "
                "Set HF_TOKEN environment variable or pass hf_token to constructor. "
                "Get a token at: https://huggingface.co/settings/tokens"
            )

        logger.info(f"Loading diarization pipeline: {self.model}")

        try:
            from pyannote.audio import Pipeline
        except ImportError:
            raise ImportError(
                "pyannote.audio is not installed. "
                "Install it with: pip install pyannote.audio"
            )

        self._pipeline = Pipeline.from_pretrained(
            self.model,
            use_auth_token=self.hf_token,
        )

        # Move to device
        import torch
        device = torch.device(self.device)
        self._pipeline.to(device)

        logger.info(f"Diarization pipeline loaded on {self.device}")

    def diarize(self, audio_path: str) -> list[SpeakerSegment]:
        """Run speaker diarization on an audio file.

        Args:
            audio_path: Path to audio file (WAV recommended).

        Returns:
            List of SpeakerSegment sorted by start time.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._load_pipeline()

        logger.info(f"Running speaker diarization: {path.name}")

        # Build kwargs for optional speaker count hints
        kwargs = {}
        if self.min_speakers is not None:
            kwargs["min_speakers"] = self.min_speakers
        if self.max_speakers is not None:
            kwargs["max_speakers"] = self.max_speakers

        # Run diarization
        diarization = self._pipeline(str(path), **kwargs)

        # Convert pyannote output to our data format
        segments: list[SpeakerSegment] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(SpeakerSegment(
                start=round(turn.start, 3),
                end=round(turn.end, 3),
                speaker=speaker,
            ))

        # Sort by start time
        segments.sort(key=lambda s: s.start)

        # Log summary
        speakers = set(s.speaker for s in segments)
        total_speech = sum(s.end - s.start for s in segments)
        logger.info(
            f"Diarization complete: {len(speakers)} speakers, "
            f"{len(segments)} segments, "
            f"total speech={format_timestamp(total_speech)}"
        )

        for spk in sorted(speakers):
            spk_time = sum(s.end - s.start for s in segments if s.speaker == spk)
            logger.info(f"  {spk}: {format_timestamp(spk_time)} of speech")

        return segments

    def get_speaker_at_time(self, segments: list[SpeakerSegment], time: float) -> str | None:
        """Find which speaker is active at a given timestamp.

        Args:
            segments: Diarization results.
            time: Timestamp in seconds.

        Returns:
            Speaker label or None if no one is speaking.
        """
        for seg in segments:
            if seg.start <= time <= seg.end:
                return seg.speaker
        return None

    def get_dominant_speaker(
        self, segments: list[SpeakerSegment], start: float, end: float
    ) -> str | None:
        """Find the speaker with most speaking time in a range.

        Useful for assigning a primary speaker to a slide segment.

        Args:
            segments: Diarization results.
            start: Range start in seconds.
            end: Range end in seconds.

        Returns:
            Speaker label with most speaking time, or None if empty.
        """
        speaker_durations: dict[str, float] = {}

        for seg in segments:
            # Calculate overlap with the requested range
            overlap_start = max(seg.start, start)
            overlap_end = min(seg.end, end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > 0:
                speaker_durations[seg.speaker] = (
                    speaker_durations.get(seg.speaker, 0.0) + overlap
                )

        if not speaker_durations:
            return None

        return max(speaker_durations, key=speaker_durations.get)

    def merge_short_segments(
        self, segments: list[SpeakerSegment], min_duration: float = 0.5
    ) -> list[SpeakerSegment]:
        """Merge consecutive segments from the same speaker and drop very short ones.

        Cleans up diarization output by combining adjacent same-speaker turns
        and removing tiny segments that are likely noise.

        Args:
            segments: Raw diarization segments.
            min_duration: Minimum segment duration in seconds.

        Returns:
            Cleaned list of SpeakerSegment.
        """
        if not segments:
            return []

        merged: list[SpeakerSegment] = [segments[0]]

        for seg in segments[1:]:
            prev = merged[-1]

            # Merge if same speaker and close in time (gap < 0.5s)
            if seg.speaker == prev.speaker and (seg.start - prev.end) < 0.5:
                merged[-1] = SpeakerSegment(
                    start=prev.start,
                    end=seg.end,
                    speaker=prev.speaker,
                )
            else:
                merged.append(seg)

        # Filter out very short segments
        filtered = [s for s in merged if (s.end - s.start) >= min_duration]

        if len(filtered) < len(segments):
            logger.debug(
                f"Merged {len(segments)} -> {len(filtered)} speaker segments"
            )

        return filtered
