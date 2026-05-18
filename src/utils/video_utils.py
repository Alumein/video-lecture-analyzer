"""Video file helper utilities."""

import json
import subprocess
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

SUPPORTED_FORMATS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}


def validate_video_file(video_path: str) -> bool:
    """Check if file exists and is a supported video format."""
    path = Path(video_path)
    return path.exists() and path.suffix.lower() in SUPPORTED_FORMATS


def get_video_info(video_path: str) -> dict:
    """Get video metadata using ffprobe.

    Args:
        video_path: Path to video file.

    Returns:
        Dict with keys: duration, fps, width, height, has_audio, codec, total_frames.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        probe = json.loads(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(f"ffprobe failed. Is FFmpeg installed? Error: {e}")

    # Find video stream
    video_stream = None
    has_audio = False
    for stream in probe.get("streams", []):
        if stream["codec_type"] == "video" and video_stream is None:
            video_stream = stream
        if stream["codec_type"] == "audio":
            has_audio = True

    if video_stream is None:
        raise ValueError(f"No video stream found in: {video_path}")

    # Parse FPS from r_frame_rate (e.g. "30/1" or "30000/1001")
    fps_parts = video_stream.get("r_frame_rate", "30/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0

    duration = float(probe.get("format", {}).get("duration", 0))

    info = {
        "duration": duration,
        "fps": round(fps, 2),
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "has_audio": has_audio,
        "codec": video_stream.get("codec_name", "unknown"),
        "total_frames": int(video_stream.get("nb_frames", 0)) or int(duration * fps),
    }

    logger.info(
        f"Video info: {info['width']}x{info['height']}, "
        f"{info['fps']} fps, {info['duration']:.1f}s, "
        f"~{info['total_frames']} frames"
    )
    return info


def extract_audio(video_path: str, output_path: str | None = None) -> str:
    """Extract audio track from video file using FFmpeg.

    Converts to 16kHz mono WAV which is optimal for Whisper.

    Args:
        video_path: Path to input video file.
        output_path: Path for output WAV file. Auto-generated if None.

    Returns:
        Path to extracted audio file.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_path is None:
        output_path = str(path.with_suffix(".wav"))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i", str(path),
        "-vn",                    # no video
        "-acodec", "pcm_s16le",   # 16-bit PCM
        "-ar", "16000",           # 16kHz (Whisper optimal)
        "-ac", "1",               # mono
        "-y",                     # overwrite
        str(output),
    ]

    logger.info(f"Extracting audio: {path.name} -> {output.name}")

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg audio extraction failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg.")

    size_mb = output.stat().st_size / 1024 / 1024
    logger.info(f"Audio extracted: {output} ({size_mb:.1f} MB)")
    return str(output)


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
