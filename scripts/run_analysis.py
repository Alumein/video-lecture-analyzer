"""CLI entry point for the Video Lecture Analyzer."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger
from src.utils.config import load_config
from src.utils.video_utils import validate_video_file
from src.integration.pipeline import Pipeline

logger = setup_logger("analyzer")

INPUT_DIR = Path("data/input")
OUTPUT_DIR = Path("data/output")


def find_video(name: str) -> Path:
    """Find video: accepts 'ders1', 'ders1.mp4', or full path."""
    p = Path(name)
    if p.exists():
        return p
    if not p.suffix:
        p = p.with_suffix(".mp4")
    in_dir = INPUT_DIR / p.name
    if in_dir.exists():
        return in_dir
    if p.exists():
        return p
    raise FileNotFoundError(
        f"Video not found: '{name}'\n"
        f"  Video dosyasini data/input/ klasorune koyun."
    )


def list_videos():
    """List available videos."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    videos = sorted(INPUT_DIR.glob("*.mp4")) + sorted(INPUT_DIR.glob("*.mkv"))
    if not videos:
        print(f"\ndata/input/ no videos found in directory.")
        return
    print(f"\nAvailable videos ({len(videos)}):")
    print("-" * 50)
    for v in videos:
        size_mb = v.stat().st_size / 1024 / 1024
        status = "  [ISLENDI]" if (OUTPUT_DIR / v.stem).exists() else ""
        print(f"  {v.name:<25} {size_mb:>8.1f} MB{status}")
    print(f"\nKullanim: python scripts/run_analysis.py ders1")


def main():
    parser = argparse.ArgumentParser(description="Video Lecture Analyzer")
    parser.add_argument("lecture", nargs="?", help="Ders adi (ders1, ders2, ...)")
    parser.add_argument("--list", action="store_true", help="List videos")
    parser.add_argument("--threshold", "-t", type=float, default=30.0)
    parser.add_argument("--config", "-c", default=None)
    parser.add_argument("--language", "-l", default=None)
    parser.add_argument("--whisper-model", default=None)
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"])
    parser.add_argument("--no-diarization", action="store_true")
    args = parser.parse_args()

    if args.list:
        list_videos()
        return

    if not args.lecture:
        print("Kullanim: python scripts/run_analysis.py ders1")
        sys.exit(1)

    try:
        video_path = find_video(args.lecture)
    except FileNotFoundError as e:
        print(f"Hata: {e}")
        sys.exit(1)

    config = load_config(args.config)
    if args.threshold != 30.0:
        config.setdefault("visual", {})["threshold"] = args.threshold
    if args.language:
        config.setdefault("audio", {})["language"] = args.language
    if args.whisper_model:
        config.setdefault("audio", {})["whisper_model"] = args.whisper_model
    if args.device:
        config.setdefault("audio", {})["device"] = args.device
    if args.no_diarization:
        config.setdefault("diarization", {})["enabled"] = False

    output_dir = OUTPUT_DIR / video_path.stem
    pipeline = Pipeline(config=config)
    result = pipeline.run(str(video_path), str(output_dir))

    print(f"\nComplete! {result['metadata']['total_segments']} segment bulundu.")
    print(f"Cikti: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
