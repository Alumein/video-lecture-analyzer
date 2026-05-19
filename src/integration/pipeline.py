"""Main processing pipeline that orchestrates all components."""

import json
import os
import time
from pathlib import Path
from typing import Callable

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.video_utils import (
    extract_audio, get_video_info, format_timestamp, validate_video_file
)
from src.visual.slide_detector import SlideDetector
from src.audio.transcriber import Transcriber
from src.audio.diarizer import Diarizer
from src.nlp.gemini_client import GeminiClient
from src.nlp.ollama_client import OllamaClient
from src.nlp.topic_segmenter import TopicSegmenter
from src.integration.merger import Merger

logger = setup_logger(__name__)

# Type for progress callback: (step_name: str, progress_pct: int) -> None
ProgressCallback = Callable[[str, int], None]


class Pipeline:
    """Main orchestrator that coordinates visual, audio, and NLP pipelines."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()

    def run(
        self,
        video_path: str,
        output_dir: str,
        on_progress: ProgressCallback | None = None,
    ) -> dict:
        """Run the full analysis pipeline.

        Args:
            video_path: Path to input video.
            output_dir: Directory for all outputs.
            on_progress: Optional callback(step_name, percent) for UI updates.

        Returns:
            Complete analysis result dict.
        """
        start_time = time.time()
        video_path = Path(video_path)
        out = Path(output_dir)

        def progress(step: str, pct: int):
            logger.info(f"[{pct}%] {step}")
            if on_progress:
                on_progress(step, pct)

        if not validate_video_file(str(video_path)):
            raise ValueError(f"Invalid video file: {video_path}")

        (out / "slides").mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info(f"Pipeline started: {video_path.name}")
        logger.info("=" * 60)

        # ── Step 1: Video info ──
        progress("Getting video info...", 5)
        video_info = get_video_info(str(video_path))

        # ── Step 2: Slide detection ──
        progress("Detecting slides...", 10)
        v_conf = self.config.get("visual", {})
        detector = SlideDetector(
            threshold=v_conf.get("threshold"),  # None = auto-detect
            min_interval=v_conf.get("min_interval", 3.0),
            resize_width=v_conf.get("resize_width", 480),
        )
        transitions = detector.detect(str(video_path))

        # ── Step 3: Extract slide images ──
        progress(f"{len(transitions)} slide images extracting...", 25)
        detector.extract_slide_images(
            str(video_path), transitions, str(out / "slides")
        )

        # ── Step 3b: OCR on slide images ──
        slide_texts = {}
        try:
            from src.visual.slide_ocr import SlideOCR
            progress("Running slide OCR...", 30)
            ocr = SlideOCR(language="eng")
            slide_texts = ocr.extract_from_slides(str(out / "slides"))
            if slide_texts:
                logger.info(f"OCR: {len(slide_texts)} slides had text extracted")
        except Exception as e:
            logger.warning(f"OCR skipped: {e}")

        # ── Step 4: Extract audio ──
        progress("Extracting audio...", 35)
        audio_path = str(out / "audio.wav")
        try:
            extract_audio(str(video_path), audio_path)
        except Exception as e:
            logger.warning(f"Audio extraction failed: {e}")
            audio_path = None

        # ── Step 5: Transcription ──
        transcript_segments = []
        if audio_path and Path(audio_path).exists():
            progress("Loading Whisper model...", 40)
            a_conf = self.config.get("audio", {})
            lang = a_conf.get("language")
            if lang in ("auto", "Auto", "AUTO", ""):
                lang = None
            try:
                transcriber = Transcriber(
                    model_size=a_conf.get("whisper_model", "large-v3"),
                    device=a_conf.get("device", "cuda"),
                    compute_type=a_conf.get("compute_type", "float16"),
                    language=lang,
                )
                progress("Transcribing audio...", 45)
                transcript_segments = transcriber.transcribe(audio_path)

                # Save SRT
                srt_content = transcriber.segments_to_srt(transcript_segments)
                srt_path = out / f"{video_path.stem}.srt"
                srt_path.write_text(srt_content, encoding="utf-8")
                logger.info(f"SRT saved: {srt_path}")

            except Exception as e:
                logger.warning(f"Transcription failed: {e}")
                progress(f"Transcription failed: {type(e).__name__}", 55)
        else:
            progress("No audio file, skipping transcription...", 55)

        # ── Step 6: Speaker diarization ──
        speaker_segments = None
        d_conf = self.config.get("diarization", {})
        if d_conf.get("enabled", True) and audio_path and Path(audio_path).exists():
            progress("Running speaker diarization...", 60)
            try:
                diarizer = Diarizer(
                    min_speakers=d_conf.get("min_speakers"),
                    max_speakers=d_conf.get("max_speakers"),
                    device=self.config.get("audio", {}).get("device", "cuda"),
                )
                speaker_segments = diarizer.diarize(audio_path)
            except Exception as e:
                logger.warning(f"Diarization failed: {e}")
                progress(f"Diarization atlandi: {type(e).__name__}", 70)
        else:
            progress("Skipping diarization...", 65)

        # ── Step 7: NLP-driven topic segmentation ──
        nlp_conf = self.config.get("nlp", {})
        llm_client = None
        llm_name = None
        merger = Merger()

        if transcript_segments:
            # Find an LLM client
            ollama_model = nlp_conf.get("ollama_model") or os.getenv("OLLAMA_MODEL")
            ollama_url = nlp_conf.get("ollama_url", "http://localhost:11434")

            if ollama_model:
                progress("Running topic analysis with Ollama...", 75)
                try:
                    client = OllamaClient(
                        model=ollama_model, base_url=ollama_url,
                        temperature=nlp_conf.get("temperature", 0.3),
                        num_ctx=nlp_conf.get("ollama_num_ctx", 8192),
                        timeout=nlp_conf.get("ollama_timeout", 600),
                    )
                    if client.is_available():
                        llm_client = client
                        llm_name = f"Ollama ({ollama_model})"
                except Exception as e:
                    logger.warning(f"Ollama connection failed: {e}")
            else:
                try:
                    client = OllamaClient(base_url=ollama_url)
                    if client.is_available():
                        llm_client = client
                        llm_name = f"Ollama ({client.model})"
                        progress(f"Ollama found ({client.model})...", 75)
                except Exception:
                    pass

            if not llm_client:
                api_key = nlp_conf.get("api_key") or os.getenv("GEMINI_API_KEY")
                if api_key:
                    progress("Running topic analysis with Gemini...", 75)
                    try:
                        llm_client = GeminiClient(
                            api_key=api_key,
                            model=nlp_conf.get("gemini_model", "gemini-2.0-flash"),
                            temperature=nlp_conf.get("temperature", 0.3),
                        )
                        llm_name = f"Gemini ({nlp_conf.get('gemini_model', 'gemini-2.0-flash')})"
                    except Exception as e:
                        logger.warning(f"Gemini init failed: {e}")

        # Build segments based on LLM topics or fallback
        if llm_client and transcript_segments:
            try:
                logger.info(f"NLP engine: {llm_name}")
                progress(f"Identifying topics with LLM ({llm_name})...", 80)

                # Merge transcript + OCR lines into a single timestamp-sorted
                # stream. The chunker assumes monotonic time; two concatenated
                # 0→end sections break its [start, end] math for chunks that
                # straddle the boundary.
                timed_lines: list[tuple[float, str]] = [
                    (s.start, f"[{format_timestamp(s.start)}] {s.text}")
                    for s in transcript_segments
                ]
                if slide_texts:
                    ocr_added = 0
                    for t in transitions:
                        if t.image_path and t.image_path in slide_texts:
                            timed_lines.append((
                                t.timestamp,
                                f"[{format_timestamp(t.timestamp)}] "
                                f"SLIDE: {slide_texts[t.image_path]}",
                            ))
                            ocr_added += 1
                    logger.info(f"OCR text merged inline: {ocr_added} slides")

                timed_lines.sort(key=lambda x: x[0])
                full_transcript = "\n".join(line for _, line in timed_lines)
                slide_timestamps = [t.timestamp for t in transitions]

                segmenter = TopicSegmenter(
                    client=llm_client,
                    max_topics=nlp_conf.get("max_topics", 15),
                )
                topics = segmenter.segment(full_transcript, slide_timestamps)

                progress("Building segments...", 90)
                # Build segments directly from LLM topics
                segments = merger.merge_by_llm_topics(
                    topics=topics,
                    transcript_segments=transcript_segments,
                    speaker_segments=speaker_segments,
                    transitions=transitions,
                    slide_texts=slide_texts,
                )
                logger.info(f"NLP segmentation: {len(segments)} segments ({llm_name})")

            except Exception as e:
                logger.warning(f"NLP segmentation failed ({llm_name}): {e}")
                progress("NLP atlandi, transcript bazli segmentasyon...", 90)
                segments = merger.merge_by_topics(
                    transcript_segments=transcript_segments,
                    speaker_segments=speaker_segments,
                    transitions=transitions,
                    video_duration=video_info["duration"],
                    min_segment_duration=180.0,
                )
        else:
            logger.info("No LLM found, using transcript-based segmentation.")
            progress("Transcript-based segmentation...", 80)
            segments = merger.merge_by_topics(
                transcript_segments=transcript_segments,
                speaker_segments=speaker_segments,
                transitions=transitions,
                video_duration=video_info["duration"],
                min_segment_duration=180.0,
            )

        # ── Save result ──
        progress("Saving results...", 92)
        elapsed = time.time() - start_time
        result = {
            "metadata": {
                "source_file": video_path.name,
                "duration": video_info["duration"],
                "duration_fmt": format_timestamp(video_info["duration"]),
                "resolution": f"{video_info['width']}x{video_info['height']}",
                "fps": video_info["fps"],
                "total_frames": video_info["total_frames"],
                "total_slides": len(transitions),
                "total_segments": len(segments),
                "processing_time": round(elapsed, 2),
                "processing_time_fmt": format_timestamp(elapsed),
                "has_transcript": len(transcript_segments) > 0,
                "has_diarization": speaker_segments is not None,
                "has_nlp": llm_client is not None,
                "has_ocr": len(slide_texts) > 0,
                "ocr_slides": len(slide_texts),
            },
            "segments": segments,
        }

        # ── Automated Evaluation ──
        if llm_client:
            progress("Running quality evaluation...", 95)
            try:
                from src.evaluation.evaluator import Evaluator
                evaluator = Evaluator(llm_client)
                eval_report = evaluator.evaluate_all(result)
                result["evaluation"] = eval_report
                logger.info(f"Evaluation: {eval_report['overall']['status']}")
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")
                result["evaluation"] = {"error": str(e)}

        result_path = out / f"{video_path.stem}_result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        progress("Complete!", 100)
        logger.info(f"Pipeline complete: {format_timestamp(elapsed)}")

        return result


