"""Automated evaluation of all pipeline components.

Measures:
    1. Slide Detection Precision/Recall (LLM-as-judge)
    2. Processing Latency
    3. Topic Segmentation F1 (LLM-as-judge)
    4. Transcription Quality / WER estimate
"""

import json
import time
from pathlib import Path

from src.utils.logger import setup_logger
from src.utils.video_utils import format_timestamp

logger = setup_logger(__name__)


class Evaluator:
    """Automated evaluation of pipeline output quality.

    Uses LLM-as-judge approach: sends pipeline outputs to the LLM
    and asks it to evaluate quality, consistency, and accuracy.
    """

    def __init__(self, llm_client):
        """
        Args:
            llm_client: LLM client (OllamaClient or GeminiClient).
        """
        self.client = llm_client

    def evaluate_all(self, result: dict, output_dir: str) -> dict:
        """Run all evaluations and generate a report.

        NOTE: These are ESTIMATES, not ground-truth measurements.
        - WER is estimated by LLM analysis of transcript quality
        - Topic F1 is estimated by LLM self-evaluation (may be biased)
        - Latency is directly measured (reliable)
        - Duplicate slides are checked programmatically (reliable)
        """
        logger.info("=" * 50)
        logger.info("Starting automated evaluation")
        logger.info("=" * 50)

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": result.get("metadata", {}).get("source_file", "unknown"),
            "disclaimer": "These metrics are estimates. WER and Topic F1 are LLM-judged approximations, not ground-truth measurements.",
        }

        # 1. Processing Latency (directly measured - reliable)
        report["latency"] = self._evaluate_latency(result)
        logger.info(f"Latency: {report['latency']['status']}")

        # 2. Transcription Quality (LLM-estimated)
        report["transcription"] = self._evaluate_transcription(result)
        logger.info(f"Transcription: {report['transcription']['status']}")

        # 3. Duplicate Slide Check (programmatic - reliable)
        report["slide_duplicates"] = self._check_duplicate_slides(result, output_dir)
        logger.info(f"Slide Duplicates: {report['slide_duplicates']['status']}")

        # 4. Topic Segmentation (LLM-estimated)
        report["topic_segmentation"] = self._evaluate_topics(result)
        logger.info(f"Topic Segmentation: {report['topic_segmentation']['status']}")

        # Overall score
        report["overall"] = self._compute_overall(report)
        logger.info(f"Overall: {report['overall']['status']}")

        return report

    # ── 1. Processing Latency ──

    def _evaluate_latency(self, result: dict) -> dict:
        """Check if processing time meets the <15 min / 60 min video target."""
        meta = result.get("metadata", {})
        proc_time = meta.get("processing_time", 0)
        duration = meta.get("duration", 1)

        # Normalize to 60-minute equivalent
        if duration > 0:
            normalized = (proc_time / duration) * 3600  # time for 60 min video
        else:
            normalized = proc_time

        target = 900  # 15 minutes in seconds
        passed = normalized <= target
        ratio = proc_time / duration if duration > 0 else 0

        return {
            "processing_time_sec": round(proc_time, 1),
            "video_duration_sec": round(duration, 1),
            "processing_time_fmt": format_timestamp(proc_time),
            "video_duration_fmt": format_timestamp(duration),
            "time_ratio": round(ratio, 2),
            "normalized_60min": round(normalized, 1),
            "target_sec": target,
            "passed": passed,
            "score": min(100, round((target / max(normalized, 1)) * 100)),
            "status": f"{'PASS' if passed else 'FAIL'} - "
                      f"{format_timestamp(proc_time)} for {format_timestamp(duration)} video "
                      f"(ratio: {ratio:.2f}x, projected 60min: {format_timestamp(normalized)})",
        }

    # ── 2. Transcription Quality ──

    def _evaluate_transcription(self, result: dict) -> dict:
        """Evaluate transcription quality using LLM-as-judge.

        Asks the LLM to assess transcript coherence, completeness,
        and estimate WER based on text quality indicators.
        """
        meta = result.get("metadata", {})
        segments = result.get("segments", [])

        if not meta.get("has_transcript"):
            return {
                "passed": False,
                "score": 0,
                "estimated_wer": 100,
                "status": "FAIL - No transcript available",
            }

        # Collect transcript samples from beginning, middle, end
        total = len(segments)
        sample_indices = [0, total // 4, total // 2, 3 * total // 4, total - 1]
        sample_indices = list(set(i for i in sample_indices if 0 <= i < total))

        samples = []
        for i in sample_indices:
            seg = segments[i]
            text = seg.get("transcript", "")[:500]
            if text:
                samples.append(f"[{seg.get('start_fmt', '')}] {text}")

        sample_text = "\n\n".join(samples)

        prompt = f"""Evaluate the quality of this speech-to-text transcription from a lecture video.

TRANSCRIPT SAMPLES (from different parts of the lecture):
{sample_text}

Evaluate and respond with ONLY this JSON:
{{
    "estimated_wer_percent": <number 0-100, your best estimate of Word Error Rate>,
    "coherence_score": <number 1-10, how coherent and readable the text is>,
    "completeness_score": <number 1-10, does it seem like complete sentences vs fragments>,
    "issues_found": ["list of specific issues like: garbled words, missing punctuation, wrong language, etc"],
    "overall_quality": "excellent|good|fair|poor"
}}"""

        try:
            eval_result = self.client.generate_json(prompt)

            wer = float(eval_result.get("estimated_wer_percent", 50))
            coherence = float(eval_result.get("coherence_score", 5))
            completeness = float(eval_result.get("completeness_score", 5))
            quality = eval_result.get("overall_quality", "unknown")
            issues = eval_result.get("issues_found", [])

            passed = wer <= 15
            score = max(0, 100 - wer)

            return {
                "passed": passed,
                "score": round(score),
                "estimated_wer": round(wer, 1),
                "coherence_score": coherence,
                "completeness_score": completeness,
                "quality": quality,
                "issues": issues,
                "total_segments": len(segments),
                "status": f"{'PASS' if passed else 'FAIL'} - "
                          f"Estimated WER: {wer:.1f}% | Quality: {quality} | "
                          f"Coherence: {coherence}/10",
            }

        except Exception as e:
            logger.warning(f"Transcription evaluation failed: {e}")
            return {
                "passed": None,
                "score": None,
                "estimated_wer": None,
                "status": f"UNKNOWN - Evaluation failed: {e}",
            }

    # ── 3. Duplicate Slide Check ──

    def _check_duplicate_slides(self, result: dict, output_dir: str) -> dict:
        """Check for duplicate slide images using perceptual hash.

        This is a programmatic check (not LLM-based) so it's reliable.
        Compares all saved slide images and reports any near-duplicates.
        """
        import cv2
        import numpy as np

        slides_dir = Path(output_dir) / "slides"
        if not slides_dir.exists():
            return {
                "passed": True,
                "score": 100,
                "total_slides": 0,
                "duplicates_found": 0,
                "status": "PASS - No slides directory",
            }

        # Load all slide images and compute pHash
        slide_files = sorted(slides_dir.glob("*.png")) + sorted(slides_dir.glob("*.jpg"))
        hashes = []

        for f in slide_files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            resized = cv2.resize(img, (16, 16))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
            dct = cv2.dct(gray)
            dct_low = dct[:8, :8]
            median = np.median(dct_low)
            ph = (dct_low > median).flatten()
            hashes.append((f.name, ph))

        # Compare all pairs
        duplicates = []
        duplicate_threshold = 0.93

        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                name_i, hash_i = hashes[i]
                name_j, hash_j = hashes[j]
                dist = np.count_nonzero(hash_i != hash_j)
                sim = 1.0 - (dist / len(hash_i))
                if sim > duplicate_threshold:
                    duplicates.append({
                        "slide_a": name_i,
                        "slide_b": name_j,
                        "similarity": round(sim * 100, 1),
                    })

        total = len(hashes)
        dup_count = len(duplicates)
        passed = dup_count == 0
        score = max(0, 100 - (dup_count * 10))

        return {
            "passed": passed,
            "score": score,
            "total_slides": total,
            "unique_slides": total - dup_count,
            "duplicates_found": dup_count,
            "duplicate_pairs": duplicates[:10],  # limit to first 10
            "status": f"{'PASS' if passed else 'NEEDS IMPROVEMENT'} - "
                      f"{total} slides, {dup_count} duplicate pairs found",
        }

    # ── 4. Topic Segmentation ──

    def _evaluate_topics(self, result: dict) -> dict:
        """Evaluate topic segmentation quality using LLM-as-judge.

        Asks a separate LLM call to evaluate:
        - Are topic titles specific and descriptive?
        - Do topics cover the full lecture duration?
        - Are summaries accurate and detailed?
        - Are there gaps or overlaps?
        """
        segments = result.get("segments", [])
        meta = result.get("metadata", {})
        duration = meta.get("duration", 0)

        if not segments:
            return {
                "passed": False,
                "score": 0,
                "status": "FAIL - No segments found",
            }

        # Check coverage
        total_covered = sum(s.get("duration", 0) for s in segments)
        coverage = (total_covered / duration * 100) if duration > 0 else 0

        # Check for gaps
        gaps = []
        for i in range(len(segments) - 1):
            end_current = segments[i].get("end_time", 0)
            start_next = segments[i + 1].get("start_time", 0)
            gap = start_next - end_current
            if gap > 60:  # gaps > 1 minute
                gaps.append({
                    "after_segment": segments[i].get("topic", ""),
                    "gap_seconds": round(gap, 1),
                })

        # LLM evaluation of topic quality
        topics_for_eval = "\n".join(
            f"{i+1}. [{s.get('start_fmt','')} - {s.get('end_fmt','')}] "
            f"Title: {s.get('topic','')}\n"
            f"   Summary: {s.get('summary','')[:200]}\n"
            f"   Key Points: {', '.join(s.get('key_points',[])[:3])}"
            for i, s in enumerate(segments)
        )

        prompt = f"""Evaluate the quality of this lecture topic segmentation.

LECTURE: {duration:.0f} seconds ({format_timestamp(duration)})
COVERAGE: {coverage:.1f}% of lecture covered
SEGMENTS ({len(segments)} total):

{topics_for_eval}

Evaluate these criteria and respond with ONLY this JSON:
{{
    "title_specificity": <1-10, are titles specific enough to understand the content?>,
    "title_descriptiveness": <1-10, do titles describe WHAT is taught, not just broad category?>,
    "summary_quality": <1-10, are summaries detailed with specific concepts/examples?>,
    "coverage_quality": <1-10, does segmentation cover the full lecture?>,
    "segmentation_granularity": <1-10, are segments the right size - not too broad, not too narrow?>,
    "overall_f1_estimate": <0-100, estimated F1 score compared to ideal human segmentation>,
    "strengths": ["list of what's done well"],
    "improvements": ["list of what could be better"]
}}"""

        try:
            eval_result = self.client.generate_json(prompt)

            specificity = float(eval_result.get("title_specificity", 5))
            descriptiveness = float(eval_result.get("title_descriptiveness", 5))
            summary_q = float(eval_result.get("summary_quality", 5))
            coverage_q = float(eval_result.get("coverage_quality", 5))
            granularity = float(eval_result.get("segmentation_granularity", 5))
            f1_estimate = float(eval_result.get("overall_f1_estimate", 50))
            strengths = eval_result.get("strengths", [])
            improvements = eval_result.get("improvements", [])

            avg_score = (specificity + descriptiveness + summary_q + coverage_q + granularity) / 5
            passed = f1_estimate >= 80

            return {
                "passed": passed,
                "score": round(f1_estimate),
                "total_segments": len(segments),
                "coverage_percent": round(coverage, 1),
                "gaps_over_1min": len(gaps),
                "gap_details": gaps,
                "title_specificity": specificity,
                "title_descriptiveness": descriptiveness,
                "summary_quality": summary_q,
                "coverage_quality": coverage_q,
                "segmentation_granularity": granularity,
                "average_quality": round(avg_score, 1),
                "f1_estimate": round(f1_estimate, 1),
                "strengths": strengths,
                "improvements": improvements,
                "status": f"{'PASS' if passed else 'NEEDS IMPROVEMENT'} - "
                          f"F1 estimate: {f1_estimate:.0f}% | Quality: {avg_score:.1f}/10 | "
                          f"Coverage: {coverage:.0f}% | Segments: {len(segments)} | "
                          f"Gaps>1min: {len(gaps)}",
            }

        except Exception as e:
            logger.warning(f"Topic evaluation failed: {e}")
            return {
                "passed": None,
                "score": None,
                "total_segments": len(segments),
                "coverage_percent": round(coverage, 1),
                "status": f"UNKNOWN - Evaluation failed: {e}",
            }

    # ── Overall ──

    def _compute_overall(self, report: dict) -> dict:
        """Compute overall pass/fail status."""
        criteria = {
            "Processing Latency ≤15min/60min": report.get("latency", {}),
            "Slide Duplicate Check": report.get("slide_duplicates", {}),
            "Topic Segmentation F1 ≥0.8": report.get("topic_segmentation", {}),
            "Transcription WER ≤15%": report.get("transcription", {}),
        }

        results = []
        for name, data in criteria.items():
            passed = data.get("passed")
            score = data.get("score", 0)
            status = "PASS" if passed else ("FAIL" if passed is False else "UNKNOWN")
            results.append({
                "criterion": name,
                "status": status,
                "score": score,
            })

        total_passed = sum(1 for r in results if r["status"] == "PASS")
        total = len(results)
        avg_score = sum(r["score"] or 0 for r in results) / max(total, 1)

        return {
            "passed": total_passed == total,
            "criteria_passed": total_passed,
            "criteria_total": total,
            "average_score": round(avg_score, 1),
            "results": results,
            "status": f"{total_passed}/{total} criteria passed | Average score: {avg_score:.0f}/100",
        }
