"""Key sentence extraction and summarization using LLM (Gemini or Ollama)."""

import json

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

SYSTEM_INSTRUCTION = """You are an expert at summarizing educational content. 
You produce clear, concise summaries and extract the most important key points. 
You always respond in valid JSON format only, with no additional text or markdown."""

SUMMARIZE_PROMPT = """Summarize the following lecture segment in {max_sentences} sentences or fewer.
The summary should capture the main ideas clearly and concisely.

TEXT:
{text}

Respond ONLY with a JSON object:
{{
  "summary": "Your concise summary here."
}}"""

KEY_POINTS_PROMPT = """Extract the {max_points} most important key points from this lecture segment.
Each key point should be a single, clear, self-contained sentence.

TEXT:
{text}

Respond ONLY with a JSON object:
{{
  "key_points": [
    "First key point",
    "Second key point"
  ]
}}"""

FULL_ANALYSIS_PROMPT = """Analyze this lecture segment and provide both a summary and key points.

TEXT:
{text}

Respond ONLY with a JSON object:
{{
  "summary": "Concise 2-3 sentence summary.",
  "key_points": [
    "First key point",
    "Second key point"
  ],
  "difficulty_level": "beginner|intermediate|advanced",
  "main_concept": "The single most important concept discussed"
}}"""


class Summarizer:
    """Generates summaries and extracts key sentences from transcript segments.

    Uses Gemini API with specialized prompts to produce concise
    summaries and identify the most important points from lecture content.

    Supports three modes:
        - summarize(): Generate a text summary
        - extract_key_points(): Extract bullet-point key points
        - full_analysis(): Get summary, key points, difficulty, and main concept
    """

    def __init__(self, client):
        """
        Args:
            client: LLM client (GeminiClient or OllamaClient).
        """
        self.client = client

    def summarize(self, text: str, max_sentences: int = 3) -> str:
        """Generate a summary of the given text.

        Args:
            text: Text to summarize.
            max_sentences: Maximum sentences in summary.

        Returns:
            Summary string.
        """
        if not text.strip():
            return ""

        prompt = SUMMARIZE_PROMPT.format(
            text=text[:15000],  # safety truncation
            max_sentences=max_sentences,
        )

        result = self.client.generate_json(
            prompt=prompt,
            system_instruction=SYSTEM_INSTRUCTION,
        )

        summary = result.get("summary", "") if isinstance(result, dict) else ""
        logger.debug(f"Generated summary: {len(summary)} chars")
        return summary

    def extract_key_points(self, text: str, max_points: int = 5) -> list[str]:
        """Extract key points from the given text.

        Args:
            text: Source text.
            max_points: Maximum number of key points.

        Returns:
            List of key point strings.
        """
        if not text.strip():
            return []

        prompt = KEY_POINTS_PROMPT.format(
            text=text[:15000],
            max_points=max_points,
        )

        result = self.client.generate_json(
            prompt=prompt,
            system_instruction=SYSTEM_INSTRUCTION,
        )

        points = result.get("key_points", []) if isinstance(result, dict) else []
        logger.debug(f"Extracted {len(points)} key points")
        return [str(p) for p in points]

    def full_analysis(self, text: str) -> dict:
        """Perform full analysis: summary, key points, difficulty, main concept.

        Args:
            text: Lecture segment text.

        Returns:
            Dict with keys: summary, key_points, difficulty_level, main_concept.
        """
        if not text.strip():
            return {
                "summary": "",
                "key_points": [],
                "difficulty_level": "unknown",
                "main_concept": "",
            }

        prompt = FULL_ANALYSIS_PROMPT.format(text=text[:15000])

        result = self.client.generate_json(
            prompt=prompt,
            system_instruction=SYSTEM_INSTRUCTION,
        )

        if not isinstance(result, dict):
            result = {}

        return {
            "summary": str(result.get("summary", "")),
            "key_points": [str(p) for p in result.get("key_points", [])],
            "difficulty_level": str(result.get("difficulty_level", "unknown")),
            "main_concept": str(result.get("main_concept", "")),
        }

    def batch_summarize(
        self,
        segments: list[dict],
        max_sentences: int = 3,
    ) -> list[dict]:
        """Summarize multiple segments in sequence.

        Args:
            segments: List of dicts with at least a "text" key.
            max_sentences: Max sentences per summary.

        Returns:
            List of dicts with added "summary" and "key_points" keys.
        """
        results = []
        for i, seg in enumerate(segments):
            text = seg.get("text", "")
            logger.info(f"Summarizing segment {i + 1}/{len(segments)}")

            analysis = self.full_analysis(text)
            result = {**seg, **analysis}
            results.append(result)

        logger.info(f"Batch summarization complete: {len(results)} segments")
        return results
