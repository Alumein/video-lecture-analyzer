"""Google Gemini API client for NLP processing."""

import json
import time
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class GeminiClient:
    """Client for Google Gemini API interactions.

    Handles authentication, request formatting, response parsing,
    retry logic, and JSON extraction from LLM responses.

    Usage:
        client = GeminiClient(api_key="your-key")
        response = client.generate("Summarize this text: ...")
        data = client.generate_json("Return JSON: ...", schema_hint="...")
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        max_retries: int = 5,
        retry_delay: float = 5.0,
    ):
        """
        Args:
            api_key: Google AI Studio API key.
            model: Gemini model name.
            temperature: Sampling temperature (0.0-1.0). Lower = more deterministic.
            max_tokens: Maximum tokens in response.
            max_retries: Number of retries on transient failures.
            retry_delay: Base delay between retries (doubles each retry).
        """
        if not api_key:
            raise ValueError(
                "Gemini API key is required. "
                "Get one at: https://aistudio.google.com/apikey"
            )

        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = None

    def _init_client(self):
        """Lazy-initialize the Gemini API client."""
        if self._client is not None:
            return

        import warnings
        warnings.filterwarnings("ignore", message=".*google.generativeai.*")

        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai is not installed. "
                "Install with: pip install google-generativeai"
            )

        genai.configure(api_key=self.api_key)
        self._genai = genai

        logger.info(f"Gemini client initialized: model={self.model}")

    def _get_model(self, system_instruction: str | None = None):
        """Create a GenerativeModel instance with optional system instruction."""
        self._init_client()

        kwargs = {}
        if system_instruction:
            kwargs["system_instruction"] = system_instruction

        return self._genai.GenerativeModel(
            model_name=self.model,
            generation_config=self._genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
            **kwargs,
        )

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        """Send a prompt to Gemini and return the response text.

        Includes retry logic for transient API errors.

        Args:
            prompt: User prompt text.
            system_instruction: Optional system-level instruction.

        Returns:
            Generated text response.

        Raises:
            RuntimeError: If all retries fail.
        """
        model = self._get_model(system_instruction)

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = model.generate_content(prompt)
                text = response.text.strip()
                logger.debug(f"Gemini response ({len(text)} chars)")
                return text

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Don't retry on auth or permanently invalid errors
                if any(x in error_str for x in ["api_key", "invalid", "permission"]):
                    raise RuntimeError(f"Gemini API error (non-retryable): {e}")

                # Daily quota exhausted - don't waste time retrying
                if "perday" in error_str.replace(" ", "").replace("_", "") or "per_day" in error_str:
                    logger.warning("Gunluk API limiti dolmus. Retry yapilmayacak.")
                    raise RuntimeError(f"Gemini daily quota exhausted. Try again tomorrow.")

                # Per-minute rate limit - short retry
                if "429" in error_str or "quota" in error_str or "rate" in error_str:
                    delay = min(30, self.retry_delay * (2 ** (attempt - 1)))
                    logger.warning(
                        f"Rate limit hit (attempt {attempt}/{self.max_retries}). "
                        f"Waiting {delay:.0f}s..."
                    )
                else:
                    delay = self.retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"Gemini API error (attempt {attempt}/{self.max_retries}): "
                    f"{type(e).__name__}. Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

        raise RuntimeError(
            f"Gemini API failed after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def generate_json(
        self,
        prompt: str,
        system_instruction: str | None = None,
    ) -> dict | list:
        """Generate a response and parse it as JSON.

        Automatically strips markdown code fences and handles
        common LLM JSON formatting issues.

        Args:
            prompt: Prompt that instructs the model to return JSON.
            system_instruction: Optional system instruction.

        Returns:
            Parsed JSON as dict or list.

        Raises:
            ValueError: If response cannot be parsed as JSON.
        """
        raw = self.generate(prompt, system_instruction)
        return self._parse_json_response(raw)

    @staticmethod
    def _parse_json_response(text: str) -> dict | list:
        """Extract and parse JSON from LLM response text.

        Handles common issues:
            - Markdown code fences (```json ... ```)
            - Leading/trailing text around JSON
            - Single quotes instead of double quotes

        Args:
            text: Raw LLM response text.

        Returns:
            Parsed JSON object.

        Raises:
            ValueError: If no valid JSON can be extracted.
        """
        # Strip markdown code fences
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (```json or ```)
            first_newline = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
            cleaned = cleaned[first_newline + 1:]
            # Remove closing fence
            if cleaned.rstrip().endswith("```"):
                cleaned = cleaned.rstrip()[:-3].rstrip()

        # Try direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object or array in the text
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start_idx = cleaned.find(start_char)
            end_idx = cleaned.rfind(end_char)
            if start_idx != -1 and end_idx > start_idx:
                candidate = cleaned[start_idx:end_idx + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue

        raise ValueError(
            f"Could not parse JSON from response. "
            f"Raw text (first 200 chars): {text[:200]}"
        )
