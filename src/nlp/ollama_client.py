"""Ollama local LLM client for NLP processing.

Ollama runs LLMs locally - no API keys, no quotas, no internet needed.

Setup:
    1. Install Ollama: https://ollama.ai/download
    2. Pull a model: ollama pull llama3.1:8b
    3. Ollama runs automatically in background

Recommended models (by VRAM):
    - llama3.1:8b     (~5GB VRAM)  - Good balance, fast
    - gemma2:9b       (~6GB VRAM)  - Good at structured output
    - mistral:7b      (~5GB VRAM)  - Fast, good at English
    - llama3.1:70b    (~40GB VRAM) - Best quality, needs big GPU
    - phi3:mini       (~3GB VRAM)  - Smallest, for low-end GPUs
"""

import json
import time

import requests

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class OllamaClient:
    """Client for Ollama local LLM API.

    Drop-in replacement for GeminiClient with the same interface:
        - generate(prompt, system_instruction) -> str
        - generate_json(prompt, system_instruction) -> dict/list
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
        max_retries: int = 2,
        num_ctx: int = 8192,
        timeout: int = 600,
    ):
        """
        Args:
            model: Ollama model name (e.g. "llama3.1:8b", "gemma2:9b").
            base_url: Ollama API URL (default localhost:11434).
            temperature: Sampling temperature (0.0-1.0).
            max_retries: Number of retries on connection errors.
            num_ctx: Context window in tokens. Larger = handles bigger
                prompts but slower. 8192 covers an 8K-char chunk plus
                response.
            timeout: HTTP timeout per request, in seconds. Large models
                that spill to CPU (e.g., qwen2.5:14b on 8GB VRAM) need
                more than the default.
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_retries = max_retries
        self.num_ctx = num_ctx
        self.timeout = timeout
        self._verified = False

    def _verify_connection(self):
        """Check that Ollama is running and model is available."""
        if self._verified:
            return

        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
        except requests.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama ({self.base_url}). "
                f"Make sure Ollama is running:\n"
                f"  1. https://ollama.ai/download download from\n"
                f"  2. 'ollama serve' run the command\n"
                f"  3. 'ollama pull {self.model}' to download the model"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama connection error: {e}")

        # Check if model is available
        model_base = self.model.split(":")[0]
        available = [m.split(":")[0] for m in models]
        if model_base not in available:
            raise RuntimeError(
                f"Model '{self.model}' not found in Ollama.\n"
                f"  Available models: {', '.join(models) if models else 'None'}\n"
                f"  To download: ollama pull {self.model}"
            )

        self._verified = True
        logger.info(f"Ollama connected: {self.model} @ {self.base_url}")

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        json_mode: bool = False,
    ) -> str:
        """Send prompt to Ollama and return response text.

        Args:
            prompt: User prompt.
            system_instruction: Optional system message.
            json_mode: If True, ask Ollama to constrain output to valid JSON.

        Returns:
            Generated text response.
        """
        self._verify_connection()

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }
        if json_mode:
            payload["format"] = "json"

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    f"Ollama request (model={self.model}, attempt {attempt}, "
                    f"{len(prompt)} chars, num_ctx={self.num_ctx}, "
                    f"json_mode={json_mode})"
                )

                resp = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()

                result = resp.json()
                text = result.get("message", {}).get("content", "").strip()

                if not text:
                    raise ValueError("Empty response from Ollama")

                logger.debug(f"Ollama response ({len(text)} chars)")
                return text

            except requests.Timeout:
                last_error = (
                    f"Request timeout after {self.timeout}s "
                    f"(model={self.model} may be loading or running on CPU)"
                )
                logger.warning(
                    f"Ollama timeout (model={self.model}, "
                    f"attempt {attempt}/{self.max_retries})"
                )
            except requests.ConnectionError:
                last_error = "Connection lost"
                logger.warning(
                    f"Ollama connection lost (model={self.model}, "
                    f"attempt {attempt}/{self.max_retries})"
                )
                self._verified = False
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Ollama error (model={self.model}, "
                    f"attempt {attempt}/{self.max_retries}): {e}"
                )

            if attempt < self.max_retries:
                time.sleep(2)

        raise RuntimeError(
            f"Ollama failed after {self.max_retries} attempts "
            f"(model={self.model}): {last_error}"
        )

    def generate_json(
        self,
        prompt: str,
        system_instruction: str | None = None,
    ) -> dict | list:
        """Generate response and parse as JSON.

        Adds explicit JSON instruction to the prompt and uses
        Ollama's JSON format mode when available.
        """
        # Reinforce JSON output in the prompt
        json_prompt = prompt + "\n\nIMPORTANT: Respond with ONLY valid JSON, no other text."

        # Add JSON format hint to system instruction
        json_system = (system_instruction or "") + (
            "\nYou MUST respond with valid JSON only. "
            "No markdown, no explanations, no code fences. Just raw JSON."
        )

        raw = self.generate(json_prompt, json_system.strip(), json_mode=True)
        return self._parse_json_response(raw)

    @staticmethod
    def _parse_json_response(text: str) -> dict | list:
        """Extract and parse JSON from LLM response."""
        cleaned = text.strip()

        # Strip markdown code fences
        if cleaned.startswith("```"):
            first_nl = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
            cleaned = cleaned[first_nl + 1:]
            if cleaned.rstrip().endswith("```"):
                cleaned = cleaned.rstrip()[:-3].rstrip()

        # Try direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in the text
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start_idx = cleaned.find(start_char)
            end_idx = cleaned.rfind(end_char)
            if start_idx != -1 and end_idx > start_idx:
                candidate = cleaned[start_idx:end_idx + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue

        raise ValueError(f"Could not parse JSON from Ollama response: {text[:200]}")

    def is_available(self) -> bool:
        """Check if Ollama is running and model is ready."""
        try:
            self._verify_connection()
            return True
        except Exception:
            return False
