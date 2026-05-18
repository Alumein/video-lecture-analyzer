"""Configuration management for Video Lecture Analyzer."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"

# Load .env from project root
load_dotenv(PROJECT_ROOT / ".env")


def load_config(config_path: str | None = None) -> dict:
    """Load configuration from YAML file and environment variables.

    Priority: env vars > custom config > default config
    """
    # Load default config
    with open(DEFAULT_CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    # Override with custom config if provided
    if config_path:
        with open(config_path) as f:
            custom = yaml.safe_load(f)
        _deep_merge(config, custom)

    # Override with environment variables
    if os.getenv("GEMINI_API_KEY"):
        config.setdefault("nlp", {})["api_key"] = os.getenv("GEMINI_API_KEY")
    if os.getenv("WHISPER_MODEL"):
        config.setdefault("audio", {})["whisper_model"] = os.getenv("WHISPER_MODEL")
    if os.getenv("DEVICE"):
        config.setdefault("audio", {})["device"] = os.getenv("DEVICE")

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base
