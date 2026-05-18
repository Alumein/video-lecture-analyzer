"""Logging setup for Video Lecture Analyzer."""

import logging
import os
import sys


def setup_logger(name: str = "lecture_analyzer") -> logging.Logger:
    """Create and configure a logger instance."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
