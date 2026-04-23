"""Logging helpers used by scripts and pipelines."""

from __future__ import annotations

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Install a basic, consistent logging configuration."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""

    return logging.getLogger(name)
