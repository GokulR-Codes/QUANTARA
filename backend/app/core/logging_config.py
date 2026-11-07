"""Structured logging configuration."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.config.settings import get_settings


def configure_logging() -> None:
    settings = get_settings()
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    except Exception:
        pass  # non-fatal if log file cannot be opened

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)
    # Suppress noisy third-party loggers
    for name in ("uvicorn.access", "multipart"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
