"""
logger.py — Centralised structured logging setup.
Import `log` from this module everywhere.
"""
from __future__ import annotations

import logging
import sys
from config import settings


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("vera_bot")
    if logger.handlers:
        return logger  # already configured (e.g., during testing)

    level = getattr(logging, settings.log_level, logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


log = _build_logger()
