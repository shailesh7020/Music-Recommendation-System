from __future__ import annotations

import logging
import os


def get_logger() -> logging.Logger:
    """Return a package logger with a predictable console handler."""

    logger = logging.getLogger("music_recommender")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.propagate = False

    configured_level = os.getenv("MUSIC_RECOMMENDER_LOG_LEVEL", "INFO").upper()
    logger.setLevel(configured_level)
    return logger
