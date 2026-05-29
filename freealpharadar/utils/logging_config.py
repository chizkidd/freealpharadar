"""Centralised logging configuration for FreeAlphaRadar.

All modules obtain their logger through :func:`get_logger` so that the entire
application shares a single, consistently-formatted logging configuration.
Logging is configured once, lazily, on first use.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED = False


def setup_logging(level: Optional[str] = None) -> None:
    """Configure the root logger for the whole application.

    This is idempotent: calling it multiple times has no additional effect.

    Args:
        level: Logging level name (e.g. ``"INFO"``, ``"DEBUG"``). When ``None``
            the value of the ``FAR_LOG_LEVEL`` environment variable is used,
            defaulting to ``"INFO"``.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved_level = (level or os.environ.get("FAR_LOG_LEVEL", "INFO")).upper()
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))

    root = logging.getLogger()
    root.setLevel(resolved_level)
    # Avoid duplicate handlers if a third party (e.g. Streamlit) already added one.
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)

    # Quieten noisy third-party libraries.
    for noisy in ("urllib3", "yfinance", "peewee", "matplotlib", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, configuring logging on first use.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    setup_logging()
    return logging.getLogger(name)
