"""Shared logging helpers for API routes and services."""

import logging
import sys

_root = logging.getLogger("atrophy_advisor")
if not _root.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    _root.addHandler(_handler)
    _root.setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"atrophy_advisor.{name}")
