"""Shared logging helpers for API routes and services."""

import logging


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"atrophy_advisor.{name}")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger
