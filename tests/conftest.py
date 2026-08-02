"""Shared pytest configuration and fixtures."""

import os
from pathlib import Path
import sys

import cv2
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Keep imports deterministic regardless of invocation directory.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# cv2.kmeans is seeded from a single process-global RNG, so its result depends on
# how many k-means calls ran earlier in the session. Without this, adding or
# removing a test silently changes the clustering in every test that follows it.
CV2_RNG_SEED = int(os.environ.get("OCT_TEST_CV2_SEED", "0"))


@pytest.fixture(autouse=True)
def _deterministic_cv2_rng() -> None:
    """Reset OpenCV's global RNG before each test so results are order-independent."""
    cv2.setRNGSeed(CV2_RNG_SEED)
