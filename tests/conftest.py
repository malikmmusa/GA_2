"""Shared pytest configuration and fixtures."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Keep imports deterministic regardless of invocation directory.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
