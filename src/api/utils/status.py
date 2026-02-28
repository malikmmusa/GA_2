"""Helpers for consistent status payloads."""

from typing import Any, Dict


def build_status_payload(status: str, **details: Any) -> Dict[str, Any]:
    return {
        "status": status,
        **details,
    }
