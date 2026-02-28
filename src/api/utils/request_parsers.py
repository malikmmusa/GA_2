"""Shared helpers for parsing structured form payloads."""

import json
from typing import Type, TypeVar

from fastapi import HTTPException
from pydantic import BaseModel, ValidationError

ParsedModel = TypeVar("ParsedModel", bound=BaseModel)


def parse_form_json(
    raw_json: str,
    model_class: Type[ParsedModel],
    *,
    field_name: str = "request_data",
) -> ParsedModel:
    """Parse and validate JSON from multipart form fields."""
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON in {field_name}: {str(exc)}",
        ) from exc

    try:
        if hasattr(model_class, "model_validate"):
            return model_class.model_validate(payload)
        return model_class(**payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid {field_name} format: {str(exc)}",
        ) from exc
