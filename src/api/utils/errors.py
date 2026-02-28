"""Shared error-handling helpers for API routes."""

import logging
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar

from fastapi import HTTPException

RouteHandler = TypeVar("RouteHandler", bound=Callable[..., Awaitable[Any]])
logger = logging.getLogger("atrophy_advisor.api.errors")


def route_error_handler(operation_name: str) -> Callable[[RouteHandler], RouteHandler]:
    """Wrap route handlers with consistent unexpected-error conversion."""

    def decorator(func: RouteHandler) -> RouteHandler:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                raise
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"{operation_name} failed: {str(exc)}",
                ) from exc
            except Exception as exc:
                logger.exception("%s unexpected error", operation_name)
                raise HTTPException(
                    status_code=500,
                    detail=f"{operation_name} failed due to an unexpected server error",
                ) from exc

        return wrapper  # type: ignore[return-value]

    return decorator
