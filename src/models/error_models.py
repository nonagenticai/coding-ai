"""Error response models for coding-ai API."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_serializer

from type_patterns import SAFE_STRING_PATTERN

from .common_api_models import ResponseStatus


class StandardErrorResponse(BaseModel):
    status: ResponseStatus = Field(default=ResponseStatus.ERROR)
    message: str = Field(..., pattern=SAFE_STRING_PATTERN)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    request_id: str = Field(..., pattern=SAFE_STRING_PATTERN)
    path: str = Field(..., pattern=SAFE_STRING_PATTERN)
    error_code: str | None = Field(None, pattern=SAFE_STRING_PATTERN)
    details: dict[str, Any] | None = Field(None)

    @field_serializer("timestamp")
    def serialize_timestamp(self, timestamp: datetime) -> str:
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        return timestamp.isoformat()


class APIValidationErrorDetail(BaseModel):
    field: str = Field(..., pattern=SAFE_STRING_PATTERN)
    message: str = Field(..., pattern=SAFE_STRING_PATTERN)
    type: str = Field(..., pattern=SAFE_STRING_PATTERN)
    input: Any = Field(None)


class ValidationErrorResponse(StandardErrorResponse):
    message: str = Field(default="Request validation failed", pattern=SAFE_STRING_PATTERN)
    error_code: str = Field(default="VALIDATION_ERROR", pattern=SAFE_STRING_PATTERN)
    validation_errors: list[APIValidationErrorDetail] = Field(...)


class Error400Response(StandardErrorResponse):
    message: str = Field(default="Bad request", pattern=SAFE_STRING_PATTERN)
    error_code: str = Field(default="BAD_REQUEST", pattern=SAFE_STRING_PATTERN)


class Error401Response(StandardErrorResponse):
    message: str = Field(default="Authentication required", pattern=SAFE_STRING_PATTERN)
    error_code: str = Field(default="UNAUTHORIZED", pattern=SAFE_STRING_PATTERN)


class Error403Response(StandardErrorResponse):
    message: str = Field(default="Insufficient permissions", pattern=SAFE_STRING_PATTERN)
    error_code: str = Field(default="FORBIDDEN", pattern=SAFE_STRING_PATTERN)


class Error404Response(StandardErrorResponse):
    message: str = Field(default="Resource not found", pattern=SAFE_STRING_PATTERN)
    error_code: str = Field(default="NOT_FOUND", pattern=SAFE_STRING_PATTERN)


class Error500Response(StandardErrorResponse):
    message: str = Field(default="Internal server error", pattern=SAFE_STRING_PATTERN)
    error_code: str = Field(default="INTERNAL_ERROR", pattern=SAFE_STRING_PATTERN)


class Error429Response(StandardErrorResponse):
    message: str = Field(default="Too many requests", pattern=SAFE_STRING_PATTERN)
    error_code: str = Field(default="RATE_LIMIT_EXCEEDED", pattern=SAFE_STRING_PATTERN)


class Error503Response(StandardErrorResponse):
    message: str = Field(default="Service temporarily unavailable", pattern=SAFE_STRING_PATTERN)
    error_code: str = Field(default="SERVICE_UNAVAILABLE", pattern=SAFE_STRING_PATTERN)
