"""Common response models for consistent API endpoint responses."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class LegacyResponseStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"


class LegacyBaseResponse(BaseModel):
    status: LegacyResponseStatus = Field(
        default=LegacyResponseStatus.SUCCESS, description="Response status"
    )
    message: str | None = Field(default=None, description="Response message")
    data: dict[str, Any] | None = Field(default=None, description="Response data")


class ValidationErrorDetail(BaseModel):
    loc: list[str] = Field(..., description="Location of the validation error")
    msg: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")


class SimpleValidationErrorResponse(BaseModel):
    detail: list[ValidationErrorDetail] = Field(..., description="List of validation errors")


class NotFoundResponse(BaseModel):
    detail: str = Field(..., description="Resource not found message")


class UnauthorizedResponse(BaseModel):
    detail: str = Field(..., description="Authentication error message")


class ForbiddenResponse(BaseModel):
    detail: str = Field(..., description="Authorization error message")


class InternalServerErrorResponse(BaseModel):
    detail: str = Field(..., description="Internal server error message")


class BadRequestResponse(BaseModel):
    detail: str = Field(..., description="Bad request error message")
    status_code: int | None = Field(400, description="HTTP status code")
    type: str | None = Field("input_validation_error", description="Error type")


class ServiceUnavailableResponse(BaseModel):
    detail: str = Field(..., description="Service unavailable error message")


class TimeoutResponse(BaseModel):
    detail: str = Field(..., description="Request timeout error message")


COMMON_RESPONSES: dict[int, dict[str, Any]] = {
    400: {"description": "Bad Request", "model": BadRequestResponse},
    401: {"description": "Unauthorized", "model": UnauthorizedResponse},
    403: {"description": "Forbidden", "model": ForbiddenResponse},
    404: {"description": "Not Found", "model": NotFoundResponse},
    408: {"description": "Request Timeout", "model": TimeoutResponse},
    422: {"description": "Validation Error", "model": SimpleValidationErrorResponse},
    500: {"description": "Internal Server Error", "model": InternalServerErrorResponse},
    503: {"description": "Service Unavailable", "model": ServiceUnavailableResponse},
}


def get_common_responses(*status_codes: int) -> dict[int, dict[str, Any]]:
    return {code: COMMON_RESPONSES[code] for code in status_codes if code in COMMON_RESPONSES}
