"""Common API models for coding-ai."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from type_patterns import SAFE_STRING_PATTERN


class ResponseStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class BaseResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "message": "Operation completed successfully",
                "timestamp": "2025-07-13T10:30:00Z",
                "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            }
        }
    )

    status: ResponseStatus = Field(..., description="Response status")
    message: str = Field(..., description="Human-readable message", pattern=SAFE_STRING_PATTERN)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Response timestamp",
    )
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        min_length=1,
        description="Unique request identifier",
    )


class SuccessResponse(BaseResponse):
    status: ResponseStatus = ResponseStatus.SUCCESS
    data: dict[str, Any] | None = Field(None, description="Response data")


class PaginationParams(BaseModel):
    page: int = Field(1, ge=1, description="Page number (1-based)")
    limit: int = Field(50, ge=1, le=1000, description="Items per page")


class PaginatedResponse(SuccessResponse):
    pagination: dict[str, Any] = Field(..., description="Pagination metadata")
    items: list[Any] = Field(..., description="Response items")

    @field_validator("pagination")
    @classmethod
    def validate_pagination(cls, v):
        required_keys = {"page", "limit", "total", "pages"}
        if not all(key in v for key in required_keys):
            raise ValueError(f"Pagination must contain: {required_keys}")
        return v


class HealthStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentHealth(BaseModel):
    name: str = Field(..., min_length=1, pattern=SAFE_STRING_PATTERN)
    status: HealthStatus = Field(...)
    message: str | None = Field(None, pattern=SAFE_STRING_PATTERN)
    last_check: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metrics: dict[str, Any] | None = Field(None)
