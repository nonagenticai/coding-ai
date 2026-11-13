"""Unified response helpers for consistent API responses."""

from typing import Any

from fastapi import HTTPException, status


def success_response(data: Any, message: str = "Success") -> dict:
    return {"status": "success", "message": message, "data": data}


def error_response(
    status_code: int,
    message: str,
    error_code: str = "INTERNAL_ERROR",
    details: dict | None = None,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "status": "error",
            "message": message,
            "error_code": error_code,
            "details": details or {},
        },
    )


def not_found(resource: str, identifier: str) -> HTTPException:
    return error_response(
        status_code=status.HTTP_404_NOT_FOUND,
        message=f"{resource} '{identifier}' not found",
        error_code=f"{resource.upper().replace(' ', '_')}_NOT_FOUND",
    )


def bad_request(message: str, details: dict | None = None) -> HTTPException:
    return error_response(
        status_code=status.HTTP_400_BAD_REQUEST,
        message=message,
        error_code="BAD_REQUEST",
        details=details,
    )


def validation_error(message: str, field: str, details: dict | None = None) -> HTTPException:
    return error_response(
        status_code=422,
        message=message,
        error_code="VALIDATION_ERROR",
        details={**(details or {}), "field": field},
    )


def internal_error(message: str, details: dict | None = None) -> HTTPException:
    return error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, message=message, details=details
    )


def unauthorized(message: str = "Authentication required") -> HTTPException:
    return error_response(
        status_code=status.HTTP_401_UNAUTHORIZED, message=message, error_code="UNAUTHORIZED"
    )


def forbidden(message: str = "Access denied") -> HTTPException:
    return error_response(
        status_code=status.HTTP_403_FORBIDDEN, message=message, error_code="FORBIDDEN"
    )


def service_unavailable(message: str = "Service temporarily unavailable") -> HTTPException:
    return error_response(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        message=message,
        error_code="SERVICE_UNAVAILABLE",
    )
