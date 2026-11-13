"""Standardized response builder for coding-ai API."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from fastapi import Request


class ResponseBuilder:
    def __init__(self, request: Request | None = None):
        self.request = request
        self.start_time = time.time()

    def success(
        self, data: Any, source: str = "local", additional_meta: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        execution_time = int((time.time() - self.start_time) * 1000)
        meta = {
            "source": source,
            "execution_time_ms": execution_time,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        if additional_meta:
            meta.update(additional_meta)
        return {"success": True, "data": data, "meta": meta}

    def error(
        self,
        error_message: str,
        error_code: str | None = None,
        suggestions: list[str] | None = None,
    ) -> dict[str, Any]:
        execution_time = int((time.time() - self.start_time) * 1000)
        error_data: dict[str, Any] = {
            "message": error_message,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        if error_code:
            error_data["code"] = error_code
        if suggestions:
            error_data["suggestions"] = suggestions
        return {
            "success": False,
            "data": None,
            "meta": {
                "execution_time_ms": execution_time,
                "timestamp": datetime.now(UTC).isoformat(),
            },
            "error": error_data,
        }


def build_success_response(
    data: Any, source: str = "local", request: Request | None = None
) -> dict[str, Any]:
    return ResponseBuilder(request).success(data, source)


def build_error_response(
    error_message: str, error_code: str | None = None, request: Request | None = None
) -> dict[str, Any]:
    return ResponseBuilder(request).error(error_message, error_code)
