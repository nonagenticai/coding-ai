"""Tests for utils/response_builder.py covering all lines.

This module tests the ResponseBuilder class and the convenience functions
build_success_response and build_error_response.
"""

from __future__ import annotations

import time

from utils.response_builder import (
    ResponseBuilder,
    build_error_response,
    build_success_response,
)

# ===================================================================
# ResponseBuilder.success
# ===================================================================


class TestResponseBuilderSuccess:
    def test_basic_success(self):
        rb = ResponseBuilder()
        result = rb.success(data={"key": "value"})
        assert result["success"] is True
        assert result["data"] == {"key": "value"}
        assert result["meta"]["source"] == "local"
        assert "execution_time_ms" in result["meta"]
        assert "timestamp" in result["meta"]

    def test_custom_source(self):
        rb = ResponseBuilder()
        result = rb.success(data="hello", source="remote")
        assert result["meta"]["source"] == "remote"

    def test_additional_meta(self):
        rb = ResponseBuilder()
        result = rb.success(data=None, additional_meta={"custom": 42})
        assert result["meta"]["custom"] == 42
        assert result["meta"]["source"] == "local"

    def test_no_additional_meta(self):
        rb = ResponseBuilder()
        result = rb.success(data=[1, 2, 3])
        assert "custom" not in result["meta"]

    def test_execution_time_positive(self):
        rb = ResponseBuilder()
        time.sleep(0.01)
        result = rb.success(data=None)
        assert result["meta"]["execution_time_ms"] >= 0

    def test_with_request(self):
        """ResponseBuilder accepts a request parameter (can be None)."""
        rb = ResponseBuilder(request=None)
        result = rb.success(data="test")
        assert result["success"] is True


# ===================================================================
# ResponseBuilder.error
# ===================================================================


class TestResponseBuilderError:
    def test_basic_error(self):
        rb = ResponseBuilder()
        result = rb.error(error_message="something went wrong")
        assert result["success"] is False
        assert result["data"] is None
        assert result["error"]["message"] == "something went wrong"
        assert "timestamp" in result["error"]
        assert "execution_time_ms" in result["meta"]

    def test_with_error_code(self):
        rb = ResponseBuilder()
        result = rb.error(error_message="not found", error_code="NOT_FOUND")
        assert result["error"]["code"] == "NOT_FOUND"

    def test_without_error_code(self):
        rb = ResponseBuilder()
        result = rb.error(error_message="generic error")
        assert "code" not in result["error"]

    def test_with_suggestions(self):
        rb = ResponseBuilder()
        result = rb.error(
            error_message="bad request",
            suggestions=["fix this", "try that"],
        )
        assert result["error"]["suggestions"] == ["fix this", "try that"]

    def test_without_suggestions(self):
        rb = ResponseBuilder()
        result = rb.error(error_message="bad request")
        assert "suggestions" not in result["error"]

    def test_all_options(self):
        rb = ResponseBuilder()
        result = rb.error(
            error_message="validation failed",
            error_code="VALIDATION_ERROR",
            suggestions=["check input"],
        )
        assert result["error"]["code"] == "VALIDATION_ERROR"
        assert result["error"]["suggestions"] == ["check input"]
        assert result["error"]["message"] == "validation failed"


# ===================================================================
# Convenience functions
# ===================================================================


class TestConvenienceFunctions:
    def test_build_success_response(self):
        result = build_success_response(data={"ok": True})
        assert result["success"] is True
        assert result["data"] == {"ok": True}

    def test_build_success_response_with_source(self):
        result = build_success_response(data=None, source="cache")
        assert result["meta"]["source"] == "cache"

    def test_build_success_response_with_request(self):
        result = build_success_response(data="hello", request=None)
        assert result["success"] is True

    def test_build_error_response(self):
        result = build_error_response(error_message="fail")
        assert result["success"] is False
        assert result["error"]["message"] == "fail"

    def test_build_error_response_with_code(self):
        result = build_error_response(error_message="fail", error_code="ERR_500")
        assert result["error"]["code"] == "ERR_500"

    def test_build_error_response_with_request(self):
        result = build_error_response(error_message="fail", request=None)
        assert result["success"] is False
