"""Edge-case and error-path tests for response helper utilities.

Complements test_responses.py by covering:
- error_response with custom details dicts
- error_response default error_code
- not_found error_code generation with special characters
- success_response with None, empty, and nested data
- validation_error details merging behaviour
- All helpers produce correct response structure keys
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from src.utils.responses import (
    bad_request,
    error_response,
    forbidden,
    internal_error,
    not_found,
    service_unavailable,
    success_response,
    unauthorized,
    validation_error,
)

# ---------------------------------------------------------------------------
# error_response edge cases (not covered by test_responses.py)
# ---------------------------------------------------------------------------


class TestErrorResponseEdgeCases:
    def test_default_error_code_is_internal_error(self):
        """error_response without error_code defaults to INTERNAL_ERROR."""
        exc = error_response(500, "Boom")
        assert exc.detail["error_code"] == "INTERNAL_ERROR"

    def test_custom_details_merged(self):
        """error_response passes through custom details dict."""
        exc = error_response(400, "Bad", details={"field": "name", "reason": "too long"})
        assert exc.detail["details"]["field"] == "name"
        assert exc.detail["details"]["reason"] == "too long"

    def test_none_details_becomes_empty_dict(self):
        """error_response with details=None produces an empty dict."""
        exc = error_response(400, "Bad")
        assert exc.detail["details"] == {}

    @pytest.mark.parametrize(
        ("status_code", "message", "error_code"),
        [
            (400, "Bad request", "BAD_REQUEST"),
            (409, "Conflict", "CONFLICT"),
            (429, "Too many requests", "RATE_LIMIT"),
            (500, "Server error", "INTERNAL_ERROR"),
        ],
        ids=["400", "409", "429", "500"],
    )
    def test_status_code_and_codes_set_correctly(self, status_code, message, error_code):
        exc = error_response(status_code, message, error_code=error_code)
        assert exc.status_code == status_code
        assert exc.detail["message"] == message
        assert exc.detail["error_code"] == error_code
        assert exc.detail["status"] == "error"

    def test_all_detail_keys_present(self):
        """error_response always includes status, message, error_code, details."""
        exc = error_response(500, "err")
        keys = set(exc.detail.keys())
        assert keys == {"status", "message", "error_code", "details"}


# ---------------------------------------------------------------------------
# not_found error_code generation
# ---------------------------------------------------------------------------


class TestNotFoundErrorCode:
    @pytest.mark.parametrize(
        ("resource", "expected_code"),
        [
            ("User", "USER_NOT_FOUND"),
            ("Test Result", "TEST_RESULT_NOT_FOUND"),
            ("analysis report", "ANALYSIS_REPORT_NOT_FOUND"),
        ],
        ids=["simple", "two-words", "lowercase"],
    )
    def test_error_code_derived_from_resource(self, resource: str, expected_code: str):
        exc = not_found(resource, "id-123")
        assert exc.detail["error_code"] == expected_code

    def test_identifier_in_message(self):
        exc = not_found("Widget", "w-999")
        assert "w-999" in exc.detail["message"]
        assert "Widget" in exc.detail["message"]


# ---------------------------------------------------------------------------
# success_response edge cases
# ---------------------------------------------------------------------------


class TestSuccessResponseEdgeCases:
    @pytest.mark.parametrize(
        "data",
        [None, "", 0, False, [], {}],
        ids=["none", "empty-str", "zero", "false", "empty-list", "empty-dict"],
    )
    def test_falsy_data_preserved(self, data):
        """success_response should preserve falsy values, not drop them."""
        result = success_response(data)
        assert result["data"] is data or result["data"] == data
        assert result["status"] == "success"

    def test_nested_data_not_modified(self):
        """success_response does not deep-copy or alter the data."""
        original = {"nested": {"key": [1, 2, 3]}}
        result = success_response(original)
        assert result["data"] is original


# ---------------------------------------------------------------------------
# validation_error details merging
# ---------------------------------------------------------------------------


class TestValidationErrorDetails:
    def test_field_always_in_details(self):
        exc = validation_error("bad", "email")
        assert exc.detail["details"]["field"] == "email"

    def test_extra_details_merged_with_field(self):
        exc = validation_error("bad", "age", details={"min": 0, "max": 200})
        d = exc.detail["details"]
        assert d["field"] == "age"
        assert d["min"] == 0
        assert d["max"] == 200

    def test_explicit_field_not_overwritten_by_extra_details(self):
        """If extra details contain 'field', the explicit field arg wins."""
        exc = validation_error("bad", "email", details={"field": "SHOULD_NOT_WIN"})
        # The explicit field parameter must always take precedence
        assert exc.detail["details"]["field"] == "email"


# ---------------------------------------------------------------------------
# bad_request edge cases
# ---------------------------------------------------------------------------


class TestBadRequestEdgeCases:
    def test_with_details(self):
        exc = bad_request("Missing field", details={"field": "name"})
        assert exc.detail["details"]["field"] == "name"

    def test_without_details(self):
        exc = bad_request("Missing field")
        assert exc.detail["details"] == {}


# ---------------------------------------------------------------------------
# Parametrized: all convenience helpers produce correct status codes and codes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("factory", "kwargs", "expected_status", "expected_code"),
    [
        (unauthorized, {}, 401, "UNAUTHORIZED"),
        (unauthorized, {"message": "Token expired"}, 401, "UNAUTHORIZED"),
        (forbidden, {}, 403, "FORBIDDEN"),
        (forbidden, {"message": "No access"}, 403, "FORBIDDEN"),
        (service_unavailable, {}, 503, "SERVICE_UNAVAILABLE"),
        (service_unavailable, {"message": "DB down"}, 503, "SERVICE_UNAVAILABLE"),
        (internal_error, {"message": "Crash"}, 500, "INTERNAL_ERROR"),
        (internal_error, {"message": "Crash", "details": {"trace": "x"}}, 500, "INTERNAL_ERROR"),
    ],
    ids=[
        "unauthorized-default",
        "unauthorized-custom",
        "forbidden-default",
        "forbidden-custom",
        "unavailable-default",
        "unavailable-custom",
        "internal-default",
        "internal-with-details",
    ],
)
def test_convenience_helpers(factory, kwargs, expected_status, expected_code):
    exc = factory(**kwargs)
    assert isinstance(exc, HTTPException)
    assert exc.status_code == expected_status
    assert exc.detail["error_code"] == expected_code
    assert exc.detail["status"] == "error"
    if "message" in kwargs:
        assert exc.detail["message"] == kwargs["message"]


# ---------------------------------------------------------------------------
# internal_error details propagation
# ---------------------------------------------------------------------------


class TestInternalErrorDetails:
    def test_details_forwarded(self):
        exc = internal_error("fail", details={"db": "pg", "query": "SELECT 1"})
        assert exc.detail["details"]["db"] == "pg"
        assert exc.detail["details"]["query"] == "SELECT 1"

    def test_no_details_gives_empty_dict(self):
        exc = internal_error("fail")
        assert exc.detail["details"] == {}
