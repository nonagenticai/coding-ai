"""Tests for response helper utilities."""

from fastapi import HTTPException

from src.utils.responses import bad_request, error_response, not_found, success_response


def test_success_response_structure():
    result = success_response({"key": "value"})
    assert result["status"] == "success"
    assert result["message"] == "Success"
    assert result["data"] == {"key": "value"}


def test_success_response_custom_message():
    result = success_response([], message="Created")
    assert result["message"] == "Created"


def test_error_response_raises_http_exception():
    exc = error_response(400, "Bad input", error_code="INVALID")
    assert isinstance(exc, HTTPException)
    assert exc.status_code == 400
    assert exc.detail["message"] == "Bad input"  # type: ignore[invalid-argument-type]
    assert exc.detail["error_code"] == "INVALID"  # type: ignore[invalid-argument-type]
    assert exc.detail["status"] == "error"  # type: ignore[invalid-argument-type]


def test_not_found_returns_404():
    exc = not_found("User", "abc-123")
    assert isinstance(exc, HTTPException)
    assert exc.status_code == 404
    assert "User" in exc.detail["message"]  # type: ignore[invalid-argument-type]
    assert "abc-123" in exc.detail["message"]  # type: ignore[invalid-argument-type]


def test_bad_request_returns_400():
    exc = bad_request("Missing field")
    assert isinstance(exc, HTTPException)
    assert exc.status_code == 400
    assert exc.detail["message"] == "Missing field"  # type: ignore[invalid-argument-type]
