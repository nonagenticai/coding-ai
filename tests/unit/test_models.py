"""Tests for Pydantic model validation."""

import pytest
from pydantic import ValidationError

from src.models.ai_test_failure_models import (
    AIAnalysisResult,
    ConfidenceLevel,
    ErrorDetails,
    FailureType,
    TestFailureContext,
    TestFailureData,
)
from src.models.error_models import StandardErrorResponse
from src.models.testing_models import TestRequest


def _make_failure_data(**overrides):
    defaults = {
        "failure_id": "test-001",
        "raw_output": "FAILED test_example - AssertionError",
        "context": TestFailureContext(test_name="test_example"),
    }
    defaults.update(overrides)
    return TestFailureData(**defaults)  # type: ignore[invalid-argument-type]


def test_test_failure_data_valid():
    data = _make_failure_data()
    assert data.failure_id == "test-001"
    assert data.context.test_name == "test_example"


def test_test_failure_data_invalid_id():
    with pytest.raises(ValidationError):
        _make_failure_data(failure_id="invalid id with spaces!")


def test_ai_analysis_result_valid():
    failure_data = _make_failure_data()
    error_details = ErrorDetails(
        error_type=FailureType.ASSERTION_ERROR,
        error_message="Expected 1 got 2",
    )
    result = AIAnalysisResult(
        analysis_id="analysis-001",
        failure_data=failure_data,
        error_details=error_details,
        analysis_confidence=ConfidenceLevel.HIGH,
        analysis_confidence_score=0.85,
    )
    assert result.analysis_id == "analysis-001"
    assert result.analysis_confidence == "high"


def test_standard_error_response_valid():
    resp = StandardErrorResponse(
        message="Something went wrong",
        request_id="req-123",
        path="/api/test",
    )
    assert resp.message == "Something went wrong"
    assert resp.status.value == "error"


def test_standard_error_response_rejects_control_chars():
    with pytest.raises(ValidationError):
        StandardErrorResponse(
            message="bad\x00message",
            request_id="req-123",
            path="/api/test",
        )


def test_test_request_allows_extra_fields():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        req = TestRequest(
            container_name="test-container",
            test_commands=["pytest"],
            foo="bar",  # type: ignore[unknown-argument]  # extra="allow" model
        )
    assert req.container_name == "test-container"
    assert req.foo == "bar"  # type: ignore[unresolved-attribute]
