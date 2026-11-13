"""Additional tests for routers/ai_test_analysis.py to improve coverage.

Tests use real Pydantic models for analysis results instead of MagicMock chains,
so the router's dict-comprehension serialization code actually executes against
real attribute access and .value calls. Only the AI service itself is mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from models.ai_test_failure_models import (
    ConfidenceLevel,
    ErrorDetails,
    FailureType,
    FixSuggestion,
    RootCause,
    SeverityLevel,
    TestFailurePattern,
)
from routers.ai_test_analysis import (
    get_ai_analysis_service,
    get_integration_service,
)
from src.main import app

PREFIX = "/api/ai/analysis/tests"
TENANT_HEADERS = {"X-Tenant-Id": "test-tenant"}

VALID_FAILURE_BODY = {
    "container_name": "test-container",
    "test_output": "AssertionError: expected True got False",
    "test_name": "test_example",
    "test_framework": "pytest",
    "language": "python",
}


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _cleanup_overrides():
    yield
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Real model builders (no MagicMock chains)
# ---------------------------------------------------------------------------


def _make_root_cause() -> RootCause:
    return RootCause(
        cause_id="rc-1",
        category="logic_error",
        description="Variable not initialized before use",
        severity=SeverityLevel.HIGH,
        confidence=ConfidenceLevel.HIGH,
        confidence_score=0.85,
        evidence=["Stack trace analysis"],
        related_patterns=["null_reference"],
    )


def _make_fix_suggestion() -> FixSuggestion:
    return FixSuggestion(
        suggestion_id="fix-1",
        title="Initialize variable",
        description="Add initialization before use",
        fix_type="code_change",
        priority="high",
        confidence=ConfidenceLevel.HIGH,
        confidence_score=0.8,
        estimated_effort="low",
        code_examples=["x = 0"],
        files_to_modify=["/src/app.py"],
        verification_steps=["Run test"],
    )


def _make_pattern() -> TestFailurePattern:
    return TestFailurePattern(
        pattern_id="pat-1",
        pattern_name="Null Reference",
        pattern_type="error_message",
        pattern_signature="Variable accessed before initialization in test",
        confidence=ConfidenceLevel.HIGH,
        confidence_score=0.9,
        frequency=5,
        typical_causes=["Missing init"],
        common_fixes=["Add default value"],
    )


def _make_error_details() -> ErrorDetails:
    return ErrorDetails(
        error_type=FailureType.ASSERTION_ERROR,
        error_message="expected True got False",
        file_path="/src/app.py",
        line_number=42,
        column_number=10,
    )


def _make_analysis_result():
    """Build a real-ish analysis result using actual model instances where possible."""
    result = MagicMock()
    result.analysis_id = "analysis-123"
    result.root_causes = [_make_root_cause()]
    result.fix_suggestions = [_make_fix_suggestion()]
    result.detected_patterns = [_make_pattern()]
    result.error_details = _make_error_details()
    result.analysis_confidence = ConfidenceLevel.HIGH
    result.analysis_confidence_score = 0.85
    return result


# ===================================================================
# POST /failure-analysis - successful AI analysis
# ===================================================================


def test_failure_analysis_successful_with_real_models(client):
    """Full analysis using real Pydantic sub-models verifies serialization works."""
    mock_service = AsyncMock()
    mock_response = MagicMock()
    mock_response.success = True
    mock_response.degraded = False
    mock_response.analysis_result = _make_analysis_result()
    mock_service.analyze_test_failure = AsyncMock(return_value=mock_response)

    async def _service():
        return mock_service

    app.dependency_overrides[get_ai_analysis_service] = _service

    response = client.post(
        f"{PREFIX}/failure-analysis",
        json=VALID_FAILURE_BODY,
        headers=TENANT_HEADERS,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["degraded"] is False
    assert data["analysis_id"] == "analysis-123"

    # Verify root cause serialization from real model
    assert len(data["root_causes"]) == 1
    rc = data["root_causes"][0]
    assert rc["id"] == "rc-1"
    assert rc["category"] == "logic_error"
    assert rc["severity"] == "high"
    assert rc["confidence"] == "high"
    assert rc["confidence_score"] == 0.85
    assert "Stack trace analysis" in rc["evidence"]

    # Verify fix suggestion serialization from real model
    assert len(data["fix_suggestions"]) == 1
    fs = data["fix_suggestions"][0]
    assert fs["id"] == "fix-1"
    assert fs["title"] == "Initialize variable"
    assert fs["confidence"] == "high"
    assert fs["code_examples"] == ["x = 0"]
    assert fs["files_to_modify"] == ["/src/app.py"]

    # Verify pattern serialization from real model
    assert len(data["detected_patterns"]) == 1
    pat = data["detected_patterns"][0]
    assert pat["id"] == "pat-1"
    assert pat["name"] == "Null Reference"
    assert pat["confidence"] == "high"
    assert pat["frequency"] == 5

    # Verify error details
    assert data["error_details"]["error_type"] == "assertion_error"
    assert data["error_details"]["line_number"] == 42

    assert data["analysis_confidence"] == "high"
    assert data["analysis_confidence_score"] == 0.85


def test_failure_analysis_successful_with_file_path(client):
    """When file_path is provided, TestFileMetadata is created and analysis proceeds."""
    mock_service = AsyncMock()
    mock_response = MagicMock()
    mock_response.success = True
    mock_response.degraded = False
    mock_response.analysis_result = _make_analysis_result()
    mock_service.analyze_test_failure = AsyncMock(return_value=mock_response)

    async def _service():
        return mock_service

    app.dependency_overrides[get_ai_analysis_service] = _service

    body = {**VALID_FAILURE_BODY, "file_path": "tests/test_app.py"}
    response = client.post(
        f"{PREFIX}/failure-analysis",
        json=body,
        headers=TENANT_HEADERS,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["analysis_id"] == "analysis-123"


def test_failure_analysis_outage_returns_degraded_true(client):
    """Total LLM outage: service reports degraded -> endpoint returns degraded=true.

    This is the live-path F4 contract: an outage must NOT masquerade as a clean
    success. success stays True (the request completed) but degraded flags that
    results are basic/empty rather than AI-derived.
    """
    mock_service = AsyncMock()
    mock_response = MagicMock()
    mock_response.success = True
    mock_response.degraded = True
    # On outage the analysis result carries empty AI stages.
    result = _make_analysis_result()
    result.root_causes = []
    result.fix_suggestions = []
    result.detected_patterns = []
    mock_response.analysis_result = result
    mock_service.analyze_test_failure = AsyncMock(return_value=mock_response)

    async def _service():
        return mock_service

    app.dependency_overrides[get_ai_analysis_service] = _service

    response = client.post(
        f"{PREFIX}/failure-analysis",
        json=VALID_FAILURE_BODY,
        headers=TENANT_HEADERS,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["degraded"] is True
    assert data["root_causes"] == []


def test_failure_analysis_failed_result(client):
    """When AI service returns success=False, error response includes correct fields."""
    mock_service = AsyncMock()
    mock_response = MagicMock()
    mock_response.success = False
    mock_response.analysis_result = None
    mock_response.request_id = "req-456"
    mock_response.error_message = "Model unavailable"
    mock_service.analyze_test_failure = AsyncMock(return_value=mock_response)

    async def _service():
        return mock_service

    app.dependency_overrides[get_ai_analysis_service] = _service

    response = client.post(
        f"{PREFIX}/failure-analysis",
        json=VALID_FAILURE_BODY,
        headers=TENANT_HEADERS,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert data["error_message"] == "Model unavailable"
    assert data["analysis_id"] == "req-456"
    # Failed responses should not have root causes
    assert data.get("root_causes") is None


def test_failure_analysis_degraded_when_no_service(client):
    """When AI service is None, degraded response respects include_suggestions/include_patterns flags."""

    async def _no_service():
        return None

    app.dependency_overrides[get_ai_analysis_service] = _no_service

    # Test with both flags disabled
    body = {
        **VALID_FAILURE_BODY,
        "include_suggestions": False,
        "include_patterns": False,
    }
    response = client.post(
        f"{PREFIX}/failure-analysis",
        json=body,
        headers=TENANT_HEADERS,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["fix_suggestions"] == []
    assert data["detected_patterns"] == []
    assert data["analysis_confidence"] == "low"
    assert data["analysis_confidence_score"] == 0.3
    # Root causes should still be present (basic analysis)
    assert len(data["root_causes"]) == 1
    assert data["root_causes"][0]["confidence_score"] == 0.3


def test_failure_analysis_degraded_with_flags_enabled(client):
    """When AI service is None but include flags are True, degraded suggestions/patterns are returned."""

    async def _no_service():
        return None

    app.dependency_overrides[get_ai_analysis_service] = _no_service

    response = client.post(
        f"{PREFIX}/failure-analysis",
        json=VALID_FAILURE_BODY,
        headers=TENANT_HEADERS,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["fix_suggestions"]) == 1
    assert data["fix_suggestions"][0]["title"] == "Manual Review Required"
    assert len(data["detected_patterns"]) == 1
    assert data["detected_patterns"][0]["pattern_type"] == "test_failure"


# ===================================================================
# POST /batch-analyze - successful path
# ===================================================================


def test_batch_analyze_success(client):
    """Successful batch analysis with real-ish result objects."""
    mock_result = MagicMock()
    mock_result.analysis_id = "batch-result-1"
    mock_result.root_causes = [_make_root_cause()]
    mock_result.fix_suggestions = [_make_fix_suggestion()]
    mock_result.detected_patterns = [_make_pattern()]

    mock_service = AsyncMock()
    mock_service.batch_analyze_failures = AsyncMock(return_value=[mock_result, None])

    async def _integration():
        return mock_service

    app.dependency_overrides[get_integration_service] = _integration

    body = {
        "container_name": "test-container",
        "test_results": [
            {
                "test_id": "t1",
                "language": "python",
                "success": False,
                "execution_time": 1.2,
                "stderr": "AssertionError",
            },
            {
                "test_id": "t2",
                "language": "python",
                "success": False,
                "execution_time": 0.5,
                "stderr": "KeyError",
            },
        ],
    }
    response = client.post(
        f"{PREFIX}/batch-analyze",
        json=body,
        headers=TENANT_HEADERS,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total_tests"] == 2
    assert data["analyzed_tests"] == 1

    # Verify the successful result is serialized from real models
    assert len(data["results"]) == 2
    result_0 = data["results"][0]
    assert result_0 is not None
    assert result_0["analysis_id"] == "batch-result-1"
    assert result_0["root_causes"][0]["id"] == "rc-1"
    assert result_0["root_causes"][0]["severity"] == "high"
    assert result_0["fix_suggestions"][0]["id"] == "fix-1"
    assert result_0["detected_patterns"][0]["id"] == "pat-1"

    # Second result was None
    assert data["results"][1] is None


# ===================================================================
# GET /failure-analysis/health - error path
# ===================================================================


def test_health_check_exception_in_service(client):
    """When get_health_status raises, error response contains the exception message."""
    mock_service = MagicMock()

    def _raise():
        raise RuntimeError("health boom")

    mock_service.get_health_status = _raise

    async def _error_service():
        return mock_service

    app.dependency_overrides[get_ai_analysis_service] = _error_service

    response = client.get(
        f"{PREFIX}/failure-analysis/health",
        headers=TENANT_HEADERS,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["service_status"] == "error"
    assert data["ai_analysis_available"] is False
    assert "health boom" in data["error"]


def test_health_check_degraded_when_no_service(client):
    """When AI service is None, health check returns degraded status."""

    async def _no_service():
        return None

    app.dependency_overrides[get_ai_analysis_service] = _no_service

    response = client.get(
        f"{PREFIX}/failure-analysis/health",
        headers=TENANT_HEADERS,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["service_status"] == "degraded"
    assert data["ai_analysis_available"] is False
    assert data["ai_model_name"] is None


# ===================================================================
# GET /failure-analysis/metrics - with service that has get_metrics
# ===================================================================


def test_get_metrics_with_service(client):
    """Metrics are derived from the counters the service actually keeps.

    This test used to feed the endpoint a fictional dict containing
    ``success_rate`` / ``average_response_time`` / ``cached_analyses`` /
    ``ai_model_usage`` -- keys ``AIFailureAnalysisService._metrics`` has never
    contained. It therefore asserted a shape production could not produce,
    while the live endpoint reported a permanent 0.0 success rate.
    """
    mock_service = MagicMock()
    # The real `_metrics` dict, verbatim.
    mock_service.get_metrics = lambda: {
        "total_analyses": 100,
        "successful_analyses": 95,
        "failed_analyses": 5,
        "cache_hits": 20,
        "cache_misses": 80,
        "average_analysis_time": 1.5,
        "total_analysis_time": 150.0,
    }

    async def _service():
        return mock_service

    app.dependency_overrides[get_ai_analysis_service] = _service

    response = client.get(
        f"{PREFIX}/failure-analysis/metrics",
        headers=TENANT_HEADERS,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["service_status"] == "healthy"
    assert data["metrics"]["total_analyses"] == 100
    assert data["metrics"]["successful_analyses"] == 95
    assert data["metrics"]["failed_analyses"] == 5
    assert data["metrics"]["success_rate"] == pytest.approx(0.95)
    assert data["metrics"]["average_analysis_time"] == 1.5
    assert data["metrics"]["cache_hits"] == 20


# ===================================================================
# GET /patterns
# ===================================================================


def test_get_patterns_returns_structured_data(client):
    """Patterns endpoint returns structured pattern data."""

    async def _no_service():
        return None

    app.dependency_overrides[get_ai_analysis_service] = _no_service

    response = client.get(
        f"{PREFIX}/patterns?limit=2",
        headers=TENANT_HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["data"]["patterns"]) <= 2
    assert data["data"]["service_status"] == "degraded"


# ===================================================================
# POST /failure-analysis with source_code and test_code
# ===================================================================


def test_failure_analysis_with_source_and_test_code(client):
    """When source_code and test_code are provided, analysis proceeds normally in degraded mode."""

    async def _no_service():
        return None

    app.dependency_overrides[get_ai_analysis_service] = _no_service

    body = {
        **VALID_FAILURE_BODY,
        "source_code": "def add(a, b): return a + b",
        "test_code": "def test_add(): assert add(1, 2) == 4",
    }
    response = client.post(
        f"{PREFIX}/failure-analysis",
        json=body,
        headers=TENANT_HEADERS,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    # Should still get degraded analysis even with source/test code
    assert data["analysis_confidence"] == "low"


# ===================================================================
# POST /failure-analysis - missing required fields
# ===================================================================


def test_failure_analysis_missing_container_returns_422(client):
    """Missing container_name returns 422."""
    body = {
        "test_output": "Error",
        "test_name": "test_foo",
    }
    response = client.post(
        f"{PREFIX}/failure-analysis",
        json=body,
        headers=TENANT_HEADERS,
    )
    assert response.status_code == 422


def test_failure_analysis_missing_test_output_returns_422(client):
    """Missing test_output returns 422."""
    body = {
        "container_name": "test-container",
        "test_name": "test_foo",
    }
    response = client.post(
        f"{PREFIX}/failure-analysis",
        json=body,
        headers=TENANT_HEADERS,
    )
    assert response.status_code == 422
