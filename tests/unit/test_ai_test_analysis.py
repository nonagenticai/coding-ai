"""Tests for AI test analysis router endpoints."""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from routers.ai_test_analysis import (
    get_ai_analysis_service,
    get_integration_service,
)
from src.main import app

PREFIX = "/api/ai/analysis/tests"

TENANT_HEADERS = {"X-Tenant-Id": "test-tenant"}

VALID_FAILURE_ANALYSIS_BODY = {
    "container_name": "test-container",
    "test_output": "AssertionError: expected True got False",
    "test_name": "test_example",
    "test_framework": "pytest",
    "language": "python",
}

VALID_BATCH_BODY = {
    "container_name": "test-container",
    "test_results": [
        {
            "test_id": "t1",
            "language": "python",
            "success": False,
            "execution_time": 1.2,
            "stderr": "AssertionError",
        }
    ],
}


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _cleanup_overrides():
    """Clean up dependency overrides after each test."""
    yield
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# POST /failure-analysis
# ---------------------------------------------------------------------------


def test_failure_analysis_no_service_returns_degraded(client):
    """When ai_service is None, the endpoint returns a degraded but successful response."""

    async def _no_service():
        return None

    app.dependency_overrides[get_ai_analysis_service] = _no_service

    response = client.post(
        f"{PREFIX}/failure-analysis",
        json=VALID_FAILURE_ANALYSIS_BODY,
        headers=TENANT_HEADERS,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["degraded"] is True
    assert data["analysis_confidence"] == "low"
    assert data["analysis_confidence_score"] == 0.3
    assert len(data["root_causes"]) >= 1
    assert data["root_causes"][0]["category"] == "unknown"


def test_failure_analysis_service_exception_returns_degraded(client):
    """When the ai_service raises an exception, router catches it and returns degraded 200."""

    mock_service = AsyncMock()
    mock_service.analyze_test_failure = AsyncMock(side_effect=RuntimeError("AI backend exploded"))

    async def _failing_service():
        return mock_service

    app.dependency_overrides[get_ai_analysis_service] = _failing_service

    response = client.post(
        f"{PREFIX}/failure-analysis",
        json=VALID_FAILURE_ANALYSIS_BODY,
        headers=TENANT_HEADERS,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["degraded"] is True
    assert len(data["root_causes"]) >= 1
    assert data["root_causes"][0]["category"] == "unknown"


def test_failure_analysis_missing_fields_returns_422(client):
    """Posting an empty body should trigger Pydantic validation (422)."""

    async def _no_service():
        return None

    app.dependency_overrides[get_ai_analysis_service] = _no_service

    response = client.post(
        f"{PREFIX}/failure-analysis",
        json={},
        headers=TENANT_HEADERS,
    )

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /batch-analyze
# ---------------------------------------------------------------------------


def test_batch_analyze_no_service_returns_503(client):
    """When integration_service is None, the endpoint raises 503."""

    async def _no_integration():
        return None

    app.dependency_overrides[get_integration_service] = _no_integration

    response = client.post(
        f"{PREFIX}/batch-analyze",
        json=VALID_BATCH_BODY,
        headers=TENANT_HEADERS,
    )

    assert response.status_code == 503


def test_batch_analyze_service_exception_returns_500(client):
    """When the integration service raises an exception, expect 500."""

    mock_service = AsyncMock()
    mock_service.batch_analyze_failures = AsyncMock(side_effect=RuntimeError("batch boom"))

    async def _failing_integration():
        return mock_service

    app.dependency_overrides[get_integration_service] = _failing_integration

    response = client.post(
        f"{PREFIX}/batch-analyze",
        json=VALID_BATCH_BODY,
        headers=TENANT_HEADERS,
    )

    assert response.status_code == 500


# ---------------------------------------------------------------------------
# GET /failure-analysis/health
# ---------------------------------------------------------------------------


def test_health_check_no_service(client):
    """Health check with no AI service should return degraded status."""

    async def _no_service():
        return None

    app.dependency_overrides[get_ai_analysis_service] = _no_service

    response = client.get(
        f"{PREFIX}/failure-analysis/health",
        headers=TENANT_HEADERS,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "ai_test_analysis"
    assert data["service_status"] == "degraded"
    assert data["ai_analysis_available"] is False


def test_health_check_with_service(client):
    """Health check with a mock service should return healthy status."""

    mock_service = AsyncMock()
    # The real shape of AIFailureAnalysisService.get_health_status(). This test
    # used to feed keys ("status", "ai_analysis_available", "ai_model_name",
    # "checks") that the service has never emitted, so it green-lit an endpoint
    # that reported "healthy / available / gpt-4" unconditionally.
    mock_service.get_health_status = lambda: {
        "service_name": "AI Failure Analysis Service",
        "model_name": "gpt-4-test",
        "active_analyses": 0,
        "circuit_breaker_open": False,
        "circuit_breaker_failures": 0,
        "metrics": {"total_analyses": 42},
    }

    async def _healthy_service():
        return mock_service

    app.dependency_overrides[get_ai_analysis_service] = _healthy_service

    response = client.get(
        f"{PREFIX}/failure-analysis/health",
        headers=TENANT_HEADERS,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "ai_test_analysis"
    assert data["service_status"] == "healthy"
    assert data["ai_analysis_available"] is True
    assert data["ai_model_name"] == "gpt-4-test"


# ---------------------------------------------------------------------------
# GET /failure-analysis/metrics
# ---------------------------------------------------------------------------


def test_get_metrics_no_service(client):
    """Metrics with no service should return degraded status with zero metrics."""

    async def _no_service():
        return None

    app.dependency_overrides[get_ai_analysis_service] = _no_service

    response = client.get(
        f"{PREFIX}/failure-analysis/metrics",
        headers=TENANT_HEADERS,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "ai_test_analysis"
    assert data["service_status"] == "degraded"
    assert data["metrics"]["total_analyses"] == 0


def test_get_metrics_error_returns_correct_format(client):
    """When the service raises, the error branch returns service/metrics/service_status keys."""

    mock_service = AsyncMock()

    # Make get_metrics raise an exception to trigger the except branch
    def _raise_metrics():
        raise RuntimeError("metrics boom")

    mock_service.get_metrics = _raise_metrics

    async def _error_service():
        return mock_service

    app.dependency_overrides[get_ai_analysis_service] = _error_service

    response = client.get(
        f"{PREFIX}/failure-analysis/metrics",
        headers=TENANT_HEADERS,
    )

    assert response.status_code == 200
    data = response.json()
    # Bug 3 fix: error branch now uses correct format matching success path
    assert data["service"] == "ai_test_analysis"
    assert data["service_status"] == "error"
    assert "metrics" in data
    assert "error" in data["metrics"]
    assert "timestamp" in data


# ---------------------------------------------------------------------------
# GET /patterns
# ---------------------------------------------------------------------------


def test_get_patterns_returns_patterns(client):
    """Patterns endpoint should return a list of known patterns."""

    async def _no_service():
        return None

    app.dependency_overrides[get_ai_analysis_service] = _no_service

    response = client.get(
        f"{PREFIX}/patterns",
        headers=TENANT_HEADERS,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    patterns = data["data"]["patterns"]
    assert isinstance(patterns, list)
    # No pattern occurrence statistics are recorded anywhere in this service,
    # so there is nothing to list. This used to assert the three hardcoded
    # patterns (frequencies 42 / 31 / 15) that the endpoint invented.
    assert patterns == []
    assert data["data"]["statistics_available"] is False


def test_get_patterns_respects_limit(client):
    """`limit` is accepted and never yields fabricated rows.

    This used to assert that limit=1 sliced the three hardcoded patterns down
    to one. With no occurrence statistics recorded there is nothing to slice,
    so the contract worth pinning is that no limit conjures data.
    """

    async def _no_service():
        return None

    app.dependency_overrides[get_ai_analysis_service] = _no_service

    response = client.get(
        f"{PREFIX}/patterns?limit=1",
        headers=TENANT_HEADERS,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["data"]["patterns"] == []
    assert data["data"]["total_patterns"] == 0
