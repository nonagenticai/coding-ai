"""Tests for AI test insights analysis endpoints."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from routers.ai_test_insights.ai_test_insights_config import (
    get_ai_test_insights_service,
)
from src.main import app

PREFIX = "/api/ai/insights/tests"

TENANT_HEADERS = {"X-Tenant-Id": "test-tenant"}

VALID_TEST_RESULT = {
    "test_id": "t1",
    "language": "python",
    "test_type": "unit",
    "success": True,
    "execution_time": 0.5,
    "stderr": "",
}

VALID_ANALYZE_BODY = {
    "container_name": "test-container",
    "test_results": [VALID_TEST_RESULT],
}

VALID_COMPREHENSIVE_BODY = {
    "container_name": "test-container",
    "test_results": [VALID_TEST_RESULT],
}


def _mock_analysis_result():
    """Build a mock TestInsightAnalysis-compatible dict for JSON serialization."""
    return {
        "analysis_id": "ana_test_123",
        "analyzed_period": "recent",
        "created_at": "2025-01-23T10:30:00Z",
        "performance_metrics": {
            "average_execution_time": 1.0,
            "median_execution_time": 0.8,
            "p95_execution_time": 2.5,
            "p99_execution_time": 3.0,
            "average_memory_usage": 50.0,
            "peak_memory_usage": 100.0,
            "average_cpu_usage": 30.0,
            "slow_tests_count": 2,
            "fast_tests_count": 8,
            "performance_trend": "stable",
            "bottleneck_categories": [],
        },
        "reliability_metrics": {
            "overall_success_rate": 0.9,
            "success_rate_by_type": {"unit": 0.95},
            "common_failure_patterns": [],
            "flaky_tests_count": 1,
            "stability_score": 85.0,
            "reliability_trend": "stable",
            "error_categories": {},
        },
        "coverage_insights": {
            "line_coverage": 80.0,
            "branch_coverage": 70.0,
            "function_coverage": 85.0,
            "uncovered_critical_paths": [],
            "redundant_tests": [],
            "coverage_gaps": [],
            "recommended_tests": [],
        },
        "quality_insights": {
            "test_quality_score": 80.0,
            "code_quality_score": 75.0,
            "critical_issues": [],
            "quality_debt": 5.0,
            "quality_recommendations": [],
            "automated_fixes": [],
        },
        "detected_patterns": [],
        "recommendations": [],
        "overall_health_score": 82.0,
        "improvement_potential": 18.0,
    }


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _cleanup_overrides():
    """Clean up dependency overrides after each test."""
    yield
    app.dependency_overrides.clear()


def _make_mock_service(
    analyze_side_effect=None,
    analyze_return=None,
    comprehensive_side_effect=None,
    comprehensive_return=None,
):
    """Create a mock AITestInsightsService with configurable behavior."""
    service = AsyncMock()

    if analyze_side_effect:
        service.analyze_test_insights = AsyncMock(side_effect=analyze_side_effect)
    elif analyze_return is not None:
        service.analyze_test_insights = AsyncMock(return_value=analyze_return)

    if comprehensive_side_effect:
        service.comprehensive_analysis = AsyncMock(side_effect=comprehensive_side_effect)
    elif comprehensive_return is not None:
        service.comprehensive_analysis = AsyncMock(return_value=comprehensive_return)

    return service


# ---------------------------------------------------------------------------
# POST /analyze
# ---------------------------------------------------------------------------


def test_analyze_empty_results_returns_400(client):
    """Posting with empty test_results should return 400 bad request."""
    mock_service = _make_mock_service()

    async def _override():
        return mock_service

    app.dependency_overrides[get_ai_test_insights_service] = _override

    body = {
        "container_name": "test-container",
        "test_results": [],
    }
    response = client.post(f"{PREFIX}/analyze", json=body, headers=TENANT_HEADERS)

    assert response.status_code == 400


def test_analyze_service_error_returns_500(client):
    """When the service raises an exception, the endpoint returns 500."""
    mock_service = _make_mock_service(analyze_side_effect=RuntimeError("AI service crashed"))

    async def _override():
        return mock_service

    app.dependency_overrides[get_ai_test_insights_service] = _override

    response = client.post(f"{PREFIX}/analyze", json=VALID_ANALYZE_BODY, headers=TENANT_HEADERS)

    assert response.status_code == 500


def test_analyze_success(client):
    """Successful analysis should return 200 with analysis data."""
    # Build a mock that returns a proper TestInsightAnalysis-like object
    mock_result = MagicMock()
    # Configure the mock to serialize properly via response_model
    result_data = _mock_analysis_result()
    for key, value in result_data.items():
        setattr(mock_result, key, value)
    # model_dump is used by FastAPI for response serialization
    mock_result.model_dump = MagicMock(return_value=result_data)

    mock_service = _make_mock_service(analyze_return=mock_result)

    async def _override():
        return mock_service

    app.dependency_overrides[get_ai_test_insights_service] = _override

    response = client.post(f"{PREFIX}/analyze", json=VALID_ANALYZE_BODY, headers=TENANT_HEADERS)

    assert response.status_code == 200
    data = response.json()
    assert data["analysis_id"] == "ana_test_123"
    assert data["analyzed_period"] == "recent"
    assert "performance_metrics" in data
    assert "reliability_metrics" in data
    assert "overall_health_score" in data


# ---------------------------------------------------------------------------
# POST /comprehensive-analysis
# ---------------------------------------------------------------------------


def test_comprehensive_empty_results_returns_400(client):
    """Posting with empty test_results should return 400 bad request."""
    mock_service = _make_mock_service()

    async def _override():
        return mock_service

    app.dependency_overrides[get_ai_test_insights_service] = _override

    body = {
        "container_name": "test-container",
        "test_results": [],
    }
    response = client.post(f"{PREFIX}/comprehensive-analysis", json=body, headers=TENANT_HEADERS)

    assert response.status_code == 400


def test_comprehensive_service_error_returns_500(client):
    """When the service raises an exception, the endpoint returns 500."""
    mock_service = _make_mock_service(
        comprehensive_side_effect=RuntimeError("comprehensive analysis exploded")
    )

    async def _override():
        return mock_service

    app.dependency_overrides[get_ai_test_insights_service] = _override

    response = client.post(
        f"{PREFIX}/comprehensive-analysis",
        json=VALID_COMPREHENSIVE_BODY,
        headers=TENANT_HEADERS,
    )

    assert response.status_code == 500


def test_comprehensive_success(client):
    """Successful comprehensive analysis should return 200 with report data."""
    comprehensive_report = {
        "analysis_id": "comp_123",
        "pattern_analysis": {"patterns_found": 3},
        "performance_insights": {"bottlenecks": 1},
        "quality_metrics": {"overall_score": 0.85},
        "recommendations": {"total": 5, "high_priority": 2},
        "timestamp": "2025-01-23T10:30:00Z",
    }

    mock_service = _make_mock_service(comprehensive_return=comprehensive_report)

    async def _override():
        return mock_service

    app.dependency_overrides[get_ai_test_insights_service] = _override

    response = client.post(
        f"{PREFIX}/comprehensive-analysis",
        json=VALID_COMPREHENSIVE_BODY,
        headers=TENANT_HEADERS,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["analysis_id"] == "comp_123"
    assert "pattern_analysis" in data
    assert "performance_insights" in data


# ---------------------------------------------------------------------------
# Validation (422)
# ---------------------------------------------------------------------------


def test_analyze_missing_body_returns_422(client):
    """Posting with no body should trigger Pydantic validation (422)."""
    mock_service = _make_mock_service()

    async def _override():
        return mock_service

    app.dependency_overrides[get_ai_test_insights_service] = _override

    response = client.post(f"{PREFIX}/analyze", json={}, headers=TENANT_HEADERS)

    assert response.status_code == 422


def test_comprehensive_missing_body_returns_422(client):
    """Posting with no body should trigger Pydantic validation (422)."""
    mock_service = _make_mock_service()

    async def _override():
        return mock_service

    app.dependency_overrides[get_ai_test_insights_service] = _override

    response = client.post(f"{PREFIX}/comprehensive-analysis", json={}, headers=TENANT_HEADERS)

    assert response.status_code == 422
