"""Regression tests for fabricated status/metrics on the AI test-analysis router.

Each endpoint below reported a value it had not measured:

* ``/failure-analysis/health`` read ``status`` / ``ai_analysis_available`` /
  ``ai_model_name`` / ``checks`` from ``get_health_status()``, which publishes
  none of those keys. Every read missed, so the endpoint answered
  "healthy / available / gpt-4" unconditionally -- including with the circuit
  breaker open.
* ``/failure-analysis/metrics`` read ``success_rate`` / ``average_response_time``
  / ``cached_analyses`` / ``ai_model_usage`` from ``_metrics``, which contains
  none of them, and hardcoded ``service_status: "healthy"``.
* ``/patterns`` returned three invented patterns with invented occurrence
  counts (42 / 31 / 15).
* ``/batch-analyze`` set ``success: True`` as a literal, so a degraded result
  (LLM unreachable) was reported as a success with no ``degraded`` flag.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from routers.ai_test_analysis import get_ai_analysis_service
from src.main import app

PREFIX = "/api/ai/analysis/tests"
TENANT_HEADERS = {"X-Tenant-Id": "test-tenant"}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _service_with_health(**overrides):
    """A mock publishing exactly the keys AIFailureAnalysisService really does."""
    health = {
        "service_name": "AI Failure Analysis Service",
        "model_name": "claude-opus-4",
        "active_analyses": 0,
        "max_concurrent_analyses": 5,
        "circuit_breaker_open": False,
        "circuit_breaker_failures": 0,
        "cache_size": 0,
        "cache_enabled": True,
        "metrics_enabled": True,
        "metrics": {},
    }
    health.update(overrides)
    svc = MagicMock()
    svc.get_health_status = lambda: health
    return svc


# --- health ---------------------------------------------------------------


def test_health_reports_degraded_when_circuit_breaker_is_open(client):
    """The whole point: an open breaker must not report healthy."""
    svc = _service_with_health(circuit_breaker_open=True, circuit_breaker_failures=7)

    async def _service():
        return svc

    app.dependency_overrides[get_ai_analysis_service] = _service

    data = client.get(f"{PREFIX}/failure-analysis/health", headers=TENANT_HEADERS).json()

    assert data["service_status"] == "degraded"
    assert data["ai_analysis_available"] is False
    assert data["checks"]["ai_service"] == "circuit_breaker_open"
    assert data["checks"]["circuit_breaker_failures"] == 7


def test_health_reports_the_real_model_name(client):
    """`ai_model_name` used to be the hardcoded string "gpt-4"."""
    svc = _service_with_health(model_name="claude-opus-4")

    async def _service():
        return svc

    app.dependency_overrides[get_ai_analysis_service] = _service

    data = client.get(f"{PREFIX}/failure-analysis/health", headers=TENANT_HEADERS).json()

    assert data["ai_model_name"] == "claude-opus-4"
    assert data["service_status"] == "healthy"
    assert data["ai_analysis_available"] is True


# --- metrics --------------------------------------------------------------


def test_success_rate_is_derived_from_the_real_counters(client):
    svc = MagicMock()
    svc.get_metrics = lambda: {
        "total_analyses": 8,
        "successful_analyses": 6,
        "failed_analyses": 2,
        "cache_hits": 3,
        "cache_misses": 5,
        "average_analysis_time": 2.0,
    }

    async def _service():
        return svc

    app.dependency_overrides[get_ai_analysis_service] = _service

    metrics = client.get(f"{PREFIX}/failure-analysis/metrics", headers=TENANT_HEADERS).json()[
        "metrics"
    ]

    assert metrics["success_rate"] == pytest.approx(0.75)
    assert metrics["failed_analyses"] == 2
    assert metrics["cache_hits"] == 3


def test_success_rate_is_undefined_before_anything_ran(client):
    """ "No analyses yet" is not "0% of them succeeded"."""
    svc = MagicMock()
    svc.get_metrics = lambda: {
        "total_analyses": 0,
        "successful_analyses": 0,
        "failed_analyses": 0,
    }

    async def _service():
        return svc

    app.dependency_overrides[get_ai_analysis_service] = _service

    metrics = client.get(f"{PREFIX}/failure-analysis/metrics", headers=TENANT_HEADERS).json()[
        "metrics"
    ]

    assert metrics["success_rate"] is None


# --- patterns -------------------------------------------------------------


def test_patterns_endpoint_invents_no_occurrence_counts(client):
    """It used to answer with three canned patterns at 42 / 31 / 15 occurrences."""

    async def _service():
        return MagicMock()

    app.dependency_overrides[get_ai_analysis_service] = _service

    body = client.get(f"{PREFIX}/patterns", headers=TENANT_HEADERS).json()
    data = body["data"]

    assert data["patterns"] == []
    assert data["total_patterns"] == 0
    assert data["statistics_available"] is False
    assert data["detail"]

    # Belt and braces: none of the invented rows may reappear. Checked against
    # `data` only -- the bare digits also occur in the response timestamp.
    serialized = str(data)
    for invented in ("assertion_error", "null_pointer", "Test Timeout", "frequency"):
        assert invented not in serialized, f"fabricated pattern data {invented!r} is back"
