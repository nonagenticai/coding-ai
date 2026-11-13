"""Tests for ai_test_insights_config.py /health and /metrics endpoints."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from src.routers.ai_test_insights.ai_test_insights_config import (
    get_service_metrics,
    health_check,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _admin_user() -> SimpleNamespace:
    return SimpleNamespace(
        preferred_username="admin",
        is_admin=lambda: True,
        effective_tenant_id="tenant-1",
    )


def _mock_service(cache_manager=None):
    svc = MagicMock()
    svc.cache_manager = cache_manager
    return svc


# ===================================================================
# /health
# ===================================================================


class TestHealthEndpoint:
    async def test_health_success_payload_keys_and_values(self):
        """Successful health check returns status, service, and timestamp keys."""
        user = _admin_user()
        service = _mock_service()

        result = await health_check(user=user, service=service, tenant_id="t1")

        assert result["status"] == "success"
        assert result["service"] == "ai_test_insights"
        assert "timestamp" in result

    async def test_health_reraises_http_exception(self):
        """HTTPExceptions raised during health check are re-raised as-is."""
        _admin_user()
        service = _mock_service()

        # Monkey-patch user to trigger an HTTPException inside the try block
        # by making preferred_username property raise
        class BrokenUser:
            @property
            def preferred_username(self):
                raise HTTPException(status_code=403, detail="Forbidden")

            def is_admin(self):
                return True

            effective_tenant_id = "t1"

        with pytest.raises(HTTPException) as exc_info:
            await health_check(user=BrokenUser(), service=service, tenant_id="t1")

        assert exc_info.value.status_code == 403

    async def test_health_unexpected_exception_returns_500(self):
        """Non-HTTP exceptions are caught and re-raised as 500."""
        _admin_user()
        service = _mock_service()

        # Make the user's preferred_username access raise a generic exception
        class ExplodingUser:
            @property
            def preferred_username(self):
                raise RuntimeError("unexpected boom")

            def is_admin(self):
                return True

            effective_tenant_id = "t1"

        with pytest.raises(HTTPException) as exc_info:
            await health_check(user=ExplodingUser(), service=service, tenant_id="t1")

        assert exc_info.value.status_code == 500
        assert "Health check failed" in str(exc_info.value.detail)


# ===================================================================
# /metrics
# ===================================================================


class TestMetricsEndpoint:
    async def test_metrics_success_payload_keys(self):
        """Successful metrics returns status, data, timestamp at top level."""
        user = _admin_user()
        service = _mock_service(cache_manager=None)

        result = await get_service_metrics(user=user, service=service, tenant_id="t1")

        assert result["status"] == "success"
        assert "data" in result
        assert "timestamp" in result

        data = result["data"]
        assert "available_features" in data
        assert "features" in data
        assert "ai_agents" in data
        assert "cache_enabled" in data

    async def test_metrics_includes_expected_fields(self):
        """Metrics data section contains all expected sub-fields."""
        user = _admin_user()
        service = _mock_service(cache_manager=MagicMock())

        result = await get_service_metrics(user=user, service=service, tenant_id="t1")

        data = result["data"]
        assert data["status"] == "operational"
        assert data["service"] == "ai_test_insights"

        # Check features dict has expected boolean fields
        features = data["features"]
        expected_features = [
            "performance_analysis",
            "reliability_analysis",
            "coverage_analysis",
            "quality_analysis",
            "pattern_detection",
            "recommendations",
            "predictions",
        ]
        for f in expected_features:
            assert f in features
            assert features[f] is True

        # Check ai_agents dict
        agents = data["ai_agents"]
        expected_agents = [
            "performance_agent",
            "reliability_agent",
            "coverage_agent",
            "quality_agent",
            "pattern_agent",
            "recommendation_agent",
        ]
        for a in expected_agents:
            assert a in agents

        # cache_enabled should be True since we provided a cache_manager
        assert data["cache_enabled"] is True
