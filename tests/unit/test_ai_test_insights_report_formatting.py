"""Tests for report_formatting_endpoints.py router endpoints.

These endpoints ARE mounted in the main app via __init__.py, so we can use TestClient.
However, they require authenticated user and service dependencies, plus the service
returns complex TestInsightAnalysis objects. We test via direct function calls for
cleaner mocking.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from src.routers.ai_test_insights.report_formatting_endpoints import (
    get_cached_analysis,
    get_cached_comprehensive_analysis,
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


def _mock_service(cached_analysis=None, cached_comprehensive=None):
    svc = MagicMock()
    svc.get_cached_analysis = AsyncMock(return_value=cached_analysis)
    svc.get_cached_comprehensive_analysis = AsyncMock(return_value=cached_comprehensive)
    return svc


# ===================================================================
# GET /analysis/{analysis_id} - cached analysis
# ===================================================================


class TestCachedAnalysis:
    async def test_cached_analysis_hit_returns_result(self):
        """When analysis is found in cache, it is returned directly."""
        fake_analysis = {"analysis_id": "ana_123", "patterns": [], "score": 95}
        service = _mock_service(cached_analysis=fake_analysis)
        user = _admin_user()

        result = await get_cached_analysis(
            analysis_id="ana_123",
            user=user,
            service=service,
            tenant_id="t1",
        )

        assert result == fake_analysis
        service.get_cached_analysis.assert_awaited_once_with("ana_123")

    async def test_cached_analysis_miss_returns_not_found(self):
        """When analysis is not in cache, endpoint raises 404."""
        service = _mock_service(cached_analysis=None)
        user = _admin_user()

        with pytest.raises(HTTPException) as exc_info:
            await get_cached_analysis(
                analysis_id="missing_id",
                user=user,
                service=service,
                tenant_id="t1",
            )

        assert exc_info.value.status_code == 404

    async def test_service_exception_maps_to_error_envelope(self):
        """When service raises an exception, endpoint returns 500."""
        service = _mock_service()
        service.get_cached_analysis = AsyncMock(side_effect=RuntimeError("cache exploded"))
        user = _admin_user()

        with pytest.raises(HTTPException) as exc_info:
            await get_cached_analysis(
                analysis_id="any_id",
                user=user,
                service=service,
                tenant_id="t1",
            )

        assert exc_info.value.status_code == 500


# ===================================================================
# GET /comprehensive-analysis/{analysis_id}
# ===================================================================


class TestCachedComprehensiveAnalysis:
    async def test_comprehensive_cached_hit_returns_payload(self):
        """When comprehensive analysis is found in cache, it is returned directly."""
        fake_report = {
            "analysis_id": "comp_ana_456",
            "pattern_analysis": {"patterns_found": 3},
            "performance_insights": {"bottlenecks": 1},
        }
        service = _mock_service(cached_comprehensive=fake_report)
        user = _admin_user()

        result = await get_cached_comprehensive_analysis(
            analysis_id="comp_ana_456",
            user=user,
            service=service,
            tenant_id="t1",
        )

        assert result == fake_report
        service.get_cached_comprehensive_analysis.assert_awaited_once_with("comp_ana_456")

    async def test_comprehensive_cached_miss_returns_not_found(self):
        """When comprehensive analysis is not in cache, endpoint raises 404."""
        service = _mock_service(cached_comprehensive=None)
        user = _admin_user()

        with pytest.raises(HTTPException) as exc_info:
            await get_cached_comprehensive_analysis(
                analysis_id="missing_comp",
                user=user,
                service=service,
                tenant_id="t1",
            )

        assert exc_info.value.status_code == 404
