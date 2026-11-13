"""Tests for test_pattern_recognition.py router endpoint (predict-failures)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from src.routers.ai_test_insights.test_pattern_recognition import (
    predict_test_failures,
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


def _mock_service(predictions=None):
    svc = MagicMock()
    svc.predict_test_failures = AsyncMock(
        return_value=predictions if predictions is not None else {}
    )
    return svc


def _prediction_request(test_results=None, code_changes=None, container_name="test-ctr"):
    return SimpleNamespace(
        container_name=container_name,
        test_results=test_results if test_results is not None else [],
        code_changes=code_changes,
    )


# ===================================================================
# POST /predict-failures
# ===================================================================


class TestPredictFailures:
    async def test_prediction_success_with_failure_set(self):
        """Successful prediction returns predictions and confidence."""
        predictions = {"python_unit": 0.3, "javascript_integration": 0.7}
        service = _mock_service(predictions=predictions)
        user = _admin_user()

        # Need non-empty test_results to pass validation
        test_results = [
            SimpleNamespace(
                test_id="t1",
                language="python",
                test_type="unit",
                success=True,
                execution_time=1.0,
                stderr="",
            ),
        ]

        request = _prediction_request(test_results=test_results)

        result = await predict_test_failures(
            request=request,
            user=user,
            service=service,
            tenant_id="t1",
        )

        assert result.predictions == predictions
        assert 0.0 <= result.confidence <= 1.0
        service.predict_test_failures.assert_awaited_once()

    async def test_empty_dataset_validation(self):
        """Empty test_results raises 400 bad request."""
        service = _mock_service()
        user = _admin_user()

        request = _prediction_request(test_results=[])

        with pytest.raises(HTTPException) as exc_info:
            await predict_test_failures(
                request=request,
                user=user,
                service=service,
                tenant_id="t1",
            )

        assert exc_info.value.status_code == 400

    async def test_service_exception_mapping(self):
        """When service.predict_test_failures raises, endpoint returns 500."""
        service = _mock_service()
        service.predict_test_failures = AsyncMock(side_effect=RuntimeError("model crashed"))
        user = _admin_user()

        test_results = [
            SimpleNamespace(
                test_id="t1",
                language="python",
                test_type="unit",
                success=True,
                execution_time=1.0,
                stderr="",
            ),
        ]
        request = _prediction_request(test_results=test_results)

        with pytest.raises(HTTPException) as exc_info:
            await predict_test_failures(
                request=request,
                user=user,
                service=service,
                tenant_id="t1",
            )

        assert exc_info.value.status_code == 500

    async def test_prediction_with_empty_predictions_result(self):
        """When service returns empty predictions, confidence is 1.0 (inverted: 1 - 0)."""
        service = _mock_service(predictions={})
        user = _admin_user()

        test_results = [
            SimpleNamespace(
                test_id="t1",
                language="python",
                test_type="unit",
                success=True,
                execution_time=1.0,
                stderr="",
            ),
        ]
        request = _prediction_request(test_results=test_results)

        result = await predict_test_failures(
            request=request,
            user=user,
            service=service,
            tenant_id="t1",
        )

        assert result.predictions == {}
        # Source inverts: confidence = 1.0 - avg_prediction; empty → 1.0 - 0.0 = 1.0
        assert result.confidence == 1.0
