"""Tests for insight_generation_services module.

Tests exercise the real generate_test_recommendations endpoint function,
including actual TestRecommendationResponse construction and high_priority_count
computation. Only the service layer is mocked (since it would require an AI backend).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from routers.ai_test_insights.insight_generation_services import (
    generate_test_recommendations,
)
from services.ai_test_insights.config import TestInsightType
from services.ai_test_insights.models import (
    TestInsightRecommendation,
)


def _make_recommendation(
    priority: str = "high", rec_id: str = "rec-1"
) -> TestInsightRecommendation:
    """Build a real TestInsightRecommendation model instead of a mock."""
    return TestInsightRecommendation(
        recommendation_id=rec_id,
        type=TestInsightType.RECOMMENDATIONS,
        priority=priority,
        confidence=0.85,
        title=f"Recommendation {rec_id}",
        description="A real recommendation for testing",
        rationale="Based on analysis data",
        implementation_steps=["Step 1", "Step 2"],
        estimated_effort="medium",
        expected_impact="Improve test quality",
        supporting_metrics={"score": 42},
        affected_components=["module_a"],
    )


@pytest.mark.asyncio
async def test_generate_recommendations_success():
    """Successful recommendation generation returns a real TestRecommendationResponse
    with correct total and high_priority_count computed from actual priority values."""
    recs = [
        _make_recommendation(priority="high", rec_id="r1"),
        _make_recommendation(priority="medium", rec_id="r2"),
        _make_recommendation(priority="high", rec_id="r3"),
    ]

    mock_service = AsyncMock()
    mock_service.generate_test_recommendations = AsyncMock(return_value=recs)

    mock_request = SimpleNamespace(
        analysis=SimpleNamespace(),
        focus_areas=["coverage"],
    )
    mock_user = SimpleNamespace(preferred_username="test-user")

    result = await generate_test_recommendations(
        request=mock_request,
        user=mock_user,
        service=mock_service,
        tenant_id="test-tenant",
    )

    # Verify the real TestRecommendationResponse model was constructed correctly
    assert result.total_recommendations == 3
    assert result.high_priority_count == 2
    assert len(result.recommendations) == 3
    # Verify actual recommendation data is preserved
    assert result.recommendations[0].recommendation_id == "r1"
    assert result.recommendations[0].priority == "high"
    assert result.recommendations[1].priority == "medium"
    # Verify generated_at is set
    assert result.generated_at is not None


@pytest.mark.asyncio
async def test_generate_recommendations_no_high_priority():
    """When no high-priority recommendations, high_priority_count is zero."""
    recs = [
        _make_recommendation(priority="low", rec_id="r1"),
        _make_recommendation(priority="medium", rec_id="r2"),
    ]

    mock_service = AsyncMock()
    mock_service.generate_test_recommendations = AsyncMock(return_value=recs)

    mock_request = SimpleNamespace(analysis=SimpleNamespace(), focus_areas=None)
    mock_user = SimpleNamespace(preferred_username="test-user")

    result = await generate_test_recommendations(
        request=mock_request,
        user=mock_user,
        service=mock_service,
        tenant_id="test-tenant",
    )

    assert result.total_recommendations == 2
    assert result.high_priority_count == 0
    assert result.recommendations[0].priority == "low"
    assert result.recommendations[1].priority == "medium"


@pytest.mark.asyncio
async def test_generate_recommendations_empty_list():
    """When service returns no recommendations, response reflects that."""
    mock_service = AsyncMock()
    mock_service.generate_test_recommendations = AsyncMock(return_value=[])

    mock_request = SimpleNamespace(analysis=SimpleNamespace(), focus_areas=None)
    mock_user = SimpleNamespace(preferred_username="test-user")

    result = await generate_test_recommendations(
        request=mock_request,
        user=mock_user,
        service=mock_service,
        tenant_id="test-tenant",
    )

    assert result.total_recommendations == 0
    assert result.high_priority_count == 0
    assert result.recommendations == []


@pytest.mark.asyncio
async def test_generate_recommendations_http_exception_passthrough():
    """HTTPExceptions from the service are re-raised without wrapping."""
    mock_service = AsyncMock()
    mock_service.generate_test_recommendations = AsyncMock(
        side_effect=HTTPException(status_code=400, detail="bad request")
    )

    mock_request = SimpleNamespace(analysis=SimpleNamespace(), focus_areas=None)
    mock_user = SimpleNamespace(preferred_username="test-user")

    with pytest.raises(HTTPException) as exc_info:
        await generate_test_recommendations(
            request=mock_request,
            user=mock_user,
            service=mock_service,
            tenant_id="test-tenant",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "bad request"


@pytest.mark.asyncio
async def test_generate_recommendations_internal_error():
    """Generic exceptions become 500 internal_error with details."""
    mock_service = AsyncMock()
    mock_service.generate_test_recommendations = AsyncMock(
        side_effect=RuntimeError("AI backend down")
    )

    mock_request = SimpleNamespace(analysis=SimpleNamespace(), focus_areas=None)
    mock_user = SimpleNamespace(preferred_username="test-user")

    with pytest.raises(HTTPException) as exc_info:
        await generate_test_recommendations(
            request=mock_request,
            user=mock_user,
            service=mock_service,
            tenant_id="test-tenant",
        )

    assert exc_info.value.status_code == 500
    # Verify the error message includes details about what failed
    assert "Recommendation generation failed" in exc_info.value.detail["message"]
    assert "AI backend down" in exc_info.value.detail["details"]["message"]


@pytest.mark.asyncio
async def test_generate_recommendations_calls_service_with_correct_args():
    """Verify the service is called with the exact analysis and focus_areas from the request."""
    mock_analysis = SimpleNamespace(overall_health_score=75.0)
    mock_service = AsyncMock()
    mock_service.generate_test_recommendations = AsyncMock(return_value=[])

    mock_request = SimpleNamespace(
        analysis=mock_analysis,
        focus_areas=["performance", "reliability"],
    )
    mock_user = SimpleNamespace(preferred_username="test-user")

    await generate_test_recommendations(
        request=mock_request,
        user=mock_user,
        service=mock_service,
        tenant_id="test-tenant",
    )

    mock_service.generate_test_recommendations.assert_awaited_once_with(
        analysis=mock_analysis,
        focus_areas=["performance", "reliability"],
    )
