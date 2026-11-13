"""Insight Generation Services Module.

This module provides AI-powered test insight generation and recommendation services.
"""

from __future__ import annotations

import logging

from coding_client.auth import User
from fastapi import APIRouter, Depends, HTTPException

from auth.tenant import get_current_user_with_tenant, require_tenant
from models.error_models import (
    Error400Response,
    Error401Response,
    Error403Response,
    Error429Response,
    Error500Response,
    Error503Response,
    ValidationErrorResponse,
)

# TODO: get_cache_manager - dependency not yet extracted
# from core.dependencies import get_cache_manager
from services.ai_test_insights.core_service import (
    AITestInsightsService,
)
from utils.responses import (
    internal_error,
)

from .ai_test_insights_config import (
    TestRecommendationRequest,
    TestRecommendationResponse,
    get_ai_test_insights_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/recommendations",
    operation_id="generate_test_recommendations",
    response_model=TestRecommendationResponse,
    summary="Generate test recommendations",
    description="Generate additional AI-powered test recommendations based on existing analysis results.",
    responses={
        200: {
            "description": "Recommendations generated successfully",
            "model": TestRecommendationResponse,
        },
        400: {
            "description": "Bad request - Invalid analysis data",
            "model": Error400Response,
        },
        401: {
            "description": "Unauthorized - Authentication required",
            "model": Error401Response,
        },
        403: {
            "description": "Forbidden - Insufficient permissions",
            "model": Error403Response,
        },
        422: {
            "description": "Validation error - Invalid request format",
            "model": ValidationErrorResponse,
            "content": {
                "application/json": {
                    "example": {
                        "status": "error",
                        "message": "Request validation failed",
                        "timestamp": "2025-01-23T10:30:00Z",
                        "request_id": "req_123456",
                        "path": "/api/ai/insights/tests/recommendations",
                        "error_code": "VALIDATION_ERROR",
                        "validation_errors": [
                            {
                                "field": "analysis",
                                "message": "Analysis object is required",
                                "type": "required",
                                "input": None,
                            }
                        ],
                    }
                }
            },
        },
        429: {
            "description": "Too many requests - Rate limit exceeded",
            "model": Error429Response,
        },
        500: {
            "description": "Internal server error - Recommendation generation failed",
            "model": Error500Response,
        },
        503: {"description": "Service unavailable", "model": Error503Response},
    },
)
async def generate_test_recommendations(
    request: TestRecommendationRequest,
    user: User = Depends(get_current_user_with_tenant),
    service: AITestInsightsService = Depends(get_ai_test_insights_service),
    tenant_id: str = Depends(require_tenant),
) -> TestRecommendationResponse:
    """Generate additional test recommendations. Requires authentication."""

    try:
        logger.info(f"Starting test recommendation generation for user {user.preferred_username}")

        # Generate recommendations
        recommendations = await service.generate_test_recommendations(
            analysis=request.analysis, focus_areas=request.focus_areas
        )

        # Count high-priority recommendations
        high_priority_count = sum(1 for rec in recommendations if rec.priority == "high")

        logger.info(f"Generated {len(recommendations)} test recommendations")

        return TestRecommendationResponse(
            recommendations=recommendations,
            total_recommendations=len(recommendations),
            high_priority_count=high_priority_count,
        )

    except HTTPException:
        # Re-raise HTTPExceptions (like parameter validation errors) without modification
        raise

    except Exception as e:
        logger.exception("Test recommendation generation failed: %s", e)
        raise internal_error("Recommendation generation failed", details={"message": str(e)})
