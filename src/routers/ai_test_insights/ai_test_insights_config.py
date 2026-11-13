"""AI Test Insights Configuration Module.

This module provides shared models, utilities, and configuration for AI test insights.
"""

import logging
from datetime import UTC, datetime
from typing import Any

from coding_client.auth import User
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from auth.tenant import get_current_user_with_tenant, require_tenant
from models.error_models import (
    Error401Response,
    Error403Response,
    Error404Response,
    Error500Response,
    Error503Response,
)

# TODO: get_cache_manager - dependency not yet extracted
# from core.dependencies import get_cache_manager
from models.testing_models import QualityResult, TestResult
from services.ai_test_insights.core_service import (
    AITestInsightsService,
    TestInsightAnalysis,
    TestInsightRecommendation,
    create_ai_test_insights_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# Shared Models
class TestInsightRequest(BaseModel):
    """Request model for AI test insights analysis."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    # Container scoping
    container_name: str = Field(..., description="Container name to scope insights analysis")

    # Test results for analysis
    test_results: list[TestResult] = Field(description="List of test results to analyze")
    quality_results: list[QualityResult] = Field(
        default_factory=list[Any], description="List of quality analysis results"
    )

    # Analysis parameters
    time_period: str = Field(
        default="recent", description="Time period for analysis (recent, week, month)"
    )
    include_predictions: bool = Field(default=True, description="Include test failure predictions")
    focus_areas: list[str] | None = Field(
        default=None, description="Specific areas to focus analysis on"
    )


class ComprehensiveAnalysisRequest(BaseModel):
    """Request model for comprehensive AI test analysis."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    # Container scoping
    container_name: str = Field(..., description="Container name to scope comprehensive analysis")

    # Test results for analysis
    test_results: list[TestResult] = Field(description="List of test results to analyze")
    quality_results: list[QualityResult] = Field(
        default_factory=list[Any], description="List of quality analysis results"
    )

    # Additional context
    code_changes: list[str] | None = Field(
        default=None, description="Recent code changes that might affect tests"
    )

    # Analysis parameters
    time_period: str = Field(
        default="recent", description="Time period for analysis (recent, week, month)"
    )
    include_predictions: bool = Field(default=True, description="Include test failure predictions")


class TestFailurePredictionRequest(BaseModel):
    """Request model for test failure prediction."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    container_name: str = Field(..., description="Container name to scope prediction analysis")
    test_results: list[TestResult] = Field(description="Historical test results for prediction")
    code_changes: list[str] | None = Field(
        default=None, description="Recent code changes that might affect tests"
    )


class TestRecommendationRequest(BaseModel):
    """Request model for generating additional test recommendations."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    container_name: str = Field(..., description="Container name to scope test recommendations")
    analysis: TestInsightAnalysis = Field(
        description="Base analysis to generate recommendations from"
    )
    focus_areas: list[str] | None = Field(
        default=None, description="Specific areas to focus recommendations on"
    )


class TestFailurePredictionResponse(BaseModel):
    """Response model for test failure predictions."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    predictions: dict[str, float] = Field(
        description="Failure probability predictions by test category"
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When predictions were generated",
    )
    confidence: float = Field(description="Overall confidence in predictions")


class TestRecommendationResponse(BaseModel):
    """Response model for test recommendations."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    recommendations: list[TestInsightRecommendation] = Field(
        description="Generated test recommendations"
    )
    total_recommendations: int = Field(description="Total number of recommendations")
    high_priority_count: int = Field(description="Number of high-priority recommendations")
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When recommendations were generated",
    )


# Shared Dependencies
async def get_ai_test_insights_service(
    # TODO: re-wire cache_manager once dependency is extracted
    # cache_manager: Any = Depends(get_cache_manager),
) -> AITestInsightsService:
    """Get AI test insights service instance."""
    return create_ai_test_insights_service(cache_manager=None)


# Service Status and Health Endpoints
@router.get(
    "/service-health",
    operation_id="ai_test_insights_health_check",
    summary="Health check for AI test insights service",
    description="Check the health status of the AI test insights service.",
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "service": "ai_test_insights",
                        "timestamp": "2025-01-23T10:30:00Z",
                    }
                }
            },
        },
        500: {
            "description": "Service health check failed",
            "model": Error500Response,
            "content": {
                "application/json": {
                    "example": {
                        "status": "error",
                        "message": "Health check failed: AI service connection error",
                        "timestamp": "2025-01-23T10:30:00Z",
                        "request_id": "req_123456",
                        "path": "/api/ai/insights/tests/health",
                        "error_code": "HEALTH_CHECK_FAILED",
                    }
                }
            },
        },
        404: {"description": "Health endpoint not found", "model": Error404Response},
        503: {
            "description": "Service unavailable - One or more dependencies are down",
            "model": Error503Response,
        },
    },
)
async def health_check(
    user: User = Depends(get_current_user_with_tenant),
    service: AITestInsightsService = Depends(get_ai_test_insights_service),
    tenant_id: str = Depends(require_tenant),
) -> dict[str, str]:
    """Health check for AI test insights service. Requires authentication."""

    try:
        logger.info(f"Health check by user {user.preferred_username}")
        # Simple health check - return expected format
        return {
            "status": "success",
            "service": "ai_test_insights",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except HTTPException:
        # Re-raise HTTPExceptions (like parameter validation errors) without modification
        raise

    except Exception as e:
        logger.exception("Health check failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get(
    "/metrics",
    operation_id="get_test_insights_metrics",
    summary="Get service metrics",
    description="Get metrics and statistics about the AI test insights service.",
    responses={
        200: {
            "description": "Service metrics retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "service": "ai_test_insights",
                        "status": "operational",
                        "features": {
                            "performance_analysis": True,
                            "reliability_analysis": True,
                            "coverage_analysis": True,
                            "quality_analysis": True,
                            "pattern_detection": True,
                            "recommendations": True,
                            "predictions": True,
                        },
                        "ai_agents": {
                            "performance_agent": True,
                            "reliability_agent": True,
                            "coverage_agent": True,
                            "quality_agent": True,
                            "pattern_agent": True,
                            "recommendation_agent": True,
                        },
                        "cache_enabled": True,
                        "timestamp": "2025-01-23T10:30:00Z",
                    }
                }
            },
        },
        401: {
            "description": "Unauthorized - Authentication required for metrics access",
            "model": Error401Response,
        },
        403: {
            "description": "Forbidden - User lacks permission to view service metrics",
            "model": Error403Response,
        },
        404: {
            "description": "Not found - Service metrics unavailable or service not configured",
            "model": Error404Response,
        },
        500: {
            "description": "Internal server error - Failed to collect metrics",
            "model": Error500Response,
        },
        503: {
            "description": "Service unavailable - Metrics collection service is down",
            "model": Error503Response,
        },
    },
)
async def get_service_metrics(
    user: User = Depends(get_current_user_with_tenant),
    service: AITestInsightsService = Depends(get_ai_test_insights_service),
    tenant_id: str = Depends(require_tenant),
) -> dict[str, Any]:
    """Get service metrics. Requires authentication."""

    try:
        logger.info(f"Getting service metrics for user {user.preferred_username}")
        # Return basic service metrics in expected format
        return {
            "status": "success",
            "data": {
                "service": "ai_test_insights",
                "status": "operational",
                "available_features": [
                    "performance_analysis",
                    "reliability_analysis",
                    "coverage_analysis",
                    "quality_analysis",
                    "pattern_detection",
                    "recommendations",
                    "predictions",
                ],
                "features": {
                    "performance_analysis": True,
                    "reliability_analysis": True,
                    "coverage_analysis": True,
                    "quality_analysis": True,
                    "pattern_detection": True,
                    "recommendations": True,
                    "predictions": True,
                },
                "ai_agents": {
                    "performance_agent": True,
                    "reliability_agent": True,
                    "coverage_agent": True,
                    "quality_agent": True,
                    "pattern_agent": True,
                    "recommendation_agent": True,
                },
                "cache_enabled": (
                    service.cache_manager is not None
                    if hasattr(service, "cache_manager")
                    else False
                ),
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except HTTPException:
        # Re-raise HTTPExceptions (like parameter validation errors) without modification
        raise

    except Exception as e:
        logger.exception("Failed to get service metrics: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")
