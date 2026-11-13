"""AI Analysis Endpoints Module.

This module provides the core AI analysis endpoints for test insights,
including basic analysis and comprehensive analysis endpoints.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from auth.tenant import require_tenant
from models.error_models import (
    Error400Response,
    Error401Response,
    Error403Response,
    Error429Response,
    Error500Response,
    Error503Response,
    ValidationErrorResponse,
)
from services.ai_test_insights.core_service import (
    AITestInsightsService,
)

# TODO: get_cache_manager - dependency not yet extracted
# from core.dependencies import get_cache_manager
from services.ai_test_insights.models import TestInsightAnalysis
from utils.responses import (
    bad_request,
    internal_error,
)

from .ai_test_insights_config import (
    ComprehensiveAnalysisRequest,
    TestInsightRequest,
    get_ai_test_insights_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/analyze",
    operation_id="analyze_test_insights",
    response_model=TestInsightAnalysis,
    summary="Analyze test insights using AI",
    description="Perform comprehensive AI-powered analysis of test execution patterns, performance metrics, and quality indicators to generate actionable insights and recommendations.",
    responses={
        200: {
            "description": "Successful analysis completed",
            "content": {
                "application/json": {
                    "example": {
                        "analysis_id": "ana_123456",
                        "patterns": [{"type": "flaky_test", "confidence": 0.85}],
                        "recommendations": [{"priority": "high", "action": "Refactor test"}],
                        "timestamp": "2025-01-23T10:30:00Z",
                    }
                }
            },
        },
        400: {
            "description": "Bad request - Invalid input data",
            "model": Error400Response,
            "content": {
                "application/json": {
                    "example": {
                        "status": "error",
                        "message": "At least one test result is required for analysis",
                        "timestamp": "2025-01-23T10:30:00Z",
                        "request_id": "req_123456",
                        "path": "/api/ai/insights/tests/analyze",
                        "error_code": "INVALID_INPUT",
                    }
                }
            },
        },
        401: {
            "description": "Unauthorized - Authentication required",
            "model": Error401Response,
            "content": {
                "application/json": {
                    "example": {
                        "status": "error",
                        "message": "Authentication token is missing or invalid",
                        "timestamp": "2025-01-23T10:30:00Z",
                        "request_id": "req_123456",
                        "path": "/api/ai/insights/tests/analyze",
                        "error_code": "UNAUTHORIZED",
                    }
                }
            },
        },
        403: {
            "description": "Forbidden - Insufficient permissions",
            "model": Error403Response,
            "content": {
                "application/json": {
                    "example": {
                        "status": "error",
                        "message": "User lacks permission to analyze test insights",
                        "timestamp": "2025-01-23T10:30:00Z",
                        "request_id": "req_123456",
                        "path": "/api/ai/insights/tests/analyze",
                        "error_code": "FORBIDDEN",
                    }
                }
            },
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
                        "path": "/api/ai/insights/tests/analyze",
                        "error_code": "VALIDATION_ERROR",
                        "validation_errors": [
                            {
                                "field": "time_period",
                                "message": "Must be one of: recent, week, month",
                                "type": "value_error",
                                "input": "invalid_period",
                            }
                        ],
                    }
                }
            },
        },
        429: {
            "description": "Too many requests - Rate limit exceeded",
            "model": Error429Response,
            "content": {
                "application/json": {
                    "example": {
                        "status": "error",
                        "message": "AI analysis rate limit exceeded. Please retry after 60 seconds",
                        "timestamp": "2025-01-23T10:30:00Z",
                        "request_id": "req_123456",
                        "path": "/api/ai/insights/tests/analyze",
                        "error_code": "RATE_LIMIT_EXCEEDED",
                        "details": {"retry_after": 60, "limit": 100, "remaining": 0},
                    }
                }
            },
        },
        500: {
            "description": "Internal server error",
            "model": Error500Response,
            "content": {
                "application/json": {
                    "example": {
                        "status": "error",
                        "message": "Analysis failed: AI service temporarily unavailable",
                        "timestamp": "2025-01-23T10:30:00Z",
                        "request_id": "req_123456",
                        "path": "/api/ai/insights/tests/analyze",
                        "error_code": "INTERNAL_ERROR",
                    }
                }
            },
        },
        503: {
            "description": "Service unavailable - AI service is down",
            "model": Error503Response,
            "content": {
                "application/json": {
                    "example": {
                        "status": "error",
                        "message": "AI service is temporarily unavailable. Please try again later",
                        "timestamp": "2025-01-23T10:30:00Z",
                        "request_id": "req_123456",
                        "path": "/api/ai/insights/tests/analyze",
                        "error_code": "SERVICE_UNAVAILABLE",
                        "details": {"retry_after": 300},
                    }
                }
            },
        },
    },
)
async def analyze_test_insights(
    request: TestInsightRequest,
    service: AITestInsightsService = Depends(get_ai_test_insights_service),
    tenant_id: str = Depends(require_tenant),
) -> TestInsightAnalysis:
    """Analyze test insights using AI-powered analysis."""

    try:
        logger.info(
            f"Starting AI test insights analysis for {len(request.test_results)} test results"
        )

        # Validate input
        if not request.test_results:
            raise bad_request("At least one test result is required for analysis")

        # Perform AI analysis
        analysis = await service.analyze_test_insights(
            test_results=request.test_results,
            quality_results=request.quality_results,
            time_period=request.time_period,
            include_predictions=request.include_predictions,
        )

        logger.info("AI test insights analysis completed successfully")

        return analysis

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("AI test insights analysis failed: %s", e)
        raise internal_error(f"Analysis failed: {str(e)}")


@router.post(
    "/comprehensive-analysis",
    operation_id="comprehensive_test_analysis",
    response_model=dict[str, Any],
    summary="Perform comprehensive AI test analysis",
    description="Perform comprehensive AI-powered analysis using all available analyzers including pattern analysis, performance insights, quality metrics, recommendations, and failure predictions.",
    responses={
        200: {
            "description": "Comprehensive analysis completed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "analysis_id": "comp_ana_123456",
                        "pattern_analysis": {"patterns_found": 5, "confidence": 0.92},
                        "performance_insights": {
                            "bottlenecks": 2,
                            "optimization_potential": 0.75,
                        },
                        "quality_metrics": {
                            "overall_score": 0.85,
                            "areas_for_improvement": 3,
                        },
                        "recommendations": {"total": 8, "high_priority": 3},
                        "failure_predictions": {"risk_score": 0.25, "at_risk_tests": 4},
                        "timestamp": "2025-01-23T10:30:00Z",
                    }
                }
            },
        },
        400: {
            "description": "Bad request - Invalid input data",
            "model": Error400Response,
            "content": {
                "application/json": {
                    "example": {
                        "status": "error",
                        "message": "At least one test result is required for analysis",
                        "timestamp": "2025-01-23T10:30:00Z",
                        "request_id": "req_123456",
                        "path": "/api/ai/insights/tests/comprehensive-analysis",
                        "error_code": "INVALID_INPUT",
                    }
                }
            },
        },
        401: {
            "description": "Unauthorized - Authentication required",
            "model": Error401Response,
        },
        403: {
            "description": "Forbidden - Insufficient permissions for comprehensive analysis",
            "model": Error403Response,
        },
        422: {
            "description": "Validation error - Invalid request format",
            "model": ValidationErrorResponse,
        },
        429: {
            "description": "Too many requests - Comprehensive analysis rate limit exceeded",
            "model": Error429Response,
            "content": {
                "application/json": {
                    "example": {
                        "status": "error",
                        "message": "Comprehensive analysis rate limit exceeded. This resource-intensive operation is limited to 10 requests per hour",
                        "timestamp": "2025-01-23T10:30:00Z",
                        "request_id": "req_123456",
                        "path": "/api/ai/insights/tests/comprehensive-analysis",
                        "error_code": "RATE_LIMIT_EXCEEDED",
                        "details": {"retry_after": 3600, "limit": 10, "remaining": 0},
                    }
                }
            },
        },
        500: {
            "description": "Internal server error during comprehensive analysis",
            "model": Error500Response,
        },
        503: {
            "description": "Service unavailable - One or more AI analyzers are unavailable",
            "model": Error503Response,
        },
    },
)
async def comprehensive_test_analysis(
    request: ComprehensiveAnalysisRequest,
    service: AITestInsightsService = Depends(get_ai_test_insights_service),
    tenant_id: str = Depends(require_tenant),
) -> dict[str, Any]:
    """Perform comprehensive AI test analysis using all specialized analyzers."""

    try:
        logger.info(
            f"Starting comprehensive AI test analysis for {len(request.test_results)} test results"
        )

        # Validate input
        if not request.test_results:
            raise bad_request("At least one test result is required for analysis")

        # Perform comprehensive analysis
        comprehensive_report = await service.comprehensive_analysis(
            test_results=request.test_results,
            quality_results=request.quality_results,
            code_changes=request.code_changes,
            time_period=request.time_period,
            include_predictions=request.include_predictions,
        )

        logger.info("Comprehensive AI test analysis completed successfully")

        return comprehensive_report

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Comprehensive AI test analysis failed: %s", e)
        raise internal_error(f"Comprehensive analysis failed: {str(e)}")
