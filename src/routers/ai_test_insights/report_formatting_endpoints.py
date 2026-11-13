"""Report Formatting Endpoints Module.

This module provides endpoints for retrieving and formatting AI test analysis reports.
"""

from __future__ import annotations

import logging
from typing import Any

from coding_client.auth import User
from fastapi import APIRouter, Depends, HTTPException

from auth.tenant import get_current_user_with_tenant, require_tenant
from models.error_models import (
    Error400Response,
    Error401Response,
    Error403Response,
    Error404Response,
    Error500Response,
    Error503Response,
)

# TODO: get_cache_manager - dependency not yet extracted
# from core.dependencies import get_cache_manager
from services.ai_test_insights.core_service import (
    AITestInsightsService,
    TestInsightAnalysis,
)
from utils.responses import (
    internal_error,
    not_found,
)

from .ai_test_insights_config import get_ai_test_insights_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/analysis/{analysis_id}",
    operation_id="get_cached_test_analysis",
    response_model=TestInsightAnalysis,
    summary="Get cached analysis results",
    description="Retrieve previously cached AI test insights analysis results.",
    responses={
        200: {
            "description": "Analysis retrieved successfully",
            "model": TestInsightAnalysis,
        },
        400: {
            "description": "Bad request - Invalid analysis ID format",
            "model": Error400Response,
        },
        401: {
            "description": "Unauthorized - Authentication required",
            "model": Error401Response,
        },
        403: {
            "description": "Forbidden - User lacks permission to access this analysis",
            "model": Error403Response,
        },
        404: {
            "description": "Not found - Analysis not found in cache",
            "model": Error404Response,
            "content": {
                "application/json": {
                    "example": {
                        "status": "error",
                        "message": "Analysis ana_123456 not found in cache",
                        "timestamp": "2025-01-23T10:30:00Z",
                        "request_id": "req_789012",
                        "path": "/api/ai/insights/tests/analysis/ana_123456",
                        "error_code": "ANALYSIS_NOT_FOUND",
                        "details": {"analysis_id": "ana_123456", "cache_ttl": 3600},
                    }
                }
            },
        },
        500: {
            "description": "Internal server error - Failed to retrieve analysis",
            "model": Error500Response,
        },
        503: {
            "description": "Service unavailable - Cache service is down",
            "model": Error503Response,
        },
    },
)
async def get_cached_analysis(
    analysis_id: str,
    user: User = Depends(get_current_user_with_tenant),
    service: AITestInsightsService = Depends(get_ai_test_insights_service),
    tenant_id: str = Depends(require_tenant),
) -> TestInsightAnalysis:
    """Get cached analysis results. Requires authentication."""

    try:
        logger.info(f"Retrieving cached analysis {analysis_id} for user {user.preferred_username}")

        # Get cached analysis
        analysis = await service.get_cached_analysis(analysis_id)

        if not analysis:
            raise not_found("Analysis", analysis_id)

        logger.info(f"Retrieved cached analysis {analysis_id}")

        return analysis

    except HTTPException:
        # Re-raise HTTPExceptions (like parameter validation errors) without modification
        raise

    except Exception as e:
        logger.exception("Failed to retrieve cached analysis: %s", e)
        raise internal_error("Failed to retrieve analysis", details={"message": str(e)})


@router.get(
    "/comprehensive-analysis/{analysis_id}",
    operation_id="get_cached_comprehensive_analysis",
    response_model=dict[str, Any],
    summary="Get cached comprehensive analysis results",
    description="Retrieve previously cached comprehensive AI test analysis results.",
    responses={
        200: {
            "description": "Comprehensive analysis retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "analysis_id": "comp_ana_123456",
                        "pattern_analysis": {},
                        "performance_insights": {},
                        "quality_metrics": {},
                        "recommendations": [],
                        "failure_predictions": {},
                        "timestamp": "2025-01-23T10:30:00Z",
                    }
                }
            },
        },
        400: {
            "description": "Bad request - Invalid analysis ID format",
            "model": Error400Response,
        },
        401: {
            "description": "Unauthorized - Authentication required",
            "model": Error401Response,
        },
        403: {
            "description": "Forbidden - User lacks permission to access this comprehensive analysis",
            "model": Error403Response,
        },
        404: {
            "description": "Not found - Comprehensive analysis not found in cache",
            "model": Error404Response,
            "content": {
                "application/json": {
                    "example": {
                        "status": "error",
                        "message": "Comprehensive analysis comp_ana_123456 not found in cache",
                        "timestamp": "2025-01-23T10:30:00Z",
                        "request_id": "req_789012",
                        "path": "/api/ai/insights/tests/comprehensive-analysis/comp_ana_123456",
                        "error_code": "COMPREHENSIVE_ANALYSIS_NOT_FOUND",
                    }
                }
            },
        },
        500: {"description": "Internal server error", "model": Error500Response},
        503: {
            "description": "Service unavailable - Cache service is down",
            "model": Error503Response,
        },
    },
)
async def get_cached_comprehensive_analysis(
    analysis_id: str,
    user: User = Depends(get_current_user_with_tenant),
    service: AITestInsightsService = Depends(get_ai_test_insights_service),
    tenant_id: str = Depends(require_tenant),
) -> dict[str, Any]:
    """Get cached comprehensive analysis results. Requires authentication."""

    try:
        logger.info(
            f"Retrieving cached comprehensive analysis {analysis_id} for user {user.preferred_username}"
        )

        # Get cached comprehensive analysis
        analysis = await service.get_cached_comprehensive_analysis(analysis_id)

        if not analysis:
            raise not_found("Comprehensive analysis", analysis_id)

        logger.info(f"Retrieved cached comprehensive analysis {analysis_id}")

        return analysis

    except HTTPException:
        # Re-raise HTTPExceptions (like parameter validation errors) without modification
        raise

    except Exception as e:
        logger.exception("Failed to retrieve cached comprehensive analysis: %s", e)
        raise internal_error(
            "Failed to retrieve comprehensive analysis", details={"message": str(e)}
        )
