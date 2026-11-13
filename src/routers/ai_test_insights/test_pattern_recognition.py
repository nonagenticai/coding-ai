"""Test Pattern Recognition Module.

This module provides AI-powered test pattern recognition and failure prediction endpoints
with container-scoped operations.
"""

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
    bad_request,
    internal_error,
)

from .ai_test_insights_config import (
    TestFailurePredictionRequest,
    TestFailurePredictionResponse,
    get_ai_test_insights_service,
)
from .shared import validate_and_get_container_for_tenant

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/predict-failures",
    operation_id="predict_test_failures",
    response_model=TestFailurePredictionResponse,
    summary="Predict test failures",
    description="Use AI to predict the likelihood of test failures based on historical data and code changes.",
    responses={
        200: {
            "description": "Failure predictions generated successfully",
            "model": TestFailurePredictionResponse,
        },
        400: {
            "description": "Bad request - Invalid or insufficient data for prediction",
            "model": Error400Response,
            "content": {
                "application/json": {
                    "example": {
                        "status": "error",
                        "message": "Historical test results are required for prediction",
                        "timestamp": "2025-01-23T10:30:00Z",
                        "request_id": "req_123456",
                        "path": "/api/ai/insights/tests/predict-failures",
                        "error_code": "INSUFFICIENT_DATA",
                    }
                }
            },
        },
        401: {
            "description": "Unauthorized - Authentication required",
            "model": Error401Response,
        },
        403: {
            "description": "Forbidden - User lacks permission to access prediction features",
            "model": Error403Response,
        },
        422: {
            "description": "Validation error - Invalid request format",
            "model": ValidationErrorResponse,
        },
        429: {
            "description": "Too many requests - Prediction rate limit exceeded",
            "model": Error429Response,
        },
        500: {
            "description": "Internal server error - Prediction model failure",
            "model": Error500Response,
            "content": {
                "application/json": {
                    "example": {
                        "status": "error",
                        "message": "Prediction failed: AI model encountered an error",
                        "timestamp": "2025-01-23T10:30:00Z",
                        "request_id": "req_123456",
                        "path": "/api/ai/insights/tests/predict-failures",
                        "error_code": "MODEL_ERROR",
                    }
                }
            },
        },
        503: {
            "description": "Service unavailable - Prediction service is down",
            "model": Error503Response,
        },
    },
)
async def predict_test_failures(
    request: TestFailurePredictionRequest,
    user: User = Depends(get_current_user_with_tenant),
    service: AITestInsightsService = Depends(get_ai_test_insights_service),
    tenant_id: str = Depends(require_tenant),
) -> TestFailurePredictionResponse:
    """Predict test failures using AI analysis with container-scoped context. Requires authentication."""

    try:
        # Tenant-aware container validation (admins can access all, users only their own)
        effective_tenant_id = None if user.is_admin() else user.effective_tenant_id
        await validate_and_get_container_for_tenant(
            request.container_name, tenant_id=effective_tenant_id
        )

        logger.info(
            f"Starting test failure prediction for container '{request.container_name}' "
            f"with {len(request.test_results)} test results"
        )

        # Validate input
        if not request.test_results:
            raise bad_request("Historical test results are required for prediction")

        # Generate predictions scoped to container
        predictions = await service.predict_test_failures(
            test_results=request.test_results, code_changes=request.code_changes
        )

        # Calculate overall confidence
        confidence = sum(predictions.values()) / len(predictions) if predictions else 0.0
        confidence = min(1.0, max(0.0, 1.0 - confidence))  # Invert for confidence

        logger.info("Test failure prediction completed successfully")

        return TestFailurePredictionResponse(predictions=predictions, confidence=confidence)

    except HTTPException:
        # Re-raise HTTPExceptions (like parameter validation errors) without modification
        raise

    except Exception as e:
        logger.exception("Test failure prediction failed: %s", e)
        raise internal_error("Prediction failed", details={"message": str(e)})
