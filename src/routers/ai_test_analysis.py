"""AI-powered test failure analysis endpoints.

API endpoints for accessing the AI-powered test failure analysis functionality,
including intelligent root cause analysis, fix suggestions, and pattern detection.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from auth.tenant import require_tenant
from models.ai_test_failure_models import (
    FailureAnalysisRequest,
    TestFailureContext,
    TestFailureData,
    TestFileMetadata,
    TestFramework,
)
from models.error_models import (
    Error400Response,
    Error401Response,
    Error404Response,
    Error500Response,
    Error503Response,
    ValidationErrorResponse,
)
from models.testing_models import TestResult
from routers.ai_test_insights.shared import validate_and_get_container_for_tenant
from utils.responses import (
    internal_error,
    service_unavailable,
    validation_error,
)


def _enum_value(v: Any) -> Any:
    """Extract the value from an enum or return the value itself if it's already a plain type.

    Models with ``use_enum_values=True`` store plain strings rather than enum instances,
    so ``.value`` would fail.  This helper handles both cases.
    """
    return v.value if hasattr(v, "value") else v


logger = logging.getLogger(__name__)

try:
    from services.ai_failure_analysis import AIFailureAnalysisService
    from services.ai_test_integration import AITestIntegrationService

    _AI_SERVICES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import AI services: {e}")
    _AI_SERVICES_AVAILABLE = False

    # Create placeholder classes for when dependencies are not available
    class AIFailureAnalysisService:
        is_available = False

        async def analyze_test_failure(self, *args: Any, **kwargs: Any) -> Any: ...

    class AITestIntegrationService:
        def __init__(self, **kwargs: Any) -> None: ...
        async def batch_analyze_failures(self, *args: Any, **kwargs: Any) -> Any: ...


router = APIRouter()


# Request/Response models for the API
class AITestAnalysisRequest(BaseModel):
    """Request model for AI test analysis."""

    container_name: str = Field(
        ..., description="Container name for test analysis context (required)"
    )

    test_output: str = Field(
        ..., description="Raw test output including failure details", max_length=100_000
    )

    test_name: str = Field(..., description="Name of the failing test", max_length=200)

    test_framework: TestFramework | None = Field(
        default=TestFramework.PYTEST, description="Test framework used"
    )

    language: str | None = Field(default="python", description="Programming language")

    source_code: str | None = Field(
        default=None, description="Source code being tested", max_length=50_000
    )

    test_code: str | None = Field(
        default=None, description="Test code that failed", max_length=50_000
    )

    file_path: str | None = Field(
        default=None,
        description="Path to the test file (relative to container)",
        max_length=1000,
    )

    include_suggestions: bool = Field(
        default=True, description="Whether to include fix suggestions"
    )

    include_patterns: bool = Field(default=True, description="Whether to include pattern detection")

    max_root_causes: int = Field(
        default=5, description="Maximum number of root causes to identify", ge=1, le=20
    )

    max_suggestions: int = Field(
        default=10, description="Maximum number of fix suggestions", ge=1, le=50
    )

    minimum_confidence: float = Field(
        default=0.1, description="Minimum confidence threshold", ge=0.0, le=1.0
    )


class AITestAnalysisResponse(BaseModel):
    """Response model for AI test analysis."""

    analysis_id: str = Field(..., description="Unique analysis identifier")
    success: bool = Field(..., description="Whether analysis was successful")
    degraded: bool = Field(
        default=False,
        description="True when returned by a fallback path (AI service unavailable or "
        "errored). success may still be True, but results are basic/non-AI.",
    )
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: datetime = Field(..., description="Response timestamp")

    # Analysis results (only present if successful)
    root_causes: list[dict[str, Any]] | None = Field(
        default=None, description="Identified root causes"
    )

    fix_suggestions: list[dict[str, Any]] | None = Field(
        default=None, description="Fix suggestions"
    )

    detected_patterns: list[dict[str, Any]] | None = Field(
        default=None, description="Detected failure patterns"
    )

    error_details: dict[str, Any] | None = Field(
        default=None, description="Error details if analysis failed"
    )

    analysis_confidence: str | None = Field(
        default=None, description="Overall confidence in analysis"
    )

    analysis_confidence_score: float | None = Field(
        default=None, description="Numerical confidence score"
    )

    # Error information (only present if failed)
    error_message: str | None = Field(default=None, description="Error message if analysis failed")


class BatchAnalysisRequest(BaseModel):
    """Request model for batch test analysis."""

    container_name: str = Field(
        ..., description="Container name for batch test analysis context (required)"
    )

    test_results: list[dict[str, Any]] = Field(
        ..., description="List of test results to analyze", max_length=50
    )

    include_suggestions: bool = Field(
        default=True, description="Whether to include fix suggestions"
    )

    include_patterns: bool = Field(default=True, description="Whether to include pattern detection")

    max_concurrent: int = Field(default=5, description="Maximum concurrent analyses", ge=1, le=10)


class BatchAnalysisResponse(BaseModel):
    """Response model for batch test analysis."""

    batch_id: str = Field(..., description="Unique batch identifier")
    total_tests: int = Field(..., description="Total number of tests")
    analyzed_tests: int = Field(..., description="Number of tests analyzed")
    processing_time: float = Field(..., description="Total processing time")
    timestamp: datetime = Field(..., description="Response timestamp")

    results: list[dict[str, Any] | None] = Field(
        ..., description="Analysis results (None for tests that couldn't be analyzed)"
    )


class HealthCheckResponse(BaseModel):
    """Response model for health check."""

    model_config = ConfigDict(protected_namespaces=())

    service: str = Field(..., description="Service name")
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Check timestamp")

    ai_analysis_available: bool = Field(..., description="Whether AI analysis is available")

    ai_model_name: str | None = Field(default=None, description="AI model being used")

    checks: dict[str, Any] = Field(
        default_factory=dict[str, Any], description="Individual component checks"
    )

    metrics: dict[str, Any] | None = Field(default=None, description="Service metrics")


# Dependency injection — module-level singleton so cache/metrics survive across requests
_ai_service: AIFailureAnalysisService | None = None


async def get_ai_analysis_service() -> AIFailureAnalysisService | None:
    """Get AI analysis service singleton with graceful degradation."""
    global _ai_service  # noqa: PLW0603
    if _ai_service is not None:
        return _ai_service
    try:
        service = AIFailureAnalysisService()
        if hasattr(service, "is_available") and not service.is_available:
            logger.warning(
                "AI analysis service is not available - API key missing or initialization failed"
            )
            return None
        _ai_service = service
        return _ai_service
    except Exception as e:
        logger.warning(
            f"Failed to create AI analysis service: {e}. Service will operate in degraded mode."
        )
        return None


async def get_integration_service() -> AITestIntegrationService | None:
    """Get integration service instance."""
    try:
        if not _AI_SERVICES_AVAILABLE:
            return None
        # TODO: docker_service and TestExecutionService belong to coding-testing
        # In the microservice, we create the integration service without a real test execution service
        return AITestIntegrationService()
    except Exception as e:
        logger.warning(f"Integration service unavailable: {e}")
        return None


@router.post(
    "/failure-analysis",
    operation_id="analyze_test_failure",
    response_model=AITestAnalysisResponse,
    summary="Analyze test failure with AI",
    description="""
    Perform container-scoped AI-powered analysis of a test failure to identify root causes and suggest fixes.

    All test failure analysis must be scoped to a container. The container_name is required to provide
    proper context for the AI analysis including codebase structure, dependencies, and related tests.
    """,
    responses={
        400: {
            "model": Error400Response,
            "description": "Bad request - Invalid input format",
        },
        401: {"model": Error401Response, "description": "Authentication required"},
        404: {"model": Error404Response, "description": "Container not found"},
        422: {
            "model": ValidationErrorResponse,
            "description": "Validation error - Invalid request parameters",
        },
        500: {
            "model": Error500Response,
            "description": "Internal server error - Analysis failed",
        },
        503: {
            "model": Error503Response,
            "description": "Service unavailable - AI service not available",
        },
    },
)
async def analyze_test_failure(
    request: AITestAnalysisRequest,
    ai_service: AIFailureAnalysisService | None = Depends(get_ai_analysis_service),
    tenant_id: str = Depends(require_tenant),
) -> AITestAnalysisResponse:
    """
    Analyze a test failure using AI-powered analysis with container context.

    This endpoint performs comprehensive analysis of test failures including:
    - Root cause identification
    - Fix suggestion generation
    - Pattern detection and classification
    - Confidence scoring

    Args:
        request: Test failure analysis request (must include container_name)
        ai_service: AI analysis service dependency

    Returns:
        Analysis results with root causes, suggestions, and patterns

    Raises:
        HTTPException: If analysis fails, container not found, or input validation fails
    """
    start_time = datetime.now(UTC)

    try:
        # Validate container exists and get container info
        container_info = await validate_and_get_container_for_tenant(
            request.container_name, tenant_id
        )

        logger.info(
            f"Starting AI test analysis for test: {request.test_name} in container: {request.container_name}"
        )

        # Check if AI service is available
        if ai_service is None:
            logger.warning("AI service not available, returning mock response")
            processing_time = (datetime.now(UTC) - start_time).total_seconds()

            # Return a degraded response with basic analysis
            return AITestAnalysisResponse(
                analysis_id=str(uuid.uuid4()),
                success=True,
                degraded=True,
                processing_time=processing_time,
                timestamp=datetime.now(UTC),
                root_causes=[
                    {
                        "id": str(uuid.uuid4()),
                        "category": "unknown",
                        "description": "AI analysis service is currently unavailable. Basic analysis only.",
                        "severity": "medium",
                        "confidence": "low",
                        "confidence_score": 0.3,
                        "evidence": ["Test output indicates failure"],
                        "related_patterns": [],
                    }
                ],
                fix_suggestions=(
                    [
                        {
                            "id": str(uuid.uuid4()),
                            "title": "Manual Review Required",
                            "description": "Please review the test output manually as AI analysis is unavailable.",
                            "fix_type": "manual",
                            "code_example": "",
                            "implementation_details": [
                                "Check test output for error messages",
                                "Review test code for issues",
                            ],
                            "effort_estimate": "variable",
                            "confidence": "low",
                            "confidence_score": 0.3,
                            "side_effects": [],
                            "verification_steps": ["Run test again", "Check logs"],
                            "documentation_links": [],
                        }
                    ]
                    if request.include_suggestions
                    else []
                ),
                detected_patterns=(
                    [
                        {
                            "id": str(uuid.uuid4()),
                            "pattern_type": "test_failure",
                            "signature": "generic_failure",
                            "confidence": "low",
                            "confidence_score": 0.3,
                            "occurrences": 1,
                            "description": "Generic test failure pattern",
                            "typical_causes": [
                                "Code error",
                                "Test configuration issue",
                            ],
                            "common_fixes": ["Review test code", "Check dependencies"],
                        }
                    ]
                    if request.include_patterns
                    else []
                ),
                analysis_confidence="low",
                analysis_confidence_score=0.3,
            )

        # Create test failure context with container information
        test_file_metadata = None
        if request.file_path:
            test_file_metadata = TestFileMetadata(
                file_path=request.file_path,
                framework=request.test_framework or TestFramework.PYTEST,
            )

        context = TestFailureContext(
            test_name=request.test_name,
            test_file=test_file_metadata,
            framework=request.test_framework or TestFramework.UNKNOWN,
            execution_environment={
                "language": request.language,
                "source_provided": request.source_code is not None,
                "test_code_provided": request.test_code is not None,
                "container_name": request.container_name,
                "codebase_path": container_info.codebase_path,
            },
        )

        # Create test failure data
        failure_data = TestFailureData(
            failure_id=str(uuid.uuid4()),
            timestamp=datetime.now(UTC),
            raw_output=request.test_output,
            context=context,
        )

        # Create analysis request
        analysis_request = FailureAnalysisRequest(
            request_id=str(uuid.uuid4()),
            failure_data=failure_data,
            include_suggestions=request.include_suggestions,
            include_patterns=request.include_patterns,
            max_root_causes=request.max_root_causes,
            max_suggestions=request.max_suggestions,
            minimum_confidence=request.minimum_confidence,
        )

        # Perform analysis
        analysis_response = await ai_service.analyze_test_failure(analysis_request)

        processing_time = (datetime.now(UTC) - start_time).total_seconds()

        if analysis_response.success and analysis_response.analysis_result:
            # Convert analysis result to API response format
            result = analysis_response.analysis_result

            return AITestAnalysisResponse(
                analysis_id=result.analysis_id,
                success=True,
                # Surface a total/partial LLM outage: the analysis completed on
                # a degraded path (agents unreachable), so results are basic/
                # empty rather than AI-derived. A genuine empty result (LLM ran,
                # found nothing) keeps degraded=False.
                degraded=bool(getattr(analysis_response, "degraded", False)),
                processing_time=processing_time,
                timestamp=datetime.now(UTC),
                root_causes=[
                    {
                        "id": cause.cause_id,
                        "category": cause.category,
                        "description": cause.description,
                        "severity": _enum_value(cause.severity),
                        "confidence": _enum_value(cause.confidence),
                        "confidence_score": cause.confidence_score,
                        "evidence": cause.evidence,
                        "related_patterns": cause.related_patterns,
                    }
                    for cause in result.root_causes
                ],
                fix_suggestions=[
                    {
                        "id": suggestion.suggestion_id,
                        "title": suggestion.title,
                        "description": suggestion.description,
                        "fix_type": suggestion.fix_type,
                        "priority": suggestion.priority,
                        "confidence": _enum_value(suggestion.confidence),
                        "confidence_score": suggestion.confidence_score,
                        "estimated_effort": suggestion.estimated_effort,
                        "code_examples": suggestion.code_examples,
                        "files_to_modify": suggestion.files_to_modify,
                        "verification_steps": suggestion.verification_steps,
                    }
                    for suggestion in result.fix_suggestions
                ],
                detected_patterns=[
                    {
                        "id": pattern.pattern_id,
                        "name": pattern.pattern_name,
                        "type": pattern.pattern_type,
                        "signature": pattern.pattern_signature,
                        "confidence": _enum_value(pattern.confidence),
                        "confidence_score": pattern.confidence_score,
                        "frequency": pattern.frequency,
                        "typical_causes": pattern.typical_causes,
                        "common_fixes": pattern.common_fixes,
                    }
                    for pattern in result.detected_patterns
                ],
                error_details={
                    "error_type": _enum_value(result.error_details.error_type),
                    "error_message": result.error_details.error_message,
                    "file_path": result.error_details.file_path,
                    "line_number": result.error_details.line_number,
                    "column_number": result.error_details.column_number,
                },
                analysis_confidence=_enum_value(result.analysis_confidence),
                analysis_confidence_score=result.analysis_confidence_score,
            )
        else:
            # Analysis failed
            return AITestAnalysisResponse(
                analysis_id=analysis_response.request_id,
                success=False,
                processing_time=processing_time,
                timestamp=datetime.now(UTC),
                error_message=analysis_response.error_message,
            )

    except ValidationError as e:
        logger.error(f"Validation error in test analysis: {str(e)}")
        raise validation_error(
            message=f"Validation error: {str(e)}",
            field="request",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Test analysis failed, returning degraded response: {str(e)}")
        processing_time = (datetime.now(UTC) - start_time).total_seconds()
        return AITestAnalysisResponse(
            analysis_id=str(uuid.uuid4()),
            success=True,
            degraded=True,
            processing_time=processing_time,
            timestamp=datetime.now(UTC),
            root_causes=[
                {
                    "id": str(uuid.uuid4()),
                    "category": "unknown",
                    "description": f"AI analysis encountered an error: {str(e)}. Basic analysis only.",
                    "severity": "medium",
                    "confidence": "low",
                    "confidence_score": 0.1,
                    "evidence": ["Test output indicates failure"],
                    "related_patterns": [],
                }
            ],
            fix_suggestions=[],
            detected_patterns=[],
            analysis_confidence="low",
            analysis_confidence_score=0.1,
        )


@router.post(
    "/batch-analyze",
    operation_id="batch_analyze_test_failures",
    response_model=BatchAnalysisResponse,
    summary="Batch analyze multiple test failures",
    description="""
    Analyze multiple test failures concurrently using AI with container context.

    All batch test analysis must be scoped to a container to provide proper context
    for understanding test failures across the codebase.
    """,
    responses={
        400: {
            "model": Error400Response,
            "description": "Bad request - Invalid input format or too many tests",
        },
        401: {"model": Error401Response, "description": "Authentication required"},
        404: {"model": Error404Response, "description": "Container not found"},
        422: {
            "model": ValidationErrorResponse,
            "description": "Validation error - Invalid test data in batch",
        },
        500: {
            "model": Error500Response,
            "description": "Internal server error - Batch analysis failed",
        },
        503: {
            "model": Error503Response,
            "description": "Service unavailable - AI integration service not available",
        },
    },
)
async def batch_analyze_failures(
    request: BatchAnalysisRequest,
    integration_service: AITestIntegrationService | None = Depends(get_integration_service),
    tenant_id: str = Depends(require_tenant),
) -> BatchAnalysisResponse:
    """
    Analyze multiple test failures concurrently with container context.

    This endpoint allows batch processing of multiple test failures for
    efficient analysis of large test suites within a specific container.

    Args:
        request: Batch analysis request (must include container_name)
        integration_service: Integration service dependency

    Returns:
        Batch analysis results

    Raises:
        HTTPException: If batch analysis fails, container not found, or integration service unavailable
    """
    start_time = datetime.now(UTC)
    batch_id = str(uuid.uuid4())

    try:
        # Validate container exists
        await validate_and_get_container_for_tenant(request.container_name, tenant_id)

        logger.info(
            f"Starting batch analysis {batch_id} for {len(request.test_results)} tests in container: {request.container_name}"
        )

        if not integration_service:
            raise service_unavailable("AI test integration service is not available")

        # Convert test results to TestResult objects
        test_results = []
        for test_data in request.test_results:
            try:
                test_result = TestResult(**test_data)
                test_results.append(test_result)
            except ValidationError as e:
                logger.warning(f"Invalid test result in batch: {str(e)}")
                continue

        # Perform batch analysis
        analysis_results = await integration_service.batch_analyze_failures(
            test_results=test_results,
            max_concurrent=request.max_concurrent,
            include_suggestions=request.include_suggestions,
            include_patterns=request.include_patterns,
        )

        processing_time = (datetime.now(UTC) - start_time).total_seconds()

        # Convert results to API format
        api_results: list[dict[str, Any] | None] = []
        for result in analysis_results:
            if result:
                # `success` used to be the literal `True` for any non-None
                # result, so a batch entry produced on the degraded fallback
                # path -- i.e. with the LLM unreachable -- reported success and
                # gave the caller no way to tell. The single-analysis endpoint
                # already surfaces `degraded` (F4); the batch endpoint dropped
                # it on the floor. Derive both from the analysis metadata.
                degraded = bool((result.additional_metadata or {}).get("degraded"))
                api_results.append(
                    {
                        "analysis_id": result.analysis_id,
                        "success": not degraded,
                        "degraded": degraded,
                        "root_causes": [
                            {
                                "id": cause.cause_id,
                                "category": cause.category,
                                "description": cause.description,
                                "severity": _enum_value(cause.severity),
                                "confidence": _enum_value(cause.confidence),
                                "confidence_score": cause.confidence_score,
                            }
                            for cause in result.root_causes
                        ],
                        "fix_suggestions": [
                            {
                                "id": suggestion.suggestion_id,
                                "title": suggestion.title,
                                "description": suggestion.description,
                                "fix_type": suggestion.fix_type,
                                "priority": suggestion.priority,
                                "confidence": _enum_value(suggestion.confidence),
                                "confidence_score": suggestion.confidence_score,
                            }
                            for suggestion in result.fix_suggestions
                        ],
                        "detected_patterns": [
                            {
                                "id": pattern.pattern_id,
                                "name": pattern.pattern_name,
                                "type": pattern.pattern_type,
                                "confidence": _enum_value(pattern.confidence),
                                "confidence_score": pattern.confidence_score,
                            }
                            for pattern in result.detected_patterns
                        ],
                    }
                )
            else:
                api_results.append(None)

        return BatchAnalysisResponse(
            batch_id=batch_id,
            total_tests=len(request.test_results),
            analyzed_tests=len([r for r in analysis_results if r is not None]),
            processing_time=processing_time,
            timestamp=datetime.now(UTC),
            results=api_results,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch analysis {batch_id} failed: {str(e)}")
        processing_time = (datetime.now(UTC) - start_time).total_seconds()

        raise internal_error(f"Batch analysis failed: {str(e)}")


@router.get(
    "/failure-analysis/health",
    operation_id="ai_test_analysis_health_check",
    response_model=dict[str, Any],
    summary="Check AI analysis service health",
    description="Get health status and metrics for the AI analysis service",
    responses={
        401: {"model": Error401Response, "description": "Authentication required"},
        404: {
            "model": Error404Response,
            "description": "Service not found - AI service not configured",
        },
        500: {
            "model": Error500Response,
            "description": "Internal server error - Health check failed",
        },
        503: {
            "model": Error503Response,
            "description": "Service unavailable - AI service is down",
        },
    },
)
async def health_check(
    ai_service: AIFailureAnalysisService | None = Depends(get_ai_analysis_service),
    tenant_id: str = Depends(require_tenant),
) -> dict[str, Any]:
    """
    Get health status of the AI analysis service.

    Returns:
        Health check response with status and metrics
    """
    try:
        if ai_service is None:
            # Service not available, return degraded response
            return {
                "service": "ai_test_analysis",
                "service_status": "degraded",
                "ai_analysis_available": False,
                "ai_model_name": None,
                "checks": {"ai_service": "not_available", "api_key": "missing"},
                "metrics": {"total_analyses": 0, "success_rate": 0.0},
                "timestamp": datetime.now(UTC).isoformat(),
            }

        health_status = (
            ai_service.get_health_status() if hasattr(ai_service, "get_health_status") else {}  # type: ignore[call-non-callable]  # ty: ignore[call-non-callable]
        )

        # `get_health_status()` reports `circuit_breaker_open` / `model_name`;
        # it has never reported `status`, `ai_analysis_available`,
        # `ai_model_name` or `checks`. Reading those keys therefore always fell
        # through to the defaults below, so this endpoint answered
        # "healthy / available / gpt-4" unconditionally -- including with the
        # circuit breaker open and every analysis failing. Derive the status
        # from the fields the service actually publishes.
        circuit_open = bool(health_status.get("circuit_breaker_open", False))
        model_name = health_status.get("model_name")
        available = not circuit_open

        return {
            "service": "ai_test_analysis",
            "service_status": "degraded" if circuit_open else "healthy",
            "ai_analysis_available": available,
            "ai_model_name": model_name,
            "checks": {
                "ai_service": "available" if available else "circuit_breaker_open",
                "circuit_breaker_failures": health_status.get("circuit_breaker_failures", 0),
                "active_analyses": health_status.get("active_analyses", 0),
            },
            "metrics": health_status.get("metrics", {}),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "service": "ai_test_analysis",
            "service_status": "error",
            "ai_analysis_available": False,
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }


@router.get(
    "/failure-analysis/metrics",
    operation_id="get_test_analysis_metrics",
    summary="Get AI analysis service metrics",
    description="Get performance metrics for the AI analysis service",
    responses={
        401: {"model": Error401Response, "description": "Authentication required"},
        404: {
            "model": Error404Response,
            "description": "Service not found - Metrics not available",
        },
        500: {
            "model": Error500Response,
            "description": "Internal server error - Failed to retrieve metrics",
        },
        503: {
            "model": Error503Response,
            "description": "Service unavailable - AI service is down",
        },
    },
)
async def get_metrics(
    ai_service: AIFailureAnalysisService | None = Depends(get_ai_analysis_service),
    tenant_id: str = Depends(require_tenant),
) -> dict[str, Any]:
    """
    Get performance metrics for the AI analysis service.

    Returns:
        Service metrics and statistics
    """
    try:
        if ai_service is None:
            # Service not available, return default metrics
            return {
                "service": "ai_test_analysis",
                "metrics": {
                    "total_analyses": 0,
                    "success_rate": 0.0,
                    "average_response_time": 0.0,
                    "cached_analyses": 0,
                    "ai_model_usage": {"total_tokens": 0, "total_requests": 0},
                },
                "service_status": "degraded",
                "timestamp": datetime.now(UTC).isoformat(),
            }

        metrics = ai_service.get_metrics() if hasattr(ai_service, "get_metrics") else {}  # type: ignore[call-non-callable]  # ty: ignore[call-non-callable]

        # The service tracks total/successful/failed analyses, cache hits and
        # average_analysis_time. It has never tracked `success_rate`,
        # `average_response_time`, `cached_analyses` or `ai_model_usage`, so
        # every one of those reads missed and this endpoint reported a
        # permanent 0.0 success rate and 0.0 response time as if measured --
        # alongside a hardcoded "healthy". Derive them from the real counters.
        total = int(metrics.get("total_analyses", 0) or 0)
        successful = int(metrics.get("successful_analyses", 0) or 0)

        return {
            "service": "ai_test_analysis",
            "metrics": {
                "total_analyses": total,
                "successful_analyses": successful,
                "failed_analyses": int(metrics.get("failed_analyses", 0) or 0),
                # Undefined rather than 0.0 until something has actually run:
                # "no analyses yet" is not "0% of them succeeded".
                "success_rate": (successful / total) if total else None,
                "average_analysis_time": metrics.get("average_analysis_time", 0.0),
                "cache_hits": int(metrics.get("cache_hits", 0) or 0),
                "cache_misses": int(metrics.get("cache_misses", 0) or 0),
            },
            "service_status": "healthy",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get metrics: {str(e)}")
        return {
            "service": "ai_test_analysis",
            "metrics": {
                "total_analyses": 0,
                "success_rate": 0.0,
                "average_response_time": 0.0,
                "cached_analyses": 0,
                "ai_model_usage": {"total_tokens": 0, "total_requests": 0},
                "error": str(e),
            },
            "service_status": "error",
            "timestamp": datetime.now(UTC).isoformat(),
        }


@router.get(
    "/patterns",
    operation_id="list_test_failure_patterns",
    summary="Get detected failure patterns",
    description="Get statistics about detected failure patterns",
    responses={
        400: {
            "model": Error400Response,
            "description": "Bad request - Invalid query parameters",
        },
        401: {"model": Error401Response, "description": "Authentication required"},
        404: {
            "model": Error404Response,
            "description": "Service not found - Pattern detector not available",
        },
        422: {
            "model": ValidationErrorResponse,
            "description": "Validation error - Invalid limit parameter",
        },
        500: {
            "model": Error500Response,
            "description": "Internal server error - Failed to retrieve patterns",
        },
    },
)
async def get_patterns(
    limit: int = Query(
        default=50, ge=1, le=200, description="Maximum number of patterns to return"
    ),
    ai_service: AIFailureAnalysisService | None = Depends(get_ai_analysis_service),
    tenant_id: str = Depends(require_tenant),
) -> dict[str, Any]:
    """
    Get statistics about detected failure patterns.

    Args:
        limit: Maximum number of patterns to return
        ai_service: AI analysis service dependency

    Returns:
        Pattern statistics and information
    """
    try:
        # This endpoint used to answer with three hardcoded patterns carrying
        # invented occurrence counts (42 / 31 / 15) and invented confidence
        # bands, under a comment reading "Return the expected format for the
        # test". Nothing measured them: no pattern occurrence statistics are
        # recorded anywhere in this service. AIPatternDetector keeps its
        # `_known_patterns` in process memory only, is never constructed on any
        # request path, and has no persistent store behind it.
        #
        # Reporting a real "no statistics are collected" is the honest answer;
        # invented frequencies are worse than none, because a consumer acts on
        # them as if they described their own codebase.
        return {
            "status": "success",
            "data": {
                "patterns": [],
                "total_patterns": 0,
                "pattern_types": [],
                "statistics_available": False,
                "detail": (
                    "This service does not record failure-pattern occurrence "
                    "statistics. Pattern detection runs per-analysis and is "
                    "not aggregated or persisted, so no frequencies can be "
                    "reported."
                ),
                "service_status": "healthy" if ai_service else "degraded",
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get patterns: {str(e)}")
        return {
            "service": "ai_test_analysis",
            "patterns": {"patterns": [], "total_patterns": 0, "error": str(e)},
            "timestamp": datetime.now(UTC).isoformat(),
        }
