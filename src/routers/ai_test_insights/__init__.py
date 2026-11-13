"""AI Test Insights API endpoints.

This module provides API endpoints for AI-powered test insights, including
test pattern analysis, performance optimization recommendations, and quality
metrics analysis using PydanticAI.

This is the main router that combines all modular AI test insights components.

⚠️ This service owns the AI *analysis* half only (``AITestInsightsService``,
PydanticAI-backed): analyze, comprehensive-analysis, predict-failures,
recommendations, cached analyses, metrics.

The intelligent test *execution* half — insights, execution-health, settings,
supported-languages, patterns, execute, execute/batch, service/{start,stop,restart}
— used to be mounted here too, on top of an ``IntelligentTestExecutionService``
that was never extracted into this service. Every one of those 12 routes answered
503 "not wired in coding-ai", for as long as they had existed, while still being
published into coding-gateway's aggregated OpenAPI and fr's generated client as
if they were capabilities. They were deleted rather than wired: coding-testing
implements this at ``/api/tests/ai/*``, and ``integration-tests/ecosystem.yaml``
already declares coding-ai -> coding-testing for it.
"""

import logging

from fastapi import APIRouter

from .ai_analysis_endpoints import router as ai_analysis_router
from .ai_test_insights_config import router as config_health_router
from .insight_generation_services import router as insight_generation_router
from .report_formatting_endpoints import router as report_formatting_router
from .test_pattern_recognition import router as pattern_recognition_router

logger = logging.getLogger(__name__)

router = APIRouter()

# Include all modular routers (tags are set at main API level as "V2 AI Test Insights")
router.include_router(ai_analysis_router)
router.include_router(pattern_recognition_router)
router.include_router(insight_generation_router)
router.include_router(report_formatting_router)
router.include_router(config_health_router)

__all__ = [
    "router",
    "ai_analysis_router",
    "pattern_recognition_router",
    "insight_generation_router",
    "report_formatting_router",
    "config_health_router",
]
