"""AI Test Insights Service Package.

This package provides AI-powered test insights and analysis functionality.
"""

from .config import DEFAULT_CONFIG, AITestInsightsConfig
from .core_service import AITestInsightsService, create_ai_test_insights_service
from .models import (
    TestCoverageInsights,
    TestInsightAnalysis,
    TestInsightRecommendation,
    TestInsightType,
    TestPatternInsight,
    TestPatternType,
    TestPerformanceMetrics,
    TestQualityInsights,
    TestReliabilityMetrics,
)

# calculate_health_score not found in pattern_recognition

__all__ = [
    "AITestInsightsService",
    "create_ai_test_insights_service",
    "TestInsightAnalysis",
    "TestPerformanceMetrics",
    "TestReliabilityMetrics",
    "TestCoverageInsights",
    "TestQualityInsights",
    "TestPatternInsight",
    "TestInsightRecommendation",
    "TestPatternType",
    "TestInsightType",
    "AITestInsightsConfig",
    "DEFAULT_CONFIG",
]
