"""Configuration and utilities for AI Test Insights Service.

This module contains configuration classes, enums, and utility functions
used throughout the AI test insights system.
"""

import logging
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class TestPatternType(StrEnum):
    __test__ = False
    """Types of test patterns detected."""

    UNIT_PATTERN = "unit_pattern"
    INTEGRATION_PATTERN = "integration_pattern"
    PERFORMANCE_PATTERN = "performance_pattern"
    FLAKY_PATTERN = "flaky_pattern"
    COVERAGE_PATTERN = "coverage_pattern"
    SECURITY_PATTERN = "security_pattern"
    RELIABILITY_PATTERN = "reliability_pattern"


class TestInsightType(StrEnum):
    __test__ = False
    """Types of test insights generated."""

    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    COVERAGE = "coverage"
    QUALITY = "quality"
    PATTERNS = "patterns"
    RECOMMENDATIONS = "recommendations"
    PREDICTIONS = "predictions"


class AITestInsightsConfig:
    """Configuration for AI Test Insights Service."""

    def __init__(
        self,
        slow_test_threshold: float = 10.0,
        flaky_test_threshold: float = 0.8,
        memory_threshold: int = 512,
        cpu_threshold: int = 80,
        coverage_threshold: int = 80,
        cache_ttl: int = 86400,
        comprehensive_cache_ttl: int = 172800,
        enable_ai_features: bool = True,
        use_ml_backend: bool = True,
        analysis_weights: dict[str, float] | None = None,
    ):
        """Initialize configuration.

        Args:
            slow_test_threshold: Threshold in seconds for slow tests
            flaky_test_threshold: Success rate threshold for flaky tests
            memory_threshold: Memory usage threshold in MB
            cpu_threshold: CPU usage threshold percentage
            coverage_threshold: Coverage threshold percentage
            cache_ttl: Cache TTL for regular analysis in seconds
            comprehensive_cache_ttl: Cache TTL for comprehensive analysis in seconds
            enable_ai_features: Whether to enable AI features
            use_ml_backend: Whether to use ML backend for LLM analysis
            analysis_weights: Custom weights for health score calculation
        """
        self.slow_test_threshold = slow_test_threshold
        self.flaky_test_threshold = flaky_test_threshold
        self.memory_threshold = memory_threshold
        self.cpu_threshold = cpu_threshold
        self.coverage_threshold = coverage_threshold
        self.cache_ttl = cache_ttl
        self.comprehensive_cache_ttl = comprehensive_cache_ttl
        self.enable_ai_features = enable_ai_features
        self.use_ml_backend = use_ml_backend

        # Default analysis weights
        self.analysis_weights = analysis_weights or {
            "performance": 0.25,
            "reliability": 0.35,
            "coverage": 0.20,
            "quality": 0.20,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "slow_test_threshold": self.slow_test_threshold,
            "flaky_test_threshold": self.flaky_test_threshold,
            "memory_threshold": self.memory_threshold,
            "cpu_threshold": self.cpu_threshold,
            "coverage_threshold": self.coverage_threshold,
            "cache_ttl": self.cache_ttl,
            "comprehensive_cache_ttl": self.comprehensive_cache_ttl,
            "enable_ai_features": self.enable_ai_features,
            "use_ml_backend": self.use_ml_backend,
            "analysis_weights": self.analysis_weights,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "AITestInsightsConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)


def clone_config(config: AITestInsightsConfig) -> AITestInsightsConfig:
    """Return a detached copy of a config instance."""
    return AITestInsightsConfig.from_dict(config.to_dict())


class AnalysisResultHandler:
    """Utility class for handling analysis results."""

    @staticmethod
    def handle_analysis_result(
        result: Any | Exception,
        analysis_type: str,
        expected_type: type,
        logger: logging.Logger,
    ) -> Any:
        """Handle analysis result with error handling."""
        if isinstance(result, Exception):
            logger.warning(f"{analysis_type} analysis failed: {result}")
            return AnalysisResultHandler._get_default_instance(expected_type)

        return result

    @staticmethod
    def _get_default_instance(expected_type: type) -> Any:
        """Get default instance for a given type."""
        from services.ai_test_insights.models import (
            TestCoverageInsights,
            TestPerformanceMetrics,
            TestQualityInsights,
            TestReliabilityMetrics,
        )

        if expected_type == TestPerformanceMetrics:
            return TestPerformanceMetrics(
                average_execution_time=0.0,
                median_execution_time=0.0,
                p95_execution_time=0.0,
                p99_execution_time=0.0,
                average_memory_usage=0.0,
                peak_memory_usage=0.0,
                average_cpu_usage=0.0,
                slow_tests_count=0,
                fast_tests_count=0,
                performance_trend="unknown",
                bottleneck_categories=[],
            )
        elif expected_type == TestReliabilityMetrics:
            return TestReliabilityMetrics(
                overall_success_rate=0.0,
                success_rate_by_type={},
                common_failure_patterns=[],
                flaky_tests_count=0,
                stability_score=0.0,
                reliability_trend="unknown",
                error_categories={},
            )
        elif expected_type == TestCoverageInsights:
            return TestCoverageInsights(
                line_coverage=0.0,
                branch_coverage=0.0,
                function_coverage=0.0,
                uncovered_critical_paths=[],
                redundant_tests=[],
                coverage_gaps=[],
                recommended_tests=[],
            )
        elif expected_type == TestQualityInsights:
            return TestQualityInsights(
                test_quality_score=0.0,
                code_quality_score=0.0,
                critical_issues=[],
                quality_debt=0.0,
                quality_recommendations=[],
                automated_fixes=[],
            )
        else:
            return []


def calculate_health_score(
    performance_score: float,
    reliability_score: float,
    coverage_score: float,
    quality_score: float,
    weights: dict[str, float] | None = None,
) -> float:
    """Calculate overall test health score."""
    default_weights = {
        "performance": 0.25,
        "reliability": 0.35,
        "coverage": 0.20,
        "quality": 0.20,
    }

    weights = weights or default_weights

    health_score = (
        performance_score * weights["performance"]
        + reliability_score * weights["reliability"]
        + coverage_score * weights["coverage"]
        + quality_score * weights["quality"]
    )

    return min(100.0, max(0.0, health_score))


def calculate_improvement_potential(
    performance_score: float,
    reliability_score: float,
    coverage_score: float,
    quality_score: float,
    pattern_impact: float = 0.0,
    target_scores: dict[str, float] | None = None,
) -> float:
    """Calculate improvement potential score."""
    default_targets = {
        "performance": 90,
        "reliability": 95,
        "coverage": 85,
        "quality": 90,
    }

    targets = target_scores or default_targets

    # Calculate gaps from target scores
    performance_gap = max(0, targets["performance"] - performance_score)
    reliability_gap = max(0, targets["reliability"] - reliability_score)
    coverage_gap = max(0, targets["coverage"] - coverage_score)
    quality_gap = max(0, targets["quality"] - quality_score)

    # Calculate potential (higher gaps = higher potential)
    potential = (
        performance_gap * 0.3
        + reliability_gap * 0.3
        + coverage_gap * 0.2
        + quality_gap * 0.2
        + min(pattern_impact, 30)  # Cap pattern impact
    )

    return min(100.0, max(0.0, potential))


def get_health_grade(score: float) -> str:
    """Get health grade based on score."""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"


# Default configuration instance
DEFAULT_CONFIG = AITestInsightsConfig()
