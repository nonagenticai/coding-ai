"""Performance Insight Analyzer for AI-powered test insights.

This module provides comprehensive performance analysis for test execution,
identifying bottlenecks, optimization opportunities, and performance trends.
It uses statistical analysis and machine learning techniques to provide
actionable insights for test performance improvement.
"""

from __future__ import annotations

import logging
import statistics
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_serializer

from models.testing_models import TestResult

logger = logging.getLogger(__name__)


class PerformanceBottleneckType(StrEnum):
    """Types of performance bottlenecks."""

    CPU_INTENSIVE = "cpu_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    IO_BOUND = "io_bound"
    NETWORK_BOUND = "network_bound"
    DATABASE_BOUND = "database_bound"
    SYNCHRONIZATION = "synchronization"
    ALGORITHMIC = "algorithmic"
    RESOURCE_CONTENTION = "resource_contention"


class PerformanceCategory(StrEnum):
    """Categories of performance analysis."""

    EXECUTION_TIME = "execution_time"
    RESOURCE_USAGE = "resource_usage"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    SCALABILITY = "scalability"
    EFFICIENCY = "efficiency"


class PerformanceInsight(BaseModel):
    """Individual performance insight."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    # Insight identification
    insight_id: str = Field(description="Unique insight identifier")
    insight_name: str = Field(description="Human-readable insight name")
    category: PerformanceCategory = Field(description="Performance category")
    bottleneck_type: PerformanceBottleneckType | None = Field(
        None, description="Type of bottleneck if applicable"
    )

    # Metrics
    severity: str = Field(description="Severity level (critical, high, medium, low)")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the insight")
    impact_score: float = Field(
        ge=0.0, le=10.0, description="Expected impact of addressing this insight"
    )

    # Performance data
    affected_tests: list[str] = Field(
        default_factory=list, description="Tests affected by this insight"
    )
    performance_delta: float = Field(description="Performance improvement potential (percentage)")
    baseline_metric: float = Field(description="Baseline performance metric value")
    target_metric: float = Field(description="Target performance metric value")

    # Analysis details
    description: str = Field(description="Detailed description of the insight")
    root_causes: list[str] = Field(default_factory=list, description="Identified root causes")
    optimization_suggestions: list[str] = Field(
        default_factory=list, description="Specific optimization suggestions"
    )
    implementation_complexity: str = Field(
        description="Implementation complexity (low, medium, high)"
    )

    # Supporting evidence
    evidence: dict[str, Any] = Field(
        default_factory=dict, description="Supporting evidence and metrics"
    )

    @computed_field
    @property
    def priority_score(self) -> float:
        """Calculate priority score based on severity, impact, and confidence."""
        severity_weights = {"critical": 10.0, "high": 8.0, "medium": 6.0, "low": 4.0}

        base_score = severity_weights.get(self.severity, 4.0)
        return base_score * 0.4 + self.impact_score * 0.4 + self.confidence * 10 * 0.2


class PerformanceStatistics(BaseModel):
    """Statistical analysis of performance metrics."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    # Basic statistics
    mean: float = Field(description="Mean execution time")
    median: float = Field(description="Median execution time")
    std_dev: float = Field(description="Standard deviation")
    variance: float = Field(description="Variance")

    # Percentiles
    p50: float = Field(description="50th percentile")
    p75: float = Field(description="75th percentile")
    p90: float = Field(description="90th percentile")
    p95: float = Field(description="95th percentile")
    p99: float = Field(description="99th percentile")

    # Range
    min_value: float = Field(description="Minimum value")
    max_value: float = Field(description="Maximum value")
    range_value: float = Field(description="Range (max - min)")

    # Distribution characteristics
    coefficient_of_variation: float = Field(description="Coefficient of variation (std_dev / mean)")
    skewness: float = Field(description="Skewness of the distribution")
    kurtosis: float = Field(description="Kurtosis of the distribution")

    @computed_field
    @property
    def is_highly_variable(self) -> bool:
        """Check if the distribution is highly variable."""
        return self.coefficient_of_variation > 0.5

    @computed_field
    @property
    def distribution_type(self) -> str:
        """Classify the distribution type."""
        if abs(self.skewness) < 0.5:
            return "normal"
        elif self.skewness > 0.5:
            return "right_skewed"
        else:
            return "left_skewed"


class PerformanceTrendAnalysis(BaseModel):
    """Analysis of performance trends over time."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    # Trend characteristics
    trend_direction: str = Field(description="Trend direction (improving, degrading, stable)")
    trend_strength: float = Field(ge=0.0, le=1.0, description="Strength of the trend")
    trend_confidence: float = Field(ge=0.0, le=1.0, description="Confidence in trend analysis")

    # Regression analysis
    slope: float = Field(description="Linear regression slope")
    r_squared: float = Field(ge=0.0, le=1.0, description="R-squared value of the trend")

    # Change metrics
    percentage_change: float = Field(description="Percentage change from start to end")
    absolute_change: float = Field(description="Absolute change from start to end")

    # Anomaly detection
    anomalies_detected: list[int] = Field(
        default_factory=list, description="Indices of detected anomalies"
    )
    anomaly_count: int = Field(description="Number of anomalies detected")

    @computed_field
    @property
    def is_significant_trend(self) -> bool:
        """Check if the trend is statistically significant."""
        return self.r_squared > 0.7 and self.trend_confidence > 0.8


class PerformanceOptimizationRecommendation(BaseModel):
    """Performance optimization recommendation."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    # Recommendation details
    recommendation_id: str = Field(description="Unique recommendation ID")
    title: str = Field(description="Recommendation title")
    description: str = Field(description="Detailed description")
    category: PerformanceCategory = Field(description="Performance category")

    # Impact assessment
    expected_improvement: float = Field(description="Expected performance improvement (percentage)")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the recommendation")
    priority: str = Field(description="Priority level (high, medium, low)")

    # Implementation details
    implementation_steps: list[str] = Field(
        default_factory=list, description="Implementation steps"
    )
    complexity: str = Field(description="Implementation complexity")
    estimated_effort: str = Field(description="Estimated effort required")

    # Dependencies and risks
    dependencies: list[str] = Field(
        default_factory=list, description="Dependencies for implementation"
    )
    risks: list[str] = Field(default_factory=list, description="Potential risks")

    # Supporting data
    supporting_evidence: dict[str, Any] = Field(
        default_factory=dict, description="Supporting evidence"
    )


class PerformanceAnalysisResult(BaseModel):
    """Result of comprehensive performance analysis."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    # Analysis metadata
    analysis_id: str = Field(description="Unique analysis ID")
    analyzed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When analysis was performed",
    )

    # Input data summary
    total_tests_analyzed: int = Field(description="Total number of tests analyzed")
    analysis_period: str = Field(description="Time period analyzed")

    # Performance statistics
    overall_statistics: PerformanceStatistics = Field(description="Overall performance statistics")
    statistics_by_language: dict[str, PerformanceStatistics] = Field(
        default_factory=dict, description="Statistics by programming language"
    )
    statistics_by_type: dict[str, PerformanceStatistics] = Field(
        default_factory=dict, description="Statistics by test type"
    )

    # Trend analysis
    trend_analysis: PerformanceTrendAnalysis = Field(description="Performance trend analysis")

    # Insights and recommendations
    performance_insights: list[PerformanceInsight] = Field(
        default_factory=list, description="Identified performance insights"
    )
    optimization_recommendations: list[PerformanceOptimizationRecommendation] = Field(
        default_factory=list, description="Optimization recommendations"
    )

    # Performance categorization
    slow_tests: list[str] = Field(default_factory=list, description="Tests classified as slow")
    fast_tests: list[str] = Field(default_factory=list, description="Tests classified as fast")
    variable_tests: list[str] = Field(
        default_factory=list, description="Tests with variable performance"
    )

    # Summary metrics
    performance_score: float = Field(ge=0.0, le=100.0, description="Overall performance score")
    optimization_potential: float = Field(
        ge=0.0, le=100.0, description="Potential for optimization"
    )

    @field_serializer("analyzed_at")
    def serialize_datetime(self, v: datetime | None) -> str | None:
        if v is None:
            return None
        return v.isoformat()

    @computed_field
    @property
    def performance_grade(self) -> str:
        """Calculate performance grade."""
        if self.performance_score >= 90:
            return "A"
        elif self.performance_score >= 80:
            return "B"
        elif self.performance_score >= 70:
            return "C"
        elif self.performance_score >= 60:
            return "D"
        else:
            return "F"


class PerformanceInsightAnalyzer:
    """Advanced performance insight analyzer."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the performance insight analyzer."""
        self.config = config or self._get_default_config()
        logger.info("Performance Insight Analyzer initialized")

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration."""
        return {
            # Performance thresholds
            "slow_test_threshold": 10.0,  # seconds
            "very_slow_test_threshold": 30.0,  # seconds
            "fast_test_threshold": 1.0,  # seconds
            "variable_cv_threshold": 0.5,  # coefficient of variation
            # Statistical analysis
            "min_samples_for_trend": 10,
            "trend_confidence_threshold": 0.8,
            "anomaly_detection_threshold": 2.0,  # standard deviations
            # Optimization thresholds
            "significant_improvement_threshold": 20.0,  # percentage
            "high_impact_threshold": 8.0,  # impact score
            "min_confidence_threshold": 0.6,
        }

    async def analyze_performance(
        self, test_results: list[TestResult]
    ) -> PerformanceAnalysisResult:
        """Perform comprehensive performance analysis."""
        analysis_id = str(uuid.uuid4())

        logger.info(
            f"Starting performance analysis {analysis_id} for {len(test_results)} test results"
        )

        # Extract execution times
        execution_times = [r.execution_time for r in test_results]

        if not execution_times:
            raise ValueError("No execution times found in test results")

        # Calculate overall statistics
        overall_stats = await self._calculate_statistics(execution_times)

        # Calculate statistics by language
        stats_by_language = await self._calculate_statistics_by_language(test_results)

        # Calculate statistics by test type
        stats_by_type = await self._calculate_statistics_by_type(test_results)

        # Perform trend analysis
        trend_analysis = await self._analyze_trends(test_results)

        # Generate performance insights
        insights = await self._generate_insights(test_results, overall_stats)

        # Generate optimization recommendations
        recommendations = await self._generate_recommendations(insights, overall_stats)

        # Categorize tests
        slow_tests, fast_tests, variable_tests = await self._categorize_tests(test_results)

        # Calculate performance score
        performance_score = await self._calculate_performance_score(overall_stats, insights)

        # Calculate optimization potential
        optimization_potential = await self._calculate_optimization_potential(insights)

        # Create result
        result = PerformanceAnalysisResult(
            analysis_id=analysis_id,
            total_tests_analyzed=len(test_results),
            analysis_period="recent",
            overall_statistics=overall_stats,
            statistics_by_language=stats_by_language,
            statistics_by_type=stats_by_type,
            trend_analysis=trend_analysis,
            performance_insights=insights,
            optimization_recommendations=recommendations,
            slow_tests=slow_tests,
            fast_tests=fast_tests,
            variable_tests=variable_tests,
            performance_score=performance_score,
            optimization_potential=optimization_potential,
        )

        logger.info(f"Performance analysis {analysis_id} completed")

        return result

    async def _calculate_statistics(self, execution_times: list[float]) -> PerformanceStatistics:
        """Calculate comprehensive statistics for execution times."""

        if not execution_times:
            raise ValueError("No execution times provided")

        # Sort for percentile calculations
        sorted_times = sorted(execution_times)
        n = len(sorted_times)

        # Basic statistics
        mean = statistics.mean(execution_times)
        median = statistics.median(execution_times)
        std_dev = statistics.stdev(execution_times) if n > 1 else 0.0
        variance = statistics.variance(execution_times) if n > 1 else 0.0

        # Percentiles
        p50 = sorted_times[int(0.50 * n)]
        p75 = sorted_times[int(0.75 * n)]
        p90 = sorted_times[int(0.90 * n)]
        p95 = sorted_times[int(0.95 * n)]
        p99 = sorted_times[int(0.99 * n)] if n > 100 else sorted_times[-1]

        # Range
        min_val = min(execution_times)
        max_val = max(execution_times)
        range_val = max_val - min_val

        # Distribution characteristics
        cv = std_dev / mean if mean > 0 else 0.0

        # Calculate skewness and kurtosis (simplified)
        if n > 2 and std_dev > 0:
            skewness = sum((x - mean) ** 3 for x in execution_times) / (n * std_dev**3)
            kurtosis = sum((x - mean) ** 4 for x in execution_times) / (n * std_dev**4) - 3
        else:
            skewness = 0.0
            kurtosis = 0.0

        return PerformanceStatistics(
            mean=mean,
            median=median,
            std_dev=std_dev,
            variance=variance,
            p50=p50,
            p75=p75,
            p90=p90,
            p95=p95,
            p99=p99,
            min_value=min_val,
            max_value=max_val,
            range_value=range_val,
            coefficient_of_variation=cv,
            skewness=skewness,
            kurtosis=kurtosis,
        )

    async def _calculate_statistics_by_language(
        self, test_results: list[TestResult]
    ) -> dict[str, PerformanceStatistics]:
        """Calculate statistics grouped by programming language."""

        language_groups = defaultdict(list)
        for result in test_results:
            language_groups[result.language].append(result.execution_time)

        stats_by_language = {}
        for language, times in language_groups.items():
            if times:
                stats_by_language[language] = await self._calculate_statistics(times)

        return stats_by_language

    async def _calculate_statistics_by_type(
        self, test_results: list[TestResult]
    ) -> dict[str, PerformanceStatistics]:
        """Calculate statistics grouped by test type."""

        type_groups = defaultdict(list)
        for result in test_results:
            test_type = result.test_type or "unknown"
            type_groups[test_type].append(result.execution_time)

        stats_by_type = {}
        for test_type, times in type_groups.items():
            if times:
                stats_by_type[test_type] = await self._calculate_statistics(times)

        return stats_by_type

    async def _analyze_trends(self, test_results: list[TestResult]) -> PerformanceTrendAnalysis:
        """Analyze performance trends over time."""

        # Sort results by creation time
        sorted_results = sorted(test_results, key=lambda r: r.created_at)

        if len(sorted_results) < self.config["min_samples_for_trend"]:
            return PerformanceTrendAnalysis(
                trend_direction="stable",
                trend_strength=0.0,
                trend_confidence=0.0,
                slope=0.0,
                r_squared=0.0,
                percentage_change=0.0,
                absolute_change=0.0,
                anomaly_count=0,
            )

        # Extract time series data
        execution_times = [r.execution_time for r in sorted_results]

        # Simple linear regression
        n = len(execution_times)
        x = list(range(n))
        y = execution_times

        # Calculate regression coefficients
        x_mean = sum(x) / n
        y_mean = sum(y) / n

        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else 0.0

        # Calculate R-squared
        y_pred = [slope * x[i] + (y_mean - slope * x_mean) for i in range(n)]
        ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((y[i] - y_mean) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        # Determine trend direction
        if abs(slope) < 0.01:
            trend_direction = "stable"
        elif slope > 0:
            trend_direction = "degrading"
        else:
            trend_direction = "improving"

        # Calculate trend strength and confidence
        trend_strength = min(1.0, abs(slope) * 10)  # Normalize slope
        trend_confidence = r_squared

        # Calculate change metrics
        first_value = execution_times[0]
        last_value = execution_times[-1]
        percentage_change = (
            ((last_value - first_value) / first_value * 100) if first_value != 0 else 0.0
        )
        absolute_change = last_value - first_value

        # Simple anomaly detection
        mean_time = statistics.mean(execution_times)
        std_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0
        threshold = self.config["anomaly_detection_threshold"]

        anomalies = []
        for i, time in enumerate(execution_times):
            if abs(time - mean_time) > threshold * std_time:
                anomalies.append(i)

        return PerformanceTrendAnalysis(
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            trend_confidence=trend_confidence,
            slope=slope,
            r_squared=r_squared,
            percentage_change=percentage_change,
            absolute_change=absolute_change,
            anomalies_detected=anomalies,
            anomaly_count=len(anomalies),
        )

    async def _generate_insights(
        self, test_results: list[TestResult], overall_stats: PerformanceStatistics
    ) -> list[PerformanceInsight]:
        """Generate performance insights."""

        insights = []

        # Slow test insight
        slow_tests = [
            r for r in test_results if r.execution_time > self.config["slow_test_threshold"]
        ]

        if slow_tests:
            insights.append(
                PerformanceInsight(
                    insight_id="perf_slow_tests",
                    insight_name="Slow Test Execution",
                    category=PerformanceCategory.EXECUTION_TIME,
                    bottleneck_type=PerformanceBottleneckType.ALGORITHMIC,
                    severity="high" if len(slow_tests) > 10 else "medium",
                    confidence=0.9,
                    impact_score=min(10.0, len(slow_tests) * 0.5),
                    affected_tests=[r.test_id for r in slow_tests],
                    performance_delta=30.0,  # Estimated improvement
                    baseline_metric=overall_stats.mean,
                    target_metric=overall_stats.mean * 0.7,
                    description=f"Identified {len(slow_tests)} slow tests exceeding {self.config['slow_test_threshold']}s threshold",
                    root_causes=[
                        "Inefficient algorithms or data structures",
                        "Database queries without optimization",
                        "Synchronous I/O operations",
                        "Excessive test setup/teardown",
                    ],
                    optimization_suggestions=[
                        "Profile slow tests to identify bottlenecks",
                        "Optimize database queries and use connection pooling",
                        "Implement asynchronous operations where possible",
                        "Use test data factories and mocking",
                    ],
                    implementation_complexity="medium",
                    evidence={
                        "slow_test_count": len(slow_tests),
                        "average_slow_test_time": statistics.mean(
                            [r.execution_time for r in slow_tests]
                        ),
                        "slowest_test_time": max(r.execution_time for r in slow_tests),
                    },
                )
            )

        # High variability insight
        if overall_stats.is_highly_variable:
            insights.append(
                PerformanceInsight(
                    insight_id="perf_high_variability",
                    insight_name="High Performance Variability",
                    category=PerformanceCategory.EFFICIENCY,
                    bottleneck_type=PerformanceBottleneckType.RESOURCE_CONTENTION,
                    severity="medium",
                    confidence=0.8,
                    impact_score=6.0,
                    affected_tests=[r.test_id for r in test_results],
                    performance_delta=20.0,
                    baseline_metric=overall_stats.coefficient_of_variation,
                    target_metric=0.3,  # Target CV
                    description=f"High performance variability detected (CV: {overall_stats.coefficient_of_variation:.2f})",
                    root_causes=[
                        "Resource contention between tests",
                        "Non-deterministic operations",
                        "Varying test data complexity",
                        "Infrastructure instability",
                    ],
                    optimization_suggestions=[
                        "Implement test isolation and resource management",
                        "Use consistent test data and mocking",
                        "Add performance monitoring and alerting",
                        "Review test infrastructure configuration",
                    ],
                    implementation_complexity="medium",
                    evidence={
                        "coefficient_of_variation": overall_stats.coefficient_of_variation,
                        "standard_deviation": overall_stats.std_dev,
                        "range": overall_stats.range_value,
                    },
                )
            )

        # Memory-intensive tests insight (mock - would need actual memory data)
        very_slow_tests = [
            r for r in test_results if r.execution_time > self.config["very_slow_test_threshold"]
        ]

        if very_slow_tests:
            insights.append(
                PerformanceInsight(
                    insight_id="perf_memory_intensive",
                    insight_name="Memory-Intensive Test Operations",
                    category=PerformanceCategory.RESOURCE_USAGE,
                    bottleneck_type=PerformanceBottleneckType.MEMORY_INTENSIVE,
                    severity="high",
                    confidence=0.7,
                    impact_score=8.0,
                    affected_tests=[r.test_id for r in very_slow_tests],
                    performance_delta=40.0,
                    baseline_metric=statistics.mean([r.execution_time for r in very_slow_tests]),
                    target_metric=statistics.mean([r.execution_time for r in very_slow_tests])
                    * 0.6,
                    description=f"Detected {len(very_slow_tests)} very slow tests potentially due to memory usage",
                    root_causes=[
                        "Large object creation and manipulation",
                        "Memory leaks in test code",
                        "Inefficient data structures",
                        "Lack of memory cleanup",
                    ],
                    optimization_suggestions=[
                        "Profile memory usage in slow tests",
                        "Implement object pooling and reuse",
                        "Use streaming for large data operations",
                        "Add explicit memory cleanup in teardown",
                    ],
                    implementation_complexity="high",
                    evidence={
                        "very_slow_test_count": len(very_slow_tests),
                        "average_very_slow_time": statistics.mean(
                            [r.execution_time for r in very_slow_tests]
                        ),
                    },
                )
            )

        # Test distribution insight
        language_stats = {}
        for result in test_results:
            if result.language not in language_stats:
                language_stats[result.language] = []
            language_stats[result.language].append(result.execution_time)

        if len(language_stats) > 1:
            # Find the slowest language
            avg_times = {lang: statistics.mean(times) for lang, times in language_stats.items()}
            slowest_lang = max(avg_times, key=lambda k: avg_times[k])
            fastest_lang = min(avg_times, key=lambda k: avg_times[k])

            if avg_times[slowest_lang] > avg_times[fastest_lang] * 2:
                insights.append(
                    PerformanceInsight(
                        insight_id="perf_language_disparity",
                        insight_name="Language Performance Disparity",
                        category=PerformanceCategory.EFFICIENCY,
                        bottleneck_type=PerformanceBottleneckType.ALGORITHMIC,
                        severity="medium",
                        confidence=0.8,
                        impact_score=5.0,
                        affected_tests=[
                            r.test_id for r in test_results if r.language == slowest_lang
                        ],
                        performance_delta=25.0,
                        baseline_metric=avg_times[slowest_lang],
                        target_metric=avg_times[fastest_lang] * 1.2,
                        description=f"Significant performance difference between {slowest_lang} and {fastest_lang} tests",
                        root_causes=[
                            "Language-specific performance characteristics",
                            "Suboptimal test implementation patterns",
                            "Different optimization levels",
                            "Framework overhead differences",
                        ],
                        optimization_suggestions=[
                            f"Review {slowest_lang} test implementations",
                            "Apply language-specific optimization techniques",
                            "Consider test framework alternatives",
                            "Implement performance benchmarking",
                        ],
                        implementation_complexity="medium",
                        evidence={
                            "slowest_language": slowest_lang,
                            "fastest_language": fastest_lang,
                            "performance_ratio": avg_times[slowest_lang] / avg_times[fastest_lang],
                        },
                    )
                )

        return insights

    async def _generate_recommendations(
        self, insights: list[PerformanceInsight], overall_stats: PerformanceStatistics
    ) -> list[PerformanceOptimizationRecommendation]:
        """Generate optimization recommendations."""

        recommendations = []

        # High-impact recommendations based on insights
        high_impact_insights = [
            i for i in insights if i.impact_score >= self.config["high_impact_threshold"]
        ]

        for insight in high_impact_insights:
            recommendations.append(
                PerformanceOptimizationRecommendation(
                    recommendation_id=f"opt_{insight.insight_id}",
                    title=f"Optimize {insight.insight_name}",
                    description=f"Address {insight.insight_name.lower()} to improve overall test performance",
                    category=insight.category,
                    expected_improvement=insight.performance_delta,
                    confidence=insight.confidence,
                    priority="high" if insight.impact_score > 8 else "medium",
                    implementation_steps=insight.optimization_suggestions,
                    complexity=insight.implementation_complexity,
                    estimated_effort="medium",
                    dependencies=[
                        "Performance profiling tools",
                        "Test infrastructure access",
                        "Development team coordination",
                    ],
                    risks=[
                        "Potential regression in test coverage",
                        "Initial implementation time investment",
                        "Need for thorough validation",
                    ],
                    supporting_evidence=insight.evidence,
                )
            )

        # General recommendations based on overall statistics
        if overall_stats.mean > self.config["slow_test_threshold"]:
            recommendations.append(
                PerformanceOptimizationRecommendation(
                    recommendation_id="opt_general_performance",
                    title="Implement Comprehensive Performance Optimization",
                    description="Overall test performance is below acceptable thresholds",
                    category=PerformanceCategory.EXECUTION_TIME,
                    expected_improvement=35.0,
                    confidence=0.85,
                    priority="high",
                    implementation_steps=[
                        "Conduct comprehensive performance audit",
                        "Implement parallel test execution",
                        "Optimize test data management",
                        "Establish performance monitoring",
                    ],
                    complexity="high",
                    estimated_effort="high",
                    dependencies=[
                        "Performance testing infrastructure",
                        "Team training on optimization techniques",
                        "CI/CD pipeline modifications",
                    ],
                    risks=[
                        "Significant initial time investment",
                        "Potential for introducing new issues",
                        "Need for extensive testing",
                    ],
                    supporting_evidence={
                        "overall_mean": overall_stats.mean,
                        "threshold": self.config["slow_test_threshold"],
                        "improvement_potential": (
                            overall_stats.mean - self.config["slow_test_threshold"]
                        )
                        / overall_stats.mean
                        * 100,
                    },
                )
            )

        # Resource utilization recommendations
        if overall_stats.is_highly_variable:
            recommendations.append(
                PerformanceOptimizationRecommendation(
                    recommendation_id="opt_resource_stability",
                    title="Improve Test Resource Stability",
                    description="Reduce performance variability through better resource management",
                    category=PerformanceCategory.RESOURCE_USAGE,
                    expected_improvement=20.0,
                    confidence=0.8,
                    priority="medium",
                    implementation_steps=[
                        "Implement resource pooling",
                        "Add test isolation mechanisms",
                        "Optimize test scheduling",
                        "Monitor resource usage patterns",
                    ],
                    complexity="medium",
                    estimated_effort="medium",
                    dependencies=[
                        "Resource monitoring tools",
                        "Test execution framework modifications",
                        "Infrastructure configuration changes",
                    ],
                    risks=[
                        "Complexity in resource management",
                        "Potential for resource conflicts",
                        "Need for careful monitoring",
                    ],
                    supporting_evidence={
                        "coefficient_of_variation": overall_stats.coefficient_of_variation,
                        "standard_deviation": overall_stats.std_dev,
                    },
                )
            )

        # Sort recommendations by priority and expected improvement
        recommendations.sort(
            key=lambda r: (
                {"high": 3, "medium": 2, "low": 1}[r.priority],
                r.expected_improvement,
            ),
            reverse=True,
        )

        return recommendations

    async def _categorize_tests(
        self, test_results: list[TestResult]
    ) -> tuple[list[str], list[str], list[str]]:
        """Categorize tests by performance characteristics."""

        execution_times = [r.execution_time for r in test_results]

        if not execution_times:
            return [], [], []

        # Calculate thresholds
        mean_time = statistics.mean(execution_times)
        std_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0

        slow_tests = []
        fast_tests = []
        variable_tests = []

        for result in test_results:
            # Slow tests
            if result.execution_time > self.config["slow_test_threshold"]:
                slow_tests.append(result.test_id)

            # Fast tests
            elif result.execution_time < self.config["fast_test_threshold"]:
                fast_tests.append(result.test_id)

            # Variable tests (tests that deviate significantly from mean)
            if std_time > 0 and abs(result.execution_time - mean_time) > 2 * std_time:
                variable_tests.append(result.test_id)

        return slow_tests, fast_tests, variable_tests

    async def _calculate_performance_score(
        self, overall_stats: PerformanceStatistics, insights: list[PerformanceInsight]
    ) -> float:
        """Calculate overall performance score."""

        # Base score from execution time
        base_score = 100.0

        # Penalize for slow average execution time
        if overall_stats.mean > self.config["slow_test_threshold"]:
            time_penalty = min(50.0, (overall_stats.mean - self.config["slow_test_threshold"]) * 5)
            base_score -= time_penalty

        # Penalize for high variability
        if overall_stats.is_highly_variable:
            variability_penalty = min(20.0, overall_stats.coefficient_of_variation * 40)
            base_score -= variability_penalty

        # Penalize for critical insights
        critical_insights = [i for i in insights if i.severity == "critical"]
        high_insights = [i for i in insights if i.severity == "high"]

        insight_penalty = len(critical_insights) * 15 + len(high_insights) * 10
        base_score -= min(30.0, insight_penalty)

        return max(0.0, min(100.0, base_score))

    async def _calculate_optimization_potential(self, insights: list[PerformanceInsight]) -> float:
        """Calculate optimization potential."""

        if not insights:
            return 0.0

        # Calculate weighted average of performance deltas
        total_weighted_delta = sum(
            insight.performance_delta * insight.confidence * insight.impact_score
            for insight in insights
        )

        total_weight = sum(insight.confidence * insight.impact_score for insight in insights)

        if total_weight == 0:
            return 0.0

        potential = total_weighted_delta / total_weight
        return min(100.0, max(0.0, potential))


# Factory function
def create_performance_insight_analyzer(
    config: dict[str, Any] | None = None,
) -> PerformanceInsightAnalyzer:
    """Create a performance insight analyzer instance."""
    return PerformanceInsightAnalyzer(config=config)
