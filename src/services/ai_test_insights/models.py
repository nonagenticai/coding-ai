"""Pydantic models for AI Test Insights Service.

This module contains all the Pydantic model classes used for data validation
and serialization in the AI test insights system.
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_serializer

from models.testing_models import QualityIssue
from services.ai_test_insights.config import TestInsightType, TestPatternType


class TestPerformanceMetrics(BaseModel):
    __test__ = False
    """Test performance metrics analysis."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    # Execution metrics
    average_execution_time: float = Field(description="Average test execution time in seconds")
    median_execution_time: float = Field(description="Median test execution time in seconds")
    p95_execution_time: float = Field(description="95th percentile execution time in seconds")
    p99_execution_time: float = Field(description="99th percentile execution time in seconds")

    # Resource usage
    average_memory_usage: float = Field(description="Average memory usage in MB")
    peak_memory_usage: float = Field(description="Peak memory usage in MB")
    average_cpu_usage: float = Field(description="Average CPU usage percentage")

    # Test distribution
    slow_tests_count: int = Field(description="Number of slow tests (above threshold)")
    fast_tests_count: int = Field(description="Number of fast tests (below threshold)")

    # Performance trends
    performance_trend: str = Field(description="Performance trend (improving, degrading, stable)")
    bottleneck_categories: list[str] = Field(
        default_factory=list, description="Categories of performance bottlenecks"
    )

    @computed_field
    @property
    def performance_score(self) -> float:
        """Calculate overall performance score (0-100)."""
        time_score = max(0, 100 - (self.average_execution_time * 10))
        memory_score = max(0, 100 - (self.average_memory_usage / 10))
        cpu_score = max(0, 100 - self.average_cpu_usage)

        return (time_score + memory_score + cpu_score) / 3


class TestReliabilityMetrics(BaseModel):
    __test__ = False
    """Test reliability metrics analysis."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    # Success rates
    overall_success_rate: float = Field(ge=0.0, le=1.0, description="Overall test success rate")
    success_rate_by_type: dict[str, float] = Field(
        default_factory=dict, description="Success rates by test type"
    )

    # Failure patterns
    common_failure_patterns: list[str] = Field(
        default_factory=list, description="Common failure patterns detected"
    )
    flaky_tests_count: int = Field(description="Number of flaky tests detected")

    # Stability metrics
    stability_score: float = Field(ge=0.0, le=100.0, description="Test stability score (0-100)")
    reliability_trend: str = Field(description="Reliability trend (improving, degrading, stable)")

    # Error analysis
    error_categories: dict[str, int] = Field(
        default_factory=dict, description="Error categories and counts"
    )

    @computed_field
    @property
    def reliability_score(self) -> float:
        """Calculate overall reliability score (0-100)."""
        return (self.overall_success_rate * 100 + self.stability_score) / 2


class TestCoverageInsights(BaseModel):
    __test__ = False
    """Test coverage insights and recommendations."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    # Coverage metrics
    line_coverage: float = Field(ge=0.0, le=100.0, description="Line coverage percentage")
    branch_coverage: float = Field(ge=0.0, le=100.0, description="Branch coverage percentage")
    function_coverage: float = Field(ge=0.0, le=100.0, description="Function coverage percentage")

    # Coverage analysis
    uncovered_critical_paths: list[str] = Field(
        default_factory=list, description="Critical paths not covered by tests"
    )
    redundant_tests: list[str] = Field(
        default_factory=list, description="Tests that may be redundant"
    )

    # Recommendations
    coverage_gaps: list[str] = Field(
        default_factory=list, description="Specific coverage gaps identified"
    )
    recommended_tests: list[str] = Field(
        default_factory=list, description="Recommended tests to add"
    )

    @computed_field
    @property
    def overall_coverage_score(self) -> float:
        """Calculate overall coverage score (0-100)."""
        return (self.line_coverage + self.branch_coverage + self.function_coverage) / 3


class TestQualityInsights(BaseModel):
    __test__ = False
    """Test quality insights and analysis."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    # Quality metrics
    test_quality_score: float = Field(ge=0.0, le=100.0, description="Overall test quality score")
    code_quality_score: float = Field(
        ge=0.0, le=100.0, description="Code quality score from analysis"
    )

    # Quality issues
    critical_issues: list[QualityIssue] = Field(
        default_factory=list, description="Critical quality issues"
    )
    quality_debt: float = Field(description="Estimated quality debt in hours")

    # Improvement suggestions
    quality_recommendations: list[str] = Field(
        default_factory=list, description="Quality improvement recommendations"
    )
    automated_fixes: list[str] = Field(
        default_factory=list, description="Issues that can be automatically resolved"
    )

    @computed_field
    @property
    def combined_quality_score(self) -> float:
        """Calculate combined quality score."""
        return (self.test_quality_score + self.code_quality_score) / 2


class TestPatternInsight(BaseModel):
    __test__ = False
    """Individual test pattern insight."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    pattern_type: TestPatternType = Field(description="Type of pattern detected")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in pattern detection")
    description: str = Field(description="Description of the pattern")
    affected_tests: list[str] = Field(
        default_factory=list, description="Tests affected by this pattern"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Recommendations for addressing the pattern"
    )
    impact_score: float = Field(ge=0.0, le=10.0, description="Impact score of the pattern")


class TestInsightRecommendation(BaseModel):
    __test__ = False
    """AI-generated test insight recommendation."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    recommendation_id: str = Field(description="Unique recommendation ID")
    type: TestInsightType = Field(description="Type of insight recommendation")
    priority: str = Field(description="Priority level (high, medium, low)")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in recommendation")

    # Recommendation details
    title: str = Field(description="Recommendation title")
    description: str = Field(description="Detailed description")
    rationale: str = Field(description="Why this recommendation is made")

    # Implementation details
    implementation_steps: list[str] = Field(
        default_factory=list, description="Steps to implement the recommendation"
    )
    estimated_effort: str = Field(description="Estimated effort (low, medium, high)")
    expected_impact: str = Field(description="Expected impact description")

    # Supporting data
    supporting_metrics: dict[str, Any] = Field(
        default_factory=dict, description="Metrics supporting the recommendation"
    )
    affected_components: list[str] = Field(
        default_factory=list, description="Components affected by the recommendation"
    )


class TestInsightAnalysis(BaseModel):
    __test__ = False
    """Comprehensive test insight analysis."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    # Analysis metadata
    analysis_id: str = Field(description="Unique analysis ID")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When analysis was created",
    )
    analyzed_period: str = Field(description="Time period analyzed")

    # Core insights
    performance_metrics: TestPerformanceMetrics = Field(description="Performance analysis results")
    reliability_metrics: TestReliabilityMetrics = Field(description="Reliability analysis results")
    coverage_insights: TestCoverageInsights = Field(description="Coverage analysis results")
    quality_insights: TestQualityInsights = Field(description="Quality analysis results")

    # Pattern analysis
    detected_patterns: list[TestPatternInsight] = Field(
        default_factory=list, description="Detected test patterns"
    )

    # AI recommendations
    recommendations: list[TestInsightRecommendation] = Field(
        default_factory=list, description="AI-generated recommendations"
    )

    # Summary metrics
    overall_health_score: float = Field(ge=0.0, le=100.0, description="Overall test health score")
    improvement_potential: float = Field(
        ge=0.0, le=100.0, description="Improvement potential score"
    )

    @field_serializer("created_at")
    def serialize_datetime(self, v: datetime | None) -> str | None:
        if v is None:
            return None
        return v.isoformat()

    @computed_field
    @property
    def health_grade(self) -> str:
        """Calculate health grade based on score."""
        if self.overall_health_score >= 90:
            return "A"
        elif self.overall_health_score >= 80:
            return "B"
        elif self.overall_health_score >= 70:
            return "C"
        elif self.overall_health_score >= 60:
            return "D"
        else:
            return "F"
