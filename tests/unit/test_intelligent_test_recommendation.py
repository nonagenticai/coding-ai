"""Tests for IntelligentRecommendationEngine covering all major code paths.

Focus: generate_recommendations flow, recommendation generators from different
analysis sources, deduplication, suite creation, roadmap generation, ROI
estimation, models, computed fields, and factory function.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from models.testing_models import TestResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_engine(config=None):
    from services.intelligent_test_recommendation import IntelligentRecommendationEngine

    return IntelligentRecommendationEngine(config=config)


def _make_test_result(success: bool = True, test_id: str = "t1") -> TestResult:
    return TestResult(
        test_id=test_id,
        success=success,
        execution_time=1.0,
        created_at=datetime.now(UTC),
    )


def _make_test_insights():
    """Create a minimal TestInsightAnalysis mock."""
    mock = MagicMock()
    mock.performance_metrics.slow_tests_count = 3
    mock.performance_metrics.performance_score = 60.0
    mock.reliability_metrics.flaky_tests_count = 2
    mock.reliability_metrics.overall_success_rate = 0.85
    mock.coverage_insights.overall_coverage_score = 70.0
    mock.coverage_insights.coverage_gaps = ["module_a", "module_b"]
    return mock


def _make_test_insights_no_issues():
    """Create TestInsightAnalysis mock with no issues."""
    mock = MagicMock()
    mock.performance_metrics.slow_tests_count = 0
    mock.reliability_metrics.flaky_tests_count = 0
    mock.coverage_insights.overall_coverage_score = 95.0
    mock.coverage_insights.coverage_gaps = []
    return mock


def _make_pattern_analysis():
    """Create a minimal TestPatternAnalysisResult mock."""
    pattern = MagicMock()
    pattern.pattern_id = "pat-1"
    pattern.pattern_name = "Anti-Pattern"
    pattern.description = "Found an anti-pattern in tests"
    pattern.impact_score = 8.0
    pattern.confidence_score = 0.9
    pattern.recommendations = ["Fix the pattern", "Refactor tests"]
    pattern.pattern_type = "anti_pattern"
    pattern.occurrence_count = 5
    pattern.frequency_ratio = 0.3

    mock = MagicMock()
    mock.critical_patterns = [pattern]
    return mock


def _make_performance_analysis():
    """Create a minimal PerformanceAnalysisResult mock."""
    rec = MagicMock()
    rec.recommendation_id = "perf-1"
    rec.title = "Optimize slow query tests"
    rec.description = "Several tests with slow database queries"
    rec.priority = "high"
    rec.expected_improvement = 40.0
    rec.confidence = 0.85
    rec.implementation_steps = ["Profile tests", "Add indexes"]
    rec.estimated_effort = "medium"
    rec.dependencies = ["Database access"]
    rec.supporting_evidence = {"slow_tests": 5}
    rec.risks = ["Potential test failures"]

    mock = MagicMock()
    mock.optimization_recommendations = [rec]
    return mock


def _make_quality_analysis():
    """Create a minimal QualityAnalysisResult mock."""
    rec = MagicMock()
    rec.recommendation_id = "qual-1"
    rec.title = "Improve code quality score"
    rec.description = "Several quality issues detected"
    rec.priority = "critical"
    rec.expected_improvement = 25.0
    rec.confidence = 0.8
    rec.implementation_steps = ["Run linter", "Fix issues"]
    rec.effort_estimate = "medium"
    rec.prerequisites = ["Linting tools"]
    rec.tools_required = ["ruff", "ty"]
    rec.supporting_evidence = {"issues_count": 15}
    rec.affected_components = ["core_module"]
    rec.risks = ["Refactoring effort"]

    mock = MagicMock()
    mock.recommendations = [rec]
    return mock


# ===================================================================
# Model tests
# ===================================================================


class TestModels:
    def test_recommendation_priority_score(self):
        from services.intelligent_test_recommendation import (
            RecommendationCategory,
            RecommendationPriority,
            RecommendationType,
            TestRecommendation,
        )

        rec = TestRecommendation(
            recommendation_id="test-1",
            title="Test Recommendation",
            description="A test recommendation",
            type=RecommendationType.TEST_OPTIMIZATION,
            priority=RecommendationPriority.CRITICAL,
            category=RecommendationCategory.IMMEDIATE_ACTION,
            expected_benefit="Big improvement",
            impact_score=9.0,
            confidence=0.9,
            estimated_effort="low",
            time_estimate="1 week",
            rationale="Because it matters",
        )

        # priority_score = (10.0 + 9.0) * 0.9 = 17.1
        assert rec.priority_score == pytest.approx(17.1)

    def test_recommendation_suite_priority_critical(self):
        from services.intelligent_test_recommendation import (
            RecommendationCategory,
            RecommendationPriority,
            RecommendationSuite,
            RecommendationType,
            TestRecommendation,
        )

        rec = TestRecommendation(
            recommendation_id="test-1",
            title="Critical Recommendation",
            description="A critical recommendation",
            type=RecommendationType.TEST_OPTIMIZATION,
            priority=RecommendationPriority.CRITICAL,
            category=RecommendationCategory.IMMEDIATE_ACTION,
            expected_benefit="Critical fix",
            impact_score=9.0,
            confidence=0.9,
            estimated_effort="low",
            time_estimate="1 week",
            rationale="Critical issue",
        )

        suite = RecommendationSuite(
            suite_id="suite-1",
            suite_name="Critical Suite",
            description="Suite with critical recs",
            recommendations=[rec],
            total_impact_score=9.0,
            average_confidence=0.9,
            estimated_total_effort="low",
        )

        assert suite.suite_priority == RecommendationPriority.CRITICAL

    def test_recommendation_suite_priority_low(self):
        from services.intelligent_test_recommendation import (
            RecommendationCategory,
            RecommendationPriority,
            RecommendationSuite,
            RecommendationType,
            TestRecommendation,
        )

        rec = TestRecommendation(
            recommendation_id="test-1",
            title="Low Priority Recommendation",
            description="A low priority recommendation",
            type=RecommendationType.TEST_OPTIMIZATION,
            priority=RecommendationPriority.LOW,
            category=RecommendationCategory.LONG_TERM_INVESTMENT,
            expected_benefit="Minor improvement",
            impact_score=3.0,
            confidence=0.5,
            estimated_effort="low",
            time_estimate="1 month",
            rationale="Nice to have",
        )

        suite = RecommendationSuite(
            suite_id="suite-2",
            suite_name="Low Suite",
            description="Suite with low recs",
            recommendations=[rec],
            total_impact_score=3.0,
            average_confidence=0.5,
            estimated_total_effort="low",
        )

        assert suite.suite_priority == RecommendationPriority.LOW

    def test_report_overall_priority_score(self):
        from services.intelligent_test_recommendation import TestRecommendationReport

        report = TestRecommendationReport(
            report_id="report-1",
            total_recommendations=0,
        )
        assert report.overall_priority_score == 0.0

    def test_report_datetime_serializer(self):
        from services.intelligent_test_recommendation import TestRecommendationReport

        report = TestRecommendationReport(
            report_id="report-1",
            total_recommendations=0,
        )
        data = report.model_dump()
        assert isinstance(data["generated_at"], str)


# ===================================================================
# Engine initialization
# ===================================================================


class TestEngineInit:
    def test_default_config(self):
        engine = _build_engine()
        assert engine.config["high_impact_threshold"] == 7.0
        assert engine.config["min_confidence"] == 0.6

    def test_custom_config(self):
        config = {"min_confidence": 0.3, "high_impact_threshold": 5.0}
        engine = _build_engine(config=config)
        assert engine.config["min_confidence"] == 0.3


# ===================================================================
# generate_recommendations – full flow
# ===================================================================


class TestGenerateRecommendations:
    @pytest.mark.asyncio
    async def test_with_all_inputs(self):
        """Full generation with all input sources."""
        engine = _build_engine()

        report = await engine.generate_recommendations(
            test_insights=_make_test_insights(),
            pattern_analysis=_make_pattern_analysis(),
            performance_analysis=_make_performance_analysis(),
            quality_analysis=_make_quality_analysis(),
            test_results=[_make_test_result(success=False)],
            focus_areas=["security"],
        )

        assert report.total_recommendations > 0
        assert report.report_id is not None

    @pytest.mark.asyncio
    async def test_with_no_inputs(self):
        """Generation with no analysis inputs still produces best practice recs."""
        engine = _build_engine()

        report = await engine.generate_recommendations()
        # Best practice recs should be generated
        assert report.total_recommendations > 0

    @pytest.mark.asyncio
    async def test_focus_areas_none(self):
        """No focus areas still produces security recommendations by default."""
        engine = _build_engine()
        report = await engine.generate_recommendations(focus_areas=None)
        assert report.total_recommendations > 0

    @pytest.mark.asyncio
    async def test_focus_areas_no_security(self):
        """Non-security focus areas skip security recommendations."""
        engine = _build_engine()
        report = await engine.generate_recommendations(focus_areas=["performance"])
        # Should still have automation best practices
        assert report.total_recommendations > 0


# ===================================================================
# Individual generators
# ===================================================================


class TestIndividualGenerators:
    @pytest.mark.asyncio
    async def test_from_test_insights_with_issues(self):
        engine = _build_engine()
        insights = _make_test_insights()

        recs = await engine._generate_from_test_insights(insights)
        # Should have performance, reliability, and coverage recs
        assert len(recs) == 3

    @pytest.mark.asyncio
    async def test_from_test_insights_no_issues(self):
        engine = _build_engine()
        insights = _make_test_insights_no_issues()

        recs = await engine._generate_from_test_insights(insights)
        assert len(recs) == 0

    @pytest.mark.asyncio
    async def test_from_pattern_analysis(self):
        engine = _build_engine()
        patterns = _make_pattern_analysis()

        recs = await engine._generate_from_pattern_analysis(patterns)
        assert len(recs) == 1
        assert recs[0].priority == "critical"

    @pytest.mark.asyncio
    async def test_from_pattern_analysis_no_critical(self):
        engine = _build_engine()
        patterns = MagicMock()
        patterns.critical_patterns = []

        recs = await engine._generate_from_pattern_analysis(patterns)
        assert len(recs) == 0

    @pytest.mark.asyncio
    async def test_from_performance_analysis(self):
        engine = _build_engine()
        perf = _make_performance_analysis()

        recs = await engine._generate_from_performance_analysis(perf)
        assert len(recs) == 1

    @pytest.mark.asyncio
    async def test_from_performance_analysis_medium_priority(self):
        engine = _build_engine()
        perf = _make_performance_analysis()
        perf.optimization_recommendations[0].priority = "medium"

        recs = await engine._generate_from_performance_analysis(perf)
        assert len(recs) == 1
        assert recs[0].priority == "medium"

    @pytest.mark.asyncio
    async def test_from_quality_analysis(self):
        engine = _build_engine()
        quality = _make_quality_analysis()

        recs = await engine._generate_from_quality_analysis(quality)
        assert len(recs) == 1

    @pytest.mark.asyncio
    async def test_from_quality_analysis_non_critical(self):
        engine = _build_engine()
        quality = _make_quality_analysis()
        quality.recommendations[0].priority = "high"

        recs = await engine._generate_from_quality_analysis(quality)
        assert len(recs) == 1
        assert recs[0].priority == "high"

    @pytest.mark.asyncio
    async def test_from_test_results_with_failures(self):
        engine = _build_engine()
        results = [
            _make_test_result(success=False, test_id="t1"),
            _make_test_result(success=True, test_id="t2"),
        ]

        recs = await engine._generate_from_test_results(results)
        assert len(recs) == 1

    @pytest.mark.asyncio
    async def test_from_test_results_no_failures(self):
        engine = _build_engine()
        results = [_make_test_result(success=True)]

        recs = await engine._generate_from_test_results(results)
        assert len(recs) == 0

    @pytest.mark.asyncio
    async def test_best_practices_default(self):
        engine = _build_engine()
        recs = await engine._generate_best_practices()
        assert len(recs) == 2  # automation + security

    @pytest.mark.asyncio
    async def test_best_practices_no_security_focus(self):
        engine = _build_engine()
        recs = await engine._generate_best_practices(focus_areas=["performance"])
        assert len(recs) == 1  # only automation


# ===================================================================
# Deduplication
# ===================================================================


class TestDeduplication:
    @pytest.mark.asyncio
    async def test_dedup_removes_duplicates(self):
        from services.intelligent_test_recommendation import (
            RecommendationCategory,
            RecommendationPriority,
            RecommendationType,
            TestRecommendation,
        )

        engine = _build_engine()

        rec1 = TestRecommendation(
            recommendation_id="dup-1",
            title="Duplicate Rec",
            description="First instance",
            type=RecommendationType.TEST_OPTIMIZATION,
            priority=RecommendationPriority.HIGH,
            category=RecommendationCategory.STRATEGIC_IMPROVEMENT,
            expected_benefit="Improvement",
            impact_score=7.0,
            confidence=0.8,
            estimated_effort="medium",
            time_estimate="2 weeks",
            rationale="Important",
        )
        rec2 = TestRecommendation(
            recommendation_id="dup-1",
            title="Duplicate Rec",
            description="Second instance",
            type=RecommendationType.TEST_OPTIMIZATION,
            priority=RecommendationPriority.HIGH,
            category=RecommendationCategory.STRATEGIC_IMPROVEMENT,
            expected_benefit="Improvement",
            impact_score=7.0,
            confidence=0.8,
            estimated_effort="medium",
            time_estimate="2 weeks",
            rationale="Important",
        )

        result = await engine._deduplicate_recommendations([rec1, rec2])
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_dedup_keeps_unique(self):
        from services.intelligent_test_recommendation import (
            RecommendationCategory,
            RecommendationPriority,
            RecommendationType,
            TestRecommendation,
        )

        engine = _build_engine()

        rec1 = TestRecommendation(
            recommendation_id="unique-1",
            title="Unique Rec 1",
            description="First unique",
            type=RecommendationType.TEST_OPTIMIZATION,
            priority=RecommendationPriority.HIGH,
            category=RecommendationCategory.STRATEGIC_IMPROVEMENT,
            expected_benefit="Improvement",
            impact_score=7.0,
            confidence=0.8,
            estimated_effort="medium",
            time_estimate="2 weeks",
            rationale="Important",
        )
        rec2 = TestRecommendation(
            recommendation_id="unique-2",
            title="Unique Rec 2",
            description="Second unique",
            type=RecommendationType.COVERAGE_IMPROVEMENT,
            priority=RecommendationPriority.MEDIUM,
            category=RecommendationCategory.STRATEGIC_IMPROVEMENT,
            expected_benefit="Coverage",
            impact_score=6.0,
            confidence=0.7,
            estimated_effort="low",
            time_estimate="1 week",
            rationale="Coverage matters",
        )

        result = await engine._deduplicate_recommendations([rec1, rec2])
        assert len(result) == 2


# ===================================================================
# Suite creation
# ===================================================================


class TestSuiteCreation:
    @pytest.mark.asyncio
    async def test_creates_suites_for_multi_type(self):
        from services.intelligent_test_recommendation import (
            RecommendationCategory,
            RecommendationPriority,
            RecommendationType,
            TestRecommendation,
        )

        engine = _build_engine()

        recs = [
            TestRecommendation(
                recommendation_id=f"opt-{i}",
                title=f"Optimization {i}",
                description=f"Optimization recommendation {i}",
                type=RecommendationType.TEST_OPTIMIZATION,
                priority=RecommendationPriority.HIGH,
                category=RecommendationCategory.STRATEGIC_IMPROVEMENT,
                expected_benefit="Faster",
                impact_score=7.0,
                confidence=0.8,
                estimated_effort="medium",
                time_estimate="2 weeks",
                rationale="Speed",
            )
            for i in range(3)
        ]

        suites = await engine._create_recommendation_suites(recs)
        assert len(suites) == 1
        assert len(suites[0].recommendations) == 3

    @pytest.mark.asyncio
    async def test_no_suite_for_single_type(self):
        from services.intelligent_test_recommendation import (
            RecommendationCategory,
            RecommendationPriority,
            RecommendationType,
            TestRecommendation,
        )

        engine = _build_engine()

        recs = [
            TestRecommendation(
                recommendation_id="single-1",
                title="Single Rec",
                description="Only one of this type",
                type=RecommendationType.SECURITY_TESTING,
                priority=RecommendationPriority.HIGH,
                category=RecommendationCategory.STRATEGIC_IMPROVEMENT,
                expected_benefit="Security",
                impact_score=8.0,
                confidence=0.85,
                estimated_effort="high",
                time_estimate="4 weeks",
                rationale="Security matters",
            )
        ]

        suites = await engine._create_recommendation_suites(recs)
        assert len(suites) == 0


# ===================================================================
# Roadmap and ROI
# ===================================================================


class TestRoadmapAndROI:
    @pytest.mark.asyncio
    async def test_implementation_roadmap(self):
        from services.intelligent_test_recommendation import (
            RecommendationCategory,
            RecommendationPriority,
            RecommendationType,
            TestRecommendation,
        )

        engine = _build_engine()

        recs = [
            TestRecommendation(
                recommendation_id="critical-1",
                title="Critical Fix",
                description="Fix critical issue",
                type=RecommendationType.TEST_REFACTORING,
                priority=RecommendationPriority.CRITICAL,
                category=RecommendationCategory.IMMEDIATE_ACTION,
                expected_benefit="Stability",
                impact_score=9.0,
                confidence=0.95,
                estimated_effort="medium",
                time_estimate="1 week",
                rationale="Critical",
            ),
            TestRecommendation(
                recommendation_id="high-1",
                title="High Priority Quick Win",
                description="Quick win recommendation",
                type=RecommendationType.TEST_OPTIMIZATION,
                priority=RecommendationPriority.HIGH,
                category=RecommendationCategory.STRATEGIC_IMPROVEMENT,
                expected_benefit="Speed",
                impact_score=7.0,
                confidence=0.8,
                estimated_effort="low",
                time_estimate="3 days",
                rationale="Quick win",
            ),
            TestRecommendation(
                recommendation_id="high-2",
                title="High Priority Medium Effort",
                description="Medium effort recommendation",
                type=RecommendationType.TEST_OPTIMIZATION,
                priority=RecommendationPriority.HIGH,
                category=RecommendationCategory.STRATEGIC_IMPROVEMENT,
                expected_benefit="Quality",
                impact_score=7.0,
                confidence=0.8,
                estimated_effort="medium",
                time_estimate="2 weeks",
                rationale="Medium effort",
            ),
            TestRecommendation(
                recommendation_id="low-1",
                title="Low Priority",
                description="Low priority recommendation",
                type=RecommendationType.AUTOMATION_IMPROVEMENT,
                priority=RecommendationPriority.LOW,
                category=RecommendationCategory.LONG_TERM_INVESTMENT,
                expected_benefit="Automation",
                impact_score=4.0,
                confidence=0.6,
                estimated_effort="high",
                time_estimate="2 months",
                rationale="Nice to have",
            ),
        ]

        short, medium, long = await engine._generate_implementation_roadmap(recs)
        assert "Critical Fix" in short
        assert "High Priority Quick Win" in short
        assert "High Priority Medium Effort" in medium
        assert "Low Priority" in long

    @pytest.mark.asyncio
    async def test_estimate_benefits_and_roi(self):
        engine = _build_engine()
        recs = []  # Empty is fine

        benefits, cost_benefit = await engine._estimate_benefits_and_roi(recs)
        assert "improved_quality" in benefits
        assert cost_benefit["total_recommendations"] == 0


# ===================================================================
# Factory function
# ===================================================================


class TestFactory:
    def test_create_engine(self):
        from services.intelligent_test_recommendation import (
            create_intelligent_test_recommendation_engine,
        )

        engine = create_intelligent_test_recommendation_engine()
        assert engine is not None

    def test_create_engine_with_config(self):
        from services.intelligent_test_recommendation import (
            create_intelligent_test_recommendation_engine,
        )

        engine = create_intelligent_test_recommendation_engine(config={"min_confidence": 0.5})
        assert engine.config["min_confidence"] == 0.5
