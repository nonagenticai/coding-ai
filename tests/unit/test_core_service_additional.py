"""Additional tests for ai_test_insights/core_service.py to improve coverage.

Tests exercise real service logic wherever possible:
- Agent initialization with different config states
- _analyze_with_llm stub behavior
- predict_test_failures with real failure rate calculations
- generate_test_recommendations with real threshold logic
- Configuration management
- Factory function
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from models.testing_models import QualityResult, TestResult
from services.ai_test_insights.config import (
    AITestInsightsConfig,
)
from services.ai_test_insights.models import (
    TestCoverageInsights,
    TestInsightAnalysis,
    TestPatternInsight,
    TestPerformanceMetrics,
    TestQualityInsights,
    TestReliabilityMetrics,
)


def _test_result(
    *,
    language: str = "python",
    success: bool = True,
    execution_time: float = 1.0,
    stderr: str = "",
    test_type: str | None = "unit",
) -> TestResult:
    return TestResult(
        test_id=f"{language}-{execution_time}",
        language=language,
        test_type=test_type,
        success=success,
        execution_time=execution_time,
        stderr=stderr,
        created_at=datetime.now(UTC),
    )


def _quality_result(overall_score: float = 80.0) -> QualityResult:
    return QualityResult(overall_score=overall_score, issues=[])


def _analysis(health_score: float = 85.0) -> TestInsightAnalysis:
    return TestInsightAnalysis(
        analysis_id="analysis-1",
        analyzed_period="recent",
        performance_metrics=TestPerformanceMetrics(
            average_execution_time=2.0,
            median_execution_time=1.5,
            p95_execution_time=4.0,
            p99_execution_time=5.0,
            average_memory_usage=200.0,
            peak_memory_usage=300.0,
            average_cpu_usage=30.0,
            slow_tests_count=1,
            fast_tests_count=9,
            performance_trend="stable",
            bottleneck_categories=[],
        ),
        reliability_metrics=TestReliabilityMetrics(
            overall_success_rate=0.9,
            success_rate_by_type={"python": 0.9},
            common_failure_patterns=[],
            flaky_tests_count=0,
            stability_score=90.0,
            reliability_trend="stable",
            error_categories={},
        ),
        coverage_insights=TestCoverageInsights(
            line_coverage=85.0,
            branch_coverage=80.0,
            function_coverage=90.0,
            uncovered_critical_paths=[],
            redundant_tests=[],
            coverage_gaps=[],
            recommended_tests=[],
        ),
        quality_insights=TestQualityInsights(
            test_quality_score=85.0,
            code_quality_score=80.0,
            critical_issues=[],
            quality_debt=1.0,
            quality_recommendations=[],
            automated_fixes=[],
        ),
        detected_patterns=[],
        recommendations=[],
        overall_health_score=health_score,
        improvement_potential=15.0,
    )


# ===================================================================
# Agent initialization
# ===================================================================


class TestAIAgentInit:
    def test_agents_init_with_ai_disabled(self):
        """When AI features are disabled, all agents should be None."""
        config = AITestInsightsConfig(enable_ai_features=False)
        from services.ai_test_insights.core_service import AITestInsightsService

        service = AITestInsightsService(config=config)
        assert service.performance_agent is None
        assert service.reliability_agent is None
        assert service.coverage_agent is None
        assert service.quality_agent is None
        assert service.pattern_agent is None
        assert service.recommendation_agent is None

    def test_agents_init_with_pydantic_ai_import_error(self):
        """When pydantic_ai import fails, agents should be None."""
        config = AITestInsightsConfig(enable_ai_features=True)

        with patch("services.ai_test_insights.core_service.HAS_PYDANTIC_AI", False):
            from services.ai_test_insights.core_service import AITestInsightsService

            service = AITestInsightsService(config=config)

        assert service.performance_agent is None
        assert service.recommendation_agent is None

    def test_agents_init_exception_sets_none(self):
        """When agent creation raises, all agents are set to None."""
        config = AITestInsightsConfig(enable_ai_features=True)

        with patch("services.ai_test_insights.core_service.HAS_PYDANTIC_AI", True):
            with patch(
                "services.ai_test_insights.core_service.Agent",
                side_effect=RuntimeError("agent init failed"),
            ):
                from services.ai_test_insights.core_service import (
                    AITestInsightsService,
                )

                service = AITestInsightsService(config=config)

        assert service.performance_agent is None
        assert service.quality_agent is None


# ===================================================================
# _analyze_with_llm
# ===================================================================


class TestAnalyzeWithLLM:
    @pytest.mark.asyncio
    async def test_returns_none_when_ml_backend_disabled(self):
        """Returns None when use_ml_backend is False."""
        config = AITestInsightsConfig(use_ml_backend=False, enable_ai_features=True)
        from services.ai_test_insights.core_service import AITestInsightsService

        service = AITestInsightsService(config=config)
        result = await service._analyze_with_llm("content", "prompt")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_ai_disabled(self):
        """Returns None when enable_ai_features is False."""
        config = AITestInsightsConfig(enable_ai_features=False, use_ml_backend=True)
        from services.ai_test_insights.core_service import AITestInsightsService

        service = AITestInsightsService(config=config)
        result = await service._analyze_with_llm("content", "prompt")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_both_enabled_stub(self):
        """Even when both enabled, the TODO stub returns None."""
        config = AITestInsightsConfig(use_ml_backend=True, enable_ai_features=True)
        from services.ai_test_insights.core_service import AITestInsightsService

        service = AITestInsightsService(config=config)
        result = await service._analyze_with_llm("content", "prompt")
        assert result is None


# ===================================================================
# analyze_test_insights - delegates to report_formatter
# ===================================================================


class TestAnalyzeTestInsights:
    @pytest.mark.asyncio
    async def test_delegates_to_report_formatter(self):
        """Verifies the service passes all arguments through to report_formatter."""
        from services.ai_test_insights.core_service import AITestInsightsService

        service = AITestInsightsService()
        expected = _analysis()

        service.report_formatter.create_comprehensive_analysis = AsyncMock(return_value=expected)

        test_results = [_test_result()]
        quality_results = [_quality_result()]

        result = await service.analyze_test_insights(
            test_results=test_results,
            quality_results=quality_results,
            time_period="last_week",
            include_predictions=False,
        )

        # Verify the correct arguments were passed through
        service.report_formatter.create_comprehensive_analysis.assert_awaited_once_with(
            test_results=test_results,
            quality_results=quality_results,
            time_period="last_week",
            include_predictions=False,
        )
        assert result.analysis_id == "analysis-1"

    @pytest.mark.asyncio
    async def test_raises_on_error(self):
        from services.ai_test_insights.core_service import AITestInsightsService

        service = AITestInsightsService()
        service.report_formatter.create_comprehensive_analysis = AsyncMock(
            side_effect=RuntimeError("boom")
        )

        with pytest.raises(RuntimeError, match="boom"):
            await service.analyze_test_insights(
                test_results=[_test_result()],
                quality_results=[_quality_result()],
            )


# ===================================================================
# comprehensive_analysis - delegates to report_formatter
# ===================================================================


class TestComprehensiveAnalysis:
    @pytest.mark.asyncio
    async def test_delegates_to_report_formatter(self):
        """Verifies all parameters are forwarded to report_formatter."""
        from services.ai_test_insights.core_service import AITestInsightsService

        service = AITestInsightsService()
        expected = {"summary": "comprehensive"}

        service.report_formatter.generate_comprehensive_report = AsyncMock(return_value=expected)

        test_results = [_test_result()]
        quality_results = [_quality_result()]
        code_changes = ["file.py"]
        analyzers = {"analyzer": "mock"}

        result = await service.comprehensive_analysis(
            test_results=test_results,
            quality_results=quality_results,
            code_changes=code_changes,
            time_period="recent",
            include_predictions=True,
            specialized_analyzers=analyzers,
        )

        service.report_formatter.generate_comprehensive_report.assert_awaited_once_with(
            test_results=test_results,
            quality_results=quality_results,
            code_changes=code_changes,
            time_period="recent",
            include_predictions=True,
            specialized_analyzers=analyzers,
        )
        assert result == expected

    @pytest.mark.asyncio
    async def test_raises_on_error(self):
        from services.ai_test_insights.core_service import AITestInsightsService

        service = AITestInsightsService()
        service.report_formatter.generate_comprehensive_report = AsyncMock(
            side_effect=RuntimeError("boom")
        )

        with pytest.raises(RuntimeError, match="boom"):
            await service.comprehensive_analysis(
                test_results=[_test_result()],
                quality_results=[_quality_result()],
            )


# ===================================================================
# predict_test_failures - real logic tests
# ===================================================================


class TestPredictTestFailures:
    @pytest.mark.asyncio
    async def test_all_passing_gives_zero_probability(self):
        """100% success rate should yield 0.0 failure probability."""
        from services.ai_test_insights.core_service import AITestInsightsService

        service = AITestInsightsService()
        results = [
            _test_result(language="python", success=True, test_type="unit"),
            _test_result(language="python", success=True, test_type="unit"),
        ]
        predictions = await service.predict_test_failures(results)
        assert predictions["python_unit"] == 0.0

    @pytest.mark.asyncio
    async def test_all_failing_gives_max_probability(self):
        """0% success rate should yield 1.0 failure probability."""
        from services.ai_test_insights.core_service import AITestInsightsService

        service = AITestInsightsService()
        results = [
            _test_result(language="go", success=False, test_type="e2e"),
            _test_result(language="go", success=False, test_type="e2e"),
        ]
        predictions = await service.predict_test_failures(results)
        assert predictions["go_e2e"] == 1.0

    @pytest.mark.asyncio
    async def test_mixed_results_give_intermediate_probability(self):
        """50% success rate should yield intermediate failure probability."""
        from services.ai_test_insights.core_service import AITestInsightsService

        service = AITestInsightsService()
        results = [
            _test_result(language="python", success=True, test_type="unit"),
            _test_result(language="python", success=False, test_type="unit"),
        ]
        predictions = await service.predict_test_failures(results)
        prob = predictions["python_unit"]
        # With 50% historical and 50% recent (same data), weighted:
        # 0.7 * 0.5 + 0.3 * 0.5 = 0.5
        assert prob == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_none_test_type_becomes_unknown(self):
        """None test_type should be grouped as 'unknown'."""
        from services.ai_test_insights.core_service import AITestInsightsService

        service = AITestInsightsService()
        results = [
            _test_result(language="rust", success=True, test_type=None),
        ]
        predictions = await service.predict_test_failures(results)
        assert "rust_unknown" in predictions

    @pytest.mark.asyncio
    async def test_multiple_language_groups_are_separate(self):
        """Different language/type combos get separate predictions."""
        from services.ai_test_insights.core_service import AITestInsightsService

        service = AITestInsightsService()
        results = [
            _test_result(language="python", success=True, test_type="unit"),
            _test_result(language="python", success=False, test_type="unit"),
            _test_result(language="java", success=True, test_type="integration"),
            _test_result(language="java", success=True, test_type="integration"),
        ]
        predictions = await service.predict_test_failures(results)
        assert "python_unit" in predictions
        assert "java_integration" in predictions
        # Python has 50% failure, Java has 0%
        assert predictions["python_unit"] > predictions["java_integration"]
        assert predictions["java_integration"] == 0.0

    @pytest.mark.asyncio
    async def test_recent_trend_weighting(self):
        """Recent failures should influence prediction more via the 0.3 weight factor."""
        from services.ai_test_insights.core_service import AITestInsightsService

        service = AITestInsightsService()
        # 8 passing followed by 2 failing -- recent trend is worse
        results = [
            _test_result(language="python", success=True, test_type="unit") for _ in range(8)
        ] + [_test_result(language="python", success=False, test_type="unit") for _ in range(2)]
        predictions = await service.predict_test_failures(results)
        prob = predictions["python_unit"]
        # Historical: 2/10 = 0.2 failure rate
        # Recent (last 5): 2/5 failing + 3/5 passing = 0.4 failure rate
        # Weighted: 0.7 * 0.2 + 0.3 * 0.4 = 0.14 + 0.12 = 0.26
        assert prob == pytest.approx(0.26)

    @pytest.mark.asyncio
    async def test_probability_clamped_to_01(self):
        """Probabilities are always between 0.0 and 1.0."""
        from services.ai_test_insights.core_service import AITestInsightsService

        service = AITestInsightsService()
        results = [_test_result(language="go", success=False, test_type="e2e")]
        predictions = await service.predict_test_failures(results)
        for prob in predictions.values():
            assert 0.0 <= prob <= 1.0


# ===================================================================
# generate_test_recommendations - real logic tests
# ===================================================================


class TestGenerateRecommendations:
    @pytest.mark.asyncio
    async def test_low_health_score_triggers_overhaul_recommendation(self):
        """Health score below 70 should produce 'Comprehensive Test Suite Overhaul'."""
        from services.ai_test_insights.core_service import AITestInsightsService

        service = AITestInsightsService()
        analysis = _analysis(health_score=50.0)
        recs = await service.generate_test_recommendations(analysis)

        assert len(recs) >= 1
        overhaul_rec = next(r for r in recs if "Overhaul" in r.title)
        assert overhaul_rec.priority == "high"
        assert overhaul_rec.confidence == 0.85
        assert overhaul_rec.supporting_metrics["health_score"] == 50.0
        assert "entire_test_suite" in overhaul_rec.affected_components

    @pytest.mark.asyncio
    async def test_high_health_score_no_overhaul(self):
        """Health score >= 70 should NOT produce the overhaul recommendation."""
        from services.ai_test_insights.core_service import AITestInsightsService

        service = AITestInsightsService()
        analysis = _analysis(health_score=85.0)
        recs = await service.generate_test_recommendations(analysis)

        assert not any("Overhaul" in r.title for r in recs)

    @pytest.mark.asyncio
    async def test_high_impact_patterns_trigger_pattern_recommendation(self):
        """Patterns with impact_score > 7 should trigger pattern recommendation."""
        from services.ai_test_insights.core_service import AITestInsightsService

        service = AITestInsightsService()
        analysis = _analysis(health_score=85.0)
        analysis.detected_patterns = [
            TestPatternInsight(
                pattern_type="reliability_pattern",
                confidence=0.9,
                description="Dangerous anti-pattern",
                affected_tests=["test_a", "test_b"],
                recommendations=["Fix it"],
                impact_score=8.5,
            )
        ]
        recs = await service.generate_test_recommendations(analysis)

        assert len(recs) == 1
        pattern_rec = recs[0]
        assert "High-Impact Test Patterns" in pattern_rec.title
        assert pattern_rec.priority == "high"
        assert pattern_rec.supporting_metrics["high_impact_patterns"] == 1

    @pytest.mark.asyncio
    async def test_low_impact_patterns_no_recommendation(self):
        """Patterns with impact_score <= 7 should NOT produce pattern recommendation."""
        from services.ai_test_insights.core_service import AITestInsightsService

        service = AITestInsightsService()
        analysis = _analysis()
        analysis.detected_patterns = [
            TestPatternInsight(
                pattern_type="unit_pattern",
                confidence=0.5,
                description="Low impact pattern",
                affected_tests=["test_a"],
                recommendations=["Fix"],
                impact_score=3.0,
            )
        ]
        recs = await service.generate_test_recommendations(analysis)
        assert not any("Pattern" in r.title for r in recs)

    @pytest.mark.asyncio
    async def test_both_health_and_pattern_recommendations(self):
        """Low health + high impact patterns should produce both recommendations."""
        from services.ai_test_insights.core_service import AITestInsightsService

        service = AITestInsightsService()
        analysis = _analysis(health_score=50.0)
        analysis.detected_patterns = [
            TestPatternInsight(
                pattern_type="reliability_pattern",
                confidence=0.9,
                description="Bad pattern",
                affected_tests=["test_a"],
                recommendations=["Fix"],
                impact_score=9.0,
            )
        ]
        recs = await service.generate_test_recommendations(analysis)

        titles = [r.title for r in recs]
        assert any("Overhaul" in t for t in titles)
        assert any("Pattern" in t for t in titles)
        assert len(recs) == 2

    @pytest.mark.asyncio
    async def test_recommendations_are_valid_models(self):
        """All returned recommendations should be valid TestInsightRecommendation instances."""
        from services.ai_test_insights.core_service import AITestInsightsService

        service = AITestInsightsService()
        analysis = _analysis(health_score=50.0)
        analysis.detected_patterns = [
            TestPatternInsight(
                pattern_type="reliability_pattern",
                confidence=0.9,
                description="Bad",
                affected_tests=["t1"],
                recommendations=["Fix"],
                impact_score=8.0,
            )
        ]
        recs = await service.generate_test_recommendations(analysis)

        for rec in recs:
            # Verify each is a real Pydantic model, not a mock
            assert rec.recommendation_id is not None
            assert rec.recommendation_id != ""
            assert rec.type is not None
            assert 0.0 <= rec.confidence <= 1.0
            assert len(rec.implementation_steps) > 0
            assert rec.estimated_effort in ("low", "medium", "high")


# ===================================================================
# Caching - delegates correctly
# ===================================================================


class TestCaching:
    @pytest.mark.asyncio
    async def test_get_cached_analysis_returns_none(self):
        from services.ai_test_insights.core_service import AITestInsightsService

        service = AITestInsightsService()
        service.report_formatter.get_cached_analysis = AsyncMock(return_value=None)
        result = await service.get_cached_analysis("id-1")
        assert result is None
        service.report_formatter.get_cached_analysis.assert_awaited_once_with("id-1")

    @pytest.mark.asyncio
    async def test_get_cached_comprehensive_analysis(self):
        from services.ai_test_insights.core_service import AITestInsightsService

        service = AITestInsightsService()
        service.report_formatter.get_cached_comprehensive_analysis = AsyncMock(
            return_value={"cached": True}
        )
        result = await service.get_cached_comprehensive_analysis("id-1")
        assert result == {"cached": True}
        service.report_formatter.get_cached_comprehensive_analysis.assert_awaited_once_with("id-1")


# ===================================================================
# Config management
# ===================================================================


class TestConfigMethods:
    def test_get_config_returns_current(self):
        from services.ai_test_insights.core_service import AITestInsightsService

        config = AITestInsightsConfig(slow_test_threshold=42.0)
        service = AITestInsightsService(config=config)
        returned = service.get_config()
        assert returned.slow_test_threshold == 42.0

    def test_update_config_known_param_reinitializes_components(self):
        """Updating a known param should change config AND reinitialize components."""
        from services.ai_test_insights.core_service import AITestInsightsService

        service = AITestInsightsService()
        old_generator = service.insight_generator
        service.update_config(slow_test_threshold=99.0)
        assert service.config.slow_test_threshold == 99.0
        # Components should be re-created
        assert service.insight_generator is not old_generator

    def test_update_config_unknown_param_warns_no_crash(self):
        from services.ai_test_insights.core_service import AITestInsightsService

        service = AITestInsightsService()
        original_threshold = service.config.slow_test_threshold
        # Should log warning but not crash, and not change known params
        service.update_config(nonexistent_param=True)
        assert service.config.slow_test_threshold == original_threshold


# ===================================================================
# Factory function
# ===================================================================


class TestFactory:
    def test_create_ai_test_insights_service(self):
        from services.ai_test_insights.core_service import (
            AITestInsightsService,
            create_ai_test_insights_service,
        )

        service = create_ai_test_insights_service()
        assert isinstance(service, AITestInsightsService)
        assert service.config is not None

    def test_create_with_config(self):
        from services.ai_test_insights.core_service import (
            create_ai_test_insights_service,
        )

        config = AITestInsightsConfig(slow_test_threshold=55.0)
        service = create_ai_test_insights_service(config=config)
        assert service.config.slow_test_threshold == 55.0
