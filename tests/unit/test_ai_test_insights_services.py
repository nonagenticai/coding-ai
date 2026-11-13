from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import services.ai_test_insights.core_service as core_service_module
from models.testing_models import QualityIssue, QualityResult, TestResult
from services.ai_test_insights.config import (
    DEFAULT_CONFIG,
    AITestInsightsConfig,
    AnalysisResultHandler,
    TestInsightType,
    calculate_health_score,
    calculate_improvement_potential,
    clone_config,
)
from services.ai_test_insights.core_service import AITestInsightsService
from services.ai_test_insights.insight_generation import TestInsightGenerator
from services.ai_test_insights.models import (
    TestCoverageInsights,
    TestInsightAnalysis,
    TestInsightRecommendation,
    TestPatternInsight,
    TestPerformanceMetrics,
    TestQualityInsights,
    TestReliabilityMetrics,
)
from services.ai_test_insights.report_formatter import TestReportFormatter


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


def _quality_result(
    *,
    overall_score: float = 80.0,
    issues: list[QualityIssue] | None = None,
) -> QualityResult:
    return QualityResult(overall_score=overall_score, issues=issues or [])


def _analysis() -> TestInsightAnalysis:
    return TestInsightAnalysis(
        analysis_id="analysis-1",
        analyzed_period="recent",
        performance_metrics=TestPerformanceMetrics(
            average_execution_time=4.0,
            median_execution_time=3.0,
            p95_execution_time=6.0,
            p99_execution_time=8.0,
            average_memory_usage=256.0,
            peak_memory_usage=384.0,
            average_cpu_usage=40.0,
            slow_tests_count=2,
            fast_tests_count=8,
            performance_trend="stable",
            bottleneck_categories=["execution_time"],
        ),
        reliability_metrics=TestReliabilityMetrics(
            overall_success_rate=0.75,
            success_rate_by_type={"python": 0.75},
            common_failure_patterns=["timeout"],
            flaky_tests_count=3,
            stability_score=72.0,
            reliability_trend="declining",
            error_categories={"timeout": 2},
        ),
        coverage_insights=TestCoverageInsights(
            line_coverage=72.0,
            branch_coverage=68.0,
            function_coverage=80.0,
            uncovered_critical_paths=["Error handling paths"],
            redundant_tests=[],
            coverage_gaps=["Overall coverage below threshold"],
            recommended_tests=["Add unit tests"],
        ),
        quality_insights=TestQualityInsights(
            test_quality_score=74.0,
            code_quality_score=70.0,
            critical_issues=[
                QualityIssue(message="bad complexity", severity="critical", category="complexity")
            ],
            quality_debt=4.0,
            quality_recommendations=["Refactor complex functions"],
            automated_fixes=["Code formatting issues"],
        ),
        detected_patterns=[
            TestPatternInsight(
                pattern_type="flaky_pattern",
                confidence=0.9,
                description="Flaky retries",
                affected_tests=["test_a"],
                recommendations=["Stabilize fixture usage"],
                impact_score=8.5,
            )
        ],
        recommendations=[],
        overall_health_score=64.0,
        improvement_potential=36.0,
    )


@pytest.mark.asyncio
async def test_test_insight_generator_analyzes_core_metrics_and_recommendations():
    generator = TestInsightGenerator(AITestInsightsConfig(slow_test_threshold=3.0))
    data = {
        "execution_times": [1.0, 2.0, 4.5] + ([8.0] * 18),
        "avg_execution_time": 6.0,
        "median_execution_time": 8.0,
        "p95_execution_time": 8.0,
        "p99_execution_time": 8.0,
        "overall_success_rate": 0.7,
        "total_tests": 60,
        "error_messages": [
            "timeout while waiting",
            "assertion failed",
            "network connection reset",
            "auth denied",
            "OOM killed",
        ],
        "results_by_language": {
            "python": [_test_result(success=True), _test_result(success=False)],
            "typescript": [_test_result(language="typescript", success=False)],
        },
        "quality_scores": [60.0, 80.0],
        "quality_issues": [
            QualityIssue(message="complex", severity="critical", category="complexity"),
            QualityIssue(message="duplicate", severity="warning", category="duplication"),
            QualityIssue(message="style", severity="info", category="style"),
        ],
    }

    performance = await generator.analyze_performance(data)
    reliability = await generator.analyze_reliability(data)
    coverage = await generator.analyze_coverage(data)
    quality = await generator.analyze_quality(data)
    recommendations = await generator.generate_recommendations(
        performance,
        reliability,
        coverage,
        quality,
        [
            TestPatternInsight(
                pattern_type="flaky_pattern",
                confidence=0.9,
                description="high impact",
                affected_tests=["t1"],
                recommendations=["fix"],
                impact_score=9.0,
            )
        ],
    )

    assert performance.slow_tests_count == 19
    assert performance.performance_trend == "stable"
    assert "multi_language_overhead" in performance.bottleneck_categories

    assert reliability.flaky_tests_count == 5
    assert reliability.error_categories["timeout"] == 1
    assert reliability.error_categories["connection"] == 1
    assert reliability.reliability_trend == "degrading"

    assert coverage.line_coverage > 80
    assert "Business logic validation coverage" in coverage.coverage_gaps
    assert "External service integration coverage" in coverage.coverage_gaps
    assert coverage.redundant_tests == ["Tests with 8.0s execution time pattern"]

    assert quality.quality_debt == 4.75
    assert "Refactor complex functions" in quality.quality_recommendations
    assert "Remove code duplication" in quality.quality_recommendations
    assert "Fix code style issues" in quality.quality_recommendations

    assert {rec.recommendation_id for rec in recommendations} == {
        "perf_001",
        "rel_001",
        "qual_001",
        "pattern_001",
    }


@pytest.mark.asyncio
async def test_report_formatter_prepares_analysis_data_and_supports_cache(monkeypatch):
    cache_manager = SimpleNamespace(set=AsyncMock(), get=AsyncMock())
    formatter = TestReportFormatter(cache_manager=cache_manager)
    test_results = [
        _test_result(language="python", success=True, execution_time=2.0),
        _test_result(language="python", success=False, execution_time=4.0, stderr="timeout"),
        _test_result(language="typescript", success=True, execution_time=6.0),
    ]
    quality_results = [
        _quality_result(
            overall_score=82.0,
            issues=[QualityIssue(message="issue", severity="warning", category="style")],
        )
    ]

    data = await formatter._prepare_analysis_data(test_results, quality_results)
    assert data["avg_execution_time"] == 4.0
    assert data["median_execution_time"] == 4.0
    assert data["p95_execution_time"] == 6.0
    assert data["overall_success_rate"] == pytest.approx(2 / 3)
    assert set(data["results_by_language"]) == {"python", "typescript"}

    analysis = _analysis()
    await formatter._cache_analysis_results("analysis-1", analysis)
    cache_manager.set.assert_awaited()

    cache_manager.get.return_value = {
        key: getattr(analysis, key) for key in analysis.__class__.model_fields
    }
    cached = await formatter.get_cached_analysis("analysis-1")
    assert cached is not None
    assert cached.analysis_id == "analysis-1"

    cache_manager.get.return_value = {"report": "ok"}
    assert await formatter.get_cached_comprehensive_analysis("analysis-1") == {"report": "ok"}


@pytest.mark.asyncio
async def test_report_formatter_creates_analysis_and_report(monkeypatch):
    formatter = TestReportFormatter()
    analysis = _analysis()
    recommendations = [
        TestInsightRecommendation(
            recommendation_id="rec-1",
            type=TestInsightType.RECOMMENDATIONS,
            priority="high",
            confidence=0.8,
            title="fix failures",
            description="desc",
            rationale="why",
            implementation_steps=["step"],
            estimated_effort="medium",
            expected_impact="high",
            supporting_metrics={},
            affected_components=["tests"],
        ),
    ]

    formatter.insight_generator.analyze_performance = AsyncMock(
        return_value=analysis.performance_metrics
    )
    formatter.insight_generator.analyze_reliability = AsyncMock(
        return_value=analysis.reliability_metrics
    )
    formatter.insight_generator.analyze_coverage = AsyncMock(
        return_value=analysis.coverage_insights
    )
    formatter.insight_generator.analyze_quality = AsyncMock(return_value=analysis.quality_insights)
    formatter.pattern_detector.detect_patterns = AsyncMock(return_value=analysis.detected_patterns)
    formatter.insight_generator.generate_recommendations = AsyncMock(
        return_value=analysis.recommendations
    )

    created = await formatter.create_comprehensive_analysis(
        test_results=[_test_result()],
        quality_results=[_quality_result()],
    )
    assert created.health_grade == "C"
    assert created.detected_patterns[0].impact_score == 8.5

    specialized = {
        "pattern_analyzer": SimpleNamespace(
            analyze_patterns=AsyncMock(
                return_value=SimpleNamespace(model_dump=lambda: {"overall_pattern_health": 88.0})
            )
        ),
        "performance_analyzer": SimpleNamespace(
            analyze_performance=AsyncMock(
                return_value=SimpleNamespace(model_dump=lambda: {"performance_score": 77.0})
            )
        ),
        "quality_analyzer": SimpleNamespace(
            analyze_quality_metrics=AsyncMock(
                return_value=SimpleNamespace(model_dump=lambda: {"overall_quality_score": 81.0})
            )
        ),
        "failure_prediction_service": SimpleNamespace(
            predict_failures=AsyncMock(return_value={"overall_risk_level": "medium"})
        ),
        "recommendation_engine": SimpleNamespace(
            generate_recommendations=AsyncMock(
                return_value=SimpleNamespace(
                    model_dump=lambda: {"total_recommendations": 2},
                    total_recommendations=2,
                    critical_recommendations=["fix"],
                )
            )
        ),
    }
    monkeypatch.setattr(
        formatter,
        "create_comprehensive_analysis",
        AsyncMock(return_value=analysis.model_copy(update={"recommendations": recommendations})),
    )

    report = await formatter.generate_comprehensive_report(
        test_results=[_test_result()],
        quality_results=[_quality_result()],
        code_changes=["src/app.py"],
        specialized_analyzers=specialized,
    )

    assert report["specialized_analyses"]["pattern_analysis"]["overall_pattern_health"] == 88.0
    assert report["summary"]["specialized_performance_score"] == 77.0
    assert report["summary"]["specialized_quality_score"] == 81.0
    assert report["summary"]["failure_risk"] == "medium"
    assert report["summary"]["total_comprehensive_recommendations"] == 2


@pytest.mark.asyncio
async def test_report_formatter_summary_markdown_and_error_handling():
    formatter = TestReportFormatter()
    analysis = _analysis().model_copy(
        update={
            "recommendations": [
                SimpleNamespace(
                    recommendation_id="rec-1",
                    title="Improve coverage",
                    priority="high",
                    confidence=0.8,
                    expected_impact="high",
                )
            ]
        }
    )

    summary = await formatter.format_analysis_summary(analysis)
    markdown = await formatter.generate_markdown_report(analysis)

    assert summary["health_overview"]["health_grade"] == "D"
    assert summary["critical_issues"]["high_impact_patterns"] == 1
    assert "# Test Insights Analysis Report" in markdown
    assert "| Performance |" in markdown

    failed = formatter._handle_result(RuntimeError("boom"), "performance", TestPerformanceMetrics)
    assert failed.performance_trend == "unknown"
    assert formatter._handle_result(RuntimeError("boom"), "patterns", list) == []


@pytest.mark.asyncio
async def test_ai_test_insights_service_predictions_recommendations_and_config_regression():
    service = AITestInsightsService(config=DEFAULT_CONFIG)

    predictions = await service.predict_test_failures(
        [
            _test_result(language="python", success=True, test_type="unit"),
            _test_result(language="python", success=False, test_type="unit"),
            _test_result(language="python", success=False, test_type="unit"),
            _test_result(language="go", success=True, test_type="integration"),
        ]
    )
    assert predictions["python_unit"] > predictions["go_integration"]

    analysis = _analysis()
    recommendations = await service.generate_test_recommendations(analysis)
    assert len(recommendations) == 2
    assert {rec.type for rec in recommendations} == {
        TestInsightType.RECOMMENDATIONS,
        TestInsightType.PATTERNS,
    }

    original_default = clone_config(DEFAULT_CONFIG)
    service.update_config(enable_ai_features=False, slow_test_threshold=123.0, unknown_value=True)
    new_service = AITestInsightsService()
    assert service.performance_agent is None
    assert new_service.config.slow_test_threshold == original_default.slow_test_threshold
    assert DEFAULT_CONFIG.slow_test_threshold == original_default.slow_test_threshold


def test_config_helpers_and_analysis_result_handler_defaults():
    cloned = clone_config(DEFAULT_CONFIG)
    cloned.slow_test_threshold = 999
    assert DEFAULT_CONFIG.slow_test_threshold != cloned.slow_test_threshold

    assert calculate_health_score(90, 80, 70, 60) == pytest.approx(76.5)
    assert calculate_improvement_potential(60, 70, 50, 80, pattern_impact=50) == 55.5

    perf_default = AnalysisResultHandler.handle_analysis_result(
        RuntimeError("bad"),
        "performance",
        TestPerformanceMetrics,
        core_service_module.logger,
    )
    assert perf_default.performance_score == 100.0
    assert AnalysisResultHandler._get_default_instance(list) == []
