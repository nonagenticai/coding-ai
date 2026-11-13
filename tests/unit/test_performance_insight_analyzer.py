"""Tests for PerformanceInsightAnalyzer and related models."""

from __future__ import annotations

import pytest

from models.testing_models import TestResult
from services.performance_insight_analyzer import (
    PerformanceAnalysisResult,
    PerformanceBottleneckType,
    PerformanceCategory,
    PerformanceInsight,
    PerformanceInsightAnalyzer,
    PerformanceStatistics,
    PerformanceTrendAnalysis,
    create_performance_insight_analyzer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(
    test_id: str = "t1",
    language: str = "python",
    test_type: str | None = "unit",
    execution_time: float = 1.0,
    success: bool = True,
    stderr: str = "",
) -> TestResult:
    return TestResult(
        test_id=test_id,
        language=language,
        test_type=test_type,
        execution_time=execution_time,
        success=success,
        stderr=stderr,
    )


def _many_results(n: int, base_time: float = 2.0) -> list[TestResult]:
    """Generate n test results with linearly increasing execution times."""
    return [_result(test_id=f"t{i}", execution_time=base_time + i * 0.1) for i in range(n)]


# ---------------------------------------------------------------------------
# Model unit tests
# ---------------------------------------------------------------------------


class TestPerformanceInsightModel:
    def test_priority_score_critical(self):
        insight = PerformanceInsight(
            insight_id="i1",
            insight_name="test",
            category=PerformanceCategory.EXECUTION_TIME,
            severity="critical",
            confidence=1.0,
            impact_score=10.0,
            performance_delta=50.0,
            baseline_metric=10.0,
            target_metric=5.0,
            description="d",
            implementation_complexity="high",
        )
        # 10*0.4 + 10*0.4 + 1.0*10*0.2 = 4+4+2 = 10.0
        assert insight.priority_score == 10.0

    def test_priority_score_low(self):
        insight = PerformanceInsight(
            insight_id="i2",
            insight_name="test",
            category=PerformanceCategory.LATENCY,
            severity="low",
            confidence=0.5,
            impact_score=2.0,
            performance_delta=10.0,
            baseline_metric=5.0,
            target_metric=4.0,
            description="d",
            implementation_complexity="low",
        )
        # 4*0.4 + 2*0.4 + 0.5*10*0.2 = 1.6+0.8+1.0 = 3.4
        assert insight.priority_score == pytest.approx(3.4)

    def test_priority_score_unknown_severity_falls_back(self):
        insight = PerformanceInsight(
            insight_id="i3",
            insight_name="test",
            category=PerformanceCategory.THROUGHPUT,
            severity="unknown",
            confidence=0.5,
            impact_score=5.0,
            performance_delta=10.0,
            baseline_metric=5.0,
            target_metric=4.0,
            description="d",
            implementation_complexity="low",
        )
        # default 4.0*0.4 + 5*0.4 + 0.5*10*0.2 = 1.6+2.0+1.0 = 4.6
        assert insight.priority_score == pytest.approx(4.6)


class TestPerformanceStatisticsModel:
    def test_is_highly_variable_true(self):
        stats = PerformanceStatistics(
            mean=2.0,
            median=2.0,
            std_dev=1.5,
            variance=2.25,
            p50=2.0,
            p75=3.0,
            p90=4.0,
            p95=4.5,
            p99=5.0,
            min_value=0.5,
            max_value=5.0,
            range_value=4.5,
            coefficient_of_variation=0.75,
            skewness=0.0,
            kurtosis=0.0,
        )
        assert stats.is_highly_variable is True

    def test_is_highly_variable_false(self):
        stats = PerformanceStatistics(
            mean=2.0,
            median=2.0,
            std_dev=0.5,
            variance=0.25,
            p50=2.0,
            p75=2.5,
            p90=3.0,
            p95=3.2,
            p99=3.5,
            min_value=1.0,
            max_value=3.5,
            range_value=2.5,
            coefficient_of_variation=0.25,
            skewness=0.0,
            kurtosis=0.0,
        )
        assert stats.is_highly_variable is False

    def test_distribution_type_normal(self):
        stats = PerformanceStatistics(
            mean=2.0,
            median=2.0,
            std_dev=0.5,
            variance=0.25,
            p50=2.0,
            p75=2.5,
            p90=3.0,
            p95=3.2,
            p99=3.5,
            min_value=1.0,
            max_value=3.5,
            range_value=2.5,
            coefficient_of_variation=0.25,
            skewness=0.3,
            kurtosis=0.0,
        )
        assert stats.distribution_type == "normal"

    def test_distribution_type_right_skewed(self):
        stats = PerformanceStatistics(
            mean=2.0,
            median=2.0,
            std_dev=0.5,
            variance=0.25,
            p50=2.0,
            p75=2.5,
            p90=3.0,
            p95=3.2,
            p99=3.5,
            min_value=1.0,
            max_value=3.5,
            range_value=2.5,
            coefficient_of_variation=0.25,
            skewness=1.0,
            kurtosis=0.0,
        )
        assert stats.distribution_type == "right_skewed"

    def test_distribution_type_left_skewed(self):
        stats = PerformanceStatistics(
            mean=2.0,
            median=2.0,
            std_dev=0.5,
            variance=0.25,
            p50=2.0,
            p75=2.5,
            p90=3.0,
            p95=3.2,
            p99=3.5,
            min_value=1.0,
            max_value=3.5,
            range_value=2.5,
            coefficient_of_variation=0.25,
            skewness=-1.0,
            kurtosis=0.0,
        )
        assert stats.distribution_type == "left_skewed"


class TestPerformanceTrendAnalysisModel:
    def test_is_significant_trend_true(self):
        t = PerformanceTrendAnalysis(
            trend_direction="degrading",
            trend_strength=0.9,
            trend_confidence=0.9,
            slope=0.5,
            r_squared=0.85,
            percentage_change=10.0,
            absolute_change=1.0,
            anomaly_count=0,
        )
        assert t.is_significant_trend is True

    def test_is_significant_trend_false_low_r2(self):
        t = PerformanceTrendAnalysis(
            trend_direction="stable",
            trend_strength=0.5,
            trend_confidence=0.9,
            slope=0.01,
            r_squared=0.3,
            percentage_change=1.0,
            absolute_change=0.1,
            anomaly_count=0,
        )
        assert t.is_significant_trend is False


class TestPerformanceAnalysisResultModel:
    def _make_stats(self):
        return PerformanceStatistics(
            mean=2.0,
            median=2.0,
            std_dev=0.5,
            variance=0.25,
            p50=2.0,
            p75=2.5,
            p90=3.0,
            p95=3.2,
            p99=3.5,
            min_value=1.0,
            max_value=3.5,
            range_value=2.5,
            coefficient_of_variation=0.25,
            skewness=0.0,
            kurtosis=0.0,
        )

    def _make_trend(self):
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

    @pytest.mark.parametrize(
        "score,expected",
        [
            (95, "A"),
            (85, "B"),
            (75, "C"),
            (65, "D"),
            (50, "F"),
        ],
    )
    def test_performance_grade(self, score, expected):
        result = PerformanceAnalysisResult(
            analysis_id="a1",
            total_tests_analyzed=10,
            analysis_period="recent",
            overall_statistics=self._make_stats(),
            trend_analysis=self._make_trend(),
            performance_score=score,
            optimization_potential=0.0,
        )
        assert result.performance_grade == expected

    def test_serialize_datetime(self):
        result = PerformanceAnalysisResult(
            analysis_id="a1",
            total_tests_analyzed=10,
            analysis_period="recent",
            overall_statistics=self._make_stats(),
            trend_analysis=self._make_trend(),
            performance_score=80.0,
            optimization_potential=0.0,
        )
        serialized = result.model_dump()
        assert isinstance(serialized["analyzed_at"], str)

    def test_serialize_datetime_none_via_serializer(self):
        """Test the serialize_datetime method directly with None."""
        result = PerformanceAnalysisResult(
            analysis_id="a1",
            total_tests_analyzed=10,
            analysis_period="recent",
            overall_statistics=self._make_stats(),
            trend_analysis=self._make_trend(),
            performance_score=80.0,
            optimization_potential=0.0,
        )
        # Call serializer directly with None to cover that branch
        assert result.serialize_datetime(None) is None


# ---------------------------------------------------------------------------
# Analyzer tests
# ---------------------------------------------------------------------------


class TestPerformanceInsightAnalyzerInit:
    def test_default_config(self):
        analyzer = PerformanceInsightAnalyzer()
        assert "slow_test_threshold" in analyzer.config
        assert analyzer.config["slow_test_threshold"] == 10.0

    def test_custom_config(self):
        cfg = {"slow_test_threshold": 5.0, "min_samples_for_trend": 5}
        analyzer = PerformanceInsightAnalyzer(config=cfg)
        assert analyzer.config["slow_test_threshold"] == 5.0


class TestCalculateStatistics:
    async def test_basic_statistics(self):
        analyzer = PerformanceInsightAnalyzer()
        times = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = await analyzer._calculate_statistics(times)
        assert stats.mean == pytest.approx(3.0)
        assert stats.min_value == 1.0
        assert stats.max_value == 5.0
        assert stats.range_value == 4.0

    async def test_single_element(self):
        analyzer = PerformanceInsightAnalyzer()
        stats = await analyzer._calculate_statistics([5.0])
        assert stats.mean == 5.0
        assert stats.std_dev == 0.0
        assert stats.variance == 0.0
        assert stats.skewness == 0.0
        assert stats.kurtosis == 0.0

    async def test_two_elements(self):
        analyzer = PerformanceInsightAnalyzer()
        stats = await analyzer._calculate_statistics([2.0, 4.0])
        assert stats.mean == pytest.approx(3.0)
        assert stats.std_dev > 0
        assert stats.skewness == 0.0  # n <= 2 branch
        assert stats.kurtosis == 0.0

    async def test_empty_raises(self):
        analyzer = PerformanceInsightAnalyzer()
        with pytest.raises(ValueError, match="No execution times"):
            await analyzer._calculate_statistics([])

    async def test_percentiles(self):
        analyzer = PerformanceInsightAnalyzer()
        times = list(range(1, 101))  # 1..100
        times = [float(t) for t in times]
        stats = await analyzer._calculate_statistics(times)
        assert stats.p50 == pytest.approx(51.0)
        assert stats.p99 == pytest.approx(100.0)

    async def test_cv_zero_mean(self):
        """When mean is 0 the CV should be 0 (edge case)."""
        analyzer = PerformanceInsightAnalyzer()
        stats = await analyzer._calculate_statistics([0.0, 0.0, 0.0])
        assert stats.coefficient_of_variation == 0.0


class TestCalculateStatisticsByLanguage:
    async def test_groups_by_language(self):
        analyzer = PerformanceInsightAnalyzer()
        results = [
            _result(language="python", execution_time=1.0),
            _result(language="python", execution_time=3.0),
            _result(language="javascript", execution_time=2.0),
        ]
        stats = await analyzer._calculate_statistics_by_language(results)
        assert "python" in stats
        assert "javascript" in stats
        assert stats["python"].mean == pytest.approx(2.0)
        assert stats["javascript"].mean == pytest.approx(2.0)


class TestCalculateStatisticsByType:
    async def test_groups_by_type(self):
        analyzer = PerformanceInsightAnalyzer()
        results = [
            _result(test_type="unit", execution_time=1.0),
            _result(test_type="integration", execution_time=5.0),
            _result(test_type=None, execution_time=2.0),
        ]
        stats = await analyzer._calculate_statistics_by_type(results)
        assert "unit" in stats
        assert "integration" in stats
        assert "unknown" in stats


class TestAnalyzeTrends:
    async def test_insufficient_samples_returns_stable(self):
        analyzer = PerformanceInsightAnalyzer()
        results = [_result(execution_time=1.0 + i) for i in range(5)]
        trend = await analyzer._analyze_trends(results)
        assert trend.trend_direction == "stable"
        assert trend.trend_strength == 0.0
        assert trend.trend_confidence == 0.0

    async def test_degrading_trend(self):
        analyzer = PerformanceInsightAnalyzer()
        # Clearly increasing execution times
        results = [_result(test_id=f"t{i}", execution_time=1.0 + i * 2.0) for i in range(15)]
        trend = await analyzer._analyze_trends(results)
        assert trend.trend_direction == "degrading"
        assert trend.slope > 0

    async def test_improving_trend(self):
        analyzer = PerformanceInsightAnalyzer()
        results = [_result(test_id=f"t{i}", execution_time=30.0 - i * 2.0) for i in range(15)]
        trend = await analyzer._analyze_trends(results)
        assert trend.trend_direction == "improving"
        assert trend.slope < 0

    async def test_stable_trend(self):
        analyzer = PerformanceInsightAnalyzer()
        results = [_result(test_id=f"t{i}", execution_time=5.0) for i in range(15)]
        trend = await analyzer._analyze_trends(results)
        assert trend.trend_direction == "stable"

    async def test_anomaly_detection(self):
        analyzer = PerformanceInsightAnalyzer()
        times = [5.0] * 14 + [100.0]  # one outlier
        results = [_result(test_id=f"t{i}", execution_time=times[i]) for i in range(15)]
        trend = await analyzer._analyze_trends(results)
        assert trend.anomaly_count > 0

    async def test_percentage_change_zero_first(self):
        analyzer = PerformanceInsightAnalyzer()
        results = [
            _result(test_id=f"t{i}", execution_time=0.0 if i == 0 else 5.0) for i in range(12)
        ]
        trend = await analyzer._analyze_trends(results)
        assert trend.percentage_change == 0.0


class TestGenerateInsights:
    async def test_slow_test_insight(self):
        analyzer = PerformanceInsightAnalyzer()
        results = [_result(test_id=f"t{i}", execution_time=15.0) for i in range(5)]
        stats = await analyzer._calculate_statistics([r.execution_time for r in results])
        insights = await analyzer._generate_insights(results, stats)
        ids = [i.insight_id for i in insights]
        assert "perf_slow_tests" in ids

    async def test_slow_test_insight_high_severity(self):
        analyzer = PerformanceInsightAnalyzer()
        results = [_result(test_id=f"t{i}", execution_time=15.0) for i in range(15)]
        stats = await analyzer._calculate_statistics([r.execution_time for r in results])
        insights = await analyzer._generate_insights(results, stats)
        slow = next(i for i in insights if i.insight_id == "perf_slow_tests")
        assert slow.severity == "high"

    async def test_no_slow_tests_no_insight(self):
        analyzer = PerformanceInsightAnalyzer()
        results = [_result(test_id=f"t{i}", execution_time=1.0) for i in range(5)]
        stats = await analyzer._calculate_statistics([r.execution_time for r in results])
        insights = await analyzer._generate_insights(results, stats)
        ids = [i.insight_id for i in insights]
        assert "perf_slow_tests" not in ids

    async def test_high_variability_insight(self):
        analyzer = PerformanceInsightAnalyzer()
        times = [1.0, 1.0, 1.0, 50.0, 50.0]
        results = [_result(test_id=f"t{i}", execution_time=times[i]) for i in range(5)]
        stats = await analyzer._calculate_statistics(times)
        insights = await analyzer._generate_insights(results, stats)
        ids = [i.insight_id for i in insights]
        assert "perf_high_variability" in ids

    async def test_very_slow_tests_insight(self):
        analyzer = PerformanceInsightAnalyzer()
        results = [_result(test_id=f"t{i}", execution_time=35.0) for i in range(3)]
        stats = await analyzer._calculate_statistics([r.execution_time for r in results])
        insights = await analyzer._generate_insights(results, stats)
        ids = [i.insight_id for i in insights]
        assert "perf_memory_intensive" in ids

    async def test_language_disparity_insight(self):
        analyzer = PerformanceInsightAnalyzer()
        results = [
            _result(test_id="t0", language="python", execution_time=1.0),
            _result(test_id="t1", language="python", execution_time=1.0),
            _result(test_id="t2", language="java", execution_time=5.0),
            _result(test_id="t3", language="java", execution_time=5.0),
        ]
        stats = await analyzer._calculate_statistics([r.execution_time for r in results])
        insights = await analyzer._generate_insights(results, stats)
        ids = [i.insight_id for i in insights]
        assert "perf_language_disparity" in ids

    async def test_no_language_disparity_when_close(self):
        analyzer = PerformanceInsightAnalyzer()
        results = [
            _result(test_id="t0", language="python", execution_time=2.0),
            _result(test_id="t1", language="java", execution_time=2.5),
        ]
        stats = await analyzer._calculate_statistics([r.execution_time for r in results])
        insights = await analyzer._generate_insights(results, stats)
        ids = [i.insight_id for i in insights]
        assert "perf_language_disparity" not in ids

    async def test_single_language_no_disparity(self):
        analyzer = PerformanceInsightAnalyzer()
        results = [
            _result(test_id=f"t{i}", language="python", execution_time=5.0) for i in range(3)
        ]
        stats = await analyzer._calculate_statistics([r.execution_time for r in results])
        insights = await analyzer._generate_insights(results, stats)
        ids = [i.insight_id for i in insights]
        assert "perf_language_disparity" not in ids


class TestGenerateRecommendations:
    async def test_high_impact_recommendations(self):
        analyzer = PerformanceInsightAnalyzer()
        insight = PerformanceInsight(
            insight_id="perf_memory_intensive",
            insight_name="Memory-Intensive Test Operations",
            category=PerformanceCategory.RESOURCE_USAGE,
            bottleneck_type=PerformanceBottleneckType.MEMORY_INTENSIVE,
            severity="high",
            confidence=0.7,
            impact_score=8.0,
            performance_delta=40.0,
            baseline_metric=35.0,
            target_metric=21.0,
            description="d",
            implementation_complexity="high",
        )
        stats = PerformanceStatistics(
            mean=2.0,
            median=2.0,
            std_dev=0.5,
            variance=0.25,
            p50=2.0,
            p75=2.5,
            p90=3.0,
            p95=3.2,
            p99=3.5,
            min_value=1.0,
            max_value=3.5,
            range_value=2.5,
            coefficient_of_variation=0.25,
            skewness=0.0,
            kurtosis=0.0,
        )
        recs = await analyzer._generate_recommendations([insight], stats)
        assert len(recs) >= 1
        assert any("opt_perf_memory_intensive" == r.recommendation_id for r in recs)

    async def test_general_recommendation_for_slow_mean(self):
        analyzer = PerformanceInsightAnalyzer()
        stats = PerformanceStatistics(
            mean=15.0,
            median=14.0,
            std_dev=2.0,
            variance=4.0,
            p50=14.0,
            p75=15.0,
            p90=16.0,
            p95=17.0,
            p99=18.0,
            min_value=10.0,
            max_value=20.0,
            range_value=10.0,
            coefficient_of_variation=0.13,
            skewness=0.0,
            kurtosis=0.0,
        )
        recs = await analyzer._generate_recommendations([], stats)
        ids = [r.recommendation_id for r in recs]
        assert "opt_general_performance" in ids

    async def test_resource_stability_recommendation(self):
        analyzer = PerformanceInsightAnalyzer()
        stats = PerformanceStatistics(
            mean=5.0,
            median=5.0,
            std_dev=4.0,
            variance=16.0,
            p50=5.0,
            p75=7.0,
            p90=9.0,
            p95=10.0,
            p99=11.0,
            min_value=1.0,
            max_value=12.0,
            range_value=11.0,
            coefficient_of_variation=0.8,
            skewness=0.0,
            kurtosis=0.0,
        )
        recs = await analyzer._generate_recommendations([], stats)
        ids = [r.recommendation_id for r in recs]
        assert "opt_resource_stability" in ids

    async def test_recommendations_sorted_by_priority(self):
        analyzer = PerformanceInsightAnalyzer()
        stats = PerformanceStatistics(
            mean=15.0,
            median=14.0,
            std_dev=10.0,
            variance=100.0,
            p50=14.0,
            p75=20.0,
            p90=25.0,
            p95=28.0,
            p99=30.0,
            min_value=1.0,
            max_value=30.0,
            range_value=29.0,
            coefficient_of_variation=0.67,
            skewness=0.0,
            kurtosis=0.0,
        )
        insight = PerformanceInsight(
            insight_id="x",
            insight_name="X",
            category=PerformanceCategory.EXECUTION_TIME,
            severity="critical",
            confidence=0.9,
            impact_score=9.0,
            performance_delta=50.0,
            baseline_metric=15.0,
            target_metric=5.0,
            description="d",
            implementation_complexity="high",
        )
        recs = await analyzer._generate_recommendations([insight], stats)
        assert len(recs) >= 2
        # First should be highest priority
        priorities = [r.priority for r in recs]
        assert priorities[0] == "high"

    async def test_no_recommendations_for_good_stats(self):
        analyzer = PerformanceInsightAnalyzer()
        stats = PerformanceStatistics(
            mean=2.0,
            median=2.0,
            std_dev=0.3,
            variance=0.09,
            p50=2.0,
            p75=2.2,
            p90=2.5,
            p95=2.6,
            p99=2.8,
            min_value=1.5,
            max_value=3.0,
            range_value=1.5,
            coefficient_of_variation=0.15,
            skewness=0.0,
            kurtosis=0.0,
        )
        recs = await analyzer._generate_recommendations([], stats)
        assert recs == []


class TestCategorizeTests:
    async def test_categorize_slow_fast_variable(self):
        analyzer = PerformanceInsightAnalyzer()
        results = [
            _result(test_id="slow1", execution_time=15.0),
            _result(test_id="fast1", execution_time=0.5),
            _result(test_id="mid1", execution_time=5.0),
            _result(test_id="mid2", execution_time=5.0),
            _result(test_id="mid3", execution_time=5.0),
            _result(test_id="mid4", execution_time=5.0),
            _result(test_id="mid5", execution_time=5.0),
            _result(test_id="outlier", execution_time=50.0),
        ]
        slow, fast, variable = await analyzer._categorize_tests(results)
        assert "slow1" in slow
        assert "fast1" in fast
        # outlier at 50.0 is far from mean (~11.3), should be variable
        assert "outlier" in variable

    async def test_empty_results(self):
        analyzer = PerformanceInsightAnalyzer()
        slow, fast, variable = await analyzer._categorize_tests([])
        assert slow == []
        assert fast == []
        assert variable == []

    async def test_single_result_no_variable(self):
        analyzer = PerformanceInsightAnalyzer()
        results = [_result(test_id="t0", execution_time=15.0)]
        slow, fast, variable = await analyzer._categorize_tests(results)
        assert "t0" in slow
        # std_dev is 0 for single element, so no variable tests
        assert variable == []


class TestCalculatePerformanceScore:
    async def test_perfect_score(self):
        analyzer = PerformanceInsightAnalyzer()
        stats = PerformanceStatistics(
            mean=2.0,
            median=2.0,
            std_dev=0.3,
            variance=0.09,
            p50=2.0,
            p75=2.2,
            p90=2.5,
            p95=2.6,
            p99=2.8,
            min_value=1.5,
            max_value=3.0,
            range_value=1.5,
            coefficient_of_variation=0.15,
            skewness=0.0,
            kurtosis=0.0,
        )
        score = await analyzer._calculate_performance_score(stats, [])
        assert score == 100.0

    async def test_penalized_for_slow_mean(self):
        analyzer = PerformanceInsightAnalyzer()
        stats = PerformanceStatistics(
            mean=15.0,
            median=14.0,
            std_dev=2.0,
            variance=4.0,
            p50=14.0,
            p75=15.0,
            p90=16.0,
            p95=17.0,
            p99=18.0,
            min_value=10.0,
            max_value=20.0,
            range_value=10.0,
            coefficient_of_variation=0.13,
            skewness=0.0,
            kurtosis=0.0,
        )
        score = await analyzer._calculate_performance_score(stats, [])
        assert score < 100.0
        assert score >= 0.0

    async def test_penalized_for_critical_insights(self):
        analyzer = PerformanceInsightAnalyzer()
        stats = PerformanceStatistics(
            mean=2.0,
            median=2.0,
            std_dev=0.3,
            variance=0.09,
            p50=2.0,
            p75=2.2,
            p90=2.5,
            p95=2.6,
            p99=2.8,
            min_value=1.5,
            max_value=3.0,
            range_value=1.5,
            coefficient_of_variation=0.15,
            skewness=0.0,
            kurtosis=0.0,
        )
        critical_insight = PerformanceInsight(
            insight_id="c1",
            insight_name="crit",
            category=PerformanceCategory.EXECUTION_TIME,
            severity="critical",
            confidence=0.9,
            impact_score=9.0,
            performance_delta=50.0,
            baseline_metric=10.0,
            target_metric=5.0,
            description="d",
            implementation_complexity="high",
        )
        score = await analyzer._calculate_performance_score(stats, [critical_insight])
        assert score < 100.0

    async def test_score_clamped_to_zero(self):
        analyzer = PerformanceInsightAnalyzer()
        stats = PerformanceStatistics(
            mean=100.0,
            median=100.0,
            std_dev=80.0,
            variance=6400.0,
            p50=100.0,
            p75=150.0,
            p90=180.0,
            p95=190.0,
            p99=200.0,
            min_value=10.0,
            max_value=200.0,
            range_value=190.0,
            coefficient_of_variation=0.8,
            skewness=0.0,
            kurtosis=0.0,
        )
        critical_insights = [
            PerformanceInsight(
                insight_id=f"c{i}",
                insight_name="crit",
                category=PerformanceCategory.EXECUTION_TIME,
                severity="critical",
                confidence=0.9,
                impact_score=9.0,
                performance_delta=50.0,
                baseline_metric=10.0,
                target_metric=5.0,
                description="d",
                implementation_complexity="high",
            )
            for i in range(5)
        ]
        score = await analyzer._calculate_performance_score(stats, critical_insights)
        assert score >= 0.0


class TestCalculateOptimizationPotential:
    async def test_no_insights(self):
        analyzer = PerformanceInsightAnalyzer()
        potential = await analyzer._calculate_optimization_potential([])
        assert potential == 0.0

    async def test_with_insights(self):
        analyzer = PerformanceInsightAnalyzer()
        insight = PerformanceInsight(
            insight_id="i1",
            insight_name="test",
            category=PerformanceCategory.EXECUTION_TIME,
            severity="high",
            confidence=0.9,
            impact_score=8.0,
            performance_delta=30.0,
            baseline_metric=10.0,
            target_metric=7.0,
            description="d",
            implementation_complexity="medium",
        )
        potential = await analyzer._calculate_optimization_potential([insight])
        assert potential == pytest.approx(30.0)

    async def test_clamped_to_100(self):
        analyzer = PerformanceInsightAnalyzer()
        insight = PerformanceInsight(
            insight_id="i1",
            insight_name="test",
            category=PerformanceCategory.EXECUTION_TIME,
            severity="high",
            confidence=1.0,
            impact_score=10.0,
            performance_delta=200.0,
            baseline_metric=10.0,
            target_metric=2.0,
            description="d",
            implementation_complexity="high",
        )
        potential = await analyzer._calculate_optimization_potential([insight])
        assert potential == 100.0

    async def test_zero_weight(self):
        analyzer = PerformanceInsightAnalyzer()
        insight = PerformanceInsight(
            insight_id="i1",
            insight_name="test",
            category=PerformanceCategory.EXECUTION_TIME,
            severity="high",
            confidence=0.0,
            impact_score=0.0,
            performance_delta=30.0,
            baseline_metric=10.0,
            target_metric=7.0,
            description="d",
            implementation_complexity="medium",
        )
        potential = await analyzer._calculate_optimization_potential([insight])
        assert potential == 0.0


class TestAnalyzePerformanceFullPipeline:
    async def test_full_analysis(self):
        analyzer = PerformanceInsightAnalyzer()
        results = [
            _result(test_id=f"t{i}", execution_time=2.0 + i * 0.5, language="python")
            for i in range(15)
        ]
        analysis = await analyzer.analyze_performance(results)
        assert analysis.total_tests_analyzed == 15
        assert analysis.analysis_period == "recent"
        assert isinstance(analysis.overall_statistics, PerformanceStatistics)
        assert isinstance(analysis.trend_analysis, PerformanceTrendAnalysis)
        assert 0.0 <= analysis.performance_score <= 100.0
        assert 0.0 <= analysis.optimization_potential <= 100.0

    async def test_empty_results_raises(self):
        analyzer = PerformanceInsightAnalyzer()
        with pytest.raises(ValueError, match="No execution times"):
            await analyzer.analyze_performance([])


class TestFactory:
    def test_create_default(self):
        analyzer = create_performance_insight_analyzer()
        assert isinstance(analyzer, PerformanceInsightAnalyzer)

    def test_create_with_config(self):
        analyzer = create_performance_insight_analyzer(config={"slow_test_threshold": 5.0})
        assert analyzer.config["slow_test_threshold"] == 5.0
