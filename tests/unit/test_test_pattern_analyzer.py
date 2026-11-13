"""Tests for TestPatternAnalyzer and related models."""

from __future__ import annotations

import pytest

from models.testing_models import QualityIssue, QualityResult, TestResult
from services.test_pattern_analyzer import (
    PatternCategory,
    PatternSeverity,
    TestPatternAnalysisResult,
    TestPatternAnalyzer,
    TestPatternMetrics,
    create_test_pattern_analyzer,
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


def _quality_result(
    overall_score: float = 80.0,
    issues: list[QualityIssue] | None = None,
) -> QualityResult:
    return QualityResult(
        overall_score=overall_score,
        issues=issues or [],
    )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestTestPatternMetricsModel:
    def test_priority_score(self):
        p = TestPatternMetrics(
            pattern_id="p1",
            pattern_name="test",
            category=PatternCategory.PERFORMANCE,
            severity=PatternSeverity.CRITICAL,
            confidence_score=0.9,
            occurrence_count=10,
            impact_score=8.0,
            frequency_ratio=0.5,
            description="d",
        )
        # 10*0.4 + 8*0.4 + 0.5*10*0.2 = 4+3.2+1 = 8.2
        assert p.priority_score == pytest.approx(8.2)

    def test_priority_score_info(self):
        p = TestPatternMetrics(
            pattern_id="p2",
            pattern_name="test",
            category=PatternCategory.COVERAGE,
            severity=PatternSeverity.INFO,
            confidence_score=0.5,
            occurrence_count=1,
            impact_score=1.0,
            frequency_ratio=0.1,
            description="d",
        )
        # 2*0.4 + 1*0.4 + 0.1*10*0.2 = 0.8+0.4+0.2 = 1.4
        assert p.priority_score == pytest.approx(1.4)

    def test_priority_score_capped_at_10(self):
        p = TestPatternMetrics(
            pattern_id="p3",
            pattern_name="test",
            category=PatternCategory.SECURITY,
            severity=PatternSeverity.CRITICAL,
            confidence_score=1.0,
            occurrence_count=100,
            impact_score=10.0,
            frequency_ratio=1.0,
            description="d",
        )
        assert p.priority_score == 10.0


class TestTestPatternAnalysisResultModel:
    def test_overall_health_excellent_no_patterns(self):
        r = TestPatternAnalysisResult(
            analysis_id="a1",
            total_tests_analyzed=10,
        )
        assert r.overall_pattern_health == "excellent"

    def test_overall_health_critical(self):
        r = TestPatternAnalysisResult(
            analysis_id="a1",
            total_tests_analyzed=10,
            detected_patterns=[
                TestPatternMetrics(
                    pattern_id="x",
                    pattern_name="x",
                    category=PatternCategory.SECURITY,
                    severity=PatternSeverity.CRITICAL,
                    confidence_score=0.9,
                    occurrence_count=5,
                    impact_score=9.0,
                    frequency_ratio=0.5,
                    description="d",
                )
            ],
            patterns_by_severity={PatternSeverity.CRITICAL: 1},
        )
        assert r.overall_pattern_health == "critical"

    def test_overall_health_poor(self):
        r = TestPatternAnalysisResult(
            analysis_id="a1",
            total_tests_analyzed=10,
            detected_patterns=[
                TestPatternMetrics(
                    pattern_id=f"x{i}",
                    pattern_name="x",
                    category=PatternCategory.RELIABILITY,
                    severity=PatternSeverity.HIGH,
                    confidence_score=0.9,
                    occurrence_count=5,
                    impact_score=8.0,
                    frequency_ratio=0.5,
                    description="d",
                )
                for i in range(3)
            ],
            patterns_by_severity={PatternSeverity.HIGH: 3},
        )
        assert r.overall_pattern_health == "poor"

    def test_overall_health_fair(self):
        r = TestPatternAnalysisResult(
            analysis_id="a1",
            total_tests_analyzed=10,
            detected_patterns=[
                TestPatternMetrics(
                    pattern_id="x",
                    pattern_name="x",
                    category=PatternCategory.PERFORMANCE,
                    severity=PatternSeverity.HIGH,
                    confidence_score=0.9,
                    occurrence_count=5,
                    impact_score=8.0,
                    frequency_ratio=0.5,
                    description="d",
                )
            ],
            patterns_by_severity={PatternSeverity.HIGH: 1},
        )
        assert r.overall_pattern_health == "fair"

    def test_overall_health_good(self):
        r = TestPatternAnalysisResult(
            analysis_id="a1",
            total_tests_analyzed=10,
            detected_patterns=[
                TestPatternMetrics(
                    pattern_id="x",
                    pattern_name="x",
                    category=PatternCategory.PERFORMANCE,
                    severity=PatternSeverity.LOW,
                    confidence_score=0.9,
                    occurrence_count=5,
                    impact_score=5.0,
                    frequency_ratio=0.3,
                    description="d",
                )
            ],
            patterns_by_severity={PatternSeverity.LOW: 1},
        )
        assert r.overall_pattern_health == "good"

    def test_serialize_datetime(self):
        r = TestPatternAnalysisResult(
            analysis_id="a1",
            total_tests_analyzed=10,
        )
        d = r.model_dump()
        assert isinstance(d["analyzed_at"], str)

    def test_serialize_datetime_none_via_serializer(self):
        """Test the serialize_datetime method directly with None."""
        r = TestPatternAnalysisResult(
            analysis_id="a1",
            total_tests_analyzed=10,
        )
        assert r.serialize_datetime(None) is None


# ---------------------------------------------------------------------------
# Analyzer internal methods
# ---------------------------------------------------------------------------


class TestDetectPerformancePatterns:
    async def test_slow_tests_detected(self):
        analyzer = TestPatternAnalyzer()
        results = [_result(test_id=f"t{i}", execution_time=15.0) for i in range(5)]
        patterns = await analyzer._detect_performance_patterns(results)
        assert any(p.pattern_id == "perf_slow_tests" for p in patterns)

    async def test_slow_tests_high_severity_when_many(self):
        analyzer = TestPatternAnalyzer()
        results = [_result(test_id=f"t{i}", execution_time=15.0) for i in range(15)]
        patterns = await analyzer._detect_performance_patterns(results)
        slow = next(p for p in patterns if p.pattern_id == "perf_slow_tests")
        assert slow.severity == PatternSeverity.HIGH

    async def test_no_slow_tests(self):
        analyzer = TestPatternAnalyzer()
        results = [_result(test_id=f"t{i}", execution_time=1.0) for i in range(5)]
        patterns = await analyzer._detect_performance_patterns(results)
        assert not any(p.pattern_id == "perf_slow_tests" for p in patterns)

    async def test_inconsistent_performance(self):
        analyzer = TestPatternAnalyzer()
        # High variance: std_dev > mean * 0.5
        times = [1.0, 1.0, 1.0, 50.0, 50.0, 50.0]
        results = [_result(test_id=f"t{i}", execution_time=times[i]) for i in range(6)]
        patterns = await analyzer._detect_performance_patterns(results)
        assert any(p.pattern_id == "perf_inconsistent" for p in patterns)

    async def test_no_inconsistency_for_few_tests(self):
        analyzer = TestPatternAnalyzer()
        results = [_result(test_id=f"t{i}", execution_time=i + 1.0) for i in range(3)]
        patterns = await analyzer._detect_performance_patterns(results)
        assert not any(p.pattern_id == "perf_inconsistent" for p in patterns)

    async def test_empty_results(self):
        analyzer = TestPatternAnalyzer()
        patterns = await analyzer._detect_performance_patterns([])
        assert patterns == []


class TestDetectReliabilityPatterns:
    async def test_flaky_tests(self):
        analyzer = TestPatternAnalyzer()
        results = [_result(test_id=f"t{i}", success=i % 3 == 0) for i in range(10)]
        patterns = await analyzer._detect_reliability_patterns(results)
        assert any(p.pattern_id == "rel_flaky_tests" for p in patterns)

    async def test_no_flaky_when_all_pass(self):
        analyzer = TestPatternAnalyzer()
        results = [_result(test_id=f"t{i}", success=True) for i in range(10)]
        patterns = await analyzer._detect_reliability_patterns(results)
        assert not any(p.pattern_id == "rel_flaky_tests" for p in patterns)

    async def test_error_clustering_timeout(self):
        analyzer = TestPatternAnalyzer()
        results = [
            _result(test_id=f"t{i}", success=False, stderr="TimeoutError: operation timed out")
            for i in range(5)
        ]
        patterns = await analyzer._detect_reliability_patterns(results)
        assert any("timeout" in p.pattern_id for p in patterns)

    async def test_error_clustering_assertion(self):
        analyzer = TestPatternAnalyzer()
        results = [
            _result(test_id=f"t{i}", success=False, stderr="AssertionError: expected X got Y")
            for i in range(5)
        ]
        patterns = await analyzer._detect_reliability_patterns(results)
        assert any("assertion" in p.pattern_id for p in patterns)

    async def test_error_clustering_connection(self):
        analyzer = TestPatternAnalyzer()
        results = [
            _result(
                test_id=f"t{i}", success=False, stderr="ConnectionRefusedError: connection refused"
            )
            for i in range(5)
        ]
        patterns = await analyzer._detect_reliability_patterns(results)
        assert any("connection" in p.pattern_id for p in patterns)

    async def test_error_clustering_null(self):
        analyzer = TestPatternAnalyzer()
        results = [
            _result(test_id=f"t{i}", success=False, stderr="NoneType has no attribute 'foo'")
            for i in range(5)
        ]
        patterns = await analyzer._detect_reliability_patterns(results)
        assert any("null_reference" in p.pattern_id for p in patterns)

    async def test_error_clustering_other(self):
        analyzer = TestPatternAnalyzer()
        results = [
            _result(test_id=f"t{i}", success=False, stderr="SomeWeirdError: blah") for i in range(5)
        ]
        patterns = await analyzer._detect_reliability_patterns(results)
        assert any("other" in p.pattern_id for p in patterns)

    async def test_error_clustering_below_threshold(self):
        analyzer = TestPatternAnalyzer()
        results = [
            _result(test_id="t0", success=False, stderr="TimeoutError"),
            _result(test_id="t1", success=False, stderr="AssertionError"),
        ]
        patterns = await analyzer._detect_reliability_patterns(results)
        # No clustering pattern because count < 3
        assert not any("cluster" in p.pattern_id for p in patterns)

    async def test_no_stderr_no_clustering(self):
        analyzer = TestPatternAnalyzer()
        results = [_result(test_id=f"t{i}", success=False, stderr="") for i in range(5)]
        patterns = await analyzer._detect_reliability_patterns(results)
        # Failed but no stderr -> no error_types dict
        assert not any("cluster" in p.pattern_id for p in patterns)


class TestDetectMaintainabilityPatterns:
    async def test_complexity_pattern_with_quality_results(self):
        analyzer = TestPatternAnalyzer()
        results = [_result(test_id=f"t{i}") for i in range(5)]
        quality = [
            _quality_result(
                overall_score=60.0,
                issues=[QualityIssue(message="High complexity in module X")],
            )
        ]
        patterns = await analyzer._detect_maintainability_patterns(results, quality)
        assert any(p.pattern_id == "maint_complexity" for p in patterns)

    async def test_no_complexity_without_quality(self):
        analyzer = TestPatternAnalyzer()
        results = [_result(test_id=f"t{i}") for i in range(5)]
        patterns = await analyzer._detect_maintainability_patterns(results, None)
        assert not any(p.pattern_id == "maint_complexity" for p in patterns)

    async def test_duplication_pattern(self):
        analyzer = TestPatternAnalyzer()
        # Create mostly distinct tests but with a few near-duplicates
        results = [
            _result(test_id="test_user_login_email"),
            _result(test_id="test_user_login_oauth"),
            _result(test_id="test_checkout_process"),
            _result(test_id="test_payment_gateway"),
            _result(test_id="test_inventory_check"),
            _result(test_id="test_order_creation"),
            _result(test_id="test_shipping_label"),
            _result(test_id="test_refund_handler"),
            _result(test_id="test_cart_item_add_a"),
            _result(test_id="test_cart_item_add_b"),
            _result(test_id="test_cart_item_add_c"),
        ]
        patterns = await analyzer._detect_maintainability_patterns(results, None)
        assert any(p.pattern_id == "maint_duplication" for p in patterns)

    async def test_no_duplication_few_tests(self):
        analyzer = TestPatternAnalyzer()
        results = [_result(test_id=f"t{i}") for i in range(5)]
        patterns = await analyzer._detect_maintainability_patterns(results, None)
        assert not any(p.pattern_id == "maint_duplication" for p in patterns)


class TestDetectCoveragePatterns:
    async def test_language_imbalance(self):
        analyzer = TestPatternAnalyzer()
        results = [_result(test_id=f"py{i}", language="python") for i in range(9)] + [
            _result(test_id="js0", language="javascript")
        ]
        patterns = await analyzer._detect_coverage_patterns(results)
        assert any(p.pattern_id == "cov_language_imbalance" for p in patterns)

    async def test_no_imbalance_single_language(self):
        analyzer = TestPatternAnalyzer()
        results = [_result(test_id=f"t{i}", language="python") for i in range(10)]
        patterns = await analyzer._detect_coverage_patterns(results)
        assert not any(p.pattern_id == "cov_language_imbalance" for p in patterns)

    async def test_no_imbalance_balanced(self):
        analyzer = TestPatternAnalyzer()
        results = [_result(test_id=f"py{i}", language="python") for i in range(5)] + [
            _result(test_id=f"js{i}", language="javascript") for i in range(5)
        ]
        patterns = await analyzer._detect_coverage_patterns(results)
        assert not any(p.pattern_id == "cov_language_imbalance" for p in patterns)


class TestDetectArchitecturePatterns:
    async def test_pyramid_imbalance(self):
        analyzer = TestPatternAnalyzer()
        results = [_result(test_id=f"u{i}", test_type="unit") for i in range(3)] + [
            _result(test_id=f"int{i}", test_type="integration") for i in range(7)
        ]
        patterns = await analyzer._detect_architecture_patterns(results)
        assert any(p.pattern_id == "arch_test_pyramid" for p in patterns)

    async def test_no_imbalance_mostly_unit(self):
        analyzer = TestPatternAnalyzer()
        results = [_result(test_id=f"u{i}", test_type="unit") for i in range(8)] + [
            _result(test_id=f"int{i}", test_type="integration") for i in range(2)
        ]
        patterns = await analyzer._detect_architecture_patterns(results)
        assert not any(p.pattern_id == "arch_test_pyramid" for p in patterns)

    async def test_single_type_no_pattern(self):
        analyzer = TestPatternAnalyzer()
        results = [_result(test_id=f"u{i}", test_type="unit") for i in range(10)]
        patterns = await analyzer._detect_architecture_patterns(results)
        assert patterns == []


class TestDetectSecurityPatterns:
    async def test_low_security_coverage(self):
        analyzer = TestPatternAnalyzer()
        results = [_result(test_id=f"t{i}", test_type="unit") for i in range(20)]
        patterns = await analyzer._detect_security_patterns(results, None)
        assert any(p.pattern_id == "sec_low_coverage" for p in patterns)

    async def test_sufficient_security_coverage(self):
        analyzer = TestPatternAnalyzer()
        results = [_result(test_id=f"sec{i}", test_type="security") for i in range(5)] + [
            _result(test_id=f"u{i}", test_type="unit") for i in range(5)
        ]
        patterns = await analyzer._detect_security_patterns(results, None)
        assert not any(p.pattern_id == "sec_low_coverage" for p in patterns)

    async def test_security_quality_issues(self):
        analyzer = TestPatternAnalyzer()
        results = [_result(test_id=f"t{i}") for i in range(5)]
        quality = [
            _quality_result(
                overall_score=50.0,
                issues=[QualityIssue(message="SQL injection vulnerability in query")],
            )
        ]
        patterns = await analyzer._detect_security_patterns(results, quality)
        assert any(p.pattern_id == "sec_quality_issues" for p in patterns)

    async def test_no_security_quality_issues(self):
        analyzer = TestPatternAnalyzer()
        results = [_result(test_id=f"t{i}") for i in range(5)]
        quality = [
            _quality_result(
                overall_score=90.0,
                issues=[QualityIssue(message="Missing docstring")],
            )
        ]
        patterns = await analyzer._detect_security_patterns(results, quality)
        assert not any(p.pattern_id == "sec_quality_issues" for p in patterns)


class TestFilterPatterns:
    def test_filters_by_confidence(self):
        analyzer = TestPatternAnalyzer()
        p = TestPatternMetrics(
            pattern_id="p1",
            pattern_name="test",
            category=PatternCategory.PERFORMANCE,
            severity=PatternSeverity.HIGH,
            confidence_score=0.3,  # below 0.6
            occurrence_count=10,
            impact_score=8.0,
            frequency_ratio=0.5,
            description="d",
        )
        assert analyzer._filter_patterns([p]) == []

    def test_filters_by_occurrence_count(self):
        analyzer = TestPatternAnalyzer()
        p = TestPatternMetrics(
            pattern_id="p1",
            pattern_name="test",
            category=PatternCategory.PERFORMANCE,
            severity=PatternSeverity.HIGH,
            confidence_score=0.9,
            occurrence_count=1,  # below 3
            impact_score=8.0,
            frequency_ratio=0.5,
            description="d",
        )
        assert analyzer._filter_patterns([p]) == []

    def test_filters_by_impact_score(self):
        analyzer = TestPatternAnalyzer()
        p = TestPatternMetrics(
            pattern_id="p1",
            pattern_name="test",
            category=PatternCategory.PERFORMANCE,
            severity=PatternSeverity.HIGH,
            confidence_score=0.9,
            occurrence_count=10,
            impact_score=3.0,  # below 5.0
            frequency_ratio=0.5,
            description="d",
        )
        assert analyzer._filter_patterns([p]) == []

    def test_passes_valid_pattern(self):
        analyzer = TestPatternAnalyzer()
        p = TestPatternMetrics(
            pattern_id="p1",
            pattern_name="test",
            category=PatternCategory.PERFORMANCE,
            severity=PatternSeverity.HIGH,
            confidence_score=0.9,
            occurrence_count=10,
            impact_score=8.0,
            frequency_ratio=0.5,
            description="d",
        )
        result = analyzer._filter_patterns([p])
        assert len(result) == 1

    def test_sorted_by_priority(self):
        analyzer = TestPatternAnalyzer()
        low = TestPatternMetrics(
            pattern_id="low",
            pattern_name="test",
            category=PatternCategory.PERFORMANCE,
            severity=PatternSeverity.MEDIUM,
            confidence_score=0.7,
            occurrence_count=5,
            impact_score=5.5,
            frequency_ratio=0.3,
            description="d",
        )
        high = TestPatternMetrics(
            pattern_id="high",
            pattern_name="test",
            category=PatternCategory.SECURITY,
            severity=PatternSeverity.CRITICAL,
            confidence_score=0.95,
            occurrence_count=10,
            impact_score=9.0,
            frequency_ratio=0.8,
            description="d",
        )
        result = analyzer._filter_patterns([low, high])
        assert result[0].pattern_id == "high"


class TestCategorization:
    def test_categorize_patterns(self):
        analyzer = TestPatternAnalyzer()
        patterns = [
            TestPatternMetrics(
                pattern_id="p1",
                pattern_name="t",
                category=PatternCategory.PERFORMANCE,
                severity=PatternSeverity.HIGH,
                confidence_score=0.9,
                occurrence_count=5,
                impact_score=7.0,
                frequency_ratio=0.5,
                description="d",
            ),
            TestPatternMetrics(
                pattern_id="p2",
                pattern_name="t",
                category=PatternCategory.PERFORMANCE,
                severity=PatternSeverity.MEDIUM,
                confidence_score=0.8,
                occurrence_count=3,
                impact_score=6.0,
                frequency_ratio=0.3,
                description="d",
            ),
            TestPatternMetrics(
                pattern_id="p3",
                pattern_name="t",
                category=PatternCategory.SECURITY,
                severity=PatternSeverity.HIGH,
                confidence_score=0.9,
                occurrence_count=5,
                impact_score=8.0,
                frequency_ratio=0.5,
                description="d",
            ),
        ]
        categorized = analyzer._categorize_patterns(patterns)
        assert categorized[PatternCategory.PERFORMANCE] == 2
        assert categorized[PatternCategory.SECURITY] == 1

    def test_categorize_by_severity(self):
        analyzer = TestPatternAnalyzer()
        patterns = [
            TestPatternMetrics(
                pattern_id="p1",
                pattern_name="t",
                category=PatternCategory.PERFORMANCE,
                severity=PatternSeverity.HIGH,
                confidence_score=0.9,
                occurrence_count=5,
                impact_score=7.0,
                frequency_ratio=0.5,
                description="d",
            ),
            TestPatternMetrics(
                pattern_id="p2",
                pattern_name="t",
                category=PatternCategory.PERFORMANCE,
                severity=PatternSeverity.HIGH,
                confidence_score=0.8,
                occurrence_count=3,
                impact_score=6.0,
                frequency_ratio=0.3,
                description="d",
            ),
        ]
        categorized = analyzer._categorize_by_severity(patterns)
        assert categorized[PatternSeverity.HIGH] == 2


class TestRecommendationsAndOpportunities:
    def test_generate_top_recommendations(self):
        analyzer = TestPatternAnalyzer()
        pattern = TestPatternMetrics(
            pattern_id="p1",
            pattern_name="t",
            category=PatternCategory.PERFORMANCE,
            severity=PatternSeverity.CRITICAL,
            confidence_score=0.9,
            occurrence_count=10,
            impact_score=9.0,
            frequency_ratio=0.8,
            description="d",
            recommendations=["Fix A", "Fix B", "Fix C"],
        )
        recs = analyzer._generate_top_recommendations([pattern])
        assert "Fix A" in recs
        assert "Fix B" in recs
        # Only top 2 per pattern
        assert len(recs) <= 10

    def test_recommendations_deduplicated(self):
        analyzer = TestPatternAnalyzer()
        p1 = TestPatternMetrics(
            pattern_id="p1",
            pattern_name="t",
            category=PatternCategory.PERFORMANCE,
            severity=PatternSeverity.CRITICAL,
            confidence_score=0.9,
            occurrence_count=10,
            impact_score=9.0,
            frequency_ratio=0.8,
            description="d",
            recommendations=["Same fix", "Other"],
        )
        p2 = TestPatternMetrics(
            pattern_id="p2",
            pattern_name="t",
            category=PatternCategory.SECURITY,
            severity=PatternSeverity.CRITICAL,
            confidence_score=0.9,
            occurrence_count=10,
            impact_score=9.0,
            frequency_ratio=0.8,
            description="d",
            recommendations=["Same fix", "Another"],
        )
        recs = analyzer._generate_top_recommendations([p1, p2])
        assert recs.count("Same fix") == 1

    def test_no_recommendations_for_low_priority(self):
        analyzer = TestPatternAnalyzer()
        p = TestPatternMetrics(
            pattern_id="p1",
            pattern_name="t",
            category=PatternCategory.COVERAGE,
            severity=PatternSeverity.LOW,
            confidence_score=0.6,
            occurrence_count=3,
            impact_score=5.0,
            frequency_ratio=0.2,
            description="d",
            recommendations=["Fix it"],
        )
        recs = analyzer._generate_top_recommendations([p])
        assert recs == []

    def test_identify_improvement_opportunities(self):
        analyzer = TestPatternAnalyzer()
        patterns = [
            TestPatternMetrics(
                pattern_id="p1",
                pattern_name="t",
                category=PatternCategory.PERFORMANCE,
                severity=PatternSeverity.HIGH,
                confidence_score=0.9,
                occurrence_count=5,
                impact_score=7.0,
                frequency_ratio=0.5,
                description="d",
            ),
            TestPatternMetrics(
                pattern_id="p2",
                pattern_name="t",
                category=PatternCategory.SECURITY,
                severity=PatternSeverity.HIGH,
                confidence_score=0.9,
                occurrence_count=5,
                impact_score=8.0,
                frequency_ratio=0.5,
                description="d",
            ),
        ]
        opps = analyzer._identify_improvement_opportunities(patterns)
        assert any("performance" in o.lower() for o in opps)
        assert any("security" in o.lower() for o in opps)

    def test_all_category_opportunities(self):
        analyzer = TestPatternAnalyzer()
        # One pattern per category
        categories = [
            PatternCategory.PERFORMANCE,
            PatternCategory.RELIABILITY,
            PatternCategory.MAINTAINABILITY,
            PatternCategory.SECURITY,
            PatternCategory.COVERAGE,
            PatternCategory.ARCHITECTURE,
        ]
        patterns = [
            TestPatternMetrics(
                pattern_id=f"p{i}",
                pattern_name="t",
                category=cat,
                severity=PatternSeverity.HIGH,
                confidence_score=0.9,
                occurrence_count=5,
                impact_score=7.0,
                frequency_ratio=0.5,
                description="d",
            )
            for i, cat in enumerate(categories)
        ]
        opps = analyzer._identify_improvement_opportunities(patterns)
        assert len(opps) == 6


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestAnalyzePatternsPipeline:
    async def test_full_analysis(self):
        analyzer = TestPatternAnalyzer()
        results = [
            _result(test_id=f"t{i}", execution_time=15.0, success=i % 3 == 0, test_type="unit")
            for i in range(20)
        ]
        analysis = await analyzer.analyze_patterns(results)
        assert analysis.total_tests_analyzed == 20
        assert "python" in analysis.languages_analyzed
        assert isinstance(analysis.detected_patterns, list)

    async def test_analysis_with_quality_results(self):
        analyzer = TestPatternAnalyzer()
        results = [_result(test_id=f"t{i}", execution_time=2.0) for i in range(5)]
        quality = [
            _quality_result(
                overall_score=50.0,
                issues=[
                    QualityIssue(message="High complexity in module"),
                    QualityIssue(message="SQL injection vulnerability"),
                ],
            )
        ]
        analysis = await analyzer.analyze_patterns(results, quality)
        assert analysis.total_tests_analyzed == 5

    async def test_patterns_classified_correctly(self):
        analyzer = TestPatternAnalyzer()
        results = [
            _result(test_id=f"t{i}", execution_time=15.0, test_type="unit") for i in range(20)
        ]
        analysis = await analyzer.analyze_patterns(results)
        for p in analysis.detected_patterns:
            assert p.category in analysis.patterns_by_category


class TestFactory:
    def test_create_default(self):
        a = create_test_pattern_analyzer()
        assert isinstance(a, TestPatternAnalyzer)

    def test_create_with_config(self):
        a = create_test_pattern_analyzer(config={"slow_test_threshold": 5.0})
        assert a.config["slow_test_threshold"] == 5.0
