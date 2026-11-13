"""Unit tests for TestPatternDetector in pattern_recognition.py.

Pure logic tests — no mocking needed. We construct data dicts and
AITestInsightsConfig instances directly.
"""

from __future__ import annotations

import pytest

from services.ai_test_insights.config import AITestInsightsConfig, TestPatternType
from services.ai_test_insights.pattern_recognition import TestPatternDetector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_data(**overrides) -> dict:
    """Return a minimal valid data dict, applying *overrides*."""
    base = {
        "avg_execution_time": 1.0,
        "peak_memory_usage": 0,
        "total_tests": 10,
        "overall_success_rate": 1.0,
        "error_messages": [],
        "quality_issues": [],
        "results_by_language": {},
        "execution_times": [],
        "p95_execution_time": 0,
        "median_execution_time": 0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# detect_patterns — orchestrator
# ---------------------------------------------------------------------------


class TestDetectPatterns:
    """Tests for the top-level detect_patterns orchestrator."""

    async def test_returns_empty_when_nothing_triggers(self):
        detector = TestPatternDetector()
        patterns = await detector.detect_patterns(_base_data())
        assert patterns == []

    async def test_aggregates_patterns_from_all_sub_detectors(self):
        """Feed data that triggers at least one pattern in every sub-detector."""
        # Need total_tests == len(execution_times) so integration ratio works
        # 40 slow (>10s) + 20 fast = 60 total, 40/60 = 67% > 30%
        exec_times = [500.0] * 40 + [0.1] * 20
        data = _base_data(
            avg_execution_time=999,  # performance: slow
            peak_memory_usage=9999,  # performance: memory
            results_by_language={"py": [1] * 30, "js": [1] * 30},  # performance: parallel
            total_tests=60,
            overall_success_rate=0.3,  # reliability: flaky
            error_messages=["timeout"] * 40,  # reliability: dominant error
            quality_issues=list(range(20)),  # coverage: quality
            execution_times=exec_times,  # integration: slow ratio
            p95_execution_time=500,
            median_execution_time=1,
        )
        detector = TestPatternDetector()
        patterns = await detector.detect_patterns(data)
        types_found = {p.pattern_type for p in patterns}
        # We expect at least performance, flaky, reliability, coverage, integration, unit
        assert TestPatternType.PERFORMANCE_PATTERN in types_found
        assert TestPatternType.FLAKY_PATTERN in types_found
        assert TestPatternType.RELIABILITY_PATTERN in types_found
        assert TestPatternType.COVERAGE_PATTERN in types_found
        assert TestPatternType.INTEGRATION_PATTERN in types_found
        assert TestPatternType.UNIT_PATTERN in types_found


# ---------------------------------------------------------------------------
# _detect_performance_patterns
# ---------------------------------------------------------------------------


class TestPerformancePatterns:
    async def test_slow_execution_detected(self):
        config = AITestInsightsConfig(slow_test_threshold=5.0)
        detector = TestPatternDetector(config)
        data = _base_data(avg_execution_time=10.0, execution_times=[10.0, 12.0])
        patterns = await detector._detect_performance_patterns(data)
        assert any(p.description == "Slow test execution pattern detected" for p in patterns)

    async def test_below_slow_threshold_no_pattern(self):
        config = AITestInsightsConfig(slow_test_threshold=5.0)
        detector = TestPatternDetector(config)
        data = _base_data(avg_execution_time=2.0)
        patterns = await detector._detect_performance_patterns(data)
        slow = [p for p in patterns if "Slow" in p.description]
        assert slow == []

    async def test_high_memory_detected(self):
        config = AITestInsightsConfig(memory_threshold=256)
        detector = TestPatternDetector(config)
        data = _base_data(peak_memory_usage=512)
        patterns = await detector._detect_performance_patterns(data)
        assert any("memory" in p.description.lower() for p in patterns)

    async def test_below_memory_threshold_no_pattern(self):
        config = AITestInsightsConfig(memory_threshold=256)
        detector = TestPatternDetector(config)
        data = _base_data(peak_memory_usage=100)
        patterns = await detector._detect_performance_patterns(data)
        mem = [p for p in patterns if "memory" in p.description.lower()]
        assert mem == []

    async def test_multi_language_parallel_suggestion(self):
        detector = TestPatternDetector()
        data = _base_data(
            results_by_language={"python": [1] * 30, "java": [1] * 30},
            total_tests=60,
        )
        patterns = await detector._detect_performance_patterns(data)
        assert any("parallel" in p.description.lower() for p in patterns)

    async def test_single_language_no_parallel_suggestion(self):
        detector = TestPatternDetector()
        data = _base_data(
            results_by_language={"python": [1] * 60},
            total_tests=60,
        )
        patterns = await detector._detect_performance_patterns(data)
        parallel = [p for p in patterns if "parallel" in p.description.lower()]
        assert parallel == []

    async def test_multi_language_few_tests_no_parallel(self):
        detector = TestPatternDetector()
        data = _base_data(
            results_by_language={"python": [1] * 5, "java": [1] * 5},
            total_tests=10,
        )
        patterns = await detector._detect_performance_patterns(data)
        parallel = [p for p in patterns if "parallel" in p.description.lower()]
        assert parallel == []


# ---------------------------------------------------------------------------
# _detect_reliability_patterns
# ---------------------------------------------------------------------------


class TestReliabilityPatterns:
    async def test_low_success_rate_flaky(self):
        config = AITestInsightsConfig(flaky_test_threshold=0.8)
        detector = TestPatternDetector(config)
        data = _base_data(overall_success_rate=0.5, total_tests=100)
        patterns = await detector._detect_reliability_patterns(data)
        flaky = [p for p in patterns if p.pattern_type == TestPatternType.FLAKY_PATTERN]
        assert len(flaky) == 1
        # Check affected_tests contains flaky_count items (capped at 5)
        assert len(flaky[0].affected_tests) <= 5

    async def test_flaky_count_calculation(self):
        config = AITestInsightsConfig(flaky_test_threshold=0.8)
        detector = TestPatternDetector(config)
        # success_rate=0.5, total_tests=100 → flaky_count = int(100 * 0.5 * 0.3) = 15 → capped at 5
        data = _base_data(overall_success_rate=0.5, total_tests=100)
        patterns = await detector._detect_reliability_patterns(data)
        flaky = [p for p in patterns if p.pattern_type == TestPatternType.FLAKY_PATTERN]
        assert len(flaky[0].affected_tests) == 5

    async def test_high_success_rate_no_flaky(self):
        config = AITestInsightsConfig(flaky_test_threshold=0.8)
        detector = TestPatternDetector(config)
        data = _base_data(overall_success_rate=0.95, total_tests=100)
        patterns = await detector._detect_reliability_patterns(data)
        flaky = [p for p in patterns if p.pattern_type == TestPatternType.FLAKY_PATTERN]
        assert flaky == []

    async def test_dominant_error_category_reliability(self):
        detector = TestPatternDetector()
        # 10 timeout errors out of 10 → 100% > 30%
        data = _base_data(error_messages=["Timeout occurred"] * 10)
        patterns = await detector._detect_reliability_patterns(data)
        rel = [p for p in patterns if p.pattern_type == TestPatternType.RELIABILITY_PATTERN]
        assert len(rel) == 1
        assert "timeout" in rel[0].description.lower()

    async def test_no_dominant_error_category(self):
        detector = TestPatternDetector()
        # 1 of each category — none > 30% of 5
        errors = ["Timeout", "AssertionError", "ConnectionError", "PermissionDenied", "MemoryError"]
        data = _base_data(error_messages=errors)
        patterns = await detector._detect_reliability_patterns(data)
        rel = [p for p in patterns if p.pattern_type == TestPatternType.RELIABILITY_PATTERN]
        assert rel == []

    async def test_empty_error_list_no_reliability_pattern(self):
        detector = TestPatternDetector()
        data = _base_data(error_messages=[])
        patterns = await detector._detect_reliability_patterns(data)
        rel = [p for p in patterns if p.pattern_type == TestPatternType.RELIABILITY_PATTERN]
        assert rel == []


# ---------------------------------------------------------------------------
# _detect_coverage_patterns
# ---------------------------------------------------------------------------


class TestCoveragePatterns:
    async def test_many_quality_issues(self):
        detector = TestPatternDetector()
        data = _base_data(quality_issues=list(range(15)))
        patterns = await detector._detect_coverage_patterns(data)
        cov = [p for p in patterns if p.pattern_type == TestPatternType.COVERAGE_PATTERN]
        assert len(cov) == 1

    async def test_few_quality_issues_no_pattern(self):
        detector = TestPatternDetector()
        data = _base_data(quality_issues=list(range(5)))
        patterns = await detector._detect_coverage_patterns(data)
        cov = [p for p in patterns if "Low coverage" in p.description]
        assert cov == []

    async def test_uneven_language_coverage(self):
        detector = TestPatternDetector()
        data = _base_data(
            results_by_language={"python": [1] * 100, "java": [1] * 10},
        )
        patterns = await detector._detect_coverage_patterns(data)
        uneven = [p for p in patterns if "Uneven" in p.description]
        assert len(uneven) == 1

    async def test_even_language_coverage_no_pattern(self):
        detector = TestPatternDetector()
        data = _base_data(
            results_by_language={"python": [1] * 10, "java": [1] * 10},
        )
        patterns = await detector._detect_coverage_patterns(data)
        uneven = [p for p in patterns if "Uneven" in p.description]
        assert uneven == []

    async def test_single_language_no_uneven_coverage(self):
        detector = TestPatternDetector()
        data = _base_data(results_by_language={"python": [1] * 50})
        patterns = await detector._detect_coverage_patterns(data)
        uneven = [p for p in patterns if "Uneven" in p.description]
        assert uneven == []


# ---------------------------------------------------------------------------
# _detect_security_patterns
# ---------------------------------------------------------------------------


class TestSecurityPatterns:
    @pytest.mark.parametrize("keyword", ["auth", "permission", "security", "token"])
    async def test_security_keyword_with_enough_tests(self, keyword):
        detector = TestPatternDetector()
        data = _base_data(
            error_messages=[f"Error: {keyword} failure"],
            total_tests=25,
        )
        patterns = await detector._detect_security_patterns(data)
        sec = [p for p in patterns if p.pattern_type == TestPatternType.SECURITY_PATTERN]
        assert len(sec) == 1

    async def test_security_keyword_few_tests_no_pattern(self):
        detector = TestPatternDetector()
        data = _base_data(error_messages=["auth error"], total_tests=5)
        patterns = await detector._detect_security_patterns(data)
        sec = [p for p in patterns if p.pattern_type == TestPatternType.SECURITY_PATTERN]
        assert sec == []

    async def test_no_security_keywords(self):
        detector = TestPatternDetector()
        data = _base_data(error_messages=["some random error"], total_tests=30)
        patterns = await detector._detect_security_patterns(data)
        sec = [p for p in patterns if p.pattern_type == TestPatternType.SECURITY_PATTERN]
        assert sec == []


# ---------------------------------------------------------------------------
# _detect_unit_test_patterns
# ---------------------------------------------------------------------------


class TestUnitTestPatterns:
    async def test_high_variance_detected(self):
        detector = TestPatternDetector()
        data = _base_data(
            execution_times=[0.1] * 10,
            p95_execution_time=200,
            median_execution_time=1,
        )
        patterns = await detector._detect_unit_test_patterns(data)
        unit = [p for p in patterns if p.pattern_type == TestPatternType.UNIT_PATTERN]
        assert len(unit) == 1

    async def test_low_variance_no_pattern(self):
        detector = TestPatternDetector()
        data = _base_data(
            execution_times=[1.0] * 10,
            p95_execution_time=2,
            median_execution_time=1,
        )
        patterns = await detector._detect_unit_test_patterns(data)
        unit = [p for p in patterns if p.pattern_type == TestPatternType.UNIT_PATTERN]
        assert unit == []

    async def test_few_execution_times_no_pattern(self):
        """<=5 execution times should not trigger unit pattern."""
        detector = TestPatternDetector()
        data = _base_data(
            execution_times=[0.1] * 5,
            p95_execution_time=200,
            median_execution_time=1,
        )
        patterns = await detector._detect_unit_test_patterns(data)
        unit = [p for p in patterns if p.pattern_type == TestPatternType.UNIT_PATTERN]
        assert unit == []

    async def test_median_zero_no_division_error(self):
        detector = TestPatternDetector()
        data = _base_data(
            execution_times=[0.0] * 10,
            p95_execution_time=100,
            median_execution_time=0,
        )
        # Should not raise ZeroDivisionError
        patterns = await detector._detect_unit_test_patterns(data)
        unit = [p for p in patterns if p.pattern_type == TestPatternType.UNIT_PATTERN]
        assert unit == []

    async def test_empty_execution_times(self):
        detector = TestPatternDetector()
        data = _base_data(execution_times=[])
        patterns = await detector._detect_unit_test_patterns(data)
        assert patterns == []


# ---------------------------------------------------------------------------
# _detect_integration_patterns
# ---------------------------------------------------------------------------


class TestIntegrationPatterns:
    async def test_high_slow_ratio(self):
        config = AITestInsightsConfig(slow_test_threshold=5.0)
        detector = TestPatternDetector(config)
        # 4 out of 10 tests are slow (40% > 30%)
        data = _base_data(
            execution_times=[6.0] * 4 + [1.0] * 6,
            total_tests=10,
        )
        patterns = await detector._detect_integration_patterns(data)
        integ = [p for p in patterns if p.pattern_type == TestPatternType.INTEGRATION_PATTERN]
        assert len(integ) == 1

    async def test_low_slow_ratio_no_pattern(self):
        config = AITestInsightsConfig(slow_test_threshold=5.0)
        detector = TestPatternDetector(config)
        # 1 out of 10 tests slow (10% < 30%)
        data = _base_data(
            execution_times=[6.0] + [1.0] * 9,
            total_tests=10,
        )
        patterns = await detector._detect_integration_patterns(data)
        integ = [p for p in patterns if p.pattern_type == TestPatternType.INTEGRATION_PATTERN]
        assert integ == []

    async def test_zero_total_tests_no_division_error(self):
        detector = TestPatternDetector()
        data = _base_data(total_tests=0, execution_times=[])
        patterns = await detector._detect_integration_patterns(data)
        assert patterns == []


# ---------------------------------------------------------------------------
# _get_slow_tests
# ---------------------------------------------------------------------------


class TestGetSlowTests:
    def test_returns_slow_test_names(self):
        config = AITestInsightsConfig(slow_test_threshold=5.0)
        detector = TestPatternDetector(config)
        data = _base_data(execution_times=[1, 2, 6, 7, 3, 8])
        result = detector._get_slow_tests(data)
        assert result == ["test_2", "test_3", "test_5"]

    def test_capped_at_five(self):
        config = AITestInsightsConfig(slow_test_threshold=1.0)
        detector = TestPatternDetector(config)
        data = _base_data(execution_times=[5.0] * 10)
        result = detector._get_slow_tests(data)
        assert len(result) == 5

    def test_empty_times(self):
        detector = TestPatternDetector()
        data = _base_data(execution_times=[])
        assert detector._get_slow_tests(data) == []


# ---------------------------------------------------------------------------
# _get_flaky_tests
# ---------------------------------------------------------------------------


class TestGetFlakyTests:
    def test_returns_up_to_five(self):
        detector = TestPatternDetector()
        result = detector._get_flaky_tests({}, 10)
        assert len(result) == 5

    def test_returns_exact_count_when_small(self):
        detector = TestPatternDetector()
        result = detector._get_flaky_tests({}, 3)
        assert result == ["flaky_test_0", "flaky_test_1", "flaky_test_2"]

    def test_zero_count(self):
        detector = TestPatternDetector()
        result = detector._get_flaky_tests({}, 0)
        assert result == []


# ---------------------------------------------------------------------------
# _analyze_error_patterns
# ---------------------------------------------------------------------------


class TestAnalyzeErrorPatterns:
    def test_timeout_category(self):
        detector = TestPatternDetector()
        result = detector._analyze_error_patterns(["Request timeout"])
        assert result == {"timeout": 1}

    def test_assertion_category(self):
        detector = TestPatternDetector()
        result = detector._analyze_error_patterns(["AssertionError: 1 != 2"])
        assert result == {"assertion": 1}

    def test_assert_keyword(self):
        detector = TestPatternDetector()
        result = detector._analyze_error_patterns(["assert failed"])
        assert result == {"assertion": 1}

    def test_connection_category(self):
        detector = TestPatternDetector()
        result = detector._analyze_error_patterns(["Connection refused"])
        assert result == {"connection": 1}

    def test_network_keyword(self):
        detector = TestPatternDetector()
        result = detector._analyze_error_patterns(["Network unreachable"])
        assert result == {"connection": 1}

    def test_permission_category(self):
        detector = TestPatternDetector()
        result = detector._analyze_error_patterns(["Permission denied"])
        assert result == {"permission": 1}

    def test_auth_keyword(self):
        detector = TestPatternDetector()
        result = detector._analyze_error_patterns(["Auth failed"])
        assert result == {"permission": 1}

    def test_memory_category(self):
        detector = TestPatternDetector()
        result = detector._analyze_error_patterns(["Memory limit exceeded"])
        assert result == {"memory": 1}

    def test_oom_keyword(self):
        detector = TestPatternDetector()
        result = detector._analyze_error_patterns(["OOM killer"])
        assert result == {"memory": 1}

    def test_other_category(self):
        detector = TestPatternDetector()
        result = detector._analyze_error_patterns(["Something weird happened"])
        assert result == {"other": 1}

    def test_mixed_errors(self):
        detector = TestPatternDetector()
        errors = [
            "Timeout occurred",
            "Timeout again",
            "AssertionError",
            "Connection lost",
            "Something else",
        ]
        result = detector._analyze_error_patterns(errors)
        assert result["timeout"] == 2
        assert result["assertion"] == 1
        assert result["connection"] == 1
        assert result["other"] == 1

    def test_empty_list(self):
        detector = TestPatternDetector()
        result = detector._analyze_error_patterns([])
        assert result == {}

    def test_empty_string_skipped(self):
        detector = TestPatternDetector()
        result = detector._analyze_error_patterns(["", ""])
        assert result == {}
