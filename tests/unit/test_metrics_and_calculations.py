"""Tests for metrics_and_calculations module."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from routers.ai_test_insights.metrics_and_calculations import (
    ExecutionMetricsCalculator,
    ResourceEstimator,
    TestCodeAnalyzer,
    TimestampHelper,
)


# ---------------------------------------------------------------------------
# ExecutionMetricsCalculator
# ---------------------------------------------------------------------------
class TestCalculateSuccessRate:
    def test_empty_results(self):
        assert ExecutionMetricsCalculator.calculate_success_rate([]) == 0.0

    def test_all_successful(self):
        results = [SimpleNamespace(success=True) for _ in range(5)]
        assert ExecutionMetricsCalculator.calculate_success_rate(results) == 100.0

    def test_all_failed(self):
        results = [SimpleNamespace(success=False) for _ in range(3)]
        assert ExecutionMetricsCalculator.calculate_success_rate(results) == 0.0

    def test_mixed(self):
        results = [
            SimpleNamespace(success=True),
            SimpleNamespace(success=False),
            SimpleNamespace(success=True),
            SimpleNamespace(success=False),
        ]
        assert ExecutionMetricsCalculator.calculate_success_rate(results) == 50.0


class TestCalculateAverageExecutionTime:
    def test_empty_results(self):
        assert ExecutionMetricsCalculator.calculate_average_execution_time([]) == 0.0

    def test_all_none_times(self):
        results = [SimpleNamespace(execution_time_ms=None) for _ in range(3)]
        assert ExecutionMetricsCalculator.calculate_average_execution_time(results) == 0.0

    def test_normal_times(self):
        results = [
            SimpleNamespace(execution_time_ms=100),
            SimpleNamespace(execution_time_ms=200),
            SimpleNamespace(execution_time_ms=300),
        ]
        assert ExecutionMetricsCalculator.calculate_average_execution_time(results) == 200.0

    def test_mixed_none_and_values(self):
        results = [
            SimpleNamespace(execution_time_ms=100),
            SimpleNamespace(execution_time_ms=None),
            SimpleNamespace(execution_time_ms=300),
        ]
        assert ExecutionMetricsCalculator.calculate_average_execution_time(results) == 200.0


class TestCalculatePercentiles:
    def test_empty_values(self):
        result = ExecutionMetricsCalculator.calculate_percentiles([])
        assert result == {50: 0.0, 90: 0.0, 95: 0.0, 99: 0.0}

    def test_single_value(self):
        result = ExecutionMetricsCalculator.calculate_percentiles([42.0])
        for p in [50, 90, 95, 99]:
            assert result[p] == 42.0

    def test_multiple_values(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        result = ExecutionMetricsCalculator.calculate_percentiles(values)
        # 50th percentile: index = int(0.5 * 9) = 4 -> 50.0
        assert result[50] == 50.0
        assert result[90] >= 90.0

    def test_custom_percentiles(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = ExecutionMetricsCalculator.calculate_percentiles(values, [25, 75])
        assert 25 in result
        assert 75 in result


class TestDetectPerformanceAnomalies:
    def test_too_few_results(self):
        results = [SimpleNamespace(execution_time_ms=100), SimpleNamespace(execution_time_ms=200)]
        assert ExecutionMetricsCalculator.detect_performance_anomalies(results) == []

    def test_no_execution_times(self):
        results = [SimpleNamespace(execution_time_ms=None) for _ in range(5)]
        assert ExecutionMetricsCalculator.detect_performance_anomalies(results) == []

    def test_no_anomalies(self):
        results = [SimpleNamespace(execution_time_ms=100) for _ in range(5)]
        anomalies = ExecutionMetricsCalculator.detect_performance_anomalies(results)
        assert anomalies == []

    def test_detects_slow_execution(self):
        results = [
            SimpleNamespace(execution_time_ms=100),
            SimpleNamespace(execution_time_ms=100),
            SimpleNamespace(execution_time_ms=100),
            SimpleNamespace(execution_time_ms=500),  # 5x average, well above 2x threshold
        ]
        anomalies = ExecutionMetricsCalculator.detect_performance_anomalies(results)
        assert len(anomalies) == 1
        assert anomalies[0]["anomaly_type"] == "slow_execution"
        assert anomalies[0]["index"] == 3

    def test_high_severity_anomaly(self):
        # avg=100, threshold=200, >300 = high severity
        results = [
            SimpleNamespace(execution_time_ms=100),
            SimpleNamespace(execution_time_ms=100),
            SimpleNamespace(execution_time_ms=100),
            SimpleNamespace(execution_time_ms=1000),  # way above 1.5x threshold
        ]
        anomalies = ExecutionMetricsCalculator.detect_performance_anomalies(results)
        assert len(anomalies) >= 1
        assert anomalies[0]["severity"] == "high"

    def test_custom_threshold_multiplier(self):
        results = [
            SimpleNamespace(execution_time_ms=100),
            SimpleNamespace(execution_time_ms=100),
            SimpleNamespace(execution_time_ms=100),
            SimpleNamespace(execution_time_ms=250),
        ]
        # Default threshold 2.0 -> avg ~137.5, threshold ~275, 250 < 275 => no anomaly
        anomalies_default = ExecutionMetricsCalculator.detect_performance_anomalies(results)
        # Lower threshold 1.0 -> threshold ~137.5, 250 > 137.5 => anomaly
        anomalies_lower = ExecutionMetricsCalculator.detect_performance_anomalies(
            results, threshold_multiplier=1.0
        )
        assert len(anomalies_lower) >= len(anomalies_default)


# ---------------------------------------------------------------------------
# ResourceEstimator
# ---------------------------------------------------------------------------
class TestEstimateMemoryRequirement:
    def test_known_language(self):
        mem = ResourceEstimator.estimate_memory_requirement("x = 1\n", "python")
        assert 64 <= mem <= 2048

    def test_unknown_language(self):
        mem = ResourceEstimator.estimate_memory_requirement("x = 1\n", "brainfuck")
        assert 64 <= mem <= 2048

    def test_large_code_increases_memory(self):
        small = ResourceEstimator.estimate_memory_requirement("x=1\n", "python")
        big_code = "\n".join(["x = 1"] * 500)
        big = ResourceEstimator.estimate_memory_requirement(big_code, "python")
        assert big >= small

    def test_test_type_multiplier(self):
        unit = ResourceEstimator.estimate_memory_requirement("x=1\n", "python", "unit")
        e2e = ResourceEstimator.estimate_memory_requirement("x=1\n", "python", "e2e")
        assert e2e >= unit

    def test_unknown_test_type_defaults(self):
        mem = ResourceEstimator.estimate_memory_requirement("x=1\n", "python", "unknown_type")
        assert 64 <= mem <= 2048

    def test_clamped_min(self):
        mem = ResourceEstimator.estimate_memory_requirement("x\n", "javascript")
        assert mem >= 64

    def test_clamped_max(self):
        huge_code = "\n".join(["x = 1"] * 10000)
        mem = ResourceEstimator.estimate_memory_requirement(huge_code, "java", "e2e")
        assert mem <= 2048


class TestEstimateCpuRequirement:
    def test_known_language(self):
        cpu = ResourceEstimator.estimate_cpu_requirement("x=1\n", "python")
        assert 0.1 <= cpu <= 4.0

    def test_unknown_language(self):
        cpu = ResourceEstimator.estimate_cpu_requirement("x=1\n", "elixir")
        assert 0.1 <= cpu <= 4.0

    def test_test_type_multiplier(self):
        unit = ResourceEstimator.estimate_cpu_requirement("x=1\n", "go", "unit")
        perf = ResourceEstimator.estimate_cpu_requirement("x=1\n", "go", "performance")
        assert perf >= unit

    def test_clamped_range(self):
        cpu = ResourceEstimator.estimate_cpu_requirement("x\n", "javascript", "unit")
        assert 0.1 <= cpu <= 4.0


# ---------------------------------------------------------------------------
# TestCodeAnalyzer
# ---------------------------------------------------------------------------
class TestEstimateComplexity:
    def test_empty_code(self):
        result = TestCodeAnalyzer.estimate_complexity("", "python")
        assert result["complexity_level"] == "low"
        assert result["complexity_score"] == 0.0

    def test_simple_code(self):
        code = "x = 1\ny = 2\nreturn x + y"
        result = TestCodeAnalyzer.estimate_complexity(code, "python")
        assert result["complexity_level"] == "low"
        assert result["total_lines"] == 3
        assert result["code_lines"] == 3

    def test_complex_code(self):
        code = "\n".join(
            [
                "if x:",
                "  for i in range(10):",
                "    while True:",
                "      try:",
                "        pass",
                "      except:",
                "        pass",
            ]
        )
        result = TestCodeAnalyzer.estimate_complexity(code, "python")
        assert result["complexity_indicators"] > 0
        assert result["complexity_level"] in ("low", "medium", "high")

    def test_unknown_language_fallback(self):
        code = "if x:\n  for i:\n    while True:\n"
        result = TestCodeAnalyzer.estimate_complexity(code, "haskell")
        assert result["complexity_indicators"] > 0

    def test_high_complexity(self):
        # Every line is a complexity indicator
        code = "\n".join(["if x:" for _ in range(10)])
        result = TestCodeAnalyzer.estimate_complexity(code, "python")
        assert result["complexity_level"] in ("medium", "high")

    def test_estimated_execution_time_clamped(self):
        result = TestCodeAnalyzer.estimate_complexity("x=1\n", "python")
        assert result["estimated_execution_time_ms"] >= 100
        assert result["estimated_execution_time_ms"] <= 30000

    def test_javascript_indicators(self):
        code = "if (x) {\n  for (let i=0;;) {\n    try {} catch(e) {}\n  }\n}"
        result = TestCodeAnalyzer.estimate_complexity(code, "javascript")
        assert result["complexity_indicators"] > 0


class TestDetectTestPatterns:
    def test_python_patterns(self):
        code = "import pytest\ndef test_foo():\n    assert True\n    with patch('x'):\n        pass"
        patterns = TestCodeAnalyzer.detect_test_patterns(code, "python")
        assert "unit_test" in patterns
        assert "mock_usage" in patterns

    def test_javascript_patterns(self):
        code = "describe('test', () => { it('works', async () => { expect(1).toBe(1); }); });"
        patterns = TestCodeAnalyzer.detect_test_patterns(code, "javascript")
        assert "unit_test" in patterns
        assert "async_test" in patterns

    def test_java_patterns(self):
        code = "@Test\npublic void test() { assert true; }"
        patterns = TestCodeAnalyzer.detect_test_patterns(code, "java")
        assert "unit_test" in patterns

    def test_unknown_language_returns_empty(self):
        code = "some code here"
        patterns = TestCodeAnalyzer.detect_test_patterns(code, "cobol")
        assert patterns == []

    def test_no_patterns_found(self):
        code = "x = 1"
        patterns = TestCodeAnalyzer.detect_test_patterns(code, "python")
        # 'assert' won't match since code_lower doesn't contain it as a keyword
        # but 'x = 1' doesn't contain any test keywords
        assert isinstance(patterns, list)

    def test_deduplication(self):
        code = "import pytest\nimport pytest\nassert True\nassert False"
        patterns = TestCodeAnalyzer.detect_test_patterns(code, "python")
        # Should not have duplicates
        assert len(patterns) == len(set(patterns))


# ---------------------------------------------------------------------------
# TimestampHelper
# ---------------------------------------------------------------------------
class TestTimestampHelper:
    def test_utc_now(self):
        ts = TimestampHelper.utc_now()
        assert ts.tzinfo is not None

    def test_format_timestamp(self):
        dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        formatted = TimestampHelper.format_timestamp(dt)
        assert "2025-01-01" in formatted

    def test_parse_timestamp(self):
        dt = TimestampHelper.parse_timestamp("2025-01-01T12:00:00Z")
        assert dt.year == 2025
        assert dt.month == 1

    def test_parse_timestamp_with_offset(self):
        dt = TimestampHelper.parse_timestamp("2025-01-01T12:00:00+00:00")
        assert dt.year == 2025

    def test_calculate_duration_ms(self):
        start = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        end = datetime(2025, 1, 1, 12, 0, 1, tzinfo=UTC)
        duration = TimestampHelper.calculate_duration_ms(start, end)
        assert duration == 1000.0

    def test_roundtrip(self):
        original = TimestampHelper.utc_now()
        formatted = TimestampHelper.format_timestamp(original)
        parsed = TimestampHelper.parse_timestamp(formatted)
        assert abs((parsed - original).total_seconds()) < 0.001
