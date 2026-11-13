"""
Metrics and calculations module for intelligent testing.
Contains metric calculation and analysis utilities.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

# TODO: IntelligentTestExecutionResult - service not yet extracted
# from services.test_execution.intelligent_test_orchestrator import (
#     IntelligentTestExecutionResult,
# )

logger = logging.getLogger(__name__)


class ExecutionMetricsCalculator:
    """Calculator for test execution metrics."""

    @staticmethod
    def calculate_success_rate(results: list) -> float:
        """Calculate success rate from execution results.

        Args:
            results: List of test execution results

        Returns:
            Success rate as percentage (0-100)
        """
        if not results:
            return 0.0

        successful_tests = sum(1 for result in results if result.success)
        return (successful_tests / len(results)) * 100

    @staticmethod
    def calculate_average_execution_time(
        results: list,
    ) -> float:
        """Calculate average execution time.

        Args:
            results: List of test execution results

        Returns:
            Average execution time in milliseconds
        """
        if not results:
            return 0.0

        execution_times = [
            result.execution_time_ms for result in results if result.execution_time_ms is not None
        ]

        if not execution_times:
            return 0.0

        return sum(execution_times) / len(execution_times)

    @staticmethod
    def calculate_percentiles(
        values: list[float], percentiles: list[int] = [50, 90, 95, 99]
    ) -> dict[int, float]:
        """Calculate percentiles for a list of values.

        Args:
            values: List of numeric values
            percentiles: List of percentile values to calculate

        Returns:
            Dictionary mapping percentile to value
        """
        if not values:
            return {p: 0.0 for p in percentiles}

        sorted_values = sorted(values)
        result = {}

        for percentile in percentiles:
            index = int((percentile / 100) * (len(sorted_values) - 1))
            result[percentile] = sorted_values[index]

        return result

    @staticmethod
    def detect_performance_anomalies(
        results: list, threshold_multiplier: float = 2.0
    ) -> list[dict[str, Any]]:
        """Detect performance anomalies in execution results.

        Args:
            results: List of test execution results
            threshold_multiplier: Multiplier for anomaly detection threshold

        Returns:
            List of detected anomalies
        """
        anomalies: list[dict[str, Any]] = []

        if len(results) < 3:  # Need at least 3 results for meaningful detection
            return anomalies

        execution_times = [
            result.execution_time_ms for result in results if result.execution_time_ms is not None
        ]

        if not execution_times:
            return anomalies

        avg_time = sum(execution_times) / len(execution_times)
        threshold = avg_time * threshold_multiplier

        for i, result in enumerate(results):
            if result.execution_time_ms is not None and result.execution_time_ms > threshold:
                anomalies.append(
                    {
                        "index": i,
                        "execution_time_ms": result.execution_time_ms,
                        "average_time_ms": avg_time,
                        "threshold_ms": threshold,
                        "anomaly_type": "slow_execution",
                        "severity": (
                            "high" if result.execution_time_ms > threshold * 1.5 else "medium"
                        ),
                    }
                )

        return anomalies


class ResourceEstimator:
    """Utility for estimating resource requirements."""

    @staticmethod
    def estimate_memory_requirement(code: str, language: str, test_type: str = "unit") -> int:
        """Estimate memory requirement in MB.

        Args:
            code: Source code
            language: Programming language
            test_type: Type of test

        Returns:
            Estimated memory requirement in MB
        """
        base_memory = {
            "python": 128,
            "javascript": 64,
            "java": 256,
            "go": 64,
            "rust": 64,
            "cpp": 64,
        }

        base = base_memory.get(language.lower(), 128)

        # Adjust based on code size
        code_lines = len(code.split("\n"))
        size_multiplier = max(1.0, code_lines / 100)

        # Adjust based on test type
        type_multipliers = {
            "unit": 1.0,
            "integration": 1.5,
            "e2e": 2.0,
            "performance": 1.8,
            "security": 1.3,
        }

        type_multiplier = type_multipliers.get(test_type, 1.0)

        estimated_memory = int(base * size_multiplier * type_multiplier)
        return min(max(estimated_memory, 64), 2048)  # Clamp between 64MB and 2GB

    @staticmethod
    def estimate_cpu_requirement(code: str, language: str, test_type: str = "unit") -> float:
        """Estimate CPU requirement (cores).

        Args:
            code: Source code
            language: Programming language
            test_type: Type of test

        Returns:
            Estimated CPU cores needed
        """
        base_cpu = {
            "python": 0.5,
            "javascript": 0.3,
            "java": 0.8,
            "go": 0.4,
            "rust": 0.6,
            "cpp": 0.7,
        }

        base = base_cpu.get(language.lower(), 0.5)

        # Adjust based on test type
        type_multipliers = {
            "unit": 1.0,
            "integration": 1.2,
            "e2e": 1.5,
            "performance": 2.0,
            "security": 1.3,
        }

        type_multiplier = type_multipliers.get(test_type, 1.0)

        estimated_cpu = base * type_multiplier
        return min(max(estimated_cpu, 0.1), 4.0)  # Clamp between 0.1 and 4 cores


class TestCodeAnalyzer:
    """Analyzer for test code quality and characteristics."""

    @staticmethod
    def estimate_complexity(code: str, language: str) -> dict[str, Any]:
        """Estimate code complexity based on simple metrics.

        Args:
            code: Source code to analyze
            language: Programming language

        Returns:
            Dictionary containing complexity metrics
        """
        lines = code.split("\n")
        non_empty_lines = [line for line in lines if line.strip()]

        # Simple complexity indicators
        complexity_indicators = {
            "python": ["if ", "for ", "while ", "try:", "except:", "with "],
            "javascript": ["if ", "for ", "while ", "try", "catch", "switch"],
            "java": ["if ", "for ", "while ", "try", "catch", "switch"],
            "go": ["if ", "for ", "switch", "select"],
            "rust": ["if ", "for ", "while ", "match", "loop"],
            "cpp": ["if ", "for ", "while ", "try", "catch", "switch"],
        }

        indicators = complexity_indicators.get(language.lower(), ["if ", "for ", "while "])

        complexity_count = 0
        for line in non_empty_lines:
            for indicator in indicators:
                if indicator in line.lower():
                    complexity_count += 1

        # Calculate complexity score
        if len(non_empty_lines) == 0:
            complexity_score = 0.0
        else:
            complexity_score = complexity_count / len(non_empty_lines)

        # Categorize complexity
        if complexity_score < 0.1:
            complexity_level = "low"
        elif complexity_score < 0.3:
            complexity_level = "medium"
        else:
            complexity_level = "high"

        return {
            "total_lines": len(lines),
            "code_lines": len(non_empty_lines),
            "complexity_indicators": complexity_count,
            "complexity_score": complexity_score,
            "complexity_level": complexity_level,
            "estimated_execution_time_ms": min(max(len(non_empty_lines) * 10, 100), 30000),
        }

    @staticmethod
    def detect_test_patterns(code: str, language: str) -> list[str]:
        """Detect common test patterns in code.

        Args:
            code: Test code to analyze
            language: Programming language

        Returns:
            List of detected test patterns
        """
        patterns = []
        code_lower = code.lower()

        # Common test patterns by language
        test_patterns = {
            "python": {
                "unit_test": ["unittest", "pytest", "assert"],
                "mock_usage": ["mock", "patch", "@mock"],
                "fixtures": ["fixture", "setup", "teardown"],
                "parameterized": ["parametrize", "@pytest.mark.parametrize"],
            },
            "javascript": {
                "unit_test": ["describe", "it(", "test(", "expect"],
                "mock_usage": ["jest.mock", "sinon", "spy"],
                "async_test": ["async", "await", "promise"],
                "fixtures": ["beforeeach", "aftereach"],
            },
            "java": {
                "unit_test": ["@test", "junit", "assert"],
                "mock_usage": ["mockito", "@mock", "when("],
                "fixtures": ["@before", "@after", "@setup"],
            },
        }

        language_patterns = test_patterns.get(language.lower(), {})

        for pattern_type, keywords in language_patterns.items():
            for keyword in keywords:
                if keyword in code_lower:
                    patterns.append(pattern_type)
                    break

        return list(set(patterns))  # Remove duplicates


class TimestampHelper:
    """Helper for timestamp operations."""

    @staticmethod
    def utc_now() -> datetime:
        """Get current UTC timestamp."""
        return datetime.now(UTC)

    @staticmethod
    def format_timestamp(dt: datetime) -> str:
        """Format timestamp as ISO string."""
        return dt.isoformat()

    @staticmethod
    def parse_timestamp(timestamp_str: str) -> datetime:
        """Parse timestamp from ISO string."""
        return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

    @staticmethod
    def calculate_duration_ms(start: datetime, end: datetime) -> float:
        """Calculate duration in milliseconds."""
        delta = end - start
        return delta.total_seconds() * 1000
