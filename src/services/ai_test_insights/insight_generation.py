"""Insight generation and scoring module for AI Test Insights Service.

This module contains logic for analyzing test data and generating insights
about performance, reliability, coverage, and quality metrics.
"""

import logging
from typing import Any

from models.testing_models import QualityIssue
from services.ai_test_insights.config import (
    DEFAULT_CONFIG,
    AITestInsightsConfig,
    TestInsightType,
    clone_config,
)
from services.ai_test_insights.models import (
    TestCoverageInsights,
    TestInsightRecommendation,
    TestPatternInsight,
    TestPerformanceMetrics,
    TestQualityInsights,
    TestReliabilityMetrics,
)

logger = logging.getLogger(__name__)


class TestInsightGenerator:
    __test__ = False
    """Generates insights from test data and analysis results."""

    def __init__(self, config: AITestInsightsConfig = DEFAULT_CONFIG):
        self.config = clone_config(config)

    async def analyze_performance(self, data: dict[str, Any]) -> TestPerformanceMetrics:
        """Analyze test performance metrics."""
        slow_tests = [
            i
            for i, t in enumerate(data.get("execution_times", []))
            if t > self.config.slow_test_threshold
        ]

        performance_trend = self._analyze_performance_trend(data)
        bottleneck_categories = self._identify_bottlenecks(data)
        avg_memory = self._estimate_memory_usage(data)
        peak_memory = avg_memory * 1.5
        avg_cpu = self._estimate_cpu_usage(data)

        return TestPerformanceMetrics(
            average_execution_time=data.get("avg_execution_time", 0.0),
            median_execution_time=data.get("median_execution_time", 0.0),
            p95_execution_time=data.get("p95_execution_time", 0.0),
            p99_execution_time=data.get("p99_execution_time", 0.0),
            average_memory_usage=avg_memory,
            peak_memory_usage=peak_memory,
            average_cpu_usage=avg_cpu,
            slow_tests_count=len(slow_tests),
            fast_tests_count=data.get("total_tests", 0) - len(slow_tests),
            performance_trend=performance_trend,
            bottleneck_categories=bottleneck_categories,
        )

    async def analyze_reliability(self, data: dict[str, Any]) -> TestReliabilityMetrics:
        """Analyze test reliability metrics."""
        success_rate = data.get("overall_success_rate", 0.0)
        success_rate_by_type = self._calculate_success_rates_by_type(data)
        flaky_tests_count = self._estimate_flaky_tests(data)
        error_categories = self._analyze_error_patterns(data.get("error_messages", []))
        common_failure_patterns = list(error_categories.keys())[:5]
        stability_score = self._calculate_stability_score(success_rate, error_categories)
        reliability_trend = self._analyze_reliability_trend(data)

        return TestReliabilityMetrics(
            overall_success_rate=success_rate,
            success_rate_by_type=success_rate_by_type,
            common_failure_patterns=common_failure_patterns,
            flaky_tests_count=flaky_tests_count,
            stability_score=stability_score,
            reliability_trend=reliability_trend,
            error_categories=error_categories,
        )

    async def analyze_coverage(self, data: dict[str, Any]) -> TestCoverageInsights:
        """Analyze test coverage insights."""
        base_coverage = self._estimate_coverage(data)
        coverage_gaps = self._identify_coverage_gaps(data, base_coverage)
        uncovered_critical_paths = self._identify_uncovered_critical_paths(data, base_coverage)
        redundant_tests = self._detect_redundant_tests(data)
        recommended_tests = self._generate_test_recommendations(data, base_coverage)

        return TestCoverageInsights(
            line_coverage=base_coverage,
            branch_coverage=base_coverage * 0.9,
            function_coverage=base_coverage * 0.95,
            uncovered_critical_paths=uncovered_critical_paths,
            redundant_tests=redundant_tests,
            coverage_gaps=coverage_gaps,
            recommended_tests=recommended_tests,
        )

    async def analyze_quality(self, data: dict[str, Any]) -> TestQualityInsights:
        """Analyze test quality insights."""
        quality_scores = data.get("quality_scores", [])
        avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 75.0

        quality_issues = data.get("quality_issues", [])
        critical_issues = [
            issue
            for issue in quality_issues
            if hasattr(issue, "severity") and issue.severity in ["critical", "error"]
        ]

        quality_debt = self._calculate_quality_debt(quality_issues)
        quality_recommendations = self._generate_quality_recommendations(
            quality_issues, avg_quality_score
        )
        automated_fixes = self._identify_automated_fixes(quality_issues)

        return TestQualityInsights(
            test_quality_score=avg_quality_score,
            code_quality_score=avg_quality_score,
            critical_issues=critical_issues,
            quality_debt=quality_debt,
            quality_recommendations=quality_recommendations,
            automated_fixes=automated_fixes,
        )

    async def generate_recommendations(
        self,
        performance_metrics: TestPerformanceMetrics,
        reliability_metrics: TestReliabilityMetrics,
        coverage_insights: TestCoverageInsights,
        quality_insights: TestQualityInsights,
        detected_patterns: list[TestPatternInsight],
    ) -> list[TestInsightRecommendation]:
        """Generate AI-powered recommendations based on analysis results."""
        recommendations = []

        perf_recs = self._generate_performance_recommendations(performance_metrics)
        recommendations.extend(perf_recs)

        rel_recs = self._generate_reliability_recommendations(reliability_metrics)
        recommendations.extend(rel_recs)

        cov_recs = self._generate_coverage_recommendations(coverage_insights)
        recommendations.extend(cov_recs)

        qual_recs = self._generate_quality_recommendations_detailed(quality_insights)
        recommendations.extend(qual_recs)

        pattern_recs = self._generate_pattern_recommendations(detected_patterns)
        recommendations.extend(pattern_recs)

        recommendations.sort(key=lambda r: (r.priority == "high", r.confidence), reverse=True)

        return recommendations

    def _analyze_performance_trend(self, data: dict[str, Any]) -> str:
        avg_time = data.get("avg_execution_time", 0)
        if avg_time > self.config.slow_test_threshold * 2:
            return "degrading"
        elif avg_time < self.config.slow_test_threshold * 0.5:
            return "improving"
        else:
            return "stable"

    def _identify_bottlenecks(self, data: dict[str, Any]) -> list[str]:
        bottlenecks = []
        avg_time = data.get("avg_execution_time", 0)
        if avg_time > self.config.slow_test_threshold:
            bottlenecks.append("execution_time")

        execution_times = data.get("execution_times", [])
        if execution_times:
            variance = self._calculate_variance(execution_times)
            if variance > avg_time:
                bottlenecks.append("inconsistent_performance")

        results_by_language = data.get("results_by_language", {})
        if len(results_by_language) > 1:
            bottlenecks.append("multi_language_overhead")

        return bottlenecks

    def _estimate_memory_usage(self, data: dict[str, Any]) -> float:
        base_memory = 128.0
        test_count = data.get("total_tests", 0)
        memory_per_test = 2.0
        return base_memory + (test_count * memory_per_test)

    def _estimate_cpu_usage(self, data: dict[str, Any]) -> float:
        avg_time = data.get("avg_execution_time", 0)
        base_cpu = 25.0
        cpu_factor = min(avg_time * 5, 50)
        return base_cpu + cpu_factor

    def _calculate_success_rates_by_type(self, data: dict[str, Any]) -> dict[str, float]:
        results_by_language = data.get("results_by_language", {})
        success_rates = {}
        for language, results in results_by_language.items():
            if results:
                success_count = sum(1 for r in results if r.success)
                success_rates[language] = success_count / len(results)
        return success_rates

    def _estimate_flaky_tests(self, data: dict[str, Any]) -> int:
        success_rate = data.get("overall_success_rate", 1.0)
        total_tests = data.get("total_tests", 0)
        failure_rate = 1 - success_rate
        flaky_ratio = 0.3
        return max(0, int(total_tests * failure_rate * flaky_ratio))

    def _analyze_error_patterns(self, error_messages: list[str]) -> dict[str, int]:
        error_categories: dict[str, int] = {}
        for error in error_messages:
            if not error:
                continue
            error_lower = error.lower()
            if "timeout" in error_lower:
                error_categories["timeout"] = error_categories.get("timeout", 0) + 1
            elif "assertion" in error_lower or "assert" in error_lower:
                error_categories["assertion"] = error_categories.get("assertion", 0) + 1
            elif "connection" in error_lower or "network" in error_lower:
                error_categories["connection"] = error_categories.get("connection", 0) + 1
            elif "permission" in error_lower or "auth" in error_lower:
                error_categories["permission"] = error_categories.get("permission", 0) + 1
            elif "memory" in error_lower or "oom" in error_lower:
                error_categories["memory"] = error_categories.get("memory", 0) + 1
            else:
                error_categories["other"] = error_categories.get("other", 0) + 1
        return error_categories

    def _calculate_stability_score(
        self, success_rate: float, error_categories: dict[str, int]
    ) -> float:
        base_score = success_rate * 100
        if error_categories:
            total_errors = sum(error_categories.values())
            max_category_errors = max(error_categories.values())
            concentration_penalty = (max_category_errors / total_errors) * 10
            base_score -= concentration_penalty
        return max(0.0, min(100.0, base_score))

    def _analyze_reliability_trend(self, data: dict[str, Any]) -> str:
        success_rate = data.get("overall_success_rate", 1.0)
        if success_rate >= 0.95:
            return "stable"
        elif success_rate >= 0.85:
            return "declining"
        else:
            return "degrading"

    def _estimate_coverage(self, data: dict[str, Any]) -> float:
        success_rate = data.get("overall_success_rate", 0.0)
        test_count = data.get("total_tests", 0)
        base_coverage = 60.0 + (success_rate * 30)
        if test_count > 50:
            coverage_bonus = min(10, test_count / 10)
            base_coverage += coverage_bonus
        return min(100.0, max(0.0, base_coverage))

    def _identify_coverage_gaps(self, data: dict[str, Any], coverage: float) -> list[str]:
        gaps = []
        if coverage < self.config.coverage_threshold:
            gaps.append("Overall coverage below threshold")
        error_categories = self._analyze_error_patterns(data.get("error_messages", []))
        if "assertion" in error_categories:
            gaps.append("Business logic validation coverage")
        if "connection" in error_categories:
            gaps.append("External service integration coverage")
        if "permission" in error_categories:
            gaps.append("Security and authorization coverage")
        return gaps

    def _identify_uncovered_critical_paths(
        self, data: dict[str, Any], coverage: float
    ) -> list[str]:
        critical_paths = []
        if coverage < 80:
            critical_paths.extend(
                ["Error handling paths", "Edge case scenarios", "Security validation flows"]
            )
        if coverage < 70:
            critical_paths.extend(
                ["Main business logic flows", "Integration points", "Data validation logic"]
            )
        return critical_paths

    def _detect_redundant_tests(self, data: dict[str, Any]) -> list[str]:
        redundant = []
        execution_times = data.get("execution_times", [])
        if len(execution_times) > 20:
            time_groups: dict[float, list[int]] = {}
            for i, time in enumerate(execution_times):
                rounded_time = round(time, 1)
                if rounded_time not in time_groups:
                    time_groups[rounded_time] = []
                time_groups[rounded_time].append(i)
            for time, indices in time_groups.items():
                if len(indices) > 5:
                    redundant.append(f"Tests with {time}s execution time pattern")
        return redundant

    def _generate_test_recommendations(self, data: dict[str, Any], coverage: float) -> list[str]:
        recommendations = []
        if coverage < 85:
            recommendations.extend(
                [
                    "Add unit tests for core business logic",
                    "Add integration tests for critical workflows",
                    "Add negative test cases for error scenarios",
                ]
            )
        error_categories = self._analyze_error_patterns(data.get("error_messages", []))
        if "connection" in error_categories:
            recommendations.append("Add tests for external service failures")
        if "permission" in error_categories:
            recommendations.append("Add comprehensive authorization tests")
        return recommendations

    def _calculate_quality_debt(self, quality_issues: list[QualityIssue]) -> float:
        debt_hours = 0.0
        for issue in quality_issues:
            if hasattr(issue, "severity"):
                if issue.severity == "critical":
                    debt_hours += 4.0
                elif issue.severity == "error":
                    debt_hours += 2.0
                elif issue.severity == "warning":
                    debt_hours += 0.5
                else:
                    debt_hours += 0.25
            else:
                debt_hours += 0.5
        return debt_hours

    def _generate_quality_recommendations(
        self, quality_issues: list[QualityIssue], score: float
    ) -> list[str]:
        recommendations = []
        if score < 80:
            recommendations.extend(
                [
                    "Address critical quality issues first",
                    "Implement automated code formatting",
                    "Add comprehensive type annotations",
                    "Improve test documentation",
                ]
            )
        issue_types: dict[str, int] = {}
        for issue in quality_issues:
            if hasattr(issue, "category"):
                category = str(issue.category)
                issue_types[category] = issue_types.get(category, 0) + 1
        if "complexity" in issue_types:
            recommendations.append("Refactor complex functions")
        if "duplication" in issue_types:
            recommendations.append("Remove code duplication")
        if "style" in issue_types:
            recommendations.append("Fix code style issues")
        return recommendations

    def _identify_automated_fixes(self, quality_issues: list[QualityIssue]) -> list[str]:
        return [
            "Code formatting issues",
            "Import organization",
            "Basic type annotations",
            "Unused variable removal",
        ]

    def _generate_performance_recommendations(
        self, metrics: TestPerformanceMetrics
    ) -> list[TestInsightRecommendation]:
        recommendations = []
        if metrics.slow_tests_count > 0:
            recommendations.append(
                TestInsightRecommendation(
                    recommendation_id="perf_001",
                    type=TestInsightType.PERFORMANCE,
                    priority="high",
                    confidence=0.9,
                    title="Optimize Slow Tests",
                    description="Several tests are executing slowly, impacting overall test suite performance",
                    rationale=f"Found {metrics.slow_tests_count} slow tests with average execution time of {metrics.average_execution_time:.2f}s",
                    implementation_steps=[
                        "Profile slow tests to identify bottlenecks",
                        "Optimize database queries and I/O operations",
                        "Consider parallel test execution",
                        "Implement test data caching",
                    ],
                    estimated_effort="medium",
                    expected_impact="Reduce test suite execution time by 30-50%",
                    supporting_metrics={
                        "slow_tests_count": metrics.slow_tests_count,
                        "avg_execution_time": metrics.average_execution_time,
                    },
                    affected_components=["test_suite", "ci_pipeline"],
                )
            )
        return recommendations

    def _generate_reliability_recommendations(
        self, metrics: TestReliabilityMetrics
    ) -> list[TestInsightRecommendation]:
        recommendations = []
        if metrics.flaky_tests_count > 0:
            recommendations.append(
                TestInsightRecommendation(
                    recommendation_id="rel_001",
                    type=TestInsightType.RELIABILITY,
                    priority="high",
                    confidence=0.85,
                    title="Address Flaky Tests",
                    description="Flaky tests are reducing test suite reliability and developer confidence",
                    rationale=f"Detected {metrics.flaky_tests_count} flaky tests with {metrics.overall_success_rate:.1%} success rate",
                    implementation_steps=[
                        "Identify root causes of test flakiness",
                        "Improve test isolation and cleanup",
                        "Add explicit waits and assertions",
                        "Consider quarantining flaky tests",
                    ],
                    estimated_effort="high",
                    expected_impact="Improve test reliability by 15-25%",
                    supporting_metrics={
                        "flaky_tests_count": metrics.flaky_tests_count,
                        "success_rate": metrics.overall_success_rate,
                    },
                    affected_components=["test_suite", "ci_pipeline"],
                )
            )
        return recommendations

    def _generate_coverage_recommendations(
        self, insights: TestCoverageInsights
    ) -> list[TestInsightRecommendation]:
        recommendations = []
        if insights.overall_coverage_score < 80:
            recommendations.append(
                TestInsightRecommendation(
                    recommendation_id="cov_001",
                    type=TestInsightType.COVERAGE,
                    priority="medium",
                    confidence=0.8,
                    title="Improve Test Coverage",
                    description="Test coverage is below recommended threshold",
                    rationale=f"Current coverage is {insights.overall_coverage_score:.1f}%, below 80% threshold",
                    implementation_steps=[
                        "Identify uncovered critical code paths",
                        "Add unit tests for core functionality",
                        "Implement integration tests for user flows",
                        "Set up coverage reporting and thresholds",
                    ],
                    estimated_effort="medium",
                    expected_impact="Increase test coverage to 85%+",
                    supporting_metrics={
                        "line_coverage": insights.line_coverage,
                        "branch_coverage": insights.branch_coverage,
                    },
                    affected_components=["test_suite", "code_quality"],
                )
            )
        return recommendations

    def _generate_quality_recommendations_detailed(
        self, insights: TestQualityInsights
    ) -> list[TestInsightRecommendation]:
        recommendations = []
        if insights.critical_issues:
            recommendations.append(
                TestInsightRecommendation(
                    recommendation_id="qual_001",
                    type=TestInsightType.QUALITY,
                    priority="high",
                    confidence=0.9,
                    title="Address Critical Quality Issues",
                    description="Critical quality issues are impacting code maintainability",
                    rationale=f"Found {len(insights.critical_issues)} critical quality issues",
                    implementation_steps=[
                        "Review and resolve critical issues",
                        "Implement automated quality checks",
                        "Add pre-commit hooks for quality",
                        "Establish quality gates in CI",
                    ],
                    estimated_effort="medium",
                    expected_impact="Improve code quality score by 20-30%",
                    supporting_metrics={
                        "critical_issues_count": len(insights.critical_issues),
                        "quality_score": insights.test_quality_score,
                    },
                    affected_components=["code_quality", "ci_pipeline"],
                )
            )
        return recommendations

    def _generate_pattern_recommendations(
        self, patterns: list[TestPatternInsight]
    ) -> list[TestInsightRecommendation]:
        recommendations = []
        high_impact_patterns = [p for p in patterns if p.impact_score > 7]
        if high_impact_patterns:
            recommendations.append(
                TestInsightRecommendation(
                    recommendation_id="pattern_001",
                    type=TestInsightType.PATTERNS,
                    priority="high",
                    confidence=0.8,
                    title="Address High-Impact Test Patterns",
                    description="High-impact patterns detected that require immediate attention",
                    rationale=f"Found {len(high_impact_patterns)} high-impact patterns",
                    implementation_steps=[
                        "Analyze high-impact patterns in detail",
                        "Develop targeted improvement plans",
                        "Implement pattern-specific solutions",
                        "Monitor pattern resolution",
                    ],
                    estimated_effort="medium",
                    expected_impact="Resolve major test suite issues",
                    supporting_metrics={"high_impact_patterns": len(high_impact_patterns)},
                    affected_components=["test_patterns", "test_reliability"],
                )
            )
        return recommendations

    def _calculate_variance(self, values: list[float]) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance


# Factory function
def create_test_insight_generator(
    config: AITestInsightsConfig = DEFAULT_CONFIG,
) -> TestInsightGenerator:
    return TestInsightGenerator(config=config)
