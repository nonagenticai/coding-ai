"""Test Pattern Analyzer for AI-powered test insights.

This module provides advanced pattern analysis for test execution data,
identifying common patterns, anti-patterns, and optimization opportunities
in test suites. It uses statistical analysis and machine learning techniques
to detect meaningful patterns in test behavior.
"""

import logging
import statistics
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_serializer

from models.testing_models import QualityResult, TestResult

logger = logging.getLogger(__name__)


class PatternSeverity(StrEnum):
    """Severity levels for detected patterns."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class PatternCategory(StrEnum):
    """Categories of test patterns."""

    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    MAINTAINABILITY = "maintainability"
    COVERAGE = "coverage"
    ARCHITECTURE = "architecture"
    SECURITY = "security"
    COMPLEXITY = "complexity"


class TestPatternMetrics(BaseModel):
    __test__ = False
    """Metrics for a specific test pattern."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    # Pattern identification
    pattern_id: str = Field(description="Unique pattern identifier")
    pattern_name: str = Field(description="Human-readable pattern name")
    pattern_type: str = Field(default="", description="Type classification of the pattern")
    category: PatternCategory = Field(description="Pattern category")
    severity: PatternSeverity = Field(description="Pattern severity level")

    # Detection metrics
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in pattern detection")
    occurrence_count: int = Field(ge=0, description="Number of times pattern was detected")
    affected_tests: list[str] = Field(
        default_factory=list, description="Tests affected by this pattern"
    )

    # Impact assessment
    impact_score: float = Field(ge=0.0, le=10.0, description="Impact score (0-10)")
    frequency_ratio: float = Field(ge=0.0, le=1.0, description="Frequency as ratio of total tests")

    # Pattern details
    description: str = Field(description="Detailed pattern description")
    symptoms: list[str] = Field(default_factory=list, description="Observable symptoms")
    root_causes: list[str] = Field(default_factory=list, description="Potential root causes")
    recommendations: list[str] = Field(
        default_factory=list, description="Recommendations for addressing pattern"
    )

    # Supporting evidence
    evidence: dict[str, Any] = Field(
        default_factory=dict, description="Supporting evidence and metrics"
    )

    @computed_field
    @property
    def priority_score(self) -> float:
        """Calculate priority score based on severity, impact, and frequency."""
        severity_weights = {
            PatternSeverity.CRITICAL: 10.0,
            PatternSeverity.HIGH: 8.0,
            PatternSeverity.MEDIUM: 6.0,
            PatternSeverity.LOW: 4.0,
            PatternSeverity.INFO: 2.0,
        }

        base_score = severity_weights[self.severity]
        weighted_score = (
            base_score * 0.4 + self.impact_score * 0.4 + self.frequency_ratio * 10.0 * 0.2
        )

        return min(10.0, weighted_score)


class TestPatternAnalysisResult(BaseModel):
    __test__ = False
    """Result of test pattern analysis."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    # Analysis metadata
    analysis_id: str = Field(description="Unique analysis identifier")
    analyzed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When analysis was performed",
    )

    # Input data summary
    total_tests_analyzed: int = Field(description="Total number of tests analyzed")
    test_types_analyzed: set[str] = Field(
        default_factory=set, description="Test types included in analysis"
    )
    languages_analyzed: set[str] = Field(
        default_factory=set, description="Languages included in analysis"
    )

    # Pattern detection results
    detected_patterns: list[TestPatternMetrics] = Field(
        default_factory=list, description="All detected patterns"
    )
    critical_patterns: list[TestPatternMetrics] = Field(
        default_factory=list,
        description="Critical patterns requiring immediate attention",
    )
    high_impact_patterns: list[TestPatternMetrics] = Field(
        default_factory=list, description="High-impact patterns"
    )

    # Pattern statistics
    patterns_by_category: dict[PatternCategory, int] = Field(
        default_factory=dict, description="Pattern counts by category"
    )
    patterns_by_severity: dict[PatternSeverity, int] = Field(
        default_factory=dict, description="Pattern counts by severity"
    )

    # Analysis insights
    top_recommendations: list[str] = Field(
        default_factory=list, description="Top recommendations based on patterns"
    )
    improvement_opportunities: list[str] = Field(
        default_factory=list, description="Identified improvement opportunities"
    )

    @field_serializer("analyzed_at")
    def serialize_datetime(self, v: datetime | None) -> str | None:
        if v is None:
            return None
        return v.isoformat()

    @computed_field
    @property
    def overall_pattern_health(self) -> str:
        """Calculate overall pattern health score."""
        if not self.detected_patterns:
            return "excellent"

        critical_count = self.patterns_by_severity.get(PatternSeverity.CRITICAL, 0)
        high_count = self.patterns_by_severity.get(PatternSeverity.HIGH, 0)

        if critical_count > 0:
            return "critical"
        elif high_count > 2:
            return "poor"
        elif high_count > 0:
            return "fair"
        else:
            return "good"


class TestPatternAnalyzer:
    __test__ = False
    """Advanced test pattern analyzer using statistical methods."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the test pattern analyzer."""
        self.config = config or self._get_default_config()
        logger.info("Test Pattern Analyzer initialized")

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration for pattern analysis."""
        return {
            # Performance thresholds
            "slow_test_threshold": 10.0,  # seconds
            "very_slow_test_threshold": 30.0,  # seconds
            "memory_threshold": 512,  # MB
            "cpu_threshold": 80,  # percentage
            # Reliability thresholds
            "flaky_test_threshold": 0.8,  # success rate
            "failure_rate_threshold": 0.1,  # 10% failure rate
            "error_clustering_threshold": 3,  # minimum errors for clustering
            # Maintainability thresholds
            "complexity_threshold": 20.0,  # cyclomatic complexity
            "duplicate_threshold": 0.15,  # 15% duplication
            "large_test_threshold": 200,  # lines of code
            # Pattern detection sensitivity
            "min_confidence_score": 0.6,
            "min_occurrence_count": 3,
            "min_impact_score": 5.0,
        }

    async def analyze_patterns(
        self,
        test_results: list[TestResult],
        quality_results: list[QualityResult] | None = None,
    ) -> TestPatternAnalysisResult:
        """Perform comprehensive pattern analysis on test data."""
        analysis_id = str(uuid.uuid4())

        logger.info(f"Starting pattern analysis {analysis_id} for {len(test_results)} test results")

        # Initialize analysis result
        result = TestPatternAnalysisResult(
            analysis_id=analysis_id,
            total_tests_analyzed=len(test_results),
            test_types_analyzed=set(r.test_type or "unknown" for r in test_results),
            languages_analyzed=set(r.language for r in test_results),
        )

        # Detect different types of patterns
        patterns = []

        # Performance patterns
        patterns.extend(await self._detect_performance_patterns(test_results))

        # Reliability patterns
        patterns.extend(await self._detect_reliability_patterns(test_results))

        # Maintainability patterns
        patterns.extend(await self._detect_maintainability_patterns(test_results, quality_results))

        # Coverage patterns
        patterns.extend(await self._detect_coverage_patterns(test_results))

        # Architecture patterns
        patterns.extend(await self._detect_architecture_patterns(test_results))

        # Security patterns
        patterns.extend(await self._detect_security_patterns(test_results, quality_results))

        # Filter patterns by confidence and impact
        filtered_patterns = self._filter_patterns(patterns)

        # Categorize patterns
        result.detected_patterns = filtered_patterns
        result.critical_patterns = [
            p for p in filtered_patterns if p.severity == PatternSeverity.CRITICAL
        ]
        result.high_impact_patterns = [p for p in filtered_patterns if p.impact_score >= 7.0]

        # Calculate statistics
        result.patterns_by_category = self._categorize_patterns(filtered_patterns)
        result.patterns_by_severity = self._categorize_by_severity(filtered_patterns)

        # Generate recommendations
        result.top_recommendations = self._generate_top_recommendations(filtered_patterns)
        result.improvement_opportunities = self._identify_improvement_opportunities(
            filtered_patterns
        )

        logger.info(
            f"Pattern analysis {analysis_id} completed: {len(filtered_patterns)} patterns detected"
        )

        return result

    async def _detect_performance_patterns(
        self, test_results: list[TestResult]
    ) -> list[TestPatternMetrics]:
        """Detect performance-related patterns."""
        patterns = []

        execution_times = [r.execution_time for r in test_results]

        if not execution_times:
            return patterns

        avg_time = statistics.mean(execution_times)
        median_time = statistics.median(execution_times)

        # Slow test pattern
        slow_tests = [
            r for r in test_results if r.execution_time > self.config["slow_test_threshold"]
        ]
        if slow_tests:
            patterns.append(
                TestPatternMetrics(
                    pattern_id="perf_slow_tests",
                    pattern_name="Slow Test Execution",
                    category=PatternCategory.PERFORMANCE,
                    severity=(
                        PatternSeverity.HIGH if len(slow_tests) > 10 else PatternSeverity.MEDIUM
                    ),
                    confidence_score=0.9,
                    occurrence_count=len(slow_tests),
                    affected_tests=[r.test_id for r in slow_tests],
                    impact_score=min(10.0, len(slow_tests) * 0.5),
                    frequency_ratio=len(slow_tests) / len(test_results),
                    description=f"Detected {len(slow_tests)} slow tests taking longer than {self.config['slow_test_threshold']}s",
                    symptoms=[
                        "Tests taking significantly longer than average",
                        "Potential I/O or computation bottlenecks",
                        "Degraded CI/CD pipeline performance",
                    ],
                    root_causes=[
                        "Database queries without optimization",
                        "Synchronous I/O operations",
                        "Inefficient algorithms or data structures",
                        "Lack of test data optimization",
                    ],
                    recommendations=[
                        "Profile slow tests to identify bottlenecks",
                        "Optimize database queries and connections",
                        "Implement parallel test execution",
                        "Use test data factories and mocking",
                    ],
                    evidence={
                        "avg_execution_time": avg_time,
                        "median_execution_time": median_time,
                        "slow_test_count": len(slow_tests),
                        "slowest_test_time": max(execution_times),
                    },
                )
            )

        # Performance inconsistency pattern
        if len(execution_times) > 5:
            std_dev = statistics.stdev(execution_times)
            if std_dev > avg_time * 0.5:  # High variance
                patterns.append(
                    TestPatternMetrics(
                        pattern_id="perf_inconsistent",
                        pattern_name="Inconsistent Test Performance",
                        category=PatternCategory.PERFORMANCE,
                        severity=PatternSeverity.MEDIUM,
                        confidence_score=0.8,
                        occurrence_count=len(test_results),
                        affected_tests=[r.test_id for r in test_results],
                        impact_score=6.0,
                        frequency_ratio=1.0,
                        description="Test execution times show high variance, indicating inconsistent performance",
                        symptoms=[
                            "Unpredictable test execution times",
                            "Some tests much faster/slower than others",
                            "Performance varies between runs",
                        ],
                        root_causes=[
                            "Resource contention",
                            "Non-deterministic operations",
                            "Varying test data complexity",
                            "Infrastructure instability",
                        ],
                        recommendations=[
                            "Standardize test data and setup",
                            "Implement resource pooling",
                            "Add performance monitoring",
                            "Review test environment configuration",
                        ],
                        evidence={
                            "execution_time_stddev": std_dev,
                            "coefficient_of_variation": std_dev / avg_time,
                            "min_time": min(execution_times),
                            "max_time": max(execution_times),
                        },
                    )
                )

        return patterns

    async def _detect_reliability_patterns(
        self, test_results: list[TestResult]
    ) -> list[TestPatternMetrics]:
        """Detect reliability-related patterns."""
        patterns = []

        # Calculate success rates
        success_rate = sum(1 for r in test_results if r.success) / len(test_results)
        failed_tests = [r for r in test_results if not r.success]

        # Flaky test pattern
        if success_rate < self.config["flaky_test_threshold"]:
            patterns.append(
                TestPatternMetrics(
                    pattern_id="rel_flaky_tests",
                    pattern_name="Flaky Test Behavior",
                    category=PatternCategory.RELIABILITY,
                    severity=PatternSeverity.HIGH,
                    confidence_score=0.85,
                    occurrence_count=len(failed_tests),
                    affected_tests=[r.test_id for r in failed_tests],
                    impact_score=8.0,
                    frequency_ratio=1 - success_rate,
                    description=f"Test success rate is {success_rate:.1%}, indicating flaky behavior",
                    symptoms=[
                        "Tests failing intermittently",
                        "Non-deterministic test results",
                        "Reduced confidence in test suite",
                    ],
                    root_causes=[
                        "Race conditions in test code",
                        "Dependency on external services",
                        "Timing-dependent assertions",
                        "Shared test state or resources",
                    ],
                    recommendations=[
                        "Identify and fix flaky tests",
                        "Implement proper test isolation",
                        "Add explicit waits and retries",
                        "Mock external dependencies",
                    ],
                    evidence={
                        "success_rate": success_rate,
                        "failed_test_count": len(failed_tests),
                        "total_tests": len(test_results),
                    },
                )
            )

        # Error clustering pattern
        if failed_tests:
            error_types = defaultdict(int)
            for test in failed_tests:
                if test.stderr:
                    # Simple error categorization
                    error_lower = test.stderr.lower()
                    if "timeout" in error_lower:
                        error_types["timeout"] += 1
                    elif "assertion" in error_lower:
                        error_types["assertion"] += 1
                    elif "connection" in error_lower:
                        error_types["connection"] += 1
                    elif "null" in error_lower or "none" in error_lower:
                        error_types["null_reference"] += 1
                    else:
                        error_types["other"] += 1

            # Find the most common error type
            if error_types:
                most_common_error = max(error_types, key=lambda k: error_types[k])
                most_common_count = error_types[most_common_error]

                if most_common_count >= self.config["error_clustering_threshold"]:
                    patterns.append(
                        TestPatternMetrics(
                            pattern_id=f"rel_error_cluster_{most_common_error}",
                            pattern_name=f"Clustered {most_common_error.title()} Errors",
                            category=PatternCategory.RELIABILITY,
                            severity=PatternSeverity.MEDIUM,
                            confidence_score=0.7,
                            occurrence_count=most_common_count,
                            affected_tests=[
                                r.test_id
                                for r in failed_tests
                                if most_common_error in r.stderr.lower()
                            ],
                            impact_score=min(10.0, most_common_count * 0.8),
                            frequency_ratio=most_common_count / len(test_results),
                            description=f"Multiple tests failing with {most_common_error} errors",
                            symptoms=[
                                f"Recurring {most_common_error} errors",
                                "Similar failure patterns across tests",
                                "Systematic rather than random failures",
                            ],
                            root_causes=[
                                "Common infrastructure issue",
                                "Shared dependency failure",
                                "Configuration problem",
                                "Environmental inconsistency",
                            ],
                            recommendations=[
                                f"Investigate root cause of {most_common_error} errors",
                                "Review shared test infrastructure",
                                "Improve error handling and reporting",
                                "Add monitoring for common failure points",
                            ],
                            evidence={
                                "error_type": most_common_error,
                                "error_count": most_common_count,
                                "error_distribution": dict(error_types),
                            },
                        )
                    )

        return patterns

    async def _detect_maintainability_patterns(
        self,
        test_results: list[TestResult],
        quality_results: list[QualityResult] | None,
    ) -> list[TestPatternMetrics]:
        """Detect maintainability-related patterns."""
        patterns = []

        # Test complexity pattern (based on quality results)
        if quality_results:
            complexity_issues = []
            for qr in quality_results:
                for issue in qr.issues:
                    if "complexity" in issue.message.lower() or "complex" in issue.message.lower():
                        complexity_issues.append(issue)

            if complexity_issues:
                patterns.append(
                    TestPatternMetrics(
                        pattern_id="maint_complexity",
                        pattern_name="High Test Complexity",
                        category=PatternCategory.MAINTAINABILITY,
                        severity=PatternSeverity.MEDIUM,
                        confidence_score=0.8,
                        occurrence_count=len(complexity_issues),
                        affected_tests=[],  # Would need mapping from quality results to tests
                        impact_score=7.0,
                        frequency_ratio=len(complexity_issues) / len(test_results),
                        description=f"Found {len(complexity_issues)} complexity-related issues in test code",
                        symptoms=[
                            "Tests are difficult to understand",
                            "Complex test setup and teardown",
                            "Multiple assertions per test",
                        ],
                        root_causes=[
                            "Tests trying to cover too much",
                            "Poor test design and structure",
                            "Lack of helper methods",
                            "Insufficient test refactoring",
                        ],
                        recommendations=[
                            "Break down complex tests into smaller ones",
                            "Extract common setup into helper methods",
                            "Follow single responsibility principle",
                            "Improve test naming and documentation",
                        ],
                        evidence={
                            "complexity_issues": len(complexity_issues),
                            "avg_quality_score": sum(qr.overall_score for qr in quality_results)
                            / len(quality_results),
                        },
                    )
                )

        # Test duplication pattern
        if len(test_results) > 10:
            # Simple duplication detection based on similar naming or structure
            test_names = [r.test_id for r in test_results]
            potential_duplicates = []

            for i, name1 in enumerate(test_names):
                for j, name2 in enumerate(test_names[i + 1 :], i + 1):
                    # Simple similarity check
                    if len(name1) > 5 and len(name2) > 5:
                        common_chars = sum(1 for c1, c2 in zip(name1, name2) if c1 == c2)
                        similarity = common_chars / max(len(name1), len(name2))
                        if similarity > 0.7:  # 70% similarity
                            potential_duplicates.append((name1, name2))

            if potential_duplicates:
                patterns.append(
                    TestPatternMetrics(
                        pattern_id="maint_duplication",
                        pattern_name="Potential Test Duplication",
                        category=PatternCategory.MAINTAINABILITY,
                        severity=PatternSeverity.LOW,
                        confidence_score=0.6,
                        occurrence_count=len(potential_duplicates),
                        affected_tests=[name for pair in potential_duplicates for name in pair],
                        impact_score=4.0,
                        frequency_ratio=len(potential_duplicates) / len(test_results),
                        description=f"Found {len(potential_duplicates)} pairs of potentially duplicate tests",
                        symptoms=[
                            "Similar test names and structure",
                            "Repeated test logic",
                            "Maintenance overhead",
                        ],
                        root_causes=[
                            "Copy-paste test creation",
                            "Lack of test refactoring",
                            "Poor test organization",
                            "Missing test utilities",
                        ],
                        recommendations=[
                            "Review and consolidate duplicate tests",
                            "Create parameterized tests",
                            "Extract common test utilities",
                            "Improve test naming conventions",
                        ],
                        evidence={
                            "duplicate_pairs": len(potential_duplicates),
                            "similarity_threshold": 0.7,
                        },
                    )
                )

        return patterns

    async def _detect_coverage_patterns(
        self, test_results: list[TestResult]
    ) -> list[TestPatternMetrics]:
        """Detect coverage-related patterns."""
        patterns = []

        # Test distribution pattern
        language_distribution = defaultdict(int)
        for result in test_results:
            language_distribution[result.language] += 1

        # Check for language imbalance
        if len(language_distribution) > 1:
            total_tests = len(test_results)
            dominant_language = max(language_distribution, key=lambda k: language_distribution[k])
            dominant_count = language_distribution[dominant_language]

            if dominant_count / total_tests > 0.8:  # 80% dominated by one language
                patterns.append(
                    TestPatternMetrics(
                        pattern_id="cov_language_imbalance",
                        pattern_name="Language Coverage Imbalance",
                        category=PatternCategory.COVERAGE,
                        severity=PatternSeverity.MEDIUM,
                        confidence_score=0.7,
                        occurrence_count=dominant_count,
                        affected_tests=[],
                        impact_score=5.0,
                        frequency_ratio=dominant_count / total_tests,
                        description=f"Test suite dominated by {dominant_language} tests ({dominant_count}/{total_tests})",
                        symptoms=[
                            "Uneven test coverage across languages",
                            "Some languages under-tested",
                            "Potential quality gaps",
                        ],
                        root_causes=[
                            "Development focus on primary language",
                            "Lack of multi-language testing strategy",
                            "Different teams using different languages",
                            "Legacy code in secondary languages",
                        ],
                        recommendations=[
                            "Develop comprehensive multi-language testing strategy",
                            "Ensure adequate coverage for all languages",
                            "Consider language-specific test requirements",
                            "Balance test investment across languages",
                        ],
                        evidence={
                            "language_distribution": dict(language_distribution),
                            "dominant_language": dominant_language,
                            "dominance_ratio": dominant_count / total_tests,
                        },
                    )
                )

        return patterns

    async def _detect_architecture_patterns(
        self, test_results: list[TestResult]
    ) -> list[TestPatternMetrics]:
        """Detect architecture-related patterns."""
        patterns = []

        # Test type distribution
        test_type_distribution = defaultdict(int)
        for result in test_results:
            test_type = result.test_type or "unknown"
            test_type_distribution[test_type] += 1

        # Check for test type imbalance
        if len(test_type_distribution) > 1:
            total_tests = len(test_results)
            unit_tests = test_type_distribution.get("unit", 0)
            integration_tests = test_type_distribution.get("integration", 0)

            # Ideal ratio is roughly 70% unit, 20% integration, 10% e2e
            unit_ratio = unit_tests / total_tests
            integration_ratio = integration_tests / total_tests

            if unit_ratio < 0.5:  # Less than 50% unit tests
                patterns.append(
                    TestPatternMetrics(
                        pattern_id="arch_test_pyramid",
                        pattern_name="Test Pyramid Imbalance",
                        category=PatternCategory.ARCHITECTURE,
                        severity=PatternSeverity.MEDIUM,
                        confidence_score=0.8,
                        occurrence_count=total_tests - unit_tests,
                        affected_tests=[],
                        impact_score=6.0,
                        frequency_ratio=1 - unit_ratio,
                        description=f"Test pyramid imbalance: {unit_ratio:.1%} unit tests, {integration_ratio:.1%} integration tests",
                        symptoms=[
                            "Too few unit tests relative to integration tests",
                            "Slower overall test execution",
                            "More complex test debugging",
                        ],
                        root_causes=[
                            "Lack of unit test discipline",
                            "Over-reliance on integration testing",
                            "Difficult to unit test legacy code",
                            "Missing test strategy guidance",
                        ],
                        recommendations=[
                            "Increase unit test coverage",
                            "Refactor code for better testability",
                            "Educate team on test pyramid principles",
                            "Establish test strategy guidelines",
                        ],
                        evidence={
                            "test_type_distribution": dict(test_type_distribution),
                            "unit_test_ratio": unit_ratio,
                            "integration_test_ratio": integration_ratio,
                        },
                    )
                )

        return patterns

    async def _detect_security_patterns(
        self,
        test_results: list[TestResult],
        quality_results: list[QualityResult] | None,
    ) -> list[TestPatternMetrics]:
        """Detect security-related patterns."""
        patterns = []

        # Security test coverage
        security_tests = [r for r in test_results if r.test_type == "security"]
        if len(security_tests) / len(test_results) < 0.05:  # Less than 5% security tests
            patterns.append(
                TestPatternMetrics(
                    pattern_id="sec_low_coverage",
                    pattern_name="Low Security Test Coverage",
                    category=PatternCategory.SECURITY,
                    severity=PatternSeverity.HIGH,
                    confidence_score=0.9,
                    occurrence_count=len(test_results) - len(security_tests),
                    affected_tests=[],
                    impact_score=8.0,
                    frequency_ratio=1 - (len(security_tests) / len(test_results)),
                    description=f"Only {len(security_tests)} security tests out of {len(test_results)} total tests",
                    symptoms=[
                        "Insufficient security testing",
                        "Potential security vulnerabilities",
                        "Lack of security validation",
                    ],
                    root_causes=[
                        "Security not prioritized in testing",
                        "Lack of security testing expertise",
                        "Missing security test requirements",
                        "No security testing framework",
                    ],
                    recommendations=[
                        "Implement comprehensive security testing",
                        "Add security test requirements",
                        "Use security testing frameworks",
                        "Include security in test planning",
                    ],
                    evidence={
                        "security_test_count": len(security_tests),
                        "total_test_count": len(test_results),
                        "security_test_ratio": len(security_tests) / len(test_results),
                    },
                )
            )

        # Security quality issues
        if quality_results:
            security_issues = []
            for qr in quality_results:
                for issue in qr.issues:
                    if any(
                        keyword in issue.message.lower()
                        for keyword in [
                            "security",
                            "vulnerability",
                            "injection",
                            "xss",
                            "csrf",
                        ]
                    ):
                        security_issues.append(issue)

            if security_issues:
                patterns.append(
                    TestPatternMetrics(
                        pattern_id="sec_quality_issues",
                        pattern_name="Security Quality Issues",
                        category=PatternCategory.SECURITY,
                        severity=PatternSeverity.HIGH,
                        confidence_score=0.85,
                        occurrence_count=len(security_issues),
                        affected_tests=[],
                        impact_score=9.0,
                        frequency_ratio=len(security_issues) / len(test_results),
                        description=f"Found {len(security_issues)} security-related quality issues",
                        symptoms=[
                            "Security vulnerabilities in code",
                            "Insecure coding patterns",
                            "Missing security validations",
                        ],
                        root_causes=[
                            "Lack of security awareness",
                            "Insecure coding practices",
                            "Missing security code review",
                            "No security static analysis",
                        ],
                        recommendations=[
                            "Address security issues immediately",
                            "Implement security code review",
                            "Add security static analysis",
                            "Provide security training",
                        ],
                        evidence={
                            "security_issues_count": len(security_issues),
                            "security_severity_distribution": {},
                        },
                    )
                )

        return patterns

    def _filter_patterns(self, patterns: list[TestPatternMetrics]) -> list[TestPatternMetrics]:
        """Filter patterns based on configuration thresholds."""
        filtered = []

        for pattern in patterns:
            # Apply confidence threshold
            if pattern.confidence_score < self.config["min_confidence_score"]:
                continue

            # Apply occurrence threshold
            if pattern.occurrence_count < self.config["min_occurrence_count"]:
                continue

            # Apply impact threshold
            if pattern.impact_score < self.config["min_impact_score"]:
                continue

            filtered.append(pattern)

        # Sort by priority score
        filtered.sort(key=lambda p: p.priority_score, reverse=True)

        return filtered

    def _categorize_patterns(
        self, patterns: list[TestPatternMetrics]
    ) -> dict[PatternCategory, int]:
        """Categorize patterns by category."""
        categorized = defaultdict(int)
        for pattern in patterns:
            categorized[pattern.category] += 1
        return dict(categorized)

    def _categorize_by_severity(
        self, patterns: list[TestPatternMetrics]
    ) -> dict[PatternSeverity, int]:
        """Categorize patterns by severity."""
        categorized = defaultdict(int)
        for pattern in patterns:
            categorized[pattern.severity] += 1
        return dict(categorized)

    def _generate_top_recommendations(self, patterns: list[TestPatternMetrics]) -> list[str]:
        """Generate top recommendations based on detected patterns."""
        recommendations = []

        # Get highest priority patterns
        high_priority = [p for p in patterns if p.priority_score >= 8.0]

        for pattern in high_priority[:5]:  # Top 5 high-priority patterns
            recommendations.extend(pattern.recommendations[:2])  # Top 2 recommendations per pattern

        # Remove duplicates while preserving order
        unique_recommendations = []
        seen = set()
        for rec in recommendations:
            if rec not in seen:
                unique_recommendations.append(rec)
                seen.add(rec)

        return unique_recommendations[:10]  # Top 10 overall recommendations

    def _identify_improvement_opportunities(self, patterns: list[TestPatternMetrics]) -> list[str]:
        """Identify improvement opportunities based on patterns."""
        opportunities = []

        # Analyze pattern categories
        category_counts = self._categorize_patterns(patterns)

        if category_counts.get(PatternCategory.PERFORMANCE, 0) > 0:
            opportunities.append("Optimize test performance to reduce execution time")

        if category_counts.get(PatternCategory.RELIABILITY, 0) > 0:
            opportunities.append("Improve test reliability to reduce flakiness")

        if category_counts.get(PatternCategory.MAINTAINABILITY, 0) > 0:
            opportunities.append("Refactor tests to improve maintainability")

        if category_counts.get(PatternCategory.SECURITY, 0) > 0:
            opportunities.append("Strengthen security testing coverage")

        if category_counts.get(PatternCategory.COVERAGE, 0) > 0:
            opportunities.append("Balance test coverage across components")

        if category_counts.get(PatternCategory.ARCHITECTURE, 0) > 0:
            opportunities.append("Restructure test architecture for better organization")

        return opportunities


# Factory function
def create_test_pattern_analyzer(
    config: dict[str, Any] | None = None,
) -> TestPatternAnalyzer:
    """Create a test pattern analyzer instance."""
    return TestPatternAnalyzer(config=config)
