"""AI Code Insights Engine.

Advanced analysis engine that combines all code analysis data to provide
intelligent insights, recommendations, and actionable intelligence about codebases.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .architecture_insights import ArchitectureInsightGenerator
from .base import (
    CodebaseInsights,
    CodeInsight,
    InsightConfidence,
    InsightPriority,
    InsightType,
)
from .code_quality_insights import (
    CrossLanguageInsightGenerator,
    PatternInsightGenerator,
    QualityInsightGenerator,
)
from .maintainability_insights import MaintainabilityInsightGenerator
from .performance_insights import PerformanceInsightGenerator
from .security_insights import SecurityInsightGenerator
from .testing_insights import RefactoringInsightGenerator

logger = logging.getLogger(__name__)


def _get_security_detector():
    """Lazy import for security detector to prevent circular imports."""
    try:
        # TODO: Replace with coding-ai security detector client when available
        from services.security_detector import get_security_detector

        return get_security_detector()
    except ImportError:
        logger.warning("Security detector not available")
        return None


def _get_similarity_detector():
    """Lazy import for similarity detector to prevent circular imports."""
    try:
        # TODO: Replace with coding-ai similarity detector client when available
        from services.similarity_detector import get_similarity_detector

        return get_similarity_detector()
    except ImportError:
        logger.warning("Similarity detector not available")
        return None


class AICodeInsightsEngine:
    """AI-powered code insights engine that aggregates all analysis data."""

    def __init__(self) -> None:
        """Initialize the AI insights engine."""
        self.similarity_detector = _get_similarity_detector()
        self.security_detector = _get_security_detector()

        # AI insight generation rules and thresholds
        self.insight_rules = self._load_insight_rules()
        self.quality_metrics = self._initialize_quality_metrics()

        # Initialize insight generators
        self._init_generators()

        logger.info("AI Code Insights Engine initialized")

    def _init_generators(self) -> None:
        """Initialize all insight generators."""
        self.architecture_generator = ArchitectureInsightGenerator(self.insight_rules)
        self.refactoring_generator = RefactoringInsightGenerator(self.insight_rules)
        self.security_generator = SecurityInsightGenerator(self.insight_rules)
        self.performance_generator = PerformanceInsightGenerator(self.insight_rules)
        self.maintainability_generator = MaintainabilityInsightGenerator(self.insight_rules)
        self.cross_language_generator = CrossLanguageInsightGenerator(self.insight_rules)
        self.quality_generator = QualityInsightGenerator(self.insight_rules)
        self.pattern_generator = PatternInsightGenerator(self.insight_rules)

    def _load_insight_rules(self) -> dict[str, Any]:
        """Load AI insight generation rules."""
        return {
            # Similarity-based insights
            "duplicate_code_threshold": 0.85,
            "refactoring_opportunity_threshold": 0.75,
            "cross_language_pattern_threshold": 0.60,
            # Security insights
            "critical_vulnerability_threshold": 3,
            "security_hotspot_threshold": 2,
            # Complexity insights
            "high_complexity_threshold": 10.0,
            "excessive_complexity_threshold": 20.0,
            # Architecture insights
            "cyclic_dependency_detection": True,
            "god_class_size_threshold": 1000,
            "function_size_threshold": 100,
            # Quality metrics
            "min_quality_score": 0.7,
            "tech_debt_threshold": 0.3,
        }

    def _initialize_quality_metrics(self) -> dict[str, float]:
        """Initialize quality metric weights."""
        return {
            "complexity_weight": 0.25,
            "security_weight": 0.30,
            "duplication_weight": 0.20,
            "maintainability_weight": 0.15,
            "test_coverage_weight": 0.10,
        }

    async def generate_insights(
        self,
        file_paths: Sequence[str | Path],
        include_experimental: bool = False,
        focus_areas: list[InsightType] | None = None,
        max_insights: int | None = None,
    ) -> CodebaseInsights:
        """Generate comprehensive AI insights for a codebase.

        Args:
            file_paths: Files to analyze
            include_experimental: Include experimental/low-confidence insights
            focus_areas: Specific insight types to focus on
            max_insights: Maximum number of insights to generate

        Returns:
            Comprehensive insights analysis
        """
        start_time = datetime.now(UTC)

        logger.info("Generating AI insights for %d files", len(file_paths))

        # Gather analysis data from all sources
        analysis_data = await self._gather_analysis_data(list(file_paths))

        # Generate insights from different perspectives
        insights: list[CodeInsight] = []

        # Architecture insights
        if not focus_areas or InsightType.ARCHITECTURE in focus_areas:
            insights.extend(await self.architecture_generator.generate(analysis_data))

        # Refactoring insights
        if not focus_areas or InsightType.REFACTORING in focus_areas:
            insights.extend(await self.refactoring_generator.generate(analysis_data))

        # Security insights
        if not focus_areas or InsightType.SECURITY in focus_areas:
            insights.extend(await self.security_generator.generate(analysis_data))

        # Performance insights
        if not focus_areas or InsightType.PERFORMANCE in focus_areas:
            insights.extend(await self.performance_generator.generate(analysis_data))

        # Maintainability insights
        if not focus_areas or InsightType.MAINTAINABILITY in focus_areas:
            insights.extend(await self.maintainability_generator.generate(analysis_data))

        # Cross-language insights
        if not focus_areas or InsightType.CROSS_LANGUAGE in focus_areas:
            insights.extend(await self.cross_language_generator.generate(analysis_data))

        # Quality insights
        if not focus_areas or InsightType.QUALITY in focus_areas:
            insights.extend(await self.quality_generator.generate(analysis_data))

        # Pattern insights
        if not focus_areas or InsightType.PATTERNS in focus_areas:
            insights.extend(await self.pattern_generator.generate(analysis_data))

        # Filter insights by confidence if needed
        if not include_experimental:
            insights = [
                insight
                for insight in insights
                if insight.confidence != InsightConfidence.EXPERIMENTAL
            ]

        # Limit insights if requested
        if max_insights:
            insights = sorted(insights, key=lambda i: i.impact_score, reverse=True)[:max_insights]

        # Analyze and organize insights
        result = await self._analyze_insights(insights, analysis_data)

        analysis_time = (datetime.now(UTC) - start_time).total_seconds()
        result.analysis_time_seconds = analysis_time

        logger.info(
            "AI insights generation completed: %d insights in %.2fs",
            len(insights),
            analysis_time,
        )

        return result

    async def _gather_analysis_data(self, file_paths: list[str | Path]) -> dict[str, Any]:
        """Gather analysis data from all available sources."""
        logger.info("Gathering analysis data from all sources")

        # Convert to Path objects
        paths = [Path(p) for p in file_paths]

        # Gather data in parallel
        # Convert back to list as the methods expect List[Union[str, Path]]
        path_list: list[str | Path] = list(paths)

        similarity_analysis: Any | None = None
        security_analysis: Any | None = None

        tasks = []

        if self.similarity_detector:
            similarity_task = self.similarity_detector.analyze_similarity(
                file_paths=path_list,
                min_similarity_score=0.3,
                detect_cross_language=True,
            )
            tasks.append(("similarity", similarity_task))

        if self.security_detector:
            security_task = self.security_detector.analyze_security_vulnerabilities(
                file_paths=path_list, include_low_confidence=True
            )
            tasks.append(("security", security_task))

        if tasks:
            # Wait for all analysis to complete
            try:
                coros = [t[1] for t in tasks]
                results = await asyncio.gather(*coros, return_exceptions=True)

                for i, (name, _) in enumerate(tasks):
                    result = results[i]
                    if isinstance(result, Exception):
                        logger.warning("%s analysis failed: %s", name.title(), result)
                    elif name == "similarity":
                        similarity_analysis = result
                    elif name == "security":
                        security_analysis = result
            except (TimeoutError, asyncio.CancelledError) as e:
                logger.error("Analysis tasks failed: %s", e)

        return {
            "file_paths": paths,
            "similarity_analysis": similarity_analysis,
            "security_analysis": security_analysis,
            "total_files": len(paths),
        }

    async def _analyze_insights(
        self,
        insights: list[CodeInsight],
        data: dict[str, Any],
    ) -> CodebaseInsights:
        """Analyze and organize generated insights."""

        result = CodebaseInsights(
            total_files_analyzed=data.get("total_files", 0),
            total_entities_analyzed=0,  # Could be calculated from parse results
            analysis_time_seconds=0.0,  # Will be set by caller
        )

        # Categorize insights
        for insight in insights:
            # By type
            if insight.insight_type not in result.insights_by_type:
                result.insights_by_type[insight.insight_type] = []
            result.insights_by_type[insight.insight_type].append(insight)

            # By priority
            if insight.priority not in result.insights_by_priority:
                result.insights_by_priority[insight.priority] = []
            result.insights_by_priority[insight.priority].append(insight)

            # High impact
            if insight.impact_score >= 5.0:
                result.high_impact_insights.append(insight)

            # Special categories
            if insight.priority == InsightPriority.CRITICAL:
                result.critical_issues.append(insight)
            elif insight.insight_type == InsightType.REFACTORING:
                result.refactoring_candidates.append(insight)
            elif insight.estimated_effort == "low" and insight.priority in [
                InsightPriority.HIGH,
                InsightPriority.MEDIUM,
            ]:
                result.quick_wins.append(insight)

            # Cross-language issues
            if insight.insight_type == InsightType.CROSS_LANGUAGE:
                result.cross_language_issues.append(insight)

        # Calculate summary metrics
        similarity_analysis = data.get("similarity_analysis")
        security_analysis = data.get("security_analysis")

        if similarity_analysis:
            result.language_distribution = similarity_analysis.language_distribution
            result.technical_debt_ratio = similarity_analysis.duplicate_code_ratio

        if security_analysis:
            result.security_risk_score = min(10.0, security_analysis.overall_risk_score)

        # Calculate quality scores
        quality_factors: list[float] = []

        # Duplication factor
        if similarity_analysis:
            duplication_factor = max(0, 1.0 - similarity_analysis.duplicate_code_ratio * 2)
            quality_factors.append(duplication_factor)

        # Security factor
        if security_analysis and security_analysis.total_files_analyzed > 0:
            vuln_ratio = (
                security_analysis.total_vulnerabilities_found
                / security_analysis.total_files_analyzed
            )
            security_factor = max(0, 1.0 - vuln_ratio * 0.1)
            quality_factors.append(security_factor)

        if quality_factors:
            result.overall_quality_score = sum(quality_factors) / len(quality_factors)
            result.maintainability_score = result.overall_quality_score

        # Sort insights by impact
        result.high_impact_insights.sort(key=lambda i: i.impact_score, reverse=True)
        result.quick_wins.sort(key=lambda i: i.impact_score, reverse=True)

        return result


class AIInsightsEngineSingleton:
    """Singleton wrapper for AI insights engine with thread-safe instance management."""

    _instance: AICodeInsightsEngine | None = None
    _lock = __import__("threading").Lock()

    @classmethod
    def get_instance(cls) -> AICodeInsightsEngine:
        """Get or create the singleton AI insights engine instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking
                if cls._instance is None:
                    cls._instance = AICodeInsightsEngine()
                    logger.info("Initialized AI insights engine singleton")
        assert cls._instance is not None
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None
            logger.debug("AI insights engine singleton reset")


def get_ai_insights_engine() -> AICodeInsightsEngine:
    """Get or create the global AI insights engine instance."""
    return AIInsightsEngineSingleton.get_instance()


def reset_ai_insights_engine() -> None:
    """Reset the AI insights engine (useful for testing)."""
    AIInsightsEngineSingleton.reset_instance()
