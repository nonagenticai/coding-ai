"""Code Quality Insights Generator.

Generates code quality insights including overall quality assessment,
cross-language patterns, and design pattern analysis.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from .base import (
    BaseInsightGenerator,
    CodeInsight,
    InsightConfidence,
    InsightPriority,
    InsightType,
)

logger = logging.getLogger(__name__)


class QualityInsightGenerator(BaseInsightGenerator):
    """Generates code quality insights."""

    async def generate(self, data: dict[str, Any]) -> list[CodeInsight]:
        """Generate code quality insights.

        Args:
            data: Analysis data containing similarity_analysis, security_analysis,
                  and other metrics.

        Returns:
            List of quality-related CodeInsight objects.
        """
        insights: list[CodeInsight] = []

        similarity_analysis = data.get("similarity_analysis")
        security_analysis = data.get("security_analysis")

        # Calculate overall quality score
        quality_components: dict[str, float] = {}

        if similarity_analysis:
            # Duplication score (lower duplication = higher quality)
            duplication_score = max(0, 1.0 - similarity_analysis.duplicate_code_ratio * 2)
            quality_components["duplication"] = duplication_score

        if security_analysis:
            # Security score (fewer vulnerabilities = higher quality)
            total_vulns = security_analysis.total_vulnerabilities_found
            total_files = security_analysis.total_files_analyzed
            vuln_ratio = total_vulns / max(total_files, 1)
            # Penalize vulnerabilities
            security_score = max(0, 1.0 - vuln_ratio * 0.1)
            quality_components["security"] = security_score

        min_quality = self.insight_rules.get("min_quality_score", 0.7)

        if quality_components:
            overall_quality = sum(quality_components.values()) / len(quality_components)

            if overall_quality < min_quality:
                insights.append(
                    CodeInsight(
                        id="quality_overall_low",
                        insight_type=InsightType.QUALITY,
                        priority=InsightPriority.HIGH,
                        confidence=InsightConfidence.HIGH,
                        title="Overall Code Quality Below Threshold",
                        description=(
                            f"Overall quality score is {overall_quality:.2f}, below recommended {min_quality}"
                        ),
                        rationale=("Low quality scores indicate multiple areas needing attention"),
                        recommendations=[
                            "Implement comprehensive code review process",
                            "Add automated quality gates to CI/CD",
                            "Establish coding standards and guidelines",
                            "Invest in developer training and tools",
                        ],
                        estimated_effort="high",
                        evidence=quality_components,
                        metrics={
                            "quality_score": overall_quality,
                            "target_score": min_quality,
                        },
                        tags=["quality", "standards", "optimization"],
                    )
                )

        return insights


class CrossLanguageInsightGenerator(BaseInsightGenerator):
    """Generates cross-language insights."""

    async def generate(self, data: dict[str, Any]) -> list[CodeInsight]:
        """Generate cross-language insights.

        Args:
            data: Analysis data containing similarity_analysis with cross-language
                  matches.

        Returns:
            List of cross-language CodeInsight objects.
        """
        insights: list[CodeInsight] = []

        similarity_analysis = data.get("similarity_analysis")
        if not similarity_analysis or not similarity_analysis.cross_language_matches:
            return insights

        threshold = self.insight_rules.get("cross_language_pattern_threshold", 0.60)

        # Strong cross-language patterns
        strong_patterns = [
            match
            for match in similarity_analysis.cross_language_matches
            if match.score >= threshold
        ]

        if strong_patterns:
            # Group by language pairs
            language_pairs: dict[tuple, list[Any]] = defaultdict(list)
            for match in strong_patterns:
                pair = tuple(sorted([match.source_language, match.target_language]))
                language_pairs[pair].append(match)

            for (lang1, lang2), matches in language_pairs.items():
                avg_similarity = sum(m.score for m in matches) / len(matches)

                insights.append(
                    CodeInsight(
                        id=f"cross_lang_{lang1}_{lang2}",
                        insight_type=InsightType.CROSS_LANGUAGE,
                        priority=InsightPriority.MEDIUM,
                        confidence=InsightConfidence.HIGH,
                        title=f"Cross-Language Pattern: {lang1.title()} - {lang2.title()}",
                        description=(
                            f"Strong similarity patterns ({avg_similarity:.2f} avg) between {lang1} and {lang2} code"
                        ),
                        rationale=(
                            "Cross-language similarities may indicate opportunities for unification or API design"
                        ),
                        recommendations=[
                            f"Consider creating shared API between {lang1} and {lang2} components",
                            "Evaluate opportunities for code generation or shared interfaces",
                            "Document cross-language interaction patterns",
                            "Consider microservice boundaries along language lines",
                        ],
                        estimated_effort="medium",
                        evidence={
                            "language_pair": [lang1, lang2],
                            "pattern_count": len(matches),
                            "avg_similarity": avg_similarity,
                        },
                        tags=["cross-language", lang1, lang2, "patterns"],
                    )
                )

        return insights


class PatternInsightGenerator(BaseInsightGenerator):
    """Generates design pattern insights."""

    async def generate(self, data: dict[str, Any]) -> list[CodeInsight]:
        """Generate design pattern insights.

        Args:
            data: Analysis data containing similarity_analysis with pattern
                  information.

        Returns:
            List of pattern-related CodeInsight objects.
        """
        insights: list[CodeInsight] = []

        similarity_analysis = data.get("similarity_analysis")
        if not similarity_analysis:
            return insights

        # Detect consistent patterns
        if similarity_analysis.cross_language_patterns:
            pattern_count = len(similarity_analysis.cross_language_patterns)
            insights.append(
                CodeInsight(
                    id="patterns_cross_language",
                    insight_type=InsightType.PATTERNS,
                    priority=InsightPriority.LOW,
                    confidence=InsightConfidence.MEDIUM,
                    title="Cross-Language Design Patterns",
                    description=(
                        f"Detected {pattern_count} consistent patterns across different languages"
                    ),
                    rationale=("Consistent patterns indicate good architectural discipline"),
                    recommendations=[
                        "Document these patterns as design guidelines",
                        "Create pattern libraries or templates",
                        "Share patterns across development teams",
                        "Consider pattern automation tools",
                    ],
                    estimated_effort="low",
                    evidence={"pattern_count": pattern_count},
                    tags=["patterns", "consistency", "architecture"],
                )
            )

        return insights
