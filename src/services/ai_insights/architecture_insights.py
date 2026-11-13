"""Architecture Insights Generator.

Generates architecture-related insights including language dominance,
structural patterns, and modularization opportunities.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import (
    BaseInsightGenerator,
    CodeInsight,
    InsightConfidence,
    InsightPriority,
    InsightType,
)

logger = logging.getLogger(__name__)


class ArchitectureInsightGenerator(BaseInsightGenerator):
    """Generates architecture-related insights."""

    async def generate(self, data: dict[str, Any]) -> list[CodeInsight]:
        """Generate architecture-related insights.

        Args:
            data: Analysis data containing similarity_analysis and other metrics.

        Returns:
            List of architecture-related CodeInsight objects.
        """
        insights: list[CodeInsight] = []

        similarity_analysis = data.get("similarity_analysis")
        if not similarity_analysis:
            return insights

        # Detect god classes/modules
        if similarity_analysis.language_distribution:
            total_files = sum(similarity_analysis.language_distribution.values())

            for (
                language,
                file_count,
            ) in similarity_analysis.language_distribution.items():
                if file_count > total_files * 0.5:  # Language dominance
                    insights.append(
                        CodeInsight(
                            id=f"arch_lang_dominance_{language}",
                            insight_type=InsightType.ARCHITECTURE,
                            priority=InsightPriority.MEDIUM,
                            confidence=InsightConfidence.HIGH,
                            title=f"{language.title()} Language Dominance",
                            description=(
                                f"The codebase is heavily dominated by {language} ({file_count}/{total_files} files)"
                            ),
                            rationale=(
                                "Language dominance may indicate opportunities for modularization or microservices"
                            ),
                            recommendations=[
                                f"Consider breaking down large {language} modules into smaller services",
                                "Evaluate opportunities for introducing other languages for specific domains",
                                "Review architecture for proper separation of concerns",
                            ],
                            estimated_effort="medium",
                            evidence={
                                "language": language,
                                "file_count": file_count,
                                "total_files": total_files,
                            },
                            tags=[
                                "architecture",
                                "language-dominance",
                                language,
                            ],
                        )
                    )

        # Detect architectural patterns from similarity matches
        if similarity_analysis.matches_by_type:
            structural_matches = similarity_analysis.matches_by_type.get("structural", [])
            if len(structural_matches) > 10:  # Many structural similarities
                insights.append(
                    CodeInsight(
                        id="arch_structural_patterns",
                        insight_type=InsightType.ARCHITECTURE,
                        priority=InsightPriority.MEDIUM,
                        confidence=InsightConfidence.MEDIUM,
                        title="Consistent Structural Patterns Detected",
                        description=(
                            f"Found {len(structural_matches)} structural similarities indicating consistent patterns"
                        ),
                        rationale=(
                            "Consistent structural patterns suggest good architectural discipline"
                        ),
                        recommendations=[
                            "Document these patterns as architectural guidelines",
                            "Consider creating templates or generators for common patterns",
                            "Ensure new code follows established patterns",
                        ],
                        estimated_effort="low",
                        evidence={"pattern_count": len(structural_matches)},
                        tags=["architecture", "patterns", "consistency"],
                    )
                )

        return insights
