"""Maintainability Insights Generator.

Generates maintainability-related insights including code duplication,
language diversity, and technical debt analysis.
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


class MaintainabilityInsightGenerator(BaseInsightGenerator):
    """Generates maintainability-related insights."""

    async def generate(self, data: dict[str, Any]) -> list[CodeInsight]:
        """Generate maintainability-related insights.

        Args:
            data: Analysis data containing similarity_analysis and other metrics.

        Returns:
            List of maintainability-related CodeInsight objects.
        """
        insights: list[CodeInsight] = []

        similarity_analysis = data.get("similarity_analysis")
        if not similarity_analysis:
            return insights

        # High duplicate ratio = maintainability issue
        if similarity_analysis.duplicate_code_ratio > 0.2:  # 20% duplication
            insights.append(
                CodeInsight(
                    id="maint_duplication_ratio",
                    insight_type=InsightType.MAINTAINABILITY,
                    priority=InsightPriority.HIGH,
                    confidence=InsightConfidence.HIGH,
                    title="High Code Duplication Affecting Maintainability",
                    description=(
                        f"Code duplication ratio is "
                        f"{similarity_analysis.duplicate_code_ratio:.1%}, "
                        "above recommended 15%"
                    ),
                    rationale=(
                        "High duplication makes code harder to maintain and increases bug risks"
                    ),
                    recommendations=[
                        "Establish code duplication monitoring in CI/CD",
                        "Create shared libraries for common functionality",
                        "Implement consistent refactoring practices",
                        "Set duplication targets and track progress",
                    ],
                    estimated_effort="high",
                    evidence={"duplication_ratio": similarity_analysis.duplicate_code_ratio},
                    metrics={
                        "target_ratio": 0.15,
                        "current_ratio": similarity_analysis.duplicate_code_ratio,
                    },
                    tags=["maintainability", "duplication", "quality"],
                )
            )

        # Language diversity insights
        if len(similarity_analysis.language_distribution) > 5:
            insights.append(
                CodeInsight(
                    id="maint_language_diversity",
                    insight_type=InsightType.MAINTAINABILITY,
                    priority=InsightPriority.MEDIUM,
                    confidence=InsightConfidence.MEDIUM,
                    title="High Language Diversity",
                    description=(
                        f"Codebase uses {len(similarity_analysis.language_distribution)} different languages"
                    ),
                    rationale=(
                        "Many languages can increase maintenance overhead and skill requirements"
                    ),
                    recommendations=[
                        "Evaluate if all languages are necessary",
                        "Standardize on core languages where possible",
                        "Ensure team expertise in all used languages",
                        "Document language-specific guidelines",
                    ],
                    estimated_effort="low",
                    evidence={"language_count": len(similarity_analysis.language_distribution)},
                    tags=["maintainability", "languages", "diversity"],
                )
            )

        return insights
