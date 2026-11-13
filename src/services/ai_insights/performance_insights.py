"""Performance Insights Generator.

Generates performance-related insights including optimization opportunities
based on code similarity and refactoring patterns.
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


class PerformanceInsightGenerator(BaseInsightGenerator):
    """Generates performance-related insights."""

    async def generate(self, data: dict[str, Any]) -> list[CodeInsight]:
        """Generate performance-related insights.

        Args:
            data: Analysis data containing similarity_analysis and other metrics.

        Returns:
            List of performance-related CodeInsight objects.
        """
        insights: list[CodeInsight] = []

        # Performance insights based on similarity patterns
        similarity_analysis = data.get("similarity_analysis")
        if similarity_analysis and similarity_analysis.refactoring_opportunities:
            potential_savings = len(similarity_analysis.refactoring_opportunities)
            if potential_savings > 5:
                insights.append(
                    CodeInsight(
                        id="perf_refactoring_savings",
                        insight_type=InsightType.PERFORMANCE,
                        priority=InsightPriority.MEDIUM,
                        confidence=InsightConfidence.MEDIUM,
                        title="Performance Optimization via Refactoring",
                        description=(
                            f"Refactoring {potential_savings} similarity groups could affect performance"
                        ),
                        rationale=(
                            "Reducing code duplication can affect cache locality and reduce memory usage"
                        ),
                        recommendations=[
                            "Consolidate duplicate algorithms for better performance",
                            "Use shared data structures to reduce memory overhead",
                            "Implement lazy loading for rarely used functionality",
                            "Profile performance before and after refactoring",
                        ],
                        estimated_effort="medium",
                        evidence={"refactoring_opportunities": potential_savings},
                        tags=["performance", "refactoring", "optimization"],
                    )
                )

        return insights
