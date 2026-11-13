"""Testing Insights Generator.

Generates testing-related insights including refactoring opportunities
that affect testability and code coverage patterns.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from .base import (
    BaseInsightGenerator,
    CodeInsight,
    InsightConfidence,
    InsightPriority,
    InsightType,
)

logger = logging.getLogger(__name__)


class RefactoringInsightGenerator(BaseInsightGenerator):
    """Generates refactoring-related insights.

    Note: This class handles refactoring insights which often relate to
    testability and code quality improvements.
    """

    async def generate(self, data: dict[str, Any]) -> list[CodeInsight]:
        """Generate refactoring-related insights.

        Args:
            data: Analysis data containing similarity_analysis and other metrics.

        Returns:
            List of refactoring-related CodeInsight objects.
        """
        insights: list[CodeInsight] = []

        similarity_analysis = data.get("similarity_analysis")
        if not similarity_analysis:
            return insights

        refactoring_threshold = self.insight_rules.get("refactoring_opportunity_threshold", 0.75)

        # High similarity = refactoring opportunity
        high_similarity_matches = [
            match
            for match in similarity_analysis.cross_language_matches
            + similarity_analysis.matches_by_score.get("high", [])
            if match.score >= refactoring_threshold
        ]

        if high_similarity_matches:
            # Group by files for better insights
            file_groups: dict[tuple, list[Any]] = defaultdict(list)
            for match in high_similarity_matches:
                key = tuple(sorted([match.source_file, match.target_file]))
                file_groups[key].append(match)

            for (file1, file2), matches in file_groups.items():
                avg_similarity = sum(m.score for m in matches) / len(matches)

                insights.append(
                    CodeInsight(
                        id=f"refactor_{hash((file1, file2))}",
                        insight_type=InsightType.REFACTORING,
                        priority=(
                            InsightPriority.HIGH if avg_similarity > 0.9 else InsightPriority.MEDIUM
                        ),
                        confidence=InsightConfidence.HIGH,
                        title="Refactoring Opportunity: Similar Code Detected",
                        description=(
                            f"High similarity ({avg_similarity:.2f}) between {Path(file1).name} and {Path(file2).name}"
                        ),
                        rationale=(
                            "Similar code indicates potential for refactoring into shared utilities"
                        ),
                        files=[file1, file2],
                        recommendations=[
                            "Extract common functionality into shared utilities",
                            "Consider creating a base class or mixin",
                            "Use composition to reduce duplication",
                            "Apply DRY (Don't Repeat Yourself) principles",
                        ],
                        mitigation_suggestions=[
                            "Create shared module for common functionality",
                            f"Refactor {Path(file1).name} and {Path(file2).name} to use shared code",
                        ],
                        estimated_effort="medium",
                        evidence={
                            "similarity_score": avg_similarity,
                            "match_count": len(matches),
                            "files": [file1, file2],
                        },
                        tags=["refactoring", "duplication", "dry"],
                    )
                )

        # Duplicate code insights
        duplicate_matches = similarity_analysis.matches_by_type.get(
            "exact_duplicate", []
        ) + similarity_analysis.matches_by_type.get("near_duplicate", [])

        if duplicate_matches:
            insights.append(
                CodeInsight(
                    id="refactor_duplicates",
                    insight_type=InsightType.REFACTORING,
                    priority=InsightPriority.HIGH,
                    confidence=InsightConfidence.VERY_HIGH,
                    title="Code Duplication Detected",
                    description=(
                        f"Found {len(duplicate_matches)} instances of duplicate or near-duplicate code"
                    ),
                    rationale=("Code duplication leads to maintenance overhead and potential bugs"),
                    recommendations=[
                        "Consolidate duplicate code into shared functions",
                        "Use inheritance or composition to reduce duplication",
                        "Implement consistent refactoring patterns",
                        "Set up code analysis tools to prevent future duplication",
                    ],
                    estimated_effort="high",
                    evidence={"duplicate_count": len(duplicate_matches)},
                    metrics={"duplication_ratio": similarity_analysis.duplicate_code_ratio},
                    tags=["refactoring", "duplication", "maintenance"],
                )
            )

        return insights
