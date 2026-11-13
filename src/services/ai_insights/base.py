"""Base classes and types for AI Insights.

This module contains the foundational enums, dataclasses, and base class
used by all insight generators.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class InsightType(StrEnum):
    """Types of AI-generated insights."""

    ARCHITECTURE = "architecture"  # High-level architectural recommendations
    REFACTORING = "refactoring"  # Code refactoring opportunities
    SECURITY = "security"  # Security-related insights
    PERFORMANCE = "performance"  # Performance optimization suggestions
    MAINTAINABILITY = "maintainability"  # Code maintainability changes
    COMPLEXITY = "complexity"  # Complexity reduction recommendations
    DEPENDENCIES = "dependencies"  # Dependency management insights
    CROSS_LANGUAGE = "cross_language"  # Multi-language integration patterns
    QUALITY = "quality"  # Code quality changes
    PATTERNS = "patterns"  # Design pattern recommendations


class InsightPriority(StrEnum):
    """Priority levels for insights."""

    CRITICAL = "critical"  # Urgent issues requiring immediate attention
    HIGH = "high"  # Important changes with significant impact
    MEDIUM = "medium"  # Moderate changes worth considering
    LOW = "low"  # Nice-to-have optimizations
    INFO = "info"  # Informational insights


class InsightConfidence(StrEnum):
    """Confidence levels for AI insights."""

    VERY_HIGH = "very_high"  # 95-100% - Strong evidence
    HIGH = "high"  # 85-94% - Good evidence
    MEDIUM = "medium"  # 70-84% - Moderate evidence
    LOW = "low"  # 50-69% - Weak evidence
    EXPERIMENTAL = "experimental"  # 0-49% - Experimental insights


@dataclass
class CodeInsight:
    """An AI-generated insight about code."""

    id: str
    insight_type: InsightType
    priority: InsightPriority
    confidence: InsightConfidence

    # Core insight information
    title: str
    description: str
    rationale: str  # Why this insight was generated

    # Affected code
    files: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)  # Entity IDs
    lines: dict[str, list[tuple[int, int]]] = field(default_factory=dict)  # file -> line ranges

    # Actionable recommendations
    recommendations: list[str] = field(default_factory=list)
    mitigation_suggestions: list[str] = field(default_factory=list)
    estimated_effort: str = "unknown"  # "low", "medium", "high"

    # Supporting evidence
    evidence: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    related_insights: list[str] = field(default_factory=list)  # IDs of related insights

    # Metadata
    generated_at: datetime = field(default_factory=datetime.now)
    tags: list[str] = field(default_factory=list)

    @property
    def impact_score(self) -> float:
        """Calculate estimated impact score."""
        priority_weights = {
            InsightPriority.CRITICAL: 10.0,
            InsightPriority.HIGH: 7.0,
            InsightPriority.MEDIUM: 4.0,
            InsightPriority.LOW: 2.0,
            InsightPriority.INFO: 0.5,
        }

        confidence_weights = {
            InsightConfidence.VERY_HIGH: 1.0,
            InsightConfidence.HIGH: 0.8,
            InsightConfidence.MEDIUM: 0.6,
            InsightConfidence.LOW: 0.4,
            InsightConfidence.EXPERIMENTAL: 0.2,
        }

        return priority_weights[self.priority] * confidence_weights[self.confidence]


@dataclass
class CodebaseInsights:
    """Complete insights analysis for a codebase."""

    total_files_analyzed: int
    total_entities_analyzed: int
    analysis_time_seconds: float

    # Categorized insights
    insights_by_type: dict[InsightType, list[CodeInsight]] = field(default_factory=dict)
    insights_by_priority: dict[InsightPriority, list[CodeInsight]] = field(default_factory=dict)
    high_impact_insights: list[CodeInsight] = field(default_factory=list)

    # Summary metrics
    overall_quality_score: float = 0.0
    technical_debt_ratio: float = 0.0
    maintainability_score: float = 0.0
    security_risk_score: float = 0.0

    # Language insights
    language_distribution: dict[str, int] = field(default_factory=dict)
    cross_language_issues: list[CodeInsight] = field(default_factory=list)

    # Recommendations dashboard
    quick_wins: list[CodeInsight] = field(default_factory=list)  # Low effort, high impact
    critical_issues: list[CodeInsight] = field(default_factory=list)
    refactoring_candidates: list[CodeInsight] = field(default_factory=list)

    # Trends and patterns
    common_patterns: list[dict[str, Any]] = field(default_factory=list)
    anti_patterns: list[dict[str, Any]] = field(default_factory=list)


class BaseInsightGenerator:
    """Base class for all insight generators.

    Provides common functionality and interface for insight generators.
    """

    def __init__(self, insight_rules: dict[str, Any]) -> None:
        """Initialize the insight generator.

        Args:
            insight_rules: Dictionary of rules and thresholds for insight generation.
        """
        self.insight_rules = insight_rules
        self._llm_enabled = True

    async def generate(self, data: dict[str, Any]) -> list[CodeInsight]:
        """Generate insights from the provided data.

        Args:
            data: Analysis data to generate insights from.

        Returns:
            List of generated CodeInsight objects.
        """
        raise NotImplementedError("Subclasses must implement generate()")

    async def _analyze_with_llm(
        self,
        content: str,
        system_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> dict[str, Any] | None:
        """Analyze content using LLM for enhanced insights.

        Args:
            content: Content to analyze
            system_prompt: System prompt with expected JSON schema
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Parsed JSON response or None if LLM unavailable
        """
        if not self._llm_enabled:
            return None

        try:
            # TODO: Replace with coding-ai LLM client when available
            from services.llm_client import analyze_with_llm

            return await analyze_with_llm(
                content=content,
                analysis_type=self.__class__.__name__,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except ImportError:
            logger.debug("LLM client not available, using rule-based analysis only")
            self._llm_enabled = False
            return None
        except Exception as e:
            logger.warning(f"LLM analysis failed, falling back to rules: {e}")
            return None
