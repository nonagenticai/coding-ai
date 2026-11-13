"""AI Insights Package.

This package provides AI-powered code insights and recommendations.
It combines multiple analysis sources to generate intelligent insights
about code quality, security, architecture, and maintainability.

Example Usage:
    from services.ai_insights import get_ai_insights_engine, InsightType

    engine = get_ai_insights_engine()
    insights = await engine.generate_insights(
        file_paths=["/path/to/code"],
        focus_areas=[InsightType.SECURITY, InsightType.QUALITY],
    )
"""

from __future__ import annotations

# Individual insight generators
from .architecture_insights import ArchitectureInsightGenerator

# Base types and dataclasses
from .base import (
    BaseInsightGenerator,
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

# Main engine
from .engine import (
    AICodeInsightsEngine,
    AIInsightsEngineSingleton,
    get_ai_insights_engine,
    reset_ai_insights_engine,
)
from .maintainability_insights import MaintainabilityInsightGenerator
from .performance_insights import PerformanceInsightGenerator
from .security_insights import SecurityInsightGenerator
from .testing_insights import RefactoringInsightGenerator

__all__ = [
    # Enums
    "InsightType",
    "InsightPriority",
    "InsightConfidence",
    # Dataclasses
    "CodeInsight",
    "CodebaseInsights",
    # Base class
    "BaseInsightGenerator",
    # Main engine
    "AICodeInsightsEngine",
    "AIInsightsEngineSingleton",
    "get_ai_insights_engine",
    "reset_ai_insights_engine",
    # Individual generators
    "ArchitectureInsightGenerator",
    "SecurityInsightGenerator",
    "PerformanceInsightGenerator",
    "MaintainabilityInsightGenerator",
    "RefactoringInsightGenerator",
    "QualityInsightGenerator",
    "CrossLanguageInsightGenerator",
    "PatternInsightGenerator",
]
