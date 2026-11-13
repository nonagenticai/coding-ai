from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.services.ai_insights.architecture_insights import ArchitectureInsightGenerator
from src.services.ai_insights.code_quality_insights import (
    CrossLanguageInsightGenerator,
    PatternInsightGenerator,
    QualityInsightGenerator,
)
from src.services.ai_insights.maintainability_insights import MaintainabilityInsightGenerator
from src.services.ai_insights.performance_insights import PerformanceInsightGenerator
from src.services.ai_insights.security_insights import SecurityInsightGenerator
from src.services.ai_insights.testing_insights import RefactoringInsightGenerator


def _match(
    score: float,
    *,
    source_language: str = "python",
    target_language: str = "typescript",
    source_file: str = "a.py",
    target_file: str = "b.ts",
) -> SimpleNamespace:
    return SimpleNamespace(
        score=score,
        source_language=source_language,
        target_language=target_language,
        source_file=source_file,
        target_file=target_file,
    )


def _vuln(file_path: str, vulnerability_type: str, remediation_advice: str) -> SimpleNamespace:
    return SimpleNamespace(
        file_path=file_path,
        vulnerability_type=vulnerability_type,
        remediation_advice=remediation_advice,
    )


@pytest.mark.asyncio
async def test_architecture_generator_emits_dominance_and_pattern_insights():
    generator = ArchitectureInsightGenerator({})
    similarity_analysis = SimpleNamespace(
        language_distribution={"python": 7, "typescript": 3},
        matches_by_type={"structural": list(range(11))},
    )

    insights = await generator.generate({"similarity_analysis": similarity_analysis})

    assert len(insights) == 2
    assert {insight.id for insight in insights} == {
        "arch_lang_dominance_python",
        "arch_structural_patterns",
    }


@pytest.mark.asyncio
async def test_quality_related_generators_cover_threshold_and_grouping_logic():
    quality_generator = QualityInsightGenerator({"min_quality_score": 0.9})
    cross_language_generator = CrossLanguageInsightGenerator(
        {"cross_language_pattern_threshold": 0.7}
    )
    pattern_generator = PatternInsightGenerator({})

    similarity_analysis = SimpleNamespace(
        duplicate_code_ratio=0.35,
        cross_language_matches=[
            _match(0.82, source_language="python", target_language="go"),
            _match(0.78, source_language="go", target_language="python"),
            _match(0.4, source_language="python", target_language="rust"),
        ],
        cross_language_patterns=["factory", "adapter"],
    )
    security_analysis = SimpleNamespace(
        total_vulnerabilities_found=8,
        total_files_analyzed=10,
    )

    quality_insights = await quality_generator.generate(
        {
            "similarity_analysis": similarity_analysis,
            "security_analysis": security_analysis,
        }
    )
    cross_language_insights = await cross_language_generator.generate(
        {"similarity_analysis": similarity_analysis}
    )
    pattern_insights = await pattern_generator.generate(
        {"similarity_analysis": similarity_analysis}
    )

    assert len(quality_insights) == 1
    assert quality_insights[0].id == "quality_overall_low"
    assert quality_insights[0].metrics["target_score"] == 0.9

    assert len(cross_language_insights) == 1
    assert cross_language_insights[0].id == "cross_lang_go_python"
    assert cross_language_insights[0].evidence["pattern_count"] == 2

    assert len(pattern_insights) == 1
    assert pattern_insights[0].id == "patterns_cross_language"


@pytest.mark.asyncio
async def test_security_generator_emits_all_security_insight_types():
    generator = SecurityInsightGenerator({})
    critical_vulnerabilities = [
        _vuln("src/a.py", "sql_injection", "parameterize queries"),
        _vuln("src/b.py", "command_injection", "avoid shell=True"),
    ]
    security_analysis = SimpleNamespace(
        vulnerabilities_by_severity={"critical": critical_vulnerabilities},
        security_hotspots=["src/a.py", "src/c.py"],
        most_common_vulnerability="sql_injection",
        vulnerabilities_by_type={"sql_injection": critical_vulnerabilities},
    )

    insights = await generator.generate({"security_analysis": security_analysis})

    assert len(insights) == 3
    assert {insight.id for insight in insights} == {
        "security_critical",
        "security_hotspots",
        "security_pattern_sql_injection",
    }
    critical = next(insight for insight in insights if insight.id == "security_critical")
    assert sorted(critical.files) == ["src/a.py", "src/b.py"]
    assert critical.mitigation_suggestions == ["parameterize queries", "avoid shell=True"]


@pytest.mark.asyncio
async def test_refactoring_generator_groups_matches_and_reports_duplicates():
    generator = RefactoringInsightGenerator({"refactoring_opportunity_threshold": 0.8})
    similarity_analysis = SimpleNamespace(
        cross_language_matches=[
            _match(0.95, source_file="api/user.py", target_file="api/user.ts"),
            _match(0.91, source_file="api/user.ts", target_file="api/user.py"),
        ],
        matches_by_score={"high": [_match(0.85, source_file="x.py", target_file="y.py")]},
        matches_by_type={"exact_duplicate": [1], "near_duplicate": [2, 3]},
        duplicate_code_ratio=0.31,
    )

    insights = await generator.generate({"similarity_analysis": similarity_analysis})

    assert len(insights) == 3
    duplicate_summary = next(insight for insight in insights if insight.id == "refactor_duplicates")
    assert duplicate_summary.metrics["duplication_ratio"] == 0.31
    grouped = [insight for insight in insights if insight.id != "refactor_duplicates"]
    assert all(insight.files for insight in grouped)


@pytest.mark.asyncio
async def test_maintainability_and_performance_generators_cover_positive_and_empty_cases():
    maintainability = MaintainabilityInsightGenerator({})
    performance = PerformanceInsightGenerator({})

    similarity_analysis = SimpleNamespace(
        duplicate_code_ratio=0.25,
        language_distribution={
            "python": 10,
            "go": 4,
            "rust": 2,
            "typescript": 5,
            "java": 3,
            "kotlin": 2,
        },
        refactoring_opportunities=list(range(6)),
    )

    maintainability_insights = await maintainability.generate(
        {"similarity_analysis": similarity_analysis}
    )
    performance_insights = await performance.generate({"similarity_analysis": similarity_analysis})
    empty_performance = await performance.generate(
        {"similarity_analysis": SimpleNamespace(refactoring_opportunities=list(range(2)))}
    )

    assert len(maintainability_insights) == 2
    assert {insight.id for insight in maintainability_insights} == {
        "maint_duplication_ratio",
        "maint_language_diversity",
    }
    assert len(performance_insights) == 1
    assert performance_insights[0].id == "perf_refactoring_savings"
    assert empty_performance == []
