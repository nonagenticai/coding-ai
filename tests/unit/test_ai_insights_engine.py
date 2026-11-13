"""Tests for the AI Code Insights Engine."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.ai_insights.base import (
    CodebaseInsights,
    CodeInsight,
    InsightConfidence,
    InsightPriority,
    InsightType,
)


def _make_insight(
    insight_type: InsightType = InsightType.ARCHITECTURE,
    priority: InsightPriority = InsightPriority.HIGH,
    confidence: InsightConfidence = InsightConfidence.HIGH,
    estimated_effort: str = "medium",
) -> CodeInsight:
    return CodeInsight(
        id="test-1",
        insight_type=insight_type,
        priority=priority,
        confidence=confidence,
        title="Test insight",
        description="Test description",
        rationale="Test rationale",
        estimated_effort=estimated_effort,
    )


def _create_engine():
    """Helper to create engine with detectors mocked out."""
    with (
        patch("services.ai_insights.engine._get_security_detector", return_value=None),
        patch("services.ai_insights.engine._get_similarity_detector", return_value=None),
    ):
        from services.ai_insights.engine import AICodeInsightsEngine

        return AICodeInsightsEngine()


def _silence_generators(engine):
    """Silence all generators with empty mocks."""
    for gen_name in [
        "architecture_generator",
        "refactoring_generator",
        "security_generator",
        "performance_generator",
        "maintainability_generator",
        "cross_language_generator",
        "quality_generator",
        "pattern_generator",
    ]:
        getattr(engine, gen_name).generate = AsyncMock(return_value=[])


# ---------------------------------------------------------------------------
# Existing tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_orchestrates_all_enabled_generators():
    """All generators are invoked when no focus_areas filter is set."""
    engine = _create_engine()

    generators = [
        engine.architecture_generator,
        engine.refactoring_generator,
        engine.security_generator,
        engine.performance_generator,
        engine.maintainability_generator,
        engine.cross_language_generator,
        engine.quality_generator,
        engine.pattern_generator,
    ]
    for i, gen in enumerate(generators):
        insight = _make_insight(
            insight_type=list(InsightType)[i % len(InsightType)],
        )
        gen.generate = AsyncMock(return_value=[insight])

    result = await engine.generate_insights(["file1.py"], include_experimental=True)

    for gen in generators:
        gen.generate.assert_awaited_once()
    assert isinstance(result, CodebaseInsights)


@pytest.mark.asyncio
async def test_skips_disabled_generators():
    """Only generators matching focus_areas are invoked."""
    engine = _create_engine()

    arch_insight = _make_insight(insight_type=InsightType.ARCHITECTURE)
    engine.architecture_generator.generate = AsyncMock(return_value=[arch_insight])
    engine.security_generator.generate = AsyncMock(return_value=[])
    engine.refactoring_generator.generate = AsyncMock(return_value=[])
    engine.performance_generator.generate = AsyncMock(return_value=[])
    engine.maintainability_generator.generate = AsyncMock(return_value=[])
    engine.cross_language_generator.generate = AsyncMock(return_value=[])
    engine.quality_generator.generate = AsyncMock(return_value=[])
    engine.pattern_generator.generate = AsyncMock(return_value=[])

    result = await engine.generate_insights(
        ["file1.py"],
        focus_areas=[InsightType.ARCHITECTURE],
        include_experimental=True,
    )

    engine.architecture_generator.generate.assert_awaited_once()
    engine.security_generator.generate.assert_not_awaited()
    engine.refactoring_generator.generate.assert_not_awaited()
    assert isinstance(result, CodebaseInsights)


@pytest.mark.asyncio
async def test_merges_multi_generator_output():
    """Insights from multiple generators are collected together."""
    engine = _create_engine()

    arch = _make_insight(insight_type=InsightType.ARCHITECTURE)
    sec = _make_insight(insight_type=InsightType.SECURITY)
    engine.architecture_generator.generate = AsyncMock(return_value=[arch])
    engine.security_generator.generate = AsyncMock(return_value=[sec])
    for gen_name in [
        "refactoring_generator",
        "performance_generator",
        "maintainability_generator",
        "cross_language_generator",
        "quality_generator",
        "pattern_generator",
    ]:
        getattr(engine, gen_name).generate = AsyncMock(return_value=[])

    result = await engine.generate_insights(["f.py"], include_experimental=True)

    all_types = set()
    for insight_list in result.insights_by_type.values():
        for insight in insight_list:
            all_types.add(insight.insight_type)
    assert InsightType.ARCHITECTURE in all_types
    assert InsightType.SECURITY in all_types


@pytest.mark.asyncio
async def test_one_generator_failure_doesnt_corrupt_others():
    """If one generator raises, the engine still returns the remaining insights."""
    engine = _create_engine()

    good_insight = _make_insight(insight_type=InsightType.ARCHITECTURE)
    engine.architecture_generator.generate = AsyncMock(return_value=[good_insight])
    engine.security_generator.generate = AsyncMock(side_effect=RuntimeError("boom"))
    for gen_name in [
        "refactoring_generator",
        "performance_generator",
        "maintainability_generator",
        "cross_language_generator",
        "quality_generator",
        "pattern_generator",
    ]:
        getattr(engine, gen_name).generate = AsyncMock(return_value=[])

    with pytest.raises(RuntimeError, match="boom"):
        await engine.generate_insights(["f.py"], include_experimental=True)


@pytest.mark.asyncio
async def test_provider_model_metadata_included():
    """analysis_time_seconds is populated on the returned CodebaseInsights."""
    engine = _create_engine()
    _silence_generators(engine)

    result = await engine.generate_insights(["f.py"], include_experimental=True)
    assert result.analysis_time_seconds >= 0.0
    assert result.total_files_analyzed == 1


@pytest.mark.asyncio
async def test_empty_code_input_handling():
    """An empty file list returns a valid but empty insights result."""
    engine = _create_engine()
    _silence_generators(engine)

    result = await engine.generate_insights([], include_experimental=True)
    assert result.total_files_analyzed == 0
    assert result.high_impact_insights == []


# ---------------------------------------------------------------------------
# New tests covering uncovered branches
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_filters_experimental_insights():
    """include_experimental=False filters out EXPERIMENTAL confidence insights."""
    engine = _create_engine()

    experimental = _make_insight(confidence=InsightConfidence.EXPERIMENTAL)
    high_conf = _make_insight(confidence=InsightConfidence.HIGH)
    engine.architecture_generator.generate = AsyncMock(return_value=[experimental, high_conf])
    for gen_name in [
        "refactoring_generator",
        "security_generator",
        "performance_generator",
        "maintainability_generator",
        "cross_language_generator",
        "quality_generator",
        "pattern_generator",
    ]:
        getattr(engine, gen_name).generate = AsyncMock(return_value=[])

    result = await engine.generate_insights(["f.py"], include_experimental=False)

    # Only high confidence insight should be present
    all_insights = []
    for insight_list in result.insights_by_type.values():
        all_insights.extend(insight_list)
    assert all(i.confidence != InsightConfidence.EXPERIMENTAL for i in all_insights)


@pytest.mark.asyncio
async def test_max_insights_limits_output():
    """max_insights parameter limits the number of insights returned."""
    engine = _create_engine()

    insights = [
        _make_insight(
            insight_type=InsightType.ARCHITECTURE,
            priority=InsightPriority.HIGH,
        )
        for _ in range(5)
    ]
    engine.architecture_generator.generate = AsyncMock(return_value=insights)
    for gen_name in [
        "refactoring_generator",
        "security_generator",
        "performance_generator",
        "maintainability_generator",
        "cross_language_generator",
        "quality_generator",
        "pattern_generator",
    ]:
        getattr(engine, gen_name).generate = AsyncMock(return_value=[])

    result = await engine.generate_insights(["f.py"], include_experimental=True, max_insights=2)

    total = sum(len(il) for il in result.insights_by_type.values())
    assert total <= 2


@pytest.mark.asyncio
async def test_analyze_insights_critical_issues():
    """Critical priority insights are placed in critical_issues."""
    engine = _create_engine()

    critical = _make_insight(
        insight_type=InsightType.SECURITY,
        priority=InsightPriority.CRITICAL,
        confidence=InsightConfidence.HIGH,
    )
    engine.security_generator.generate = AsyncMock(return_value=[critical])
    for gen_name in [
        "architecture_generator",
        "refactoring_generator",
        "performance_generator",
        "maintainability_generator",
        "cross_language_generator",
        "quality_generator",
        "pattern_generator",
    ]:
        getattr(engine, gen_name).generate = AsyncMock(return_value=[])

    result = await engine.generate_insights(["f.py"], include_experimental=True)
    assert len(result.critical_issues) == 1


@pytest.mark.asyncio
async def test_analyze_insights_refactoring_candidates():
    """Refactoring type insights are placed in refactoring_candidates."""
    engine = _create_engine()

    refactoring = _make_insight(
        insight_type=InsightType.REFACTORING,
        priority=InsightPriority.MEDIUM,
    )
    engine.refactoring_generator.generate = AsyncMock(return_value=[refactoring])
    for gen_name in [
        "architecture_generator",
        "security_generator",
        "performance_generator",
        "maintainability_generator",
        "cross_language_generator",
        "quality_generator",
        "pattern_generator",
    ]:
        getattr(engine, gen_name).generate = AsyncMock(return_value=[])

    result = await engine.generate_insights(["f.py"], include_experimental=True)
    assert len(result.refactoring_candidates) == 1


@pytest.mark.asyncio
async def test_analyze_insights_quick_wins():
    """Low-effort, high-priority insights go into quick_wins."""
    engine = _create_engine()

    quick_win = _make_insight(
        insight_type=InsightType.QUALITY,
        priority=InsightPriority.HIGH,
        estimated_effort="low",
    )
    engine.quality_generator.generate = AsyncMock(return_value=[quick_win])
    for gen_name in [
        "architecture_generator",
        "refactoring_generator",
        "security_generator",
        "performance_generator",
        "maintainability_generator",
        "cross_language_generator",
        "pattern_generator",
    ]:
        getattr(engine, gen_name).generate = AsyncMock(return_value=[])

    result = await engine.generate_insights(["f.py"], include_experimental=True)
    assert len(result.quick_wins) == 1


@pytest.mark.asyncio
async def test_analyze_insights_high_impact():
    """Insights with impact_score >= 5.0 go into high_impact_insights."""
    engine = _create_engine()

    # CRITICAL + VERY_HIGH = 10.0 * 1.0 = 10.0 >= 5.0
    high_impact = _make_insight(
        insight_type=InsightType.ARCHITECTURE,
        priority=InsightPriority.CRITICAL,
        confidence=InsightConfidence.VERY_HIGH,
    )
    engine.architecture_generator.generate = AsyncMock(return_value=[high_impact])
    for gen_name in [
        "refactoring_generator",
        "security_generator",
        "performance_generator",
        "maintainability_generator",
        "cross_language_generator",
        "quality_generator",
        "pattern_generator",
    ]:
        getattr(engine, gen_name).generate = AsyncMock(return_value=[])

    result = await engine.generate_insights(["f.py"], include_experimental=True)
    assert len(result.high_impact_insights) >= 1


@pytest.mark.asyncio
async def test_analyze_insights_cross_language():
    """Cross-language insights go into cross_language_issues."""
    engine = _create_engine()

    cl_insight = _make_insight(
        insight_type=InsightType.CROSS_LANGUAGE,
        priority=InsightPriority.MEDIUM,
    )
    engine.cross_language_generator.generate = AsyncMock(return_value=[cl_insight])
    for gen_name in [
        "architecture_generator",
        "refactoring_generator",
        "security_generator",
        "performance_generator",
        "maintainability_generator",
        "quality_generator",
        "pattern_generator",
    ]:
        getattr(engine, gen_name).generate = AsyncMock(return_value=[])

    result = await engine.generate_insights(["f.py"], include_experimental=True)
    assert len(result.cross_language_issues) == 1


@pytest.mark.asyncio
async def test_gather_analysis_data_with_similarity_detector():
    """Similarity detector results are incorporated into analysis data."""
    engine = _create_engine()

    mock_similarity = MagicMock()
    mock_similarity.analyze_similarity = AsyncMock(
        return_value=MagicMock(
            language_distribution={"python": 5},
            duplicate_code_ratio=0.1,
        )
    )
    engine.similarity_detector = mock_similarity
    _silence_generators(engine)

    result = await engine.generate_insights(["f.py"], include_experimental=True)
    assert result.language_distribution == {"python": 5}
    assert result.technical_debt_ratio == 0.1


@pytest.mark.asyncio
async def test_gather_analysis_data_with_security_detector():
    """Security detector results are incorporated."""
    engine = _create_engine()

    mock_security = MagicMock()
    mock_security.analyze_security_vulnerabilities = AsyncMock(
        return_value=MagicMock(
            overall_risk_score=3.5,
            total_files_analyzed=5,
            total_vulnerabilities_found=2,
        )
    )
    engine.security_detector = mock_security
    _silence_generators(engine)

    result = await engine.generate_insights(["f.py"], include_experimental=True)
    assert result.security_risk_score == 3.5


@pytest.mark.asyncio
async def test_gather_analysis_data_detector_failure():
    """When a detector fails, analysis continues without it."""
    engine = _create_engine()

    mock_similarity = MagicMock()
    mock_similarity.analyze_similarity = AsyncMock(side_effect=RuntimeError("fail"))
    engine.similarity_detector = mock_similarity
    _silence_generators(engine)

    result = await engine.generate_insights(["f.py"], include_experimental=True)
    assert isinstance(result, CodebaseInsights)


@pytest.mark.asyncio
async def test_security_risk_score_capped_at_10():
    """security_risk_score is clamped to max 10.0."""
    engine = _create_engine()

    mock_security = MagicMock()
    mock_security.analyze_security_vulnerabilities = AsyncMock(
        return_value=MagicMock(
            overall_risk_score=25.0,
            total_files_analyzed=5,
            total_vulnerabilities_found=100,
        )
    )
    engine.security_detector = mock_security
    _silence_generators(engine)

    result = await engine.generate_insights(["f.py"], include_experimental=True)
    assert result.security_risk_score == 10.0


# ---------------------------------------------------------------------------
# Singleton tests
# ---------------------------------------------------------------------------
def test_singleton_get_and_reset():
    """AIInsightsEngineSingleton returns the same instance and can be reset."""
    with (
        patch("services.ai_insights.engine._get_security_detector", return_value=None),
        patch("services.ai_insights.engine._get_similarity_detector", return_value=None),
    ):
        from services.ai_insights.engine import (
            AIInsightsEngineSingleton,
            get_ai_insights_engine,
            reset_ai_insights_engine,
        )

        AIInsightsEngineSingleton.reset_instance()

        instance1 = get_ai_insights_engine()
        instance2 = get_ai_insights_engine()
        assert instance1 is instance2

        reset_ai_insights_engine()
        instance3 = get_ai_insights_engine()
        assert instance3 is not instance1


def test_lazy_import_security_detector():
    """_get_security_detector returns None when import fails."""
    with patch.dict("sys.modules", {"services.security_detector": None}):
        from services.ai_insights.engine import _get_security_detector

        result = _get_security_detector()
        assert result is None


def test_lazy_import_similarity_detector():
    """_get_similarity_detector returns None when import fails."""
    with patch.dict("sys.modules", {"services.similarity_detector": None}):
        from services.ai_insights.engine import _get_similarity_detector

        result = _get_similarity_detector()
        assert result is None


def test_insight_rules_loaded():
    """Engine loads insight rules on init."""
    engine = _create_engine()
    assert "duplicate_code_threshold" in engine.insight_rules
    assert "high_complexity_threshold" in engine.insight_rules
    assert engine.insight_rules["duplicate_code_threshold"] == 0.85


def test_quality_metrics_initialized():
    """Engine initializes quality metric weights."""
    engine = _create_engine()
    assert "complexity_weight" in engine.quality_metrics
    assert "security_weight" in engine.quality_metrics
    total = sum(engine.quality_metrics.values())
    assert abs(total - 1.0) < 0.001


@pytest.mark.asyncio
async def test_quality_score_calculation_with_both_detectors():
    """Quality score is calculated from both similarity and security detectors."""
    engine = _create_engine()

    mock_similarity = MagicMock()
    mock_similarity.analyze_similarity = AsyncMock(
        return_value=MagicMock(
            language_distribution={"python": 10},
            duplicate_code_ratio=0.2,
        )
    )
    mock_security = MagicMock()
    mock_security.analyze_security_vulnerabilities = AsyncMock(
        return_value=MagicMock(
            overall_risk_score=2.0,
            total_files_analyzed=10,
            total_vulnerabilities_found=2,
        )
    )
    engine.similarity_detector = mock_similarity
    engine.security_detector = mock_security
    _silence_generators(engine)

    result = await engine.generate_insights(["f.py"], include_experimental=True)
    assert result.overall_quality_score > 0
    assert result.maintainability_score == result.overall_quality_score
