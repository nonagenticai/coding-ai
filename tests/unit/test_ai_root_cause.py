"""Tests for AIRootCauseEngine covering all uncovered methods and branches.

Focus: initialization, analyze_root_causes flow, all sub-analysis methods,
pattern matching, enhancement, consolidation, caching, and fallback.

NOTE: ErrorDetails uses Pydantic's use_enum_values=True, which converts
StrEnum values to plain strings. Source methods that call .value on these
fields will raise AttributeError, which is caught by try/except blocks.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.ai_test_failure_models import (
    ConfidenceLevel,
    ErrorDetails,
    FailureType,
    RootCause,
    SeverityLevel,
    TestFailureContext,
    TestFailureData,
    TestFramework,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_failure_data(
    raw_output: str = "AssertionError: expected True got False",
    failure_id: str = "fail-1",
    execution_time: float | None = None,
    memory_usage: int | None = None,
) -> TestFailureData:
    return TestFailureData(
        failure_id=failure_id,
        raw_output=raw_output,
        context=TestFailureContext(
            test_name="test_example",
            framework=TestFramework.PYTEST,
        ),
        execution_time=execution_time,
        memory_usage=memory_usage,
    )


def _make_error_details(
    error_type: FailureType = FailureType.ASSERTION_ERROR,
    error_message: str = "expected True got False",
    file_path: str | None = None,
    line_number: int | None = None,
    traceback: str | None = None,
    stack_trace: list[str] | None = None,
) -> ErrorDetails:
    return ErrorDetails(
        error_type=error_type,
        error_message=error_message,
        file_path=file_path,
        line_number=line_number,
        traceback=traceback,
        stack_trace=stack_trace,
    )


def _make_root_cause(
    confidence_score: float = 0.5,
    category: str = "unknown",
    description: str = "Test failed with assertion_error: expected True got False",
) -> RootCause:
    if confidence_score >= 0.9:
        level = ConfidenceLevel.VERY_HIGH
    elif confidence_score >= 0.75:
        level = ConfidenceLevel.HIGH
    elif confidence_score >= 0.5:
        level = ConfidenceLevel.MEDIUM
    elif confidence_score >= 0.25:
        level = ConfidenceLevel.LOW
    else:
        level = ConfidenceLevel.VERY_LOW
    return RootCause(
        cause_id="cause-1",
        category=category,
        description=description,
        severity=SeverityLevel.MEDIUM,
        confidence=level,
        confidence_score=confidence_score,
        evidence=["Error type: assertion_error"],
    )


def _build_engine():
    """Build AIRootCauseEngine with mocked LLM agent runners.

    Migration: provider config is gone; LLM transport is built by
    ``make_agent_runner`` (returning a shim). We patch that factory to hand
    out ``MagicMock`` runners with ``AsyncMock`` ``run`` so tests can set
    return values per call site.
    """

    def _runner_factory(*args: object, **kwargs: object) -> MagicMock:
        runner = MagicMock()
        runner.run = AsyncMock()
        return runner

    with patch("services.ai_root_cause.make_agent_runner", side_effect=_runner_factory):
        from services.ai_root_cause import AIRootCauseEngine

        engine = AIRootCauseEngine()
    return engine


def _build_context(**kwargs):
    from services.ai_root_cause import AIAnalysisContext

    defaults = dict(
        failure_data=_make_failure_data(),
        error_details=_make_error_details(),
    )
    defaults.update(kwargs)
    return AIAnalysisContext(**defaults)


# ===================================================================
# Initialization
# ===================================================================


class TestInit:
    def test_default_init(self):
        """Engine initializes with the registry schema_id and empty caches."""
        engine = _build_engine()
        # Provider strings (e.g. "openai:gpt-4o") are gone; selection is now
        # server-side in inference-router. ``model_name`` defaults to the
        # registry schema_id used as the X-Schema-Id header for routing.
        assert engine.model_name == "test_failure_analysis"
        assert engine._analysis_cache == {}
        assert engine._knowledge_base is not None

    def test_custom_model_name(self):
        """``model_name`` override is preserved for diagnostic continuity."""

        def _factory(*args: object, **kwargs: object) -> MagicMock:
            runner = MagicMock()
            runner.run = AsyncMock()
            return runner

        with patch("services.ai_root_cause.make_agent_runner", side_effect=_factory):
            from services.ai_root_cause import AIRootCauseEngine

            engine = AIRootCauseEngine(model_name="custom-routing-key")
        assert engine.model_name == "custom-routing-key"

    def test_knowledge_base_has_expected_keys(self):
        engine = _build_engine()
        kb = engine._knowledge_base
        assert "common_patterns" in kb
        assert "severity_mappings" in kb
        assert "confidence_factors" in kb
        assert "assertion_failures" in kb["common_patterns"]
        assert "import_errors" in kb["common_patterns"]
        assert "runtime_errors" in kb["common_patterns"]


# ===================================================================
# analyze_root_causes
# ===================================================================


class TestAnalyzeRootCauses:
    @pytest.mark.asyncio
    async def test_full_analysis_flow(self):
        """All sub-analyses are called and results consolidated."""
        engine = _build_engine()
        ctx = _build_context()

        cause = _make_root_cause(confidence_score=0.6)

        with patch.object(engine, "_create_cache_key", return_value="flow-key"):
            with patch.object(engine, "_analyze_syntactic_issues", return_value=[cause]):
                with patch.object(engine, "_analyze_semantic_issues", return_value=[]):
                    with patch.object(engine, "_analyze_environmental_issues", return_value=[]):
                        with patch.object(engine, "_analyze_integration_issues", return_value=[]):
                            with patch.object(
                                engine, "_analyze_performance_issues", return_value=[]
                            ):
                                with patch.object(
                                    engine, "_analyze_known_patterns", return_value=[]
                                ):
                                    results = await engine.analyze_root_causes(ctx)

        assert len(results) >= 1
        assert results[0].confidence_score >= 0.1

    @pytest.mark.asyncio
    async def test_caching(self):
        """Second call returns cached results."""
        engine = _build_engine()
        ctx = _build_context()

        cause = _make_root_cause(confidence_score=0.6)

        with patch.object(engine, "_create_cache_key", return_value="test-cache-key"):
            with patch.object(engine, "_analyze_syntactic_issues", return_value=[cause]):
                with patch.object(engine, "_analyze_semantic_issues", return_value=[]):
                    with patch.object(engine, "_analyze_environmental_issues", return_value=[]):
                        with patch.object(engine, "_analyze_integration_issues", return_value=[]):
                            with patch.object(
                                engine, "_analyze_performance_issues", return_value=[]
                            ):
                                with patch.object(
                                    engine, "_analyze_known_patterns", return_value=[]
                                ):
                                    results1 = await engine.analyze_root_causes(ctx)

            # Second call should use cache
            ctx2 = _build_context()
            results2 = await engine.analyze_root_causes(ctx2)

        assert results1 is results2

    @pytest.mark.asyncio
    async def test_confidence_filtering(self):
        """Low-confidence causes are filtered."""
        engine = _build_engine()
        ctx = _build_context()

        low = _make_root_cause(confidence_score=0.05)
        high = _make_root_cause(
            confidence_score=0.7,
            description="High confidence cause with sufficient length to pass validation",
        )
        high.cause_id = "cause-2"

        with patch.object(engine, "_create_cache_key", return_value="filter-key"):
            with patch.object(engine, "_analyze_syntactic_issues", return_value=[low, high]):
                with patch.object(engine, "_analyze_semantic_issues", return_value=[]):
                    with patch.object(engine, "_analyze_environmental_issues", return_value=[]):
                        with patch.object(engine, "_analyze_integration_issues", return_value=[]):
                            with patch.object(
                                engine, "_analyze_performance_issues", return_value=[]
                            ):
                                with patch.object(
                                    engine, "_analyze_known_patterns", return_value=[]
                                ):
                                    results = await engine.analyze_root_causes(
                                        ctx, minimum_confidence=0.5
                                    )

        assert all(c.confidence_score >= 0.5 for c in results)

    @pytest.mark.asyncio
    async def test_max_causes_limit(self):
        """Results are limited to max_causes."""
        engine = _build_engine()
        ctx = _build_context()

        causes = []
        for i in range(10):
            c = _make_root_cause(
                confidence_score=0.5 + i * 0.01,
                description=f"Cause number {i} with enough detail to pass validation checks",
            )
            c.cause_id = f"cause-{i}"
            causes.append(c)

        with patch.object(engine, "_create_cache_key", return_value="limit-key"):
            with patch.object(engine, "_analyze_syntactic_issues", return_value=causes):
                with patch.object(engine, "_analyze_semantic_issues", return_value=[]):
                    with patch.object(engine, "_analyze_environmental_issues", return_value=[]):
                        with patch.object(engine, "_analyze_integration_issues", return_value=[]):
                            with patch.object(
                                engine, "_analyze_performance_issues", return_value=[]
                            ):
                                with patch.object(
                                    engine, "_analyze_known_patterns", return_value=[]
                                ):
                                    results = await engine.analyze_root_causes(ctx, max_causes=3)

        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_exception_returns_fallback(self):
        """Exception during analysis returns fallback."""
        engine = _build_engine()
        ctx = _build_context()

        fallback = _make_root_cause(
            confidence_score=0.2,
            description="Fallback analysis result due to AI failure detected",
        )
        fallback.evidence = ["Fallback analysis due to AI analysis failure"]

        with patch.object(engine, "_create_cache_key", side_effect=RuntimeError("boom")):
            with patch.object(engine, "_create_fallback_analysis", return_value=[fallback]):
                results = await engine.analyze_root_causes(ctx)

        assert len(results) == 1
        assert "Fallback" in results[0].evidence[0]


# ===================================================================
# Outage vs genuine-empty distinction (F4 honesty fix)
# ===================================================================


class TestOutageVsGenuineEmpty:
    """The engine must NOT conflate a total LLM outage with a real empty result.

    - OUTAGE  (every LLM agent raises): return the degraded fallback marker,
      not a silent empty list. `_create_fallback_analysis` is reachable.
    - GENUINE EMPTY (LLM ran, found nothing): return `[]` — success, not
      degraded.
    """

    def _set_all_agents(self, engine, *, run):
        """Replace all five LLM agent runners with a mock using ``run``."""
        for name in (
            "syntactic_agent",
            "semantic_agent",
            "environmental_agent",
            "integration_agent",
            "performance_agent",
        ):
            agent = MagicMock()
            agent.run = run
            setattr(engine, name, agent)

    @pytest.mark.asyncio
    async def test_total_llm_outage_returns_degraded_fallback(self):
        """Every LLM agent unreachable -> degraded fallback, NOT empty success."""
        engine = _build_engine()
        # raw_output matches no known pattern, so pattern analysis is empty and
        # the ONLY thing that can populate results is the LLM plane.
        ctx = _build_context(
            failure_data=_make_failure_data(raw_output="totally unrelated output blob"),
        )

        self._set_all_agents(
            engine, run=AsyncMock(side_effect=RuntimeError("LLM plane unreachable"))
        )

        results = await engine.analyze_root_causes(ctx)

        # Degraded: the reachable fallback marker is returned (not []).
        assert results != []
        assert results[0].cause_id.startswith("fallback_")
        assert any(
            c.evidence and "Fallback analysis due to AI analysis failure" in c.evidence[0]
            for c in results
        )

    @pytest.mark.asyncio
    async def test_outage_result_is_not_cached(self):
        """A degraded outage result must not be cached (so recovery re-runs)."""
        engine = _build_engine()
        ctx = _build_context(
            failure_data=_make_failure_data(raw_output="totally unrelated output blob"),
        )
        self._set_all_agents(
            engine, run=AsyncMock(side_effect=RuntimeError("LLM plane unreachable"))
        )

        await engine.analyze_root_causes(ctx)

        assert engine._analysis_cache == {}

    @pytest.mark.asyncio
    async def test_outage_preserves_pattern_matches(self):
        """On outage, real non-LLM pattern matches survive alongside the marker."""
        engine = _build_engine()
        ctx = _build_context(
            failure_data=_make_failure_data(
                raw_output="ModuleNotFoundError: No module named 'foobar'"
            ),
            error_details=_make_error_details(error_type=FailureType.IMPORT_ERROR),
        )
        self._set_all_agents(
            engine, run=AsyncMock(side_effect=RuntimeError("LLM plane unreachable"))
        )

        results = await engine.analyze_root_causes(ctx)

        assert results[0].cause_id.startswith("fallback_")
        # The regex-matched dependency cause is retained.
        assert any("module_not_found" in c.cause_id for c in results)

    @pytest.mark.asyncio
    async def test_llm_healthy_but_no_causes_returns_empty(self):
        """LLM ran fine and found nothing -> genuine empty list, NOT degraded."""
        engine = _build_engine()
        ctx = _build_context(
            failure_data=_make_failure_data(raw_output="totally unrelated output blob"),
        )

        empty_result = MagicMock()
        empty_result.output = []
        self._set_all_agents(engine, run=AsyncMock(return_value=empty_result))

        results = await engine.analyze_root_causes(ctx)

        # Genuine empty: no fabricated fallback cause.
        assert results == []

    @pytest.mark.asyncio
    async def test_partial_outage_returns_real_causes_no_fallback(self):
        """If at least one agent succeeds, no degraded fallback is injected."""
        engine = _build_engine()
        ctx = _build_context(
            failure_data=_make_failure_data(raw_output="totally unrelated output blob"),
        )

        good = _make_root_cause(
            confidence_score=0.7,
            description="A real root cause produced by the one healthy agent here",
        )
        good_result = MagicMock()
        good_result.output = [good]

        # Syntactic succeeds; the other four are unreachable.
        engine.syntactic_agent = MagicMock()
        engine.syntactic_agent.run = AsyncMock(return_value=good_result)
        for name in (
            "semantic_agent",
            "environmental_agent",
            "integration_agent",
            "performance_agent",
        ):
            agent = MagicMock()
            agent.run = AsyncMock(side_effect=RuntimeError("LLM plane unreachable"))
            setattr(engine, name, agent)

        results = await engine.analyze_root_causes(ctx)

        assert results != []
        # No degraded fallback marker — we have a real result.
        assert not any(c.cause_id.startswith("fallback_") for c in results)


# ===================================================================
# Sub-analysis methods (error paths -- .value on str raises AttributeError)
# ===================================================================


class TestSubAnalyses:
    @pytest.mark.asyncio
    async def test_syntactic_analysis_error_returns_empty(self):
        """Syntactic analysis error returns empty list."""
        engine = _build_engine()
        ctx = _build_context()
        results = await engine._analyze_syntactic_issues(ctx)
        assert results == []

    @pytest.mark.asyncio
    async def test_semantic_analysis_error_returns_empty(self):
        """Semantic analysis error returns empty list."""
        engine = _build_engine()
        ctx = _build_context()
        results = await engine._analyze_semantic_issues(ctx)
        assert results == []

    @pytest.mark.asyncio
    async def test_environmental_analysis_error_returns_empty(self):
        """Environmental analysis error returns empty list."""
        engine = _build_engine()
        ctx = _build_context()
        results = await engine._analyze_environmental_issues(ctx)
        assert results == []

    @pytest.mark.asyncio
    async def test_integration_analysis_error_returns_empty(self):
        """Integration analysis error returns empty list."""
        engine = _build_engine()
        ctx = _build_context()
        results = await engine._analyze_integration_issues(ctx)
        assert results == []

    @pytest.mark.asyncio
    async def test_performance_analysis_error_returns_empty(self):
        """Performance analysis error returns empty list."""
        engine = _build_engine()
        ctx = _build_context()
        results = await engine._analyze_performance_issues(ctx)
        assert results == []


# ===================================================================
# Sub-analysis methods with mocked agent calls (success paths)
# ===================================================================


class TestSubAnalysesSuccess:
    @pytest.mark.asyncio
    async def test_syntactic_analysis_success(self):
        """Syntactic analysis with mocked agent run returns enhanced causes."""
        engine = _build_engine()
        ctx = _build_context(
            error_details=_make_error_details(
                file_path="/src/app.py",
                line_number=42,
                traceback="Traceback ...\nSyntaxError: invalid syntax",
            ),
            code_snippets=["def foo():\n    pass"],
        )

        cause = _make_root_cause(confidence_score=0.7)
        mock_result = MagicMock()
        mock_result.output = [cause]

        with patch.object(engine, "_prepare_syntactic_input", return_value="mocked input"):
            engine.syntactic_agent = AsyncMock()
            engine.syntactic_agent.run = AsyncMock(return_value=mock_result)
            results = await engine._analyze_syntactic_issues(ctx)

        assert len(results) == 1
        assert "syntactic" in results[0].cause_id

    @pytest.mark.asyncio
    async def test_semantic_analysis_success(self):
        """Semantic analysis with mocked agent run returns enhanced causes."""
        engine = _build_engine()
        ctx = _build_context(
            error_details=_make_error_details(
                stack_trace=["File app.py, line 1", "  assert True == False"],
            ),
        )

        cause = _make_root_cause(confidence_score=0.6)
        mock_result = MagicMock()
        mock_result.output = [cause]

        with patch.object(engine, "_prepare_semantic_input", return_value="mocked input"):
            engine.semantic_agent = AsyncMock()
            engine.semantic_agent.run = AsyncMock(return_value=mock_result)
            results = await engine._analyze_semantic_issues(ctx)

        assert len(results) == 1
        assert "semantic" in results[0].cause_id

    @pytest.mark.asyncio
    async def test_environmental_analysis_success(self):
        """Environmental analysis with mocked agent run returns enhanced causes."""
        engine = _build_engine()
        ctx = _build_context(
            environment_info={"os": "linux", "python": "3.12"},
        )

        cause = _make_root_cause(confidence_score=0.5)
        mock_result = MagicMock()
        mock_result.output = [cause]

        with patch.object(engine, "_prepare_environmental_input", return_value="mocked input"):
            engine.environmental_agent = AsyncMock()
            engine.environmental_agent.run = AsyncMock(return_value=mock_result)
            results = await engine._analyze_environmental_issues(ctx)

        assert len(results) == 1
        assert "environmental" in results[0].cause_id

    @pytest.mark.asyncio
    async def test_integration_analysis_success(self):
        """Integration analysis with mocked agent run returns enhanced causes."""
        engine = _build_engine()
        ctx = _build_context(
            dependency_graph={"app": ["db", "cache"]},
            error_details=_make_error_details(
                traceback="Traceback ...\nConnectionError: refused",
            ),
        )

        cause = _make_root_cause(confidence_score=0.6)
        mock_result = MagicMock()
        mock_result.output = [cause]

        with patch.object(engine, "_prepare_integration_input", return_value="mocked input"):
            engine.integration_agent = AsyncMock()
            engine.integration_agent.run = AsyncMock(return_value=mock_result)
            results = await engine._analyze_integration_issues(ctx)

        assert len(results) == 1
        assert "integration" in results[0].cause_id

    @pytest.mark.asyncio
    async def test_performance_analysis_success(self):
        """Performance analysis with mocked agent run returns enhanced causes."""
        engine = _build_engine()
        ctx = _build_context(
            failure_data=_make_failure_data(execution_time=120.5, memory_usage=1024000),
        )

        cause = _make_root_cause(confidence_score=0.5)
        mock_result = MagicMock()
        mock_result.output = [cause]

        with patch.object(engine, "_prepare_performance_input", return_value="mocked input"):
            engine.performance_agent = AsyncMock()
            engine.performance_agent.run = AsyncMock(return_value=mock_result)
            results = await engine._analyze_performance_issues(ctx)

        assert len(results) == 1
        assert "performance" in results[0].cause_id


# ===================================================================
# _prepare_*_input methods
# ===================================================================


class TestPrepareInputs:
    def test_prepare_syntactic_input_basic(self):
        """Syntactic input includes error type and message."""
        engine = _build_engine()
        ctx = _build_context(
            error_details=_make_error_details(
                error_type=FailureType.SYNTAX_ERROR,
                error_message="invalid syntax",
            ),
        )
        result = engine._prepare_syntactic_input(ctx)
        assert "syntax_error" in result
        assert "invalid syntax" in result

    def test_prepare_syntactic_input_with_all_fields(self):
        """Syntactic input with file_path, line_number, traceback, code_snippets."""
        engine = _build_engine()
        ctx = _build_context(
            error_details=_make_error_details(
                error_type=FailureType.SYNTAX_ERROR,
                error_message="invalid syntax",
                file_path="/src/app.py",
                line_number=42,
                traceback="Traceback line",
            ),
        )
        ctx.code_snippets = ["def foo(): pass"]

        result = engine._prepare_syntactic_input(ctx)
        assert "syntax_error" in result
        assert "/src/app.py" in result
        assert "42" in result
        assert "Traceback line" in result
        assert "def foo(): pass" in result

    def test_prepare_semantic_input(self):
        """Semantic input includes test name and stack trace."""
        engine = _build_engine()
        ctx = _build_context(
            error_details=_make_error_details(
                error_type=FailureType.ASSERTION_ERROR,
                error_message="not equal",
                stack_trace=["line 1", "line 2"],
            ),
        )

        result = engine._prepare_semantic_input(ctx)
        assert "test_example" in result  # test_name from _make_failure_data
        assert "assertion_error" in result
        assert "line 1" in result

    def test_prepare_environmental_input(self):
        """Environmental input includes framework and environment info."""
        engine = _build_engine()
        ctx = _build_context(
            error_details=_make_error_details(
                error_type=FailureType.CONFIGURATION_ERROR,
                error_message="missing config",
            ),
        )
        ctx.environment_info = {"os": "linux"}

        result = engine._prepare_environmental_input(ctx)
        assert "configuration_error" in result
        assert "pytest" in result
        assert "linux" in result

    def test_prepare_environmental_input_no_extras(self):
        """Environmental input without environment_info or execution_environment."""
        engine = _build_engine()
        ctx = _build_context(
            error_details=_make_error_details(
                error_type=FailureType.CONFIGURATION_ERROR,
                error_message="missing config",
            ),
        )
        ctx.environment_info = None

        result = engine._prepare_environmental_input(ctx)
        assert "configuration_error" in result

    def test_prepare_integration_input(self):
        """Integration input includes dependency graph and traceback."""
        engine = _build_engine()
        ctx = _build_context(
            error_details=_make_error_details(
                error_type=FailureType.DEPENDENCY_ERROR,
                error_message="connection refused",
                traceback="ConnectionError: refused",
            ),
        )
        ctx.dependency_graph = {"app": ["db"]}

        result = engine._prepare_integration_input(ctx)
        assert "dependency_error" in result
        assert "db" in result
        assert "ConnectionError" in result

    def test_prepare_integration_input_no_extras(self):
        """Integration input without dependency_graph or traceback."""
        engine = _build_engine()
        ctx = _build_context(
            error_details=_make_error_details(
                error_type=FailureType.RUNTIME_ERROR,
                error_message="something broke",
            ),
        )
        ctx.dependency_graph = None

        result = engine._prepare_integration_input(ctx)
        assert "runtime_error" in result

    def test_prepare_performance_input(self):
        """Performance input includes execution time and memory usage."""
        engine = _build_engine()
        ctx = _build_context(
            failure_data=_make_failure_data(execution_time=30.5, memory_usage=2048000),
            error_details=_make_error_details(
                error_type=FailureType.TIMEOUT_ERROR,
                error_message="test timed out",
            ),
        )

        result = engine._prepare_performance_input(ctx)
        assert "timeout_error" in result
        assert "30.5" in result
        assert "2048000" in result

    def test_prepare_performance_input_no_extras(self):
        """Performance input without execution_time or memory_usage."""
        engine = _build_engine()
        ctx = _build_context(
            error_details=_make_error_details(
                error_type=FailureType.RUNTIME_ERROR,
                error_message="error",
            ),
        )

        result = engine._prepare_performance_input(ctx)
        assert "runtime_error" in result


# ===================================================================
# _analyze_known_patterns
# ===================================================================


class TestAnalyzeKnownPatterns:
    @pytest.mark.asyncio
    async def test_pattern_match_module_not_found(self):
        """ModuleNotFoundError pattern is matched with correct category."""
        engine = _build_engine()
        ctx = _build_context(
            failure_data=_make_failure_data(
                raw_output="ModuleNotFoundError: No module named 'foobar'"
            ),
            error_details=_make_error_details(error_type=FailureType.IMPORT_ERROR),
        )
        results = await engine._analyze_known_patterns(ctx)
        assert len(results) >= 1
        assert any("module_not_found" in c.cause_id for c in results)
        assert results[0].category == "dependency_missing"

    @pytest.mark.asyncio
    async def test_pattern_match_attribute_error(self):
        """AttributeError pattern is matched."""
        engine = _build_engine()
        ctx = _build_context(
            failure_data=_make_failure_data(
                raw_output="AttributeError: 'NoneType' object has no attribute 'foo'"
            ),
            error_details=_make_error_details(error_type=FailureType.RUNTIME_ERROR),
        )
        results = await engine._analyze_known_patterns(ctx)
        assert len(results) >= 1
        assert results[0].category == "null_reference"

    @pytest.mark.asyncio
    async def test_no_pattern_match(self):
        """No matching pattern returns empty list."""
        engine = _build_engine()
        ctx = _build_context(
            failure_data=_make_failure_data(raw_output="some completely unrelated output xyz"),
            error_details=_make_error_details(error_type=FailureType.RUNTIME_ERROR),
        )
        results = await engine._analyze_known_patterns(ctx)
        assert results == []

    @pytest.mark.asyncio
    async def test_default_context_pattern_match(self):
        """Default context with matching pattern returns results."""
        engine = _build_engine()
        ctx = _build_context(
            failure_data=_make_failure_data(raw_output="ModuleNotFoundError: No module named 'x'"),
        )
        results = await engine._analyze_known_patterns(ctx)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_pattern_error_returns_empty(self):
        """Exception in pattern matching returns empty list."""
        engine = _build_engine()
        ctx = _build_context()
        engine._knowledge_base = None  # type: ignore[assignment]
        results = await engine._analyze_known_patterns(ctx)
        assert results == []


# ===================================================================
# _matches_pattern
# ===================================================================


class TestMatchesPattern:
    def test_empty_pattern_returns_false(self):
        engine = _build_engine()
        assert engine._matches_pattern("any text", "") is False

    def test_valid_regex_match(self):
        engine = _build_engine()
        assert (
            engine._matches_pattern(
                "ModuleNotFoundError: No module named 'x'", r"ModuleNotFoundError.*No module named"
            )
            is True
        )

    def test_valid_regex_no_match(self):
        engine = _build_engine()
        assert engine._matches_pattern("all good", r"ModuleNotFoundError") is False

    def test_invalid_regex_falls_back_to_substring(self):
        engine = _build_engine()
        # Invalid regex pattern but text contains it as substring
        assert engine._matches_pattern("hello [world", "[world") is True

    def test_invalid_regex_no_substring(self):
        engine = _build_engine()
        assert engine._matches_pattern("hello world", "[xyz") is False

    def test_case_insensitive(self):
        engine = _build_engine()
        assert engine._matches_pattern("keyerror: missing", r"KeyError.*") is True


# ===================================================================
# _create_pattern_based_cause
# ===================================================================


class TestCreatePatternBasedCause:
    def _ctx_with_mock_error_details(self, error_type_value="assertion_error"):
        ctx = _build_context()
        ed = MagicMock()
        ed.error_type.value = error_type_value
        ed.error_message = "some error"
        ctx.error_details = ed
        return ctx

    def test_creates_cause_with_valid_category(self):
        engine = _build_engine()
        ctx = self._ctx_with_mock_error_details("runtime_error")
        pattern_info = {
            "pattern": r"KeyError.*",
            "cause": "unknown",
            "confidence": 0.7,
        }
        result = engine._create_pattern_based_cause("key_error", pattern_info, ctx)
        assert "pattern_key_error" in result.cause_id
        assert result.confidence_score == 0.7
        assert result.category == "unknown"

    def test_high_confidence_maps_to_very_high(self):
        engine = _build_engine()
        ctx = self._ctx_with_mock_error_details()
        pattern_info = {"cause": "unknown", "confidence": 0.95}
        result = engine._create_pattern_based_cause("test", pattern_info, ctx)
        assert result.confidence == ConfidenceLevel.VERY_HIGH

    def test_medium_confidence(self):
        engine = _build_engine()
        ctx = self._ctx_with_mock_error_details()
        pattern_info = {"cause": "unknown", "confidence": 0.6}
        result = engine._create_pattern_based_cause("test", pattern_info, ctx)
        assert result.confidence == ConfidenceLevel.MEDIUM

    def test_low_confidence(self):
        engine = _build_engine()
        ctx = self._ctx_with_mock_error_details()
        pattern_info = {"cause": "unknown", "confidence": 0.3}
        result = engine._create_pattern_based_cause("test", pattern_info, ctx)
        assert result.confidence == ConfidenceLevel.LOW

    def test_very_low_confidence(self):
        engine = _build_engine()
        ctx = self._ctx_with_mock_error_details()
        pattern_info = {"cause": "unknown", "confidence": 0.1}
        result = engine._create_pattern_based_cause("test", pattern_info, ctx)
        assert result.confidence == ConfidenceLevel.VERY_LOW

    def test_default_confidence(self):
        engine = _build_engine()
        ctx = self._ctx_with_mock_error_details()
        pattern_info = {"cause": "unknown"}
        result = engine._create_pattern_based_cause("test", pattern_info, ctx)
        assert result.confidence_score == 0.5


# ===================================================================
# _enhance_cause_with_context
# ===================================================================


class TestEnhanceCause:
    def test_syntactic_boost(self):
        engine = _build_engine()
        ctx = _build_context()
        cause = _make_root_cause(confidence_score=0.7)

        result = engine._enhance_cause_with_context(cause, ctx, "syntactic")
        assert result.confidence_score == pytest.approx(min(0.7 * 1.1, 1.0))
        assert "syntactic" in result.cause_id

    def test_pattern_boost(self):
        engine = _build_engine()
        ctx = _build_context()
        cause = _make_root_cause(confidence_score=0.7)

        result = engine._enhance_cause_with_context(cause, ctx, "pattern")
        assert result.confidence_score == pytest.approx(min(0.7 * 1.1, 1.0))

    def test_semantic_reduction(self):
        engine = _build_engine()
        ctx = _build_context()
        cause = _make_root_cause(confidence_score=0.7)

        result = engine._enhance_cause_with_context(cause, ctx, "semantic")
        assert result.confidence_score == pytest.approx(0.7 * 0.9)

    def test_integration_reduction(self):
        engine = _build_engine()
        ctx = _build_context()
        cause = _make_root_cause(confidence_score=0.7)

        result = engine._enhance_cause_with_context(cause, ctx, "integration")
        assert result.confidence_score == pytest.approx(0.7 * 0.9)

    def test_environmental_no_change(self):
        """Environmental type doesn't have a special multiplier."""
        engine = _build_engine()
        ctx = _build_context()
        cause = _make_root_cause(confidence_score=0.7)

        result = engine._enhance_cause_with_context(cause, ctx, "environmental")
        assert result.confidence_score == pytest.approx(0.7)

    def test_confidence_level_very_high(self):
        engine = _build_engine()
        ctx = _build_context()
        cause = _make_root_cause(confidence_score=0.95)

        result = engine._enhance_cause_with_context(cause, ctx, "syntactic")
        assert result.confidence == ConfidenceLevel.VERY_HIGH

    def test_confidence_level_high(self):
        engine = _build_engine()
        ctx = _build_context()
        cause = _make_root_cause(confidence_score=0.75)

        result = engine._enhance_cause_with_context(cause, ctx, "syntactic")
        assert result.confidence in (ConfidenceLevel.VERY_HIGH, ConfidenceLevel.HIGH)

    def test_confidence_level_medium(self):
        engine = _build_engine()
        ctx = _build_context()
        cause = _make_root_cause(confidence_score=0.6)

        result = engine._enhance_cause_with_context(cause, ctx, "environmental")
        assert result.confidence == ConfidenceLevel.MEDIUM

    def test_confidence_level_low(self):
        engine = _build_engine()
        ctx = _build_context()
        cause = _make_root_cause(confidence_score=0.3)

        result = engine._enhance_cause_with_context(cause, ctx, "semantic")
        # 0.3 * 0.9 = 0.27
        assert result.confidence == ConfidenceLevel.LOW

    def test_confidence_level_very_low(self):
        engine = _build_engine()
        ctx = _build_context()
        cause = _make_root_cause(confidence_score=0.1)

        result = engine._enhance_cause_with_context(cause, ctx, "semantic")
        # 0.1 * 0.9 = 0.09
        assert result.confidence == ConfidenceLevel.VERY_LOW

    def test_adds_evidence_when_empty(self):
        engine = _build_engine()
        ctx = _build_context()
        cause = _make_root_cause(confidence_score=0.5)
        cause.evidence = []

        result = engine._enhance_cause_with_context(cause, ctx, "syntactic")
        assert len(result.evidence) == 1
        assert "syntactic" in result.evidence[0]

    def test_appends_evidence_when_existing(self):
        engine = _build_engine()
        ctx = _build_context()
        cause = _make_root_cause(confidence_score=0.5)
        cause.evidence = ["existing evidence"]

        result = engine._enhance_cause_with_context(cause, ctx, "syntactic")
        assert len(result.evidence) == 2

    def test_confidence_capped_at_one(self):
        engine = _build_engine()
        ctx = _build_context()
        cause = _make_root_cause(confidence_score=0.99)

        result = engine._enhance_cause_with_context(cause, ctx, "syntactic")
        assert result.confidence_score <= 1.0


# ===================================================================
# _consolidate_causes
# ===================================================================


class TestConsolidateCauses:
    def test_empty_list(self):
        engine = _build_engine()
        assert engine._consolidate_causes([]) == []

    def test_no_duplicates(self):
        engine = _build_engine()
        c1 = _make_root_cause(
            confidence_score=0.6,
            description="First root cause identified during analysis process",
        )
        c2 = _make_root_cause(
            confidence_score=0.7,
            description="Second root cause identified during analysis process",
        )
        c2.cause_id = "cause-2"

        result = engine._consolidate_causes([c1, c2])
        assert len(result) == 2
        # Sorted by confidence
        assert result[0].confidence_score >= result[1].confidence_score

    def test_merges_duplicates(self):
        engine = _build_engine()
        c1 = _make_root_cause(
            confidence_score=0.5,
            description="Same root cause description repeated for testing consolidation",
        )
        c2 = _make_root_cause(
            confidence_score=0.7,
            description="Same root cause description repeated for testing consolidation",
        )
        c2.cause_id = "cause-2"
        c2.evidence = ["additional evidence"]

        result = engine._consolidate_causes([c1, c2])
        assert len(result) == 1
        assert result[0].confidence_score == 0.7

    def test_merged_confidence_level_updated(self):
        """Confidence level is updated based on merged score."""
        engine = _build_engine()
        c1 = _make_root_cause(
            confidence_score=0.5,
            description="Same root cause for testing confidence level updates here",
        )
        c2 = _make_root_cause(
            confidence_score=0.95,
            description="Same root cause for testing confidence level updates here",
        )
        c2.cause_id = "cause-2"
        c2.evidence = ["more evidence"]

        result = engine._consolidate_causes([c1, c2])
        assert result[0].confidence == ConfidenceLevel.VERY_HIGH

    def test_merged_low_confidence(self):
        engine = _build_engine()
        c1 = _make_root_cause(
            confidence_score=0.3,
            description="Low confidence cause for merging test purposes here",
        )
        c2 = _make_root_cause(
            confidence_score=0.35,
            description="Low confidence cause for merging test purposes here",
        )
        c2.cause_id = "cause-2"
        c2.evidence = ["more"]

        result = engine._consolidate_causes([c1, c2])
        assert result[0].confidence == ConfidenceLevel.LOW

    def test_merged_very_low_confidence(self):
        engine = _build_engine()
        c1 = _make_root_cause(
            confidence_score=0.1,
            description="Very low confidence cause for testing consolidation work",
        )
        c2 = _make_root_cause(
            confidence_score=0.15,
            description="Very low confidence cause for testing consolidation work",
        )
        c2.cause_id = "cause-2"
        c2.evidence = ["more"]

        result = engine._consolidate_causes([c1, c2])
        assert result[0].confidence == ConfidenceLevel.VERY_LOW


# ===================================================================
# _create_cache_key
# ===================================================================


class TestCacheKey:
    def test_cache_key_deterministic(self):
        """Cache key is deterministic for the same context."""
        engine = _build_engine()
        ctx = _build_context()

        key1 = engine._create_cache_key(ctx)
        key2 = engine._create_cache_key(ctx)
        assert key1 == key2
        assert "fail-1" in key1
        assert "assertion_error" in key1

    def test_cache_key_includes_error_message(self):
        """Cache key includes error message."""
        engine = _build_engine()
        ctx = _build_context(
            error_details=_make_error_details(error_message="something specific"),
        )
        key = engine._create_cache_key(ctx)
        assert "something specific" in key


# ===================================================================
# _create_fallback_analysis
# ===================================================================


class TestFallbackAnalysis:
    def test_returns_single_fallback_cause(self):
        """_create_fallback_analysis returns a single cause with VERY_LOW confidence."""
        engine = _build_engine()
        ctx = _build_context()
        results = engine._create_fallback_analysis(ctx)
        assert len(results) == 1
        assert results[0].category == "unknown"
        assert results[0].confidence_score == 0.2
        assert results[0].confidence == ConfidenceLevel.VERY_LOW
        assert "Fallback" in results[0].evidence[0]

    def test_fallback_includes_error_details(self):
        """Fallback analysis description includes error type and message."""
        engine = _build_engine()
        ctx = _build_context(
            error_details=_make_error_details(
                error_type=FailureType.ASSERTION_ERROR,
                error_message="expected True got False",
            ),
        )
        results = engine._create_fallback_analysis(ctx)
        assert "assertion_error" in results[0].description
        assert "expected True got False" in results[0].description


# ===================================================================
# AIAnalysisContext
# ===================================================================


class TestAIAnalysisContext:
    def test_default_fields(self):
        from services.ai_root_cause import AIAnalysisContext

        ctx = AIAnalysisContext(
            failure_data=_make_failure_data(),
            error_details=_make_error_details(),
        )
        assert ctx.code_snippets is None
        assert ctx.environment_info is None
        assert ctx.historical_failures is None
        assert ctx.dependency_graph is None

    def test_all_fields(self):
        from services.ai_root_cause import AIAnalysisContext

        ctx = AIAnalysisContext(
            failure_data=_make_failure_data(),
            error_details=_make_error_details(),
            code_snippets=["snippet"],
            environment_info={"os": "linux"},
            historical_failures=["prev-fail"],
            dependency_graph={"a": ["b"]},
        )
        assert ctx.code_snippets == ["snippet"]
        assert ctx.environment_info == {"os": "linux"}
        assert ctx.historical_failures == ["prev-fail"]
        assert ctx.dependency_graph == {"a": ["b"]}
