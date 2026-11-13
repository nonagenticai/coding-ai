"""Tests for AITestFailureAnalyzer covering all major code paths.

Focus: analyze_failure flow, error detail extraction, assertion detail extraction,
root cause analysis, fix suggestion generation, pattern detection, confidence
calculation, helper methods, and fallback paths.

NOTE: ErrorDetails and other models use Pydantic's use_enum_values=True,
which converts StrEnum values to plain strings. Source methods that call
.value on these fields will raise AttributeError, which is caught by
try/except blocks and causes graceful degradation (fallback results).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.ai_test_failure_models import (
    ConfidenceLevel,
    ErrorDetails,
    FailureAnalysisRequest,
    FailureType,
    FixSuggestion,
    RootCause,
    SeverityLevel,
    TestFailureContext,
    TestFailureData,
    TestFailurePattern,
    TestFramework,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_failure_data(
    raw_output: str = "AssertionError: expected True got False",
    failure_id: str = "fail-1",
) -> TestFailureData:
    return TestFailureData(
        failure_id=failure_id,
        raw_output=raw_output,
        context=TestFailureContext(
            test_name="test_example",
            framework=TestFramework.PYTEST,
        ),
    )


def _make_request(
    raw_output: str = "AssertionError: expected True got False",
    failure_id: str = "fail-1",
    request_id: str = "req-1",
    include_suggestions: bool = True,
    include_patterns: bool = True,
    max_root_causes: int = 5,
    max_suggestions: int = 10,
) -> FailureAnalysisRequest:
    return FailureAnalysisRequest(
        request_id=request_id,
        failure_data=_make_failure_data(raw_output=raw_output, failure_id=failure_id),
        include_suggestions=include_suggestions,
        include_patterns=include_patterns,
        max_root_causes=max_root_causes,
        max_suggestions=max_suggestions,
    )


def _make_root_cause(confidence_score: float = 0.3) -> RootCause:
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
        category="unknown",
        description="Test failed with assertion_error: expected True got False",
        severity=SeverityLevel.MEDIUM,
        confidence=level,
        confidence_score=confidence_score,
    )


def _build_analyzer():
    """Build AITestFailureAnalyzer with mocked LLM agent runners.

    The migration to coding-client BaseLLMAgent moved Agent construction into
    ``make_agent_runner``; we patch that to keep tests offline. Each
    ``self.*_agent`` ends up as a ``MagicMock`` with an ``AsyncMock`` ``run``
    attribute so individual tests can set ``run.return_value``.
    """

    def _runner_factory(*args: object, **kwargs: object) -> MagicMock:
        runner = MagicMock()
        runner.run = AsyncMock()
        return runner

    with patch(
        "services.ai_test_failure_analyzer.make_agent_runner",
        side_effect=_runner_factory,
    ):
        from services.ai_test_failure_analyzer import AITestFailureAnalyzer

        analyzer = AITestFailureAnalyzer()
    return analyzer


# ===================================================================
# analyze_failure
# ===================================================================


class TestAnalyzeFailure:
    @pytest.mark.asyncio
    async def test_analyze_failure_success_minimal(self):
        """Success path with minimal options (no AI agent calls)."""
        analyzer = _build_analyzer()
        request = _make_request(
            raw_output='File "test.py", line 10\nAssertionError: 1 != 2',
            include_suggestions=False,
            include_patterns=False,
        )

        # Mock root cause agent to avoid real AI call
        analyzer.root_cause_agent = MagicMock()
        analyzer.root_cause_agent.run = AsyncMock(return_value=MagicMock(output=[]))

        response = await analyzer.analyze_failure(request)
        assert response.success is True
        assert response.analysis_result is not None

    @pytest.mark.asyncio
    async def test_analyze_failure_with_root_causes(self):
        """Analyze failure with AI root cause analysis (mocked)."""
        analyzer = _build_analyzer()
        request = _make_request(include_suggestions=False, include_patterns=False)

        cause = _make_root_cause(0.3)
        analyzer.root_cause_agent = MagicMock()
        analyzer.root_cause_agent.run = AsyncMock(return_value=MagicMock(output=[cause]))

        response = await analyzer.analyze_failure(request)
        assert response.success is True

    @pytest.mark.asyncio
    async def test_analyze_failure_with_suggestions(self):
        """Analyze failure with AI fix suggestions (mocked)."""
        analyzer = _build_analyzer()
        request = _make_request(include_suggestions=True, include_patterns=False)

        # Root cause agent
        analyzer.root_cause_agent = MagicMock()
        analyzer.root_cause_agent.run = AsyncMock(return_value=MagicMock(output=[]))

        fix = FixSuggestion(
            suggestion_id="fix-1",
            title="Fix the assertion error",
            description="Change the assertion to match the actual value returned",
            fix_type="code_change",
            priority="medium",
            confidence=ConfidenceLevel.MEDIUM,
            confidence_score=0.5,
            estimated_effort="low",
        )
        analyzer.fix_suggestion_agent = MagicMock()
        analyzer.fix_suggestion_agent.run = AsyncMock(return_value=MagicMock(output=[fix]))

        response = await analyzer.analyze_failure(request)
        assert response.success is True

    @pytest.mark.asyncio
    async def test_analyze_failure_with_patterns(self):
        """Analyze failure with AI pattern detection (mocked).

        NOTE: _detect_patterns calls _create_failure_signature which uses
        .error_type.value on a string, raising AttributeError. We mock
        _detect_patterns directly to test the analyze_failure flow.
        """
        analyzer = _build_analyzer()
        request = _make_request(include_suggestions=False, include_patterns=True)

        # Root cause agent
        analyzer.root_cause_agent = MagicMock()
        analyzer.root_cause_agent.run = AsyncMock(return_value=MagicMock(output=[]))

        pattern = TestFailurePattern(
            pattern_id="pat-1",
            pattern_name="Assertion Error Pattern",
            pattern_signature="assertion_error|expected|pytest_output",
            pattern_type="error_message",
            confidence=ConfidenceLevel.MEDIUM,
            confidence_score=0.6,
        )

        # Mock _detect_patterns to bypass the .value bug in _create_failure_signature
        with patch.object(analyzer, "_detect_patterns", return_value=[pattern]):
            response = await analyzer.analyze_failure(request)
        assert response.success is True

    @pytest.mark.asyncio
    async def test_analyze_failure_exception(self):
        """Exception during analysis returns error response."""
        analyzer = _build_analyzer()
        request = _make_request()

        with patch.object(analyzer, "_extract_error_details", side_effect=RuntimeError("boom")):
            response = await analyzer.analyze_failure(request)
        assert response.success is False
        assert "boom" in response.error_message


# ===================================================================
# _extract_error_details
# ===================================================================


class TestExtractErrorDetails:
    @pytest.mark.asyncio
    async def test_basic_extraction(self):
        fd = _make_failure_data(raw_output='File "test.py", line 10\nAssertionError: 1 != 2')
        analyzer = _build_analyzer()
        details = await analyzer._extract_error_details(fd)
        assert details.error_type == FailureType.ASSERTION_ERROR
        assert details.file_path == "test.py"
        assert details.line_number == 10

    @pytest.mark.asyncio
    async def test_traceback_extraction(self):
        fd = _make_failure_data(
            raw_output='Traceback (most recent call last):\n  File "test.py", line 5\n    x = 1/0\nZeroDivisionError: division by zero'
        )
        analyzer = _build_analyzer()
        details = await analyzer._extract_error_details(fd)
        assert details.traceback is not None

    @pytest.mark.asyncio
    async def test_no_traceback(self):
        fd = _make_failure_data(raw_output="FAILED test_example - assertion error")
        analyzer = _build_analyzer()
        details = await analyzer._extract_error_details(fd)
        assert details.stack_trace is None


# ===================================================================
# _extract_assertion_details
# ===================================================================


class TestExtractAssertionDetails:
    @pytest.mark.asyncio
    async def test_no_assertion_in_output(self):
        fd = _make_failure_data(raw_output="RuntimeError: something bad")
        analyzer = _build_analyzer()
        result = await analyzer._extract_assertion_details(fd)
        assert result is None

    @pytest.mark.asyncio
    async def test_assertion_with_values(self):
        fd = _make_failure_data(raw_output="AssertionError: 42 != 43\nassertEqual used")
        analyzer = _build_analyzer()
        result = await analyzer._extract_assertion_details(fd)
        assert result is not None
        assert result.assertion_type == "assertEqual"

    @pytest.mark.asyncio
    async def test_assertion_unknown_type(self):
        fd = _make_failure_data(raw_output="assert x == y")
        analyzer = _build_analyzer()
        result = await analyzer._extract_assertion_details(fd)
        assert result is not None
        assert result.assertion_type == "unknown"


# ===================================================================
# _classify_error_type
# ===================================================================


class TestClassifyErrorType:
    def test_assertion_error(self):
        analyzer = _build_analyzer()
        assert analyzer._classify_error_type("AssertionError: x") == FailureType.ASSERTION_ERROR

    def test_syntax_error(self):
        analyzer = _build_analyzer()
        assert analyzer._classify_error_type("SyntaxError: invalid") == FailureType.SYNTAX_ERROR

    def test_import_error(self):
        analyzer = _build_analyzer()
        assert analyzer._classify_error_type("ImportError: no module") == FailureType.IMPORT_ERROR

    def test_module_not_found(self):
        analyzer = _build_analyzer()
        assert analyzer._classify_error_type("ModuleNotFoundError: foo") == FailureType.IMPORT_ERROR

    def test_timeout_error(self):
        analyzer = _build_analyzer()
        assert analyzer._classify_error_type("timeout exceeded") == FailureType.TIMEOUT_ERROR

    def test_timed_out(self):
        analyzer = _build_analyzer()
        assert analyzer._classify_error_type("operation timed out") == FailureType.TIMEOUT_ERROR

    def test_memory_error(self):
        analyzer = _build_analyzer()
        assert (
            analyzer._classify_error_type("MemoryError: out of memory") == FailureType.MEMORY_ERROR
        )

    def test_network_error(self):
        analyzer = _build_analyzer()
        assert (
            analyzer._classify_error_type("ConnectionError: network down")
            == FailureType.NETWORK_ERROR
        )

    def test_dependency_error(self):
        analyzer = _build_analyzer()
        assert (
            analyzer._classify_error_type("dependency not found package missing")
            == FailureType.DEPENDENCY_ERROR
        )

    def test_configuration_error(self):
        analyzer = _build_analyzer()
        assert (
            analyzer._classify_error_type("configuration invalid")
            == FailureType.CONFIGURATION_ERROR
        )

    def test_generic_error(self):
        analyzer = _build_analyzer()
        assert (
            analyzer._classify_error_type("ValueError: error occurred") == FailureType.RUNTIME_ERROR
        )

    def test_unknown_error(self):
        analyzer = _build_analyzer()
        assert analyzer._classify_error_type("all is well") == FailureType.UNKNOWN_ERROR


# ===================================================================
# _extract_error_message
# ===================================================================


class TestExtractErrorMessage:
    def test_pattern_with_error_type(self):
        analyzer = _build_analyzer()
        msg = analyzer._extract_error_message("ValueError: bad input")
        assert "bad input" in msg

    def test_failed_pattern(self):
        analyzer = _build_analyzer()
        msg = analyzer._extract_error_message("FAILED test_foo - some reason")
        assert len(msg) > 0

    def test_fallback_line_with_error(self):
        analyzer = _build_analyzer()
        msg = analyzer._extract_error_message("line1\nthis has error word\nline3")
        assert len(msg) > 0

    def test_unknown_message(self):
        analyzer = _build_analyzer()
        msg = analyzer._extract_error_message("xyz abc 123 def")
        assert msg == "Unknown error message"


# ===================================================================
# _extract_traceback and _parse_stack_trace
# ===================================================================


class TestTracebackAndStack:
    def test_extract_traceback(self):
        analyzer = _build_analyzer()
        output = 'Traceback (most recent call last):\n  File "test.py", line 5\nValueError: x'
        result = analyzer._extract_traceback(output)
        assert result is not None

    def test_extract_traceback_none(self):
        analyzer = _build_analyzer()
        result = analyzer._extract_traceback("no traceback here")
        assert result is None

    def test_parse_stack_trace_empty(self):
        analyzer = _build_analyzer()
        assert analyzer._parse_stack_trace("") == []

    def test_parse_stack_trace(self):
        analyzer = _build_analyzer()
        entries = analyzer._parse_stack_trace('  File "a.py", line 1\n  x = 1')
        assert len(entries) == 2


# ===================================================================
# _extract_location_info
# ===================================================================


class TestExtractLocationInfo:
    def test_file_line_format(self):
        analyzer = _build_analyzer()
        fp, ln, col = analyzer._extract_location_info('File "test.py", line 42')
        assert fp == "test.py"
        assert ln == 42

    def test_file_line_column_format(self):
        analyzer = _build_analyzer()
        fp, ln, col = analyzer._extract_location_info("test.py:42:5:")
        assert fp == "test.py"
        assert ln == 42
        assert col == 5

    def test_no_location(self):
        analyzer = _build_analyzer()
        fp, ln, col = analyzer._extract_location_info("no location at all")
        assert fp is None
        assert ln is None
        assert col is None


# ===================================================================
# Assertion detail helpers
# ===================================================================


class TestAssertionHelpers:
    def test_classify_assertion_type_assert_equal(self):
        analyzer = _build_analyzer()
        assert analyzer._classify_assertion_type("assertEqual(x, y)") == "assertEqual"

    def test_classify_assertion_type_assert_true(self):
        analyzer = _build_analyzer()
        assert analyzer._classify_assertion_type("assertTrue(x)") == "assertTrue"

    def test_classify_assertion_type_unknown(self):
        analyzer = _build_analyzer()
        assert analyzer._classify_assertion_type("some random output") == "unknown"

    def test_extract_assertion_values_expected_actual(self):
        analyzer = _build_analyzer()
        exp, act = analyzer._extract_assertion_values("Expected: 42\nActual: 43")
        assert exp == "42"
        assert act == "43"

    def test_extract_assertion_values_none(self):
        analyzer = _build_analyzer()
        exp, act = analyzer._extract_assertion_values("no values here")
        assert exp is None
        assert act is None

    def test_extract_comparison_operator_ne(self):
        analyzer = _build_analyzer()
        assert analyzer._extract_comparison_operator("x != y") == "!="

    def test_extract_comparison_operator_eq(self):
        analyzer = _build_analyzer()
        assert analyzer._extract_comparison_operator("x == y") == "=="

    def test_extract_comparison_operator_none(self):
        analyzer = _build_analyzer()
        # "abc" contains no operators
        assert analyzer._extract_comparison_operator("abc") is None

    def test_extract_assertion_message(self):
        analyzer = _build_analyzer()
        msg = analyzer._extract_assertion_message("AssertionError: custom message")
        assert msg is not None

    def test_extract_assertion_message_filters_comparison(self):
        analyzer = _build_analyzer()
        msg = analyzer._extract_assertion_message("AssertionError: 1 != 2")
        # Should be filtered because it contains !=
        assert msg is None

    def test_extract_assertion_message_none(self):
        analyzer = _build_analyzer()
        msg = analyzer._extract_assertion_message("xyz abc 123")
        assert msg is None


# ===================================================================
# _calculate_value_difference
# ===================================================================


class TestCalculateValueDifference:
    def test_numerical_difference(self):
        analyzer = _build_analyzer()
        result = analyzer._calculate_value_difference("10", "15")
        assert result is not None
        assert "5" in result

    def test_string_difference(self):
        analyzer = _build_analyzer()
        result = analyzer._calculate_value_difference("hello", "world")
        assert result is not None

    def test_length_difference(self):
        analyzer = _build_analyzer()
        result = analyzer._calculate_value_difference("ab", "abc")
        assert result is not None

    def test_no_difference(self):
        analyzer = _build_analyzer()
        result = analyzer._calculate_value_difference("same", "same")
        assert result is None


# ===================================================================
# Context preparation
# ===================================================================


class TestContextPreparation:
    def test_prepare_analysis_context(self):
        analyzer = _build_analyzer()
        fd = _make_failure_data()
        ed = ErrorDetails(
            error_type=FailureType.ASSERTION_ERROR,
            error_message="expected True got False",
            file_path="/src/app.py",
            line_number=42,
            traceback="File app.py line 42",
        )

        result = analyzer._prepare_analysis_context(fd, ed)
        assert "test_example" in result
        assert "/src/app.py" in result
        assert "42" in result
        assert "Traceback:" in result

    def test_prepare_fix_context(self):
        analyzer = _build_analyzer()
        fd = _make_failure_data()
        ed = ErrorDetails(
            error_type=FailureType.ASSERTION_ERROR,
            error_message="expected True got False",
        )
        causes = [_make_root_cause(0.3)]

        result = analyzer._prepare_fix_context(fd, ed, causes)
        assert "Root Causes:" in result

    def test_prepare_pattern_context(self):
        analyzer = _build_analyzer()
        fd = _make_failure_data()
        ed = ErrorDetails(
            error_type=FailureType.ASSERTION_ERROR,
            error_message="expected True got False",
            stack_trace=["frame1", "frame2"],
        )

        result = analyzer._prepare_pattern_context(fd, ed)
        assert "Stack Trace:" in result


# ===================================================================
# Failure signature and confidence
#
# NOTE: _create_failure_signature calls .value on Pydantic-stored enum
# strings, which will raise AttributeError due to use_enum_values=True.
# ===================================================================


class TestSignatureAndConfidence:
    def test_create_failure_signature_returns_pipe_separated(self):
        """_create_failure_signature returns a pipe-separated string of
        error_type, truncated message, framework, and filename."""
        analyzer = _build_analyzer()
        fd = _make_failure_data()
        ed = ErrorDetails(
            error_type=FailureType.ASSERTION_ERROR,
            error_message="expected True got False",
            file_path="/src/app.py",
        )

        sig = analyzer._create_failure_signature(fd, ed)
        parts = sig.split("|")
        assert len(parts) >= 3
        assert "assertion_error" in parts[0].lower() or "ASSERTION_ERROR" in parts[0]
        assert "expected True got False" in parts[1]
        assert "app.py" in parts[-1]

    def test_confidence_very_low(self):
        analyzer = _build_analyzer()
        level, score = analyzer._calculate_overall_confidence([], [], [])
        assert level == ConfidenceLevel.VERY_LOW
        assert score == 0.0

    def test_confidence_medium(self):
        analyzer = _build_analyzer()
        cause = _make_root_cause(0.6)
        level, score = analyzer._calculate_overall_confidence([cause], [], [])
        assert level in (ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH)

    def test_confidence_very_high(self):
        analyzer = _build_analyzer()
        cause = _make_root_cause(0.95)
        level, score = analyzer._calculate_overall_confidence([cause], [], [])
        assert level in (ConfidenceLevel.VERY_HIGH, ConfidenceLevel.HIGH)

    def test_confidence_with_all_inputs(self):
        analyzer = _build_analyzer()
        cause = _make_root_cause(0.3)
        fix = FixSuggestion(
            suggestion_id="fix-1",
            title="Fix the assertion error",
            description="Change the assertion to match actual value returned",
            fix_type="code_change",
            priority="medium",
            confidence=ConfidenceLevel.LOW,
            confidence_score=0.3,
            estimated_effort="low",
        )
        pattern = TestFailurePattern(
            pattern_id="pat-1",
            pattern_name="Assertion Error Pattern",
            pattern_signature="assertion_error|expected|pytest_pattern",
            pattern_type="error_message",
            confidence=ConfidenceLevel.LOW,
            confidence_score=0.3,
        )
        level, score = analyzer._calculate_overall_confidence([cause], [fix], [pattern])
        assert score > 0.0


# ===================================================================
# AI agent calls: outage RAISES (no fabricated fallback)
#
# On an LLM-plane outage each stage now raises LLMAgentUnavailableError
# instead of silently returning fabricated confidence-0.3 content and
# reporting a clean success. analyze_failure() records this as degraded.
# ===================================================================


class TestAgentCallsRaiseOnOutage:
    @pytest.mark.asyncio
    async def test_root_cause_agent_error_raises(self):
        """Root cause agent error raises LLMAgentUnavailableError (no fabrication)."""
        from services.ai_test_failure_analyzer import LLMAgentUnavailableError

        analyzer = _build_analyzer()
        fd = _make_failure_data()
        ed = ErrorDetails(
            error_type=FailureType.ASSERTION_ERROR,
            error_message="expected True got False",
        )

        analyzer.root_cause_agent = MagicMock()
        analyzer.root_cause_agent.run = AsyncMock(side_effect=RuntimeError("ai fail"))

        with pytest.raises(LLMAgentUnavailableError):
            await analyzer._analyze_root_causes(fd, ed, 5, 0.1)

    @pytest.mark.asyncio
    async def test_fix_suggestion_agent_error_raises(self):
        """Fix suggestion agent error raises LLMAgentUnavailableError."""
        from services.ai_test_failure_analyzer import LLMAgentUnavailableError

        analyzer = _build_analyzer()
        fd = _make_failure_data()
        ed = ErrorDetails(
            error_type=FailureType.ASSERTION_ERROR,
            error_message="expected True got False",
        )

        analyzer.fix_suggestion_agent = MagicMock()
        analyzer.fix_suggestion_agent.run = AsyncMock(side_effect=RuntimeError("ai fail"))

        with pytest.raises(LLMAgentUnavailableError):
            await analyzer._generate_fix_suggestions(fd, ed, [], 5, 0.1)

    @pytest.mark.asyncio
    async def test_pattern_agent_error_raises(self):
        """Pattern agent error raises LLMAgentUnavailableError."""
        from services.ai_test_failure_analyzer import LLMAgentUnavailableError

        analyzer = _build_analyzer()
        fd = _make_failure_data()
        ed = ErrorDetails(
            error_type=FailureType.ASSERTION_ERROR,
            error_message="expected True got False",
        )

        analyzer.pattern_agent = MagicMock()
        analyzer.pattern_agent.run = AsyncMock(side_effect=RuntimeError("ai fail"))

        with pytest.raises(LLMAgentUnavailableError):
            await analyzer._detect_patterns(fd, ed, 0.1)


# ===================================================================
# analyze_failure: outage -> degraded=True; genuine-empty -> degraded=False
# ===================================================================


class TestAnalyzeFailureDegradedSignal:
    @pytest.mark.asyncio
    async def test_total_outage_marks_response_degraded(self):
        """Every agent unreachable -> success=True but degraded=True, empty results."""
        analyzer = _build_analyzer()
        request = _make_request(include_suggestions=True, include_patterns=True)

        outage = AsyncMock(side_effect=RuntimeError("LLM plane unreachable"))
        for name in ("root_cause_agent", "fix_suggestion_agent", "pattern_agent"):
            agent = MagicMock()
            agent.run = outage
            setattr(analyzer, name, agent)

        response = await analyzer.analyze_failure(request)

        assert response.success is True
        assert response.degraded is True
        # No fabricated content — the degraded stages are empty.
        assert response.analysis_result.root_causes == []
        assert response.analysis_result.fix_suggestions == []
        assert response.analysis_result.detected_patterns == []
        # Flag also carried in metadata so it survives the service layer.
        assert response.analysis_result.additional_metadata == {"degraded": True}

    @pytest.mark.asyncio
    async def test_healthy_but_empty_is_not_degraded(self):
        """Agents run and find nothing -> success=True, degraded=False, empty."""
        analyzer = _build_analyzer()
        request = _make_request(include_suggestions=True, include_patterns=False)

        empty = MagicMock(output=[])
        analyzer.root_cause_agent = MagicMock()
        analyzer.root_cause_agent.run = AsyncMock(return_value=empty)
        analyzer.fix_suggestion_agent = MagicMock()
        analyzer.fix_suggestion_agent.run = AsyncMock(return_value=empty)

        response = await analyzer.analyze_failure(request)

        assert response.success is True
        assert response.degraded is False
        assert response.analysis_result.root_causes == []
        assert response.analysis_result.additional_metadata is None

    @pytest.mark.asyncio
    async def test_partial_outage_is_degraded(self):
        """Root cause succeeds but fix-suggestion agent down -> degraded=True."""
        analyzer = _build_analyzer()
        request = _make_request(include_suggestions=True, include_patterns=False)

        cause = _make_root_cause(0.3)
        analyzer.root_cause_agent = MagicMock()
        analyzer.root_cause_agent.run = AsyncMock(return_value=MagicMock(output=[cause]))
        analyzer.fix_suggestion_agent = MagicMock()
        analyzer.fix_suggestion_agent.run = AsyncMock(
            side_effect=RuntimeError("LLM plane unreachable")
        )

        response = await analyzer.analyze_failure(request)

        assert response.success is True
        assert response.degraded is True
        # Real root cause preserved; the failed stage is empty (not fabricated).
        assert len(response.analysis_result.root_causes) >= 1
        assert response.analysis_result.fix_suggestions == []


# ===================================================================
# Root cause agent - confidence filtering
# ===================================================================


class TestRootCauseFiltering:
    @pytest.mark.asyncio
    async def test_filters_low_confidence(self):
        analyzer = _build_analyzer()
        fd = _make_failure_data()
        ed = ErrorDetails(
            error_type=FailureType.ASSERTION_ERROR,
            error_message="expected True got False",
        )

        low = _make_root_cause(0.05)
        high = _make_root_cause(0.3)
        high.cause_id = "cause-2"

        analyzer.root_cause_agent = MagicMock()
        analyzer.root_cause_agent.run = AsyncMock(return_value=MagicMock(output=[low, high]))

        results = await analyzer._analyze_root_causes(fd, ed, 5, 0.2)
        assert all(c.confidence_score >= 0.2 for c in results)

    @pytest.mark.asyncio
    async def test_limits_max_causes(self):
        analyzer = _build_analyzer()
        fd = _make_failure_data()
        ed = ErrorDetails(
            error_type=FailureType.ASSERTION_ERROR,
            error_message="expected True got False",
        )

        causes = [_make_root_cause(0.3) for _ in range(10)]
        for i, c in enumerate(causes):
            c.cause_id = f"cause-{i}"

        analyzer.root_cause_agent = MagicMock()
        analyzer.root_cause_agent.run = AsyncMock(return_value=MagicMock(output=causes))

        results = await analyzer._analyze_root_causes(fd, ed, 2, 0.1)
        assert len(results) <= 2
