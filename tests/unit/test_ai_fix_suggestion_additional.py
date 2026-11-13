"""Additional tests for AIFixSuggestionEngine covering remaining uncovered lines.

Focus: _prepare_*_input methods (success paths with mocked context),
_generate_immediate/preventive/best_practice_fixes success paths,
_generate_pattern_based_fixes with circular_import match,
and _create_cache_key success.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.ai_test_failure_models import (
    ConfidenceLevel,
    ErrorDetails,
    FailureType,
    FixSuggestion,
    RootCause,
    SeverityLevel,
    TestFailureContext,
    TestFailureData,
    TestFramework,
)


def _make_failure_data(
    raw_output: str = "AssertionError: expected True got False",
    test_name: str = "test_example",
    framework: TestFramework = TestFramework.PYTEST,
) -> TestFailureData:
    return TestFailureData(
        failure_id="fail-1",
        raw_output=raw_output,
        context=TestFailureContext(
            test_name=test_name,
            framework=framework,
        ),
    )


def _make_error_details(
    error_message: str = "expected True got False",
    error_type: FailureType = FailureType.ASSERTION_ERROR,
    file_path: str | None = None,
    line_number: int | None = None,
) -> ErrorDetails:
    return ErrorDetails(
        error_type=error_type,
        error_message=error_message,
        file_path=file_path,
        line_number=line_number,
    )


def _make_root_cause(confidence_score: float = 0.5) -> RootCause:
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
        evidence=["Test evidence"],
    )


def _make_fix_suggestion(confidence_score: float = 0.5) -> FixSuggestion:
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
    return FixSuggestion(
        suggestion_id="fix-1",
        title="Fix the issue found during analysis",
        description="This is a detailed description of the fix suggestion for testing",
        fix_type="code_change",
        priority="medium",
        confidence=level,
        confidence_score=confidence_score,
        estimated_effort="medium",
    )


def _build_engine():
    with patch("services.ai_fix_suggestion.make_agent_runner"):
        from services.ai_fix_suggestion import AIFixSuggestionEngine

        engine = AIFixSuggestionEngine()
    return engine


def _build_context(**kwargs):
    from services.ai_fix_suggestion import FixContext

    defaults = dict(
        failure_data=_make_failure_data(),
        error_details=_make_error_details(),
        root_causes=[],
    )
    defaults.update(kwargs)
    return FixContext(**defaults)


# ===================================================================
# _prepare_*_input
#
# These build their context from FULLY VALIDATED models, exactly as production
# does. Do NOT substitute MagicMock for failure_data.context / error_details:
# a mock grows a ``.value`` attribute on demand, so it hides source code that
# assumes the enum fields are enum instances. They are not -- the models set
# use_enum_values=True, so they arrive as plain strings and ``.value`` raises
# AttributeError at runtime while a mock-based test stays green.
# ===================================================================


class TestPrepareImmediateFixInput:
    def test_with_all_fields(self):
        engine = _build_engine()
        ctx = _build_context(
            failure_data=_make_failure_data(test_name="test_foo"),
            error_details=_make_error_details(
                error_type=FailureType.ASSERTION_ERROR,
                error_message="expected 1 got 2",
                file_path="/src/app.py",
                line_number=10,
            ),
            root_causes=[_make_root_cause(confidence_score=0.7)],
            code_context={"/src/app.py": "def foo(): pass"},
        )

        result = engine._prepare_immediate_fix_input(ctx)
        assert "test_foo" in result
        assert "assertion_error" in result
        assert "/src/app.py" in result
        assert "10" in result
        assert "def foo(): pass" in result

    def test_minimal_fields(self):
        engine = _build_engine()
        ctx = _build_context(
            failure_data=_make_failure_data(test_name="test_bar"),
            error_details=_make_error_details(
                error_type=FailureType.RUNTIME_ERROR,
                error_message="boom",
            ),
        )

        result = engine._prepare_immediate_fix_input(ctx)
        assert "test_bar" in result
        assert "runtime_error" in result


class TestPreparePreventiveFixInput:
    def test_with_all_fields(self):
        engine = _build_engine()
        ctx = _build_context(
            failure_data=_make_failure_data(test_name="test_foo", framework=TestFramework.PYTEST),
            error_details=_make_error_details(
                error_type=FailureType.IMPORT_ERROR,
                error_message="no module",
            ),
            root_causes=[_make_root_cause()],
            historical_fixes=[{"description": "Fixed import"}],
            project_structure={"src": ["app.py"]},
        )

        result = engine._prepare_preventive_fix_input(ctx)
        assert "test_foo" in result
        assert "import_error" in result
        assert "pytest" in result
        assert "Fixed import" in result

    def test_minimal_fields(self):
        engine = _build_engine()
        ctx = _build_context(
            failure_data=_make_failure_data(test_name="test_min", framework=TestFramework.JEST),
            error_details=_make_error_details(
                error_type=FailureType.RUNTIME_ERROR,
                error_message="x",
            ),
        )

        result = engine._prepare_preventive_fix_input(ctx)
        assert "test_min" in result
        assert "jest" in result


class TestPrepareBestPracticeInput:
    def test_with_all_fields(self):
        engine = _build_engine()
        ctx = _build_context(
            failure_data=_make_failure_data(test_name="test_bp", framework=TestFramework.PYTEST),
            error_details=_make_error_details(
                error_type=FailureType.ASSERTION_ERROR,
                error_message="mismatch",
            ),
            code_context={"/tests/test_app.py": "def test_add(): assert 1==2"},
            team_preferences={"style": "ruff-format"},
        )

        result = engine._prepare_best_practice_input(ctx)
        assert "test_bp" in result
        assert "pytest" in result
        assert "def test_add" in result
        assert "ruff-format" in result

    def test_minimal_fields(self):
        engine = _build_engine()
        ctx = _build_context(
            failure_data=_make_failure_data(test_name="test_min", framework=TestFramework.UNITTEST),
            error_details=_make_error_details(
                error_type=FailureType.UNKNOWN_ERROR,
                error_message="x",
            ),
        )

        result = engine._prepare_best_practice_input(ctx)
        assert "unittest" in result
        assert "unknown_error" in result


# ===================================================================
# _generate_immediate/preventive/best_practice_fixes success paths
# ===================================================================


class TestSubGeneratorSuccessPaths:
    @pytest.mark.asyncio
    async def test_immediate_fixes_success(self):
        engine = _build_engine()
        ctx = _build_context()
        fix = _make_fix_suggestion(confidence_score=0.6)
        mock_result = MagicMock()
        mock_result.output = [fix]

        with patch.object(engine, "_prepare_immediate_fix_input", return_value="input"):
            engine.immediate_fix_agent = AsyncMock()
            engine.immediate_fix_agent.run = AsyncMock(return_value=mock_result)
            results = await engine._generate_immediate_fixes(ctx)

        assert len(results) == 1
        assert "immediate" in results[0].suggestion_id

    @pytest.mark.asyncio
    async def test_preventive_fixes_success(self):
        engine = _build_engine()
        ctx = _build_context()
        fix = _make_fix_suggestion(confidence_score=0.5)
        mock_result = MagicMock()
        mock_result.output = [fix]

        with patch.object(engine, "_prepare_preventive_fix_input", return_value="input"):
            engine.preventive_fix_agent = AsyncMock()
            engine.preventive_fix_agent.run = AsyncMock(return_value=mock_result)
            results = await engine._generate_preventive_fixes(ctx)

        assert len(results) == 1
        assert "preventive" in results[0].suggestion_id

    @pytest.mark.asyncio
    async def test_best_practice_fixes_success(self):
        engine = _build_engine()
        ctx = _build_context()
        fix = _make_fix_suggestion(confidence_score=0.5)
        mock_result = MagicMock()
        mock_result.output = [fix]

        with patch.object(engine, "_prepare_best_practice_input", return_value="input"):
            engine.best_practice_agent = AsyncMock()
            engine.best_practice_agent.run = AsyncMock(return_value=mock_result)
            results = await engine._generate_best_practice_fixes(ctx)

        assert len(results) == 1
        assert "best_practice" in results[0].suggestion_id


# ===================================================================
# _generate_pattern_based_fixes - circular_import match
# ===================================================================


class TestPatternBasedFixesAdditional:
    @pytest.mark.asyncio
    async def test_circular_import_match(self):
        engine = _build_engine()
        ctx = _build_context(
            error_details=_make_error_details(
                error_message="circular import from partially initialized module"
            )
        )
        results = await engine._generate_pattern_based_fixes(ctx)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_none_expected_type_match(self):
        """Tests error_message matching 'none' and 'expected'."""
        engine = _build_engine()
        ctx = _build_context(
            error_details=_make_error_details(
                error_message="expected value but got none instead of actual"
            )
        )
        results = await engine._generate_pattern_based_fixes(ctx)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_type_mismatch_match(self):
        engine = _build_engine()
        ctx = _build_context(
            error_details=_make_error_details(
                error_message="type expected is int class but actual is str class"
            )
        )
        results = await engine._generate_pattern_based_fixes(ctx)
        assert len(results) >= 1


# ===================================================================
# _consolidate_suggestions - merge with existing code_examples
# ===================================================================


class TestConsolidateAdditional:
    def test_merge_code_examples_into_existing(self):
        engine = _build_engine()
        s1 = _make_fix_suggestion(confidence_score=0.5)
        s1.title = "Same title for consolidation testing"
        s1.code_examples = ["example1()"]

        s2 = _make_fix_suggestion(confidence_score=0.6)
        s2.suggestion_id = "fix-2"
        s2.title = "Same title for consolidation testing"
        s2.description = "A completely different description for the alternative approach here"
        s2.code_examples = ["example2()"]

        result = engine._consolidate_suggestions([s1, s2])
        assert len(result) == 1
        assert "example1()" in result[0].code_examples
        assert "example2()" in result[0].code_examples

    def test_merge_code_examples_none_to_existing(self):
        engine = _build_engine()
        s1 = _make_fix_suggestion(confidence_score=0.5)
        s1.title = "Same title for null examples test"
        s1.code_examples = None

        s2 = _make_fix_suggestion(confidence_score=0.6)
        s2.suggestion_id = "fix-2"
        s2.title = "Same title for null examples test"
        s2.description = "A completely different description for the alternative approach here"
        s2.code_examples = ["new_example()"]

        result = engine._consolidate_suggestions([s1, s2])
        assert len(result) == 1
        assert "new_example()" in result[0].code_examples
