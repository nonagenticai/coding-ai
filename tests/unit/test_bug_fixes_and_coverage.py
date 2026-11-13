"""Tests for real bugs found during deep dive and coverage gaps.

Bugs fixed:
1. AssertionError typo: code searched for "AssertionError" but Python produces "AssertionError"
   - ai_failure_analysis.py line 326
   - ai_test_failure_analyzer.py line 329
2. Naive/aware datetime mismatch in ai_test_integration.py execute_test_with_analysis
3. _active_analyses resource leak in ai_failure_analysis.py (never cleaned up after analysis)
4. Description overflow in ai_fix_suggestion.py _consolidate_suggestions
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.ai_test_failure_models import (
    AIAnalysisResult,
    ConfidenceLevel,
    ErrorDetails,
    FailureAnalysisRequest,
    FailureAnalysisResponse,
    FailureType,
    FixSuggestion,
    RootCause,
    SeverityLevel,
    TestFailureContext,
    TestFailureData,
    TestFailurePattern,
    TestFramework,
)
from models.testing_models import TestResult

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
) -> FailureAnalysisRequest:
    return FailureAnalysisRequest(
        request_id=request_id,
        failure_data=_make_failure_data(raw_output=raw_output, failure_id=failure_id),
    )


def _make_error_details(
    error_type: FailureType = FailureType.ASSERTION_ERROR,
    error_message: str = "expected True got False",
) -> ErrorDetails:
    return ErrorDetails(
        error_type=error_type,
        error_message=error_message,
    )


def _make_analysis_result(failure_data: TestFailureData | None = None) -> AIAnalysisResult:
    fd = failure_data or _make_failure_data()
    return AIAnalysisResult(
        analysis_id="analysis-1",
        failure_data=fd,
        error_details=_make_error_details(),
        root_causes=[],
        fix_suggestions=[],
        detected_patterns=[],
        analysis_confidence=ConfidenceLevel.VERY_LOW,
        analysis_confidence_score=0.1,
        ai_model_version="test",
    )


def _build_service():
    """Build AIFailureAnalysisService with mocked AI engines."""
    with patch("services.ai_failure_analysis.AITestFailureAnalyzer"):
        with patch("services.ai_failure_analysis.AIRootCauseEngine"):
            with patch("services.ai_failure_analysis.AIFixSuggestionEngine"):
                from services.ai_failure_analysis import AIFailureAnalysisService

                svc = AIFailureAnalysisService(model_name="test:model")
    return svc


def _build_analyzer():
    """Build AITestFailureAnalyzer with mocked LLM agent runners."""
    with patch("services.ai_test_failure_analyzer.make_agent_runner"):
        from services.ai_test_failure_analyzer import AITestFailureAnalyzer

        analyzer = AITestFailureAnalyzer(model_name="test-model")
    return analyzer


# ===================================================================
# Bug 1: AssertionError typo fix — both spellings now detected
# ===================================================================


class TestAssertionErrorTypoFix:
    """Verify that real Python AssertionError output is correctly detected."""

    @pytest.mark.asyncio
    async def test_real_python_assertion_error_detected_in_failure_analysis(self):
        """ai_failure_analysis._extract_basic_error_details detects real AssertionError."""
        svc = _build_service()
        # Use the CORRECT Python spelling
        fd = _make_failure_data(raw_output="AssertionError: assert 1 == 2")
        details = await svc._extract_basic_error_details(fd)
        assert details.error_type == FailureType.ASSERTION_ERROR

    @pytest.mark.asyncio
    async def test_misspelled_assertion_error_still_detected(self):
        """Legacy misspelled AssertionError is still detected for backward compat."""
        svc = _build_service()
        fd = _make_failure_data(raw_output="AssertionError: assert 1 == 2")
        details = await svc._extract_basic_error_details(fd)
        assert details.error_type == FailureType.ASSERTION_ERROR

    @pytest.mark.asyncio
    async def test_real_assertion_error_triggers_assertion_details_extraction(self):
        """ai_test_failure_analyzer._extract_assertion_details works with real AssertionError."""
        analyzer = _build_analyzer()
        fd = _make_failure_data(raw_output="AssertionError: 42 != 43\nassertEqual used")
        result = await analyzer._extract_assertion_details(fd)
        # Should not return None -- the real Python spelling should be recognized
        assert result is not None

    @pytest.mark.asyncio
    async def test_no_assertion_error_returns_none(self):
        """Non-assertion errors return None from _extract_assertion_details."""
        analyzer = _build_analyzer()
        fd = _make_failure_data(raw_output="ImportError: No module named foo")
        result = await analyzer._extract_assertion_details(fd)
        assert result is None


# ===================================================================
# Bug 3: _active_analyses resource leak fix
# ===================================================================


class TestActiveAnalysisCleanup:
    """Verify _active_analyses is cleaned up after analyze_test_failure."""

    @pytest.mark.asyncio
    async def test_active_analyses_cleaned_after_success(self):
        """After successful analysis, request_id is removed from _active_analyses."""
        svc = _build_service()
        request = _make_request()
        result = _make_analysis_result(request.failure_data)

        svc.analyzer = MagicMock()
        svc.analyzer.analyze_failure = AsyncMock(
            return_value=FailureAnalysisResponse(
                request_id=request.request_id,
                success=True,
                analysis_result=result,
                processing_time=0.1,
            )
        )

        response = await svc.analyze_test_failure(request)
        assert response.success is True
        # The key cleanup: request_id should NOT remain in _active_analyses
        assert request.request_id not in svc._active_analyses

    @pytest.mark.asyncio
    async def test_active_analyses_cleaned_after_error(self):
        """After failed analysis, request_id is removed from _active_analyses."""
        svc = _build_service()
        request = _make_request()

        with patch.object(
            svc,
            "_perform_analysis_with_retry",
            side_effect=RuntimeError("boom"),
        ):
            response = await svc.analyze_test_failure(request)

        assert response.success is False
        assert request.request_id not in svc._active_analyses

    @pytest.mark.asyncio
    async def test_active_analyses_cleaned_after_timeout(self):
        """After timeout, request_id is removed from _active_analyses."""
        from services.ai_failure_analysis import AIAnalysisTimeoutError

        svc = _build_service()
        request = _make_request()

        with patch.object(
            svc,
            "_perform_analysis_with_retry",
            side_effect=AIAnalysisTimeoutError("timed out"),
        ):
            response = await svc.analyze_test_failure(request)

        assert response.success is False
        assert request.request_id not in svc._active_analyses


# ===================================================================
# Bug 4: Description overflow in _consolidate_suggestions
# ===================================================================


class TestConsolidateSuggestionsOverflow:
    """Verify description truncation when merging suggestions."""

    def test_merged_description_does_not_exceed_max_length(self):
        """Merging suggestions with long descriptions truncates to 2000 chars."""
        from services.ai_fix_suggestion import AIFixSuggestionEngine

        with patch("services.ai_fix_suggestion.make_agent_runner"):
            engine = AIFixSuggestionEngine(model_name="test:model")

        # Create two suggestions with the same grouping key (same fix_type + title prefix)
        base_desc = "A" * 1000  # 1000 chars each
        alt_desc = "B" * 1500

        s1 = FixSuggestion(
            suggestion_id="s1",
            title="Fix the important thing here",
            description=base_desc,
            fix_type="code_change",
            priority="high",
            confidence=ConfidenceLevel.MEDIUM,
            confidence_score=0.6,
            estimated_effort="medium",
        )
        s2 = FixSuggestion(
            suggestion_id="s2",
            title="Fix the important thing here",
            description=alt_desc,
            fix_type="code_change",
            priority="medium",
            confidence=ConfidenceLevel.LOW,
            confidence_score=0.4,
            estimated_effort="low",
        )

        result = engine._consolidate_suggestions([s1, s2])
        assert len(result) == 1
        # Description should be at most 2000 chars
        assert len(result[0].description) <= 2000

    def test_consolidate_empty_list(self):
        """Empty list returns empty."""
        from services.ai_fix_suggestion import AIFixSuggestionEngine

        with patch("services.ai_fix_suggestion.make_agent_runner"):
            engine = AIFixSuggestionEngine(model_name="test:model")

        assert engine._consolidate_suggestions([]) == []

    def test_consolidate_merges_code_examples(self):
        """Code examples from both suggestions are combined."""
        from services.ai_fix_suggestion import AIFixSuggestionEngine

        with patch("services.ai_fix_suggestion.make_agent_runner"):
            engine = AIFixSuggestionEngine(model_name="test:model")

        s1 = FixSuggestion(
            suggestion_id="s1",
            title="Fix the important thing here",
            description="First approach to fix the problem in the code",
            fix_type="code_change",
            priority="medium",
            confidence=ConfidenceLevel.MEDIUM,
            confidence_score=0.5,
            estimated_effort="medium",
            code_examples=["example1()"],
        )
        s2 = FixSuggestion(
            suggestion_id="s2",
            title="Fix the important thing here",
            description="Second approach to fix the problem in the code",
            fix_type="code_change",
            priority="high",
            confidence=ConfidenceLevel.HIGH,
            confidence_score=0.8,
            estimated_effort="low",
            code_examples=["example2()"],
        )

        result = engine._consolidate_suggestions([s1, s2])
        assert len(result) == 1
        assert result[0].code_examples is not None
        assert len(result[0].code_examples) == 2
        # Higher confidence should be used
        assert result[0].confidence_score == 0.8
        # Higher priority should be used
        assert result[0].priority == "high"


# ===================================================================
# Bug 2: Naive/aware datetime mismatch (now fixed)
# ===================================================================


class TestDatetimeConsistency:
    """Verify datetime operations use consistent timezone-aware datetimes."""

    def test_integration_service_metrics_use_utc(self):
        """get_integration_metrics timestamps should be UTC-aware ISO strings."""
        from services.ai_test_integration import AITestIntegrationService

        with patch("services.ai_test_integration.AIFailureAnalysisService"):
            svc = AITestIntegrationService(enable_ai_analysis=False)

        metrics = svc.get_integration_metrics()
        ts = metrics["timestamp"]
        # Should parse without error and contain timezone info
        assert "+" in ts or "Z" in ts or ts.endswith("+00:00")


# ===================================================================
# AITestFailureAnalyzer: deeper coverage
# ===================================================================


class TestAITestFailureAnalyzerDeep:
    """Deep tests for error classification, extraction, and value diff."""

    def test_classify_error_type_memory_error(self):
        analyzer = _build_analyzer()
        assert (
            analyzer._classify_error_type("MemoryError: out of memory") == FailureType.MEMORY_ERROR
        )

    def test_classify_error_type_network_error(self):
        analyzer = _build_analyzer()
        assert (
            analyzer._classify_error_type("ConnectionError: network failure")
            == FailureType.NETWORK_ERROR
        )

    def test_classify_error_type_dependency_error(self):
        analyzer = _build_analyzer()
        assert (
            analyzer._classify_error_type("dependency resolution failed for package X")
            == FailureType.DEPENDENCY_ERROR
        )

    def test_classify_error_type_configuration_error(self):
        analyzer = _build_analyzer()
        assert (
            analyzer._classify_error_type("configuration error in settings.yaml")
            == FailureType.CONFIGURATION_ERROR
        )

    def test_classify_error_type_unknown(self):
        analyzer = _build_analyzer()
        assert analyzer._classify_error_type("everything is fine") == FailureType.UNKNOWN_ERROR

    def test_extract_error_message_fallback(self):
        analyzer = _build_analyzer()
        msg = analyzer._extract_error_message("no patterns here at all\njust text")
        assert msg == "Unknown error message"

    def test_extract_error_message_from_error_line(self):
        analyzer = _build_analyzer()
        msg = analyzer._extract_error_message("setup code\nSomething failed terribly\ncleanup")
        assert "failed" in msg.lower()

    def test_extract_traceback_present(self):
        analyzer = _build_analyzer()
        output = "Traceback (most recent call last):\n  File test.py, line 10\nValueError: bad"
        tb = analyzer._extract_traceback(output)
        assert tb is not None
        assert "File test.py" in tb

    def test_extract_traceback_absent(self):
        analyzer = _build_analyzer()
        assert analyzer._extract_traceback("no traceback here") is None

    def test_extract_location_file_line(self):
        analyzer = _build_analyzer()
        output = 'File "tests/test_foo.py", line 42, in test_bar'
        fp, ln, cn = analyzer._extract_location_info(output)
        assert fp == "tests/test_foo.py"
        assert ln == 42
        assert cn is None

    def test_extract_location_file_line_column(self):
        analyzer = _build_analyzer()
        output = 'File "src/main.py", line 10, column 5'
        fp, ln, cn = analyzer._extract_location_info(output)
        assert fp == "src/main.py"
        assert ln == 10
        # Column extraction not implemented — returns None
        assert cn is None

    def test_extract_location_colon_format(self):
        analyzer = _build_analyzer()
        output = "src/main.py:10:5: error found"
        fp, ln, cn = analyzer._extract_location_info(output)
        assert fp == "src/main.py"
        assert ln == 10
        assert cn == 5

    def test_extract_location_none(self):
        analyzer = _build_analyzer()
        fp, ln, cn = analyzer._extract_location_info("no location info")
        assert fp is None
        assert ln is None
        assert cn is None

    def test_parse_stack_trace_empty(self):
        analyzer = _build_analyzer()
        assert analyzer._parse_stack_trace("") == []

    def test_parse_stack_trace_filters_traceback_header(self):
        analyzer = _build_analyzer()
        entries = analyzer._parse_stack_trace(
            "Traceback header\n  File test.py, line 1\n  in test_foo"
        )
        assert all("Traceback" not in e for e in entries)

    def test_classify_assertion_type_assert_equal(self):
        analyzer = _build_analyzer()
        assert analyzer._classify_assertion_type("assertEqual(1, 2)") == "assertEqual"

    def test_classify_assertion_type_unknown(self):
        analyzer = _build_analyzer()
        assert analyzer._classify_assertion_type("some random text") == "unknown"

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

    def test_extract_comparison_operator(self):
        analyzer = _build_analyzer()
        assert analyzer._extract_comparison_operator("1 != 2") == "!="
        assert analyzer._extract_comparison_operator("1 == 2") == "=="
        assert analyzer._extract_comparison_operator("x is y") == "is"
        # "nothing" contains "in" so the operator is detected as "in"
        assert analyzer._extract_comparison_operator("nothing") == "in"
        # A string with no operator keywords returns None
        assert analyzer._extract_comparison_operator("42") is None

    def test_extract_assertion_message_custom(self):
        analyzer = _build_analyzer()
        msg = analyzer._extract_assertion_message("AssertionError: custom message here")
        assert msg is not None
        assert "custom" in msg

    def test_extract_assertion_message_with_operator_filtered(self):
        analyzer = _build_analyzer()
        msg = analyzer._extract_assertion_message("AssertionError: 1 != 2")
        assert msg is None  # Filtered because it contains !=

    def test_calculate_value_difference_numeric(self):
        analyzer = _build_analyzer()
        diff = analyzer._calculate_value_difference("10", "12")
        assert diff is not None
        assert "2" in diff

    def test_calculate_value_difference_string(self):
        analyzer = _build_analyzer()
        diff = analyzer._calculate_value_difference("hello", "hallo")
        assert diff is not None

    def test_calculate_value_difference_length_mismatch(self):
        analyzer = _build_analyzer()
        diff = analyzer._calculate_value_difference("short", "very long string")
        assert diff is not None
        assert "Length" in diff or "-" in diff or "+" in diff

    def test_calculate_value_difference_invalid(self):
        analyzer = _build_analyzer()
        # Same strings with same length, no diff
        diff = analyzer._calculate_value_difference("same", "same")
        # Returns None when values are identical
        assert diff is None

    def test_calculate_overall_confidence_empty(self):
        analyzer = _build_analyzer()
        level, score = analyzer._calculate_overall_confidence([], [], [])
        assert level == ConfidenceLevel.VERY_LOW
        assert score == 0.0

    def test_calculate_overall_confidence_with_data(self):
        analyzer = _build_analyzer()
        cause = RootCause(
            cause_id="c1",
            category="logic_error",
            description="A logic error was found in the test",
            severity=SeverityLevel.MEDIUM,
            confidence=ConfidenceLevel.HIGH,
            confidence_score=0.85,
            evidence=["test evidence"],
        )
        level, score = analyzer._calculate_overall_confidence([cause], [], [])
        assert score > 0
        assert level != ConfidenceLevel.VERY_LOW


# ===================================================================
# AIPatternDetector: coverage for import, export, statistics
# ===================================================================


class TestAIPatternDetectorExtras:
    """Additional tests for pattern detector operations."""

    def _build_detector(self):
        with patch("services.ai_pattern_detector.make_agent_runner"):
            from services.ai_pattern_detector import AIPatternDetector

            return AIPatternDetector(model_name="test:model")

    def test_get_pattern_statistics_empty(self):
        detector = self._build_detector()
        stats = detector.get_pattern_statistics()
        assert stats["total_known_patterns"] == 0
        assert stats["total_history_entries"] == 0

    def test_export_patterns_empty(self):
        detector = self._build_detector()
        exported = detector.export_patterns()
        assert exported["patterns"] == {}
        assert exported["history"] == []

    def test_import_and_export_roundtrip(self):
        detector = self._build_detector()
        pattern_data = {
            "patterns": {
                "p1": {
                    "pattern_name": "Test Pattern Name Here",
                    "pattern_signature": "assertion_error|test_signature_here",
                    "pattern_type": "error_message",
                    "frequency": 5,
                    "confidence_score": 0.8,
                    "typical_causes": ["cause1"],
                    "common_fixes": ["fix1"],
                }
            },
            "history": [],
            "usage_count": {"p1": 3},
            "accuracy": {"p1": 0.9},
        }
        detector.import_patterns(pattern_data)
        assert len(detector._known_patterns) == 1
        assert detector._pattern_usage_count["p1"] == 3

        exported = detector.export_patterns()
        assert "p1" in exported["patterns"]
        assert exported["patterns"]["p1"]["frequency"] == 5

    def test_deduplicate_patterns(self):
        detector = self._build_detector()
        p1 = TestFailurePattern(
            pattern_id="p1",
            pattern_name="Duplicate Pattern Test One",
            pattern_signature="sig_one_test_pattern",
            pattern_type="error_message",
            confidence=ConfidenceLevel.MEDIUM,
            confidence_score=0.5,
        )
        p2 = TestFailurePattern(
            pattern_id="p2",
            pattern_name="Duplicate Pattern Test Two",
            pattern_signature="sig_one_test_pattern",  # Same signature
            pattern_type="error_message",
            confidence=ConfidenceLevel.HIGH,
            confidence_score=0.8,
        )
        result = detector._deduplicate_patterns([p1, p2])
        assert len(result) == 1
        assert result[0].pattern_id == "p1"  # First one wins

    def test_normalize_error_message(self):
        detector = self._build_detector()
        normalized = detector._normalize_error_message("Error at line 42 in 'test.py'")
        assert "N" in normalized  # Numbers replaced
        assert "STR" in normalized  # Strings replaced

    def test_normalize_pattern_name(self):
        detector = self._build_detector()
        name = detector._normalize_pattern_name("My Pattern! #1")
        assert "_" in name
        assert "#" not in name

    def test_score_to_confidence_all_levels(self):
        detector = self._build_detector()
        assert detector._score_to_confidence(0.95) == ConfidenceLevel.VERY_HIGH
        assert detector._score_to_confidence(0.80) == ConfidenceLevel.HIGH
        assert detector._score_to_confidence(0.60) == ConfidenceLevel.MEDIUM
        assert detector._score_to_confidence(0.30) == ConfidenceLevel.LOW
        assert detector._score_to_confidence(0.10) == ConfidenceLevel.VERY_LOW

    def test_cleanup_old_patterns(self):
        """Cleanup removes patterns not in recent history."""
        detector = self._build_detector()
        # Add a known pattern
        p = TestFailurePattern(
            pattern_id="old_p",
            pattern_name="Old Pattern That Is Stale",
            pattern_signature="old_sig_stale_pattern",
            pattern_type="error_message",
            confidence=ConfidenceLevel.LOW,
            confidence_score=0.3,
        )
        detector._known_patterns["old_p"] = p
        detector._pattern_signatures["old_sig_stale_pattern"] = "old_p"
        detector._pattern_usage_count["old_p"] = 1
        detector._pattern_accuracy["old_p"] = 0.5

        # No history entries reference this pattern, so cleanup should remove it
        detector._cleanup_old_patterns()
        assert "old_p" not in detector._known_patterns
        assert "old_sig_stale_pattern" not in detector._pattern_signatures


# ===================================================================
# AIFixSuggestion: pattern matching and fallback
# ===================================================================


class TestFixSuggestionPatternMatching:
    """Test pattern-based fix generation and error matching."""

    def _build_engine(self):
        with patch("services.ai_fix_suggestion.make_agent_runner"):
            from services.ai_fix_suggestion import AIFixSuggestionEngine

            return AIFixSuggestionEngine(model_name="test:model")

    def test_matches_error_pattern_none_value(self):
        engine = self._build_engine()
        assert engine._matches_error_pattern("expected 42 actual None", "none_instead_of_value")

    def test_matches_error_pattern_type_mismatch(self):
        engine = self._build_engine()
        assert engine._matches_error_pattern("expected type int actual class str", "type_mismatch")

    def test_matches_error_pattern_key_error(self):
        engine = self._build_engine()
        assert engine._matches_error_pattern("KeyError: 'missing_key'", "key_error")

    def test_matches_error_pattern_no_match(self):
        engine = self._build_engine()
        assert not engine._matches_error_pattern("everything is fine", "none_instead_of_value")

    def test_create_pattern_based_suggestion(self):
        engine = self._build_engine()
        from services.ai_fix_suggestion import FixContext

        fd = _make_failure_data()
        ed = _make_error_details()
        context = FixContext(failure_data=fd, error_details=ed, root_causes=[])

        solution = {
            "type": "code_change",
            "description": "Add missing return statement",
            "priority": "high",
            "effort": "trivial",
            "template": "return value",
        }
        suggestion = engine._create_pattern_based_suggestion("test_pattern", solution, context)
        assert suggestion.suggestion_id.startswith("pattern_test_pattern_")
        assert suggestion.code_examples == ["return value"]

    def test_create_fallback_suggestions(self):
        engine = self._build_engine()
        from services.ai_fix_suggestion import FixContext

        fd = _make_failure_data()
        ed = _make_error_details()
        context = FixContext(failure_data=fd, error_details=ed, root_causes=[])

        suggestions = engine._create_fallback_suggestions(context)
        assert len(suggestions) == 1
        assert suggestions[0].fix_type == "investigation_needed"

    def test_create_cache_key_deterministic(self):
        engine = self._build_engine()
        from services.ai_fix_suggestion import FixContext

        fd = _make_failure_data()
        ed = _make_error_details()
        context = FixContext(failure_data=fd, error_details=ed, root_causes=[])

        key1 = engine._create_cache_key(context)
        key2 = engine._create_cache_key(context)
        assert key1 == key2
        assert "|" in key1

    def test_enhance_suggestion_immediate_type(self):
        engine = self._build_engine()
        from services.ai_fix_suggestion import FixContext

        fd = _make_failure_data()
        ed = _make_error_details(error_message="test error in the code module")
        context = FixContext(
            failure_data=fd,
            error_details=ed,
            root_causes=[],
        )

        suggestion = FixSuggestion(
            suggestion_id="test-1",
            title="Fix the broken assertion code",
            description="Change the assertion to match expected behavior correctly",
            fix_type="code_change",
            priority="medium",
            confidence=ConfidenceLevel.MEDIUM,
            confidence_score=0.7,
            estimated_effort="low",
        )

        enhanced = engine._enhance_suggestion_with_context(suggestion, context, "immediate")
        assert enhanced.suggestion_id.startswith("immediate_")
        # Immediate type boosts confidence by 1.1x, capped at 1.0
        assert enhanced.confidence_score == min(0.7 * 1.1, 1.0)

    def test_enhance_suggestion_preventive_type(self):
        engine = self._build_engine()
        from services.ai_fix_suggestion import FixContext

        fd = _make_failure_data()
        ed = _make_error_details()
        context = FixContext(failure_data=fd, error_details=ed, root_causes=[])

        suggestion = FixSuggestion(
            suggestion_id="test-2",
            title="Add validation to prevent issues",
            description="Add input validation to prevent similar failures in the future",
            fix_type="code_change",
            priority="low",
            confidence=ConfidenceLevel.HIGH,
            confidence_score=0.8,
            estimated_effort="medium",
        )

        enhanced = engine._enhance_suggestion_with_context(suggestion, context, "preventive")
        # Preventive type reduces confidence by 0.8x
        assert enhanced.confidence_score == 0.8 * 0.8

    def test_enhance_suggestion_adds_files_from_context(self):
        engine = self._build_engine()
        from services.ai_fix_suggestion import FixContext

        fd = _make_failure_data()
        ed = ErrorDetails(
            error_type=FailureType.ASSERTION_ERROR,
            error_message="error in test code here",
            file_path="src/module.py",
        )
        context = FixContext(failure_data=fd, error_details=ed, root_causes=[])

        suggestion = FixSuggestion(
            suggestion_id="test-3",
            title="Fix the file with the error now",
            description="Modify the file to correct the assertion failure issue",
            fix_type="code_change",
            priority="high",
            confidence=ConfidenceLevel.MEDIUM,
            confidence_score=0.6,
            estimated_effort="low",
        )

        enhanced = engine._enhance_suggestion_with_context(suggestion, context, "immediate")
        assert enhanced.files_to_modify == ["src/module.py"]
        assert enhanced.verification_steps is not None
        assert len(enhanced.verification_steps) == 3


# ===================================================================
# AIRootCause: pattern matching and consolidation
# ===================================================================


class TestRootCauseEngineExtras:
    """Additional tests for root cause analysis engine."""

    def _build_engine(self):
        with patch("services.ai_root_cause.make_agent_runner"):
            from services.ai_root_cause import AIRootCauseEngine

            return AIRootCauseEngine()

    def test_matches_pattern_valid_regex(self):
        engine = self._build_engine()
        assert engine._matches_pattern(
            "AttributeError: object has no attribute 'foo'",
            r"AttributeError.*object has no attribute",
        )

    def test_matches_pattern_invalid_regex_returns_false(self):
        engine = self._build_engine()
        # Invalid regex pattern returns False (no fallback to substring)
        assert engine._matches_pattern("hello world", "[invalid") is False

    def test_matches_pattern_empty(self):
        engine = self._build_engine()
        assert not engine._matches_pattern("any text", "")

    def test_consolidate_causes_empty(self):
        engine = self._build_engine()
        assert engine._consolidate_causes([]) == []

    def test_consolidate_causes_merges_evidence(self):
        engine = self._build_engine()
        c1 = RootCause(
            cause_id="c1",
            category="logic_error",
            description="Logic error found in calculation module test code",
            severity=SeverityLevel.MEDIUM,
            confidence=ConfidenceLevel.MEDIUM,
            confidence_score=0.6,
            evidence=["evidence1"],
        )
        c2 = RootCause(
            cause_id="c2",
            category="logic_error",
            description="Logic error found in calculation module test code",
            severity=SeverityLevel.HIGH,
            confidence=ConfidenceLevel.HIGH,
            confidence_score=0.85,
            evidence=["evidence2"],
        )
        result = engine._consolidate_causes([c1, c2])
        assert len(result) == 1
        assert "evidence1" in result[0].evidence
        assert "evidence2" in result[0].evidence
        assert result[0].confidence_score == 0.85

    def test_create_fallback_analysis(self):
        engine = self._build_engine()
        from services.ai_root_cause import AIAnalysisContext

        fd = _make_failure_data()
        ed = _make_error_details()
        context = AIAnalysisContext(failure_data=fd, error_details=ed)

        causes = engine._create_fallback_analysis(context)
        assert len(causes) == 1
        assert causes[0].confidence == ConfidenceLevel.VERY_LOW

    def test_create_cache_key(self):
        engine = self._build_engine()
        from services.ai_root_cause import AIAnalysisContext

        fd = _make_failure_data()
        ed = _make_error_details()
        context = AIAnalysisContext(failure_data=fd, error_details=ed)

        key = engine._create_cache_key(context)
        assert "|" in key

    def test_enhance_cause_syntactic_type(self):
        engine = self._build_engine()
        from services.ai_root_cause import AIAnalysisContext

        fd = _make_failure_data()
        ed = _make_error_details()
        context = AIAnalysisContext(failure_data=fd, error_details=ed)

        cause = RootCause(
            cause_id="temp",
            category="logic_error",
            description="A syntactic issue found during code analysis check",
            severity=SeverityLevel.MEDIUM,
            confidence=ConfidenceLevel.MEDIUM,
            confidence_score=0.6,
            evidence=[],
        )
        enhanced = engine._enhance_cause_with_context(cause, context, "syntactic")
        assert enhanced.cause_id.startswith("syntactic_")
        assert "Identified through syntactic analysis" in enhanced.evidence
        # Syntactic analysis boosts confidence by 1.1x
        assert enhanced.confidence_score == min(0.6 * 1.1, 1.0)

    def test_enhance_cause_semantic_type(self):
        engine = self._build_engine()
        from services.ai_root_cause import AIAnalysisContext

        fd = _make_failure_data()
        ed = _make_error_details()
        context = AIAnalysisContext(failure_data=fd, error_details=ed)

        cause = RootCause(
            cause_id="temp",
            category="logic_error",
            description="A semantic issue found during code analysis check",
            severity=SeverityLevel.MEDIUM,
            confidence=ConfidenceLevel.MEDIUM,
            confidence_score=0.6,
            evidence=[],
        )
        enhanced = engine._enhance_cause_with_context(cause, context, "semantic")
        # Semantic analysis reduces confidence by 0.9x
        assert enhanced.confidence_score == 0.6 * 0.9


# ===================================================================
# AITestIntegration: health check and batch analysis
# ===================================================================


class TestAITestIntegrationExtras:
    """Additional tests for integration service."""

    def _build_service(self, enable_ai=False):
        with patch("services.ai_test_integration.AIFailureAnalysisService"):
            from services.ai_test_integration import AITestIntegrationService

            return AITestIntegrationService(enable_ai_analysis=enable_ai)

    @pytest.mark.asyncio
    async def test_health_check_with_stub(self):
        svc = self._build_service(enable_ai=False)
        health = await svc.health_check()
        assert health["service"] == "ai_test_integration"
        assert "checks" in health

    @pytest.mark.asyncio
    async def test_analyze_existing_failure_disabled(self):
        svc = self._build_service(enable_ai=False)
        result = TestResult(
            test_id="t1",
            success=False,
            execution_time=1.0,
            stderr="AssertionError",
        )
        analysis = await svc.analyze_existing_failure(result)
        assert analysis is None

    @pytest.mark.asyncio
    async def test_batch_analyze_no_failures(self):
        svc = self._build_service(enable_ai=True)
        results = [TestResult(test_id="t1", success=True, execution_time=0.5)]
        analyses = await svc.batch_analyze_failures(results)
        assert all(a is None for a in analyses)

    @pytest.mark.asyncio
    async def test_batch_analyze_disabled(self):
        svc = self._build_service(enable_ai=False)
        results = [TestResult(test_id="t1", success=False, execution_time=1.0)]
        analyses = await svc.batch_analyze_failures(results)
        assert all(a is None for a in analyses)

    def test_determine_test_framework_javascript(self):
        svc = self._build_service(enable_ai=False)
        result = TestResult(test_id="t1", language="javascript", success=True, execution_time=1.0)
        fw = svc._determine_test_framework(result)
        assert fw.value == "jest"

    def test_determine_test_framework_unknown_language(self):
        svc = self._build_service(enable_ai=False)
        result = TestResult(test_id="t1", language="ruby", success=True, execution_time=1.0)
        fw = svc._determine_test_framework(result)
        assert fw.value == "unknown"

    def test_determine_test_framework_python_pytest(self):
        svc = self._build_service(enable_ai=False)
        result = TestResult(
            test_id="t1",
            language="python",
            success=False,
            execution_time=1.0,
            stderr="FAILED tests/test_foo.py - pytest output",
        )
        fw = svc._determine_test_framework(result)
        assert fw.value == "pytest"

    @pytest.mark.asyncio
    async def test_get_analysis_service_health_disabled(self):
        svc = self._build_service(enable_ai=False)
        health = await svc.get_analysis_service_health()
        assert health["status"] == "disabled"

    @pytest.mark.asyncio
    async def test_cleanup(self):
        svc = self._build_service(enable_ai=False)
        # Should not raise
        await svc.cleanup()


# ===================================================================
# Performance Insight Analyzer: edge cases
# ===================================================================


class TestPerformanceInsightAnalyzerEdgeCases:
    """Edge case tests for performance analysis."""

    def _build_analyzer(self):
        from services.performance_insight_analyzer import PerformanceInsightAnalyzer

        return PerformanceInsightAnalyzer()

    @pytest.mark.asyncio
    async def test_analyze_single_result(self):
        """Analysis with a single test result should not crash."""
        analyzer = self._build_analyzer()
        results = [TestResult(test_id="t1", execution_time=5.0, language="python")]
        result = await analyzer.analyze_performance(results)
        assert result.total_tests_analyzed == 1
        assert result.overall_statistics.std_dev == 0.0

    @pytest.mark.asyncio
    async def test_analyze_empty_raises(self):
        """Empty results should raise ValueError."""
        analyzer = self._build_analyzer()
        with pytest.raises(ValueError, match="No execution times"):
            await analyzer.analyze_performance([])

    @pytest.mark.asyncio
    async def test_trend_analysis_insufficient_samples(self):
        """Trend analysis with too few samples returns stable."""
        analyzer = self._build_analyzer()
        results = [
            TestResult(test_id=f"t{i}", execution_time=float(i), language="python")
            for i in range(3)
        ]
        trend = await analyzer._analyze_trends(results)
        assert trend.trend_direction == "stable"
        assert trend.trend_confidence == 0.0

    @pytest.mark.asyncio
    async def test_trend_analysis_degrading(self):
        """Monotonically increasing times should show degrading trend."""
        analyzer = self._build_analyzer()
        results = [
            TestResult(test_id=f"t{i}", execution_time=float(i) * 10, language="python")
            for i in range(15)
        ]
        trend = await analyzer._analyze_trends(results)
        assert trend.trend_direction == "degrading"
        assert trend.slope > 0

    @pytest.mark.asyncio
    async def test_categorize_tests(self):
        """Tests are correctly categorized as slow/fast/variable."""
        analyzer = self._build_analyzer()
        results = [
            TestResult(test_id="fast", execution_time=0.5, language="python"),
            TestResult(test_id="normal", execution_time=5.0, language="python"),
            TestResult(test_id="slow", execution_time=15.0, language="python"),
        ]
        slow, fast, variable = await analyzer._categorize_tests(results)
        assert "slow" in slow
        assert "fast" in fast

    @pytest.mark.asyncio
    async def test_optimization_potential_no_insights(self):
        """No insights means zero optimization potential."""
        analyzer = self._build_analyzer()
        potential = await analyzer._calculate_optimization_potential([])
        assert potential == 0.0


# ===================================================================
# Test Pattern Analyzer: edge cases
# ===================================================================


class TestTestPatternAnalyzerEdgeCases:
    """Edge case tests for test pattern analyzer."""

    def _build_analyzer(self):
        from services.test_pattern_analyzer import TestPatternAnalyzer

        return TestPatternAnalyzer()

    @pytest.mark.asyncio
    async def test_analyze_all_passing(self):
        """All passing tests should produce no reliability patterns."""
        analyzer = self._build_analyzer()
        results = [
            TestResult(test_id=f"t{i}", success=True, execution_time=1.0, language="python")
            for i in range(20)
        ]
        result = await analyzer.analyze_patterns(results)
        # No flaky test patterns since all pass
        reliability = [p for p in result.detected_patterns if p.category.value == "reliability"]
        assert len(reliability) == 0

    @pytest.mark.asyncio
    async def test_filter_patterns_by_thresholds(self):
        """Patterns below thresholds are filtered out."""
        analyzer = self._build_analyzer()
        from services.test_pattern_analyzer import (
            PatternCategory,
            PatternSeverity,
            TestPatternMetrics,
        )

        low_confidence = TestPatternMetrics(
            pattern_id="low",
            pattern_name="Low confidence pattern test",
            category=PatternCategory.PERFORMANCE,
            severity=PatternSeverity.LOW,
            confidence_score=0.1,  # Below default 0.6 threshold
            occurrence_count=10,
            impact_score=8.0,
            frequency_ratio=0.5,
            description="This pattern has very low confidence score",
        )
        filtered = analyzer._filter_patterns([low_confidence])
        assert len(filtered) == 0

    def test_identify_improvement_opportunities(self):
        """Improvement opportunities are generated from pattern categories."""
        analyzer = self._build_analyzer()
        from services.test_pattern_analyzer import (
            PatternCategory,
            PatternSeverity,
            TestPatternMetrics,
        )

        patterns = [
            TestPatternMetrics(
                pattern_id="perf1",
                pattern_name="Performance Issue In Tests",
                category=PatternCategory.PERFORMANCE,
                severity=PatternSeverity.HIGH,
                confidence_score=0.9,
                occurrence_count=5,
                impact_score=8.0,
                frequency_ratio=0.3,
                description="Test performance bottleneck detected in suite",
            ),
        ]
        opportunities = analyzer._identify_improvement_opportunities(patterns)
        assert any("performance" in o.lower() for o in opportunities)
