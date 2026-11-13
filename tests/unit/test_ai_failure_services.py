"""Tests for AI failure analysis services.

Covers: ai_failure_analysis.py, ai_test_failure_analyzer.py,
        ai_pattern_detector.py, ai_root_cause.py, ai_fix_suggestion.py
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from models.ai_test_failure_models import (
    ConfidenceLevel,
    ErrorDetails,
    FailureAnalysisRequest,
    FailureType,
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


def _make_analysis_request(
    raw_output: str = "AssertionError: expected True got False",
    failure_id: str = "fail-1",
) -> FailureAnalysisRequest:
    return FailureAnalysisRequest(
        request_id="req-1",
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


def _make_root_cause(confidence_score: float = 0.3) -> RootCause:
    level = ConfidenceLevel.LOW
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
        evidence=["Error type: assertion_error"],
    )


# ===================================================================
# AIFailureAnalysisService tests
# ===================================================================


class TestAIFailureAnalysisServiceHealth:
    def test_failure_analysis_health_status(self):
        """get_health_status returns expected keys."""
        with patch("services.ai_failure_analysis.AITestFailureAnalyzer"):
            with patch("services.ai_failure_analysis.AIRootCauseEngine"):
                with patch("services.ai_failure_analysis.AIFixSuggestionEngine"):
                    from services.ai_failure_analysis import AIFailureAnalysisService

                    svc = AIFailureAnalysisService(model_name="test:model")

        health = svc.get_health_status()
        assert "service_name" in health
        assert "circuit_breaker_open" in health
        assert health["model_name"] == "test:model"

    def test_failure_analysis_metrics_aggregation(self):
        """Metrics are updated on success / failure calls."""
        with patch("services.ai_failure_analysis.AITestFailureAnalyzer"):
            with patch("services.ai_failure_analysis.AIRootCauseEngine"):
                with patch("services.ai_failure_analysis.AIFixSuggestionEngine"):
                    from services.ai_failure_analysis import AIFailureAnalysisService

                    svc = AIFailureAnalysisService(model_name="test:model")

        svc._update_metrics(success=True, duration=1.0)
        svc._update_metrics(success=False, duration=0.5, error_type="timeout")

        metrics = svc.get_metrics()
        assert metrics["total_analyses"] == 2
        assert metrics["successful_analyses"] == 1
        assert metrics["failed_analyses"] == 1
        assert metrics["timeout_errors"] == 1

    def test_failure_analysis_degraded_service(self):
        """If engine init fails the service falls to degraded mode."""
        with patch(
            "services.ai_failure_analysis.AITestFailureAnalyzer",
            side_effect=RuntimeError("init fail"),
        ):
            from services.ai_failure_analysis import AIFailureAnalysisService

            svc = AIFailureAnalysisService(model_name="bad:model")

        assert svc.is_available is False
        assert svc.analyzer is None


# ===================================================================
# AITestFailureAnalyzer tests
# ===================================================================


class TestAITestFailureAnalyzer:
    def test_analyzer_structured_output_from_failure_input(self):
        """_classify_error_type returns expected FailureType for known patterns."""
        with patch("services.ai_test_failure_analyzer.make_agent_runner"):
            from services.ai_test_failure_analyzer import AITestFailureAnalyzer

            analyzer = AITestFailureAnalyzer()

        assert (
            analyzer._classify_error_type("AssertionError: x != y") == FailureType.ASSERTION_ERROR
        )
        assert analyzer._classify_error_type("SyntaxError: invalid") == FailureType.SYNTAX_ERROR
        assert analyzer._classify_error_type("ImportError: no module") == FailureType.IMPORT_ERROR
        assert analyzer._classify_error_type("timeout exceeded") == FailureType.TIMEOUT_ERROR

    def test_analyzer_missing_logs_handling(self):
        """_extract_error_message returns fallback when output has no error."""
        with patch("services.ai_test_failure_analyzer.make_agent_runner"):
            from services.ai_test_failure_analyzer import AITestFailureAnalyzer

            analyzer = AITestFailureAnalyzer()

        msg = analyzer._extract_error_message("all good, nothing bad here")
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_analyzer_confidence_categorization(self):
        """_calculate_overall_confidence maps scores to correct levels."""
        with patch("services.ai_test_failure_analyzer.make_agent_runner"):
            from services.ai_test_failure_analyzer import AITestFailureAnalyzer

            analyzer = AITestFailureAnalyzer()

        # No inputs -> VERY_LOW
        level, score = analyzer._calculate_overall_confidence([], [], [])
        assert level == ConfidenceLevel.VERY_LOW
        assert score == 0.0

        # High-confidence cause
        cause = _make_root_cause(confidence_score=0.95)
        level, score = analyzer._calculate_overall_confidence([cause], [], [])
        assert level in (ConfidenceLevel.VERY_HIGH, ConfidenceLevel.HIGH)
        assert score > 0.5


# ===================================================================
# AIPatternDetector tests
# ===================================================================


class TestAIPatternDetector:
    async def test_pattern_detector_frequency_grouping(self):
        """A known pattern is returned and its frequency incremented on re-match."""
        with patch("services.ai_pattern_detector.make_agent_runner"):
            from services.ai_pattern_detector import AIPatternDetector

            detector = AIPatternDetector()

        pattern = TestFailurePattern(
            pattern_id="p1",
            pattern_name="Timeout Pattern",
            pattern_signature="timeout_error|normalized|pytest",
            pattern_type="error_message",
            confidence=ConfidenceLevel.MEDIUM,
            confidence_score=0.6,
        )
        detector._known_patterns["p1"] = pattern

        failure_data = _make_failure_data(raw_output="timeout occurred")
        error_details = _make_error_details(
            error_type=FailureType.TIMEOUT_ERROR,
            error_message="timeout occurred",
        )

        # Signature building must work against VALIDATED models (use_enum_values
        # stores the enum fields as plain strings).
        sig = detector._create_failure_signature(failure_data, error_details)
        assert "timeout_error" in sig
        assert "pytest" in sig
        detector._pattern_signatures[sig] = "p1"

        # Similarity says "not similar", so the only way to get a hit is the
        # exact-signature lookup -- that is what this test pins.
        detector.similarity_analyzer = MagicMock()
        detector.similarity_analyzer.run = AsyncMock(return_value=MagicMock(output=0.0))

        matched = await detector._match_known_patterns(failure_data, error_details)

        assert [p.pattern_id for p in matched] == ["p1"]
        assert pattern.frequency == 2, "exact-signature match must increment frequency"

        # A second match increments again -- proving the counter is real state,
        # not a constant.
        matched_again = await detector._match_known_patterns(failure_data, error_details)
        assert [p.pattern_id for p in matched_again] == ["p1"]
        assert pattern.frequency == 3

    def test_pattern_detector_deduping(self):
        """_deduplicate_patterns removes duplicates by signature."""
        with patch("services.ai_pattern_detector.make_agent_runner"):
            from services.ai_pattern_detector import AIPatternDetector

            detector = AIPatternDetector()

        p1 = TestFailurePattern(
            pattern_id="p1",
            pattern_name="Dup Pattern A",
            pattern_signature="error_type:assertion_error_sig",
            pattern_type="error_message",
            confidence=ConfidenceLevel.MEDIUM,
            confidence_score=0.5,
        )
        p2 = TestFailurePattern(
            pattern_id="p2",
            pattern_name="Dup Pattern B",
            pattern_signature="error_type:assertion_error_sig",
            pattern_type="error_message",
            confidence=ConfidenceLevel.HIGH,
            confidence_score=0.8,
        )
        p3 = TestFailurePattern(
            pattern_id="p3",
            pattern_name="Unique Pattern",
            pattern_signature="error_type:timeout_error_unique_sig",
            pattern_type="error_message",
            confidence=ConfidenceLevel.LOW,
            confidence_score=0.3,
        )

        deduped = detector._deduplicate_patterns([p1, p2, p3])
        assert len(deduped) == 2
        signatures = [p.pattern_signature for p in deduped]
        assert "error_type:assertion_error_sig" in signatures
        assert "error_type:timeout_error_unique_sig" in signatures

    def test_pattern_detector_unknown_fallback(self):
        """_score_to_confidence maps edge values properly."""
        with patch("services.ai_pattern_detector.make_agent_runner"):
            from services.ai_pattern_detector import AIPatternDetector

            detector = AIPatternDetector()

        assert detector._score_to_confidence(0.95) == ConfidenceLevel.VERY_HIGH
        assert detector._score_to_confidence(0.8) == ConfidenceLevel.HIGH
        assert detector._score_to_confidence(0.6) == ConfidenceLevel.MEDIUM
        assert detector._score_to_confidence(0.3) == ConfidenceLevel.LOW
        assert detector._score_to_confidence(0.1) == ConfidenceLevel.VERY_LOW


# ===================================================================
# AIRootCauseEngine tests
# ===================================================================


class TestAIRootCauseEngine:
    def test_root_cause_extraction(self):
        """_create_fallback_analysis returns the degraded marker cause (F4, PR #39).

        This is the path taken when the whole LLM plane is unreachable, so it must
        never raise and must be self-identifying as low-trust: an outage result that
        looked like a confident answer would be indistinguishable from a real one.
        """
        with patch("services.ai_root_cause.make_agent_runner"):
            from services.ai_root_cause import AIAnalysisContext, AIRootCauseEngine

            engine = AIRootCauseEngine()

        fd = _make_failure_data()
        ed = _make_error_details()
        ctx = AIAnalysisContext(failure_data=fd, error_details=ed)

        fallback = engine._create_fallback_analysis(ctx)
        assert len(fallback) == 1
        cause = fallback[0]
        assert cause.confidence == ConfidenceLevel.VERY_LOW
        assert cause.confidence_score == 0.2
        assert cause.category == "unknown"
        assert cause.cause_id.startswith("fallback_")
        # The evidence must say this came from a failure, not from analysis.
        assert any("Fallback analysis" in e for e in cause.evidence)
        # The real error must survive into the degraded result rather than be dropped.
        assert ed.error_message in cause.description

    def test_root_cause_ambiguous_input(self):
        """_matches_pattern handles empty pattern gracefully."""
        with patch("services.ai_root_cause.make_agent_runner"):
            from services.ai_root_cause import AIRootCauseEngine

            engine = AIRootCauseEngine()

        assert engine._matches_pattern("some text", "") is False
        assert engine._matches_pattern("ModuleNotFoundError", r"ModuleNotFoundError") is True

    def test_root_cause_confidence_ranking(self):
        """_consolidate_causes sorts by confidence descending."""
        with patch("services.ai_root_cause.make_agent_runner"):
            from services.ai_root_cause import AIRootCauseEngine

            engine = AIRootCauseEngine()

        low_cause = _make_root_cause(confidence_score=0.3)
        low_cause.category = "logic_error"
        low_cause.description = "Low confidence cause that exceeds ten characters easily"

        high_cause = _make_root_cause(confidence_score=0.8)
        high_cause.cause_id = "cause-2"
        high_cause.confidence = ConfidenceLevel.HIGH
        high_cause.category = "type_mismatch"
        high_cause.description = "High confidence cause that exceeds ten characters easily"

        consolidated = engine._consolidate_causes([low_cause, high_cause])
        assert consolidated[0].confidence_score >= consolidated[-1].confidence_score


# ===================================================================
# AIFixSuggestionEngine tests
# ===================================================================


class TestAIFixSuggestionEngine:
    def test_fix_suggestion_generation(self):
        """_create_fallback_suggestions returns a valid FixSuggestion list."""
        with patch("services.ai_fix_suggestion.make_agent_runner"):
            from services.ai_fix_suggestion import AIFixSuggestionEngine, FixContext

            engine = AIFixSuggestionEngine()

        fd = _make_failure_data()
        ed = _make_error_details()
        ctx = FixContext(
            failure_data=fd,
            error_details=ed,
            root_causes=[],
        )

        fallback = engine._create_fallback_suggestions(ctx)
        assert len(fallback) == 1
        assert fallback[0].confidence == ConfidenceLevel.LOW

    def test_fix_suggestion_empty_context(self):
        """_matches_error_pattern with unknown pattern returns False."""
        with patch("services.ai_fix_suggestion.make_agent_runner"):
            from services.ai_fix_suggestion import AIFixSuggestionEngine

            engine = AIFixSuggestionEngine()

        assert engine._matches_error_pattern("some random error", "nonexistent_pattern") is False

    def test_fix_suggestion_provider_failure(self):
        """_consolidate_suggestions with empty list returns empty."""
        with patch("services.ai_fix_suggestion.make_agent_runner"):
            from services.ai_fix_suggestion import AIFixSuggestionEngine

            engine = AIFixSuggestionEngine()

        result = engine._consolidate_suggestions([])
        assert result == []
