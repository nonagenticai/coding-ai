"""Tests for AIPatternDetector."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.ai_test_failure_models import (
    ConfidenceLevel,
    ErrorDetails,
    FailureType,
    TestFailureContext,
    TestFailureData,
    TestFailurePattern,
    TestFramework,
)


def _make_detector(**kwargs):
    """Create an AIPatternDetector with mocked agent initialization."""
    with patch("services.ai_pattern_detector.AIPatternDetector._init_pattern_agents"):
        from services.ai_pattern_detector import AIPatternDetector

        return AIPatternDetector(**kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# These factories construct models through NORMAL pydantic validation, exactly
# as production does. The models set use_enum_values=True, so the enum fields
# land as plain strings -- do NOT switch these to ``model_construct``. Bypassing
# validation keeps enum instances alive and hides any source code that assumes
# it can call ``.value`` on those fields (which would raise AttributeError in
# production while the suite stayed green).
# ---------------------------------------------------------------------------


def _failure_data(
    failure_id: str = "fail-1",
    test_name: str = "test_foo",
    framework: TestFramework = TestFramework.PYTEST,
    raw_output: str = "FAILED test_foo - AssertionError",
) -> TestFailureData:
    ctx = TestFailureContext(
        test_name=test_name,
        framework=framework,
    )
    return TestFailureData(
        failure_id=failure_id,
        raw_output=raw_output,
        timestamp=datetime.now(UTC),
        context=ctx,
    )


def _error_details(
    error_type: FailureType = FailureType.ASSERTION_ERROR,
    error_message: str = "expected 1 but got 2",
    file_path: str | None = None,
    stack_trace: list[str] | None = None,
) -> ErrorDetails:
    return ErrorDetails(
        error_type=error_type,
        error_message=error_message,
        file_path=file_path,
        stack_trace=stack_trace,
    )


def _pattern(
    pattern_id: str = "pat-1",
    pattern_name: str = "Test Pattern One",
    pattern_signature: str = "assertion_error|expected_N_but_got_N|pytest",
    pattern_type: str = "error_message",
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
    confidence_score: float = 0.8,
    frequency: int = 1,
    typical_causes: list[str] | None = None,
    common_fixes: list[str] | None = None,
    associated_failures: list[str] | None = None,
) -> TestFailurePattern:
    return TestFailurePattern(
        pattern_id=pattern_id,
        pattern_name=pattern_name,
        pattern_signature=pattern_signature,
        pattern_type=pattern_type,
        confidence=confidence,
        confidence_score=confidence_score,
        frequency=frequency,
        typical_causes=typical_causes or [],
        common_fixes=common_fixes or [],
        associated_failures=associated_failures or [],
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestAIPatternDetectorInit:
    def test_default_model(self):
        # Migration: provider strings ("openai:gpt-4o-mini") were dropped when
        # we routed coding-ai through coding-client.BaseLLMAgent. The default
        # ``model_name`` is now the registry schema_id used as the
        # X-Schema-Id header.
        detector = _make_detector()
        assert detector.model_name == "test_failure_analysis"

    def test_custom_model(self):
        # ``model_name`` override is preserved verbatim for diagnostic logs.
        detector = _make_detector(model_name="custom-routing-key")
        assert detector.model_name == "custom-routing-key"

    def test_initial_state(self):
        detector = _make_detector()
        assert detector._known_patterns == {}
        assert detector._pattern_history == []
        assert detector._pattern_signatures == {}
        assert detector._similarity_threshold == 0.7


# ---------------------------------------------------------------------------
# Helper / utility methods
# ---------------------------------------------------------------------------


class TestNormalizeErrorMessage:
    def test_replaces_numbers(self):
        detector = _make_detector()
        result = detector._normalize_error_message("expected 42 but got 99")
        assert "42" not in result
        assert "99" not in result
        assert "N" in result

    def test_replaces_quoted_strings(self):
        detector = _make_detector()
        result = detector._normalize_error_message("key 'hello world' not found")
        assert "hello world" not in result
        assert "STR" in result

    def test_collapses_whitespace(self):
        detector = _make_detector()
        result = detector._normalize_error_message("a   b   c")
        assert "   " not in result

    def test_truncates_long_messages(self):
        detector = _make_detector()
        long_msg = "x" * 200
        result = detector._normalize_error_message(long_msg)
        assert len(result) <= 100


class TestNormalizePatternName:
    def test_lowercases_and_replaces_specials(self):
        detector = _make_detector()
        result = detector._normalize_pattern_name("My Pattern (v2)")
        assert result == "my_pattern__v2_"

    def test_truncates(self):
        detector = _make_detector()
        result = detector._normalize_pattern_name("a" * 100)
        assert len(result) <= 50


class TestScoreToConfidence:
    @pytest.mark.parametrize(
        "score,expected",
        [
            (0.95, ConfidenceLevel.VERY_HIGH),
            (0.8, ConfidenceLevel.HIGH),
            (0.6, ConfidenceLevel.MEDIUM),
            (0.3, ConfidenceLevel.LOW),
            (0.1, ConfidenceLevel.VERY_LOW),
        ],
    )
    def test_mapping(self, score, expected):
        detector = _make_detector()
        assert detector._score_to_confidence(score) == expected


class TestCreateFailureSignature:
    def test_basic_signature(self):
        detector = _make_detector()
        fd = _failure_data()
        ed = _error_details()
        sig = detector._create_failure_signature(fd, ed)
        assert "assertion_error" in sig
        assert "pytest" in sig

    def test_with_file_path(self):
        detector = _make_detector()
        fd = _failure_data()
        ed = _error_details(file_path="/src/foo/bar.py")
        sig = detector._create_failure_signature(fd, ed)
        assert "bar.py" in sig

    def test_without_file_path(self):
        detector = _make_detector()
        fd = _failure_data()
        ed = _error_details(file_path=None)
        sig = detector._create_failure_signature(fd, ed)
        parts = sig.split("|")
        assert len(parts) == 3


class TestCreatePatternSignature:
    def test_uses_existing_signature(self):
        detector = _make_detector()
        pat = _pattern(pattern_signature="existing_sig_1234567890")
        fd = _failure_data()
        ed = _error_details()
        sig = detector._create_pattern_signature(pat, fd, ed)
        assert sig == "existing_sig_1234567890"

    def test_creates_new_when_no_existing(self):
        """When pattern_signature is falsy, a new one is created."""
        detector = _make_detector()
        # An empty pattern_signature cannot be produced through validation
        # (min_length=10), so build this one case with model_construct. Only the
        # signature is off-spec; the enum-bearing models stay fully validated.
        pat = _pattern().model_copy(update={"pattern_signature": ""})
        fd = _failure_data()
        ed = _error_details()
        sig = detector._create_pattern_signature(pat, fd, ed)
        # Empty string is falsy, so it should build a new signature
        assert "|" in sig
        assert "error_message" in sig


class TestPreparePatternInput:
    def test_basic_input(self):
        detector = _make_detector()
        fd = _failure_data()
        ed = _error_details()
        result = detector._prepare_pattern_input(fd, ed)
        assert "test_foo" in result
        assert "assertion_error" in result

    def test_with_file_path(self):
        detector = _make_detector()
        fd = _failure_data()
        ed = _error_details(file_path="/src/bar.py")
        result = detector._prepare_pattern_input(fd, ed)
        assert "File: /src/bar.py" in result

    def test_with_stack_trace(self):
        detector = _make_detector()
        fd = _failure_data()
        ed = _error_details(stack_trace=["frame1", "frame2"])
        result = detector._prepare_pattern_input(fd, ed)
        assert "Stack Trace:" in result
        assert "frame1" in result


class TestPrepareSimilarityInput:
    def test_basic_input(self):
        detector = _make_detector()
        fd = _failure_data()
        ed = _error_details()
        kp = _pattern(typical_causes=["bad logic"])
        result = detector._prepare_similarity_input(fd, ed, kp)
        assert "Current Failure" in result
        assert "Known Pattern" in result
        assert "bad logic" in result

    def test_without_typical_causes(self):
        detector = _make_detector()
        fd = _failure_data()
        ed = _error_details()
        kp = _pattern(typical_causes=None)
        result = detector._prepare_similarity_input(fd, ed, kp)
        assert "Typical Causes" not in result


class TestPrepareClassificationInput:
    def test_basic_input(self):
        detector = _make_detector()
        pat = _pattern()
        ed = _error_details()
        result = detector._prepare_classification_input(pat, ed)
        assert "Pattern Name:" in result
        assert "Error Type:" in result

    def test_with_stack_trace(self):
        detector = _make_detector()
        pat = _pattern()
        ed = _error_details(stack_trace=["frame1", "frame2", "frame3"])
        result = detector._prepare_classification_input(pat, ed)
        assert "Stack Trace Sample:" in result
        assert "frame1" in result


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplicatePatterns:
    def test_removes_duplicates(self):
        detector = _make_detector()
        p1 = _pattern(
            pattern_id="a",
            pattern_signature="assertion_error|expected_N_but_got_N|pytest",
        )
        p2 = _pattern(
            pattern_id="b",
            pattern_signature="assertion_error|expected_N_but_got_N|pytest",
        )
        result = detector._deduplicate_patterns([p1, p2])
        assert len(result) == 1

    def test_keeps_different_signatures(self):
        detector = _make_detector()
        p1 = _pattern(pattern_id="a", pattern_signature="sig_aaaaaaaaaa")
        p2 = _pattern(pattern_id="b", pattern_signature="sig_bbbbbbbbbb")
        result = detector._deduplicate_patterns([p1, p2])
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Learn from failure
# ---------------------------------------------------------------------------


class TestLearnFromFailure:
    async def test_adds_new_pattern(self):
        detector = _make_detector()
        fd = _failure_data()
        ed = _error_details()
        pat = _pattern()
        await detector._learn_from_failure(fd, ed, [pat])
        assert pat.pattern_id in detector._known_patterns
        assert len(detector._pattern_history) == 1

    async def test_increments_existing_pattern_frequency(self):
        detector = _make_detector()
        fd = _failure_data()
        ed = _error_details()
        pat = _pattern()
        # First learning
        await detector._learn_from_failure(fd, ed, [pat])
        initial_freq = detector._known_patterns[pat.pattern_id].frequency
        # Second learning with same pattern
        await detector._learn_from_failure(fd, ed, [pat])
        assert detector._known_patterns[pat.pattern_id].frequency == initial_freq + 1

    async def test_appends_failure_id_to_existing(self):
        detector = _make_detector()
        fd1 = _failure_data(failure_id="fail-1")
        fd2 = _failure_data(failure_id="fail-2")
        ed = _error_details()
        pat = _pattern()
        await detector._learn_from_failure(fd1, ed, [pat])
        await detector._learn_from_failure(fd2, ed, [pat])
        assoc = detector._known_patterns[pat.pattern_id].associated_failures
        assert "fail-2" in assoc

    async def test_updates_usage_count(self):
        detector = _make_detector()
        fd = _failure_data()
        ed = _error_details()
        pat = _pattern()
        await detector._learn_from_failure(fd, ed, [pat])
        assert detector._pattern_usage_count[pat.pattern_id] == 1

    async def test_handles_none_associated_failures(self):
        detector = _make_detector()
        fd = _failure_data(failure_id="fail-1")
        ed = _error_details()
        pat = _pattern()
        # Register pat first so it exists
        await detector._learn_from_failure(fd, ed, [pat])
        # Set associated_failures to None to hit the branch
        detector._known_patterns[pat.pattern_id].associated_failures = None
        fd2 = _failure_data(failure_id="fail-2")
        await detector._learn_from_failure(fd2, ed, [pat])
        # Should have created a new list
        assoc = detector._known_patterns[pat.pattern_id].associated_failures
        assert assoc is not None


# ---------------------------------------------------------------------------
# Cleanup old patterns
# ---------------------------------------------------------------------------


class TestCleanupOldPatterns:
    def test_removes_old_patterns(self):
        detector = _make_detector()
        # Add a pattern with old history
        pat = _pattern()
        detector._known_patterns[pat.pattern_id] = pat
        detector._pattern_signatures[pat.pattern_signature] = pat.pattern_id
        detector._pattern_usage_count[pat.pattern_id] = 5
        detector._pattern_accuracy[pat.pattern_id] = 0.9
        old_time = (datetime.now(UTC) - timedelta(days=60)).isoformat()
        detector._pattern_history = [
            {
                "timestamp": old_time,
                "failure_id": "f1",
                "error_type": "assertion_error",
                "error_message": "msg",
                "patterns": [pat.pattern_id],
            },
        ]
        detector._cleanup_old_patterns()
        assert pat.pattern_id not in detector._known_patterns
        assert pat.pattern_signature not in detector._pattern_signatures
        assert pat.pattern_id not in detector._pattern_usage_count
        assert pat.pattern_id not in detector._pattern_accuracy

    def test_keeps_recent_patterns(self):
        detector = _make_detector()
        pat = _pattern()
        detector._known_patterns[pat.pattern_id] = pat
        detector._pattern_signatures[pat.pattern_signature] = pat.pattern_id
        recent_time = datetime.now(UTC).isoformat()
        detector._pattern_history = [
            {
                "timestamp": recent_time,
                "failure_id": "f1",
                "error_type": "assertion_error",
                "error_message": "msg",
                "patterns": [pat.pattern_id],
            },
        ]
        detector._cleanup_old_patterns()
        assert pat.pattern_id in detector._known_patterns

    def test_handles_error_gracefully(self):
        detector = _make_detector()
        detector._pattern_history = [{"timestamp": "invalid-date"}]
        # Should not raise, just log error
        detector._cleanup_old_patterns()


# ---------------------------------------------------------------------------
# Statistics / Export / Import
# ---------------------------------------------------------------------------


class TestGetPatternStatistics:
    def test_empty(self):
        detector = _make_detector()
        stats = detector.get_pattern_statistics()
        assert stats["total_known_patterns"] == 0
        assert stats["total_history_entries"] == 0
        assert stats["similarity_threshold"] == 0.7

    def test_with_data(self):
        detector = _make_detector()
        pat = _pattern()
        detector._known_patterns[pat.pattern_id] = pat
        detector._pattern_usage_count[pat.pattern_id] = 3
        detector._pattern_accuracy[pat.pattern_id] = 0.85
        stats = detector.get_pattern_statistics()
        assert stats["total_known_patterns"] == 1
        assert len(stats["most_common_patterns"]) == 1


class TestExportPatterns:
    def test_export(self):
        detector = _make_detector()
        pat = _pattern()
        detector._known_patterns[pat.pattern_id] = pat
        detector._pattern_usage_count[pat.pattern_id] = 2
        detector._pattern_accuracy[pat.pattern_id] = 0.9
        detector._pattern_history = [{"foo": "bar"}]
        exported = detector.export_patterns()
        assert pat.pattern_id in exported["patterns"]
        assert exported["patterns"][pat.pattern_id]["pattern_name"] == pat.pattern_name
        assert exported["usage_count"][pat.pattern_id] == 2
        assert exported["history"] == [{"foo": "bar"}]


class TestImportPatterns:
    def test_import(self):
        detector = _make_detector()
        data = {
            "patterns": {
                "pat-imported": {
                    "pattern_name": "Imported Pattern Name",
                    "pattern_signature": "imported_sig_1234567890",
                    "pattern_type": "error_message",
                    "frequency": 5,
                    "confidence_score": 0.85,
                    "typical_causes": ["cause1"],
                    "common_fixes": ["fix1"],
                }
            },
            "history": [{"h": 1}],
            "usage_count": {"pat-imported": 10},
            "accuracy": {"pat-imported": 0.9},
        }
        detector.import_patterns(data)
        assert "pat-imported" in detector._known_patterns
        assert detector._known_patterns["pat-imported"].frequency == 5
        assert detector._pattern_signatures["imported_sig_1234567890"] == "pat-imported"
        assert detector._pattern_history == [{"h": 1}]
        assert detector._pattern_usage_count["pat-imported"] == 10
        assert detector._pattern_accuracy["pat-imported"] == 0.9

    def test_import_empty(self):
        detector = _make_detector()
        detector.import_patterns({})
        assert detector._known_patterns == {}

    def test_import_handles_error(self):
        detector = _make_detector()
        # Invalid data should not raise
        detector.import_patterns({"patterns": {"bad": "not_a_dict_with_required_keys"}})
        # Should have logged error but not crashed


# ---------------------------------------------------------------------------
# Detect patterns (with mocked AI agents)
# ---------------------------------------------------------------------------


class TestDetectPatterns:
    async def test_detect_patterns_returns_sorted(self):
        detector = _make_detector()

        # Mock the AI-calling methods
        pat_a = _pattern(
            pattern_id="a",
            pattern_signature="sig_aaaaaaaaaa",
            confidence_score=0.6,
        )
        pat_b = _pattern(
            pattern_id="b",
            pattern_signature="sig_bbbbbbbbbb",
            confidence_score=0.9,
        )
        detector._extract_patterns_ai = AsyncMock(return_value=[pat_a])
        detector._match_known_patterns = AsyncMock(return_value=[pat_b])
        detector._learn_from_failure = AsyncMock()

        fd = _failure_data()
        ed = _error_details()
        result = await detector.detect_patterns(fd, ed)
        assert len(result) == 2
        # Sorted by confidence_score descending
        assert result[0].confidence_score >= result[1].confidence_score

    async def test_detect_patterns_handles_exception(self):
        detector = _make_detector()
        detector._extract_patterns_ai = AsyncMock(side_effect=RuntimeError("boom"))
        fd = _failure_data()
        ed = _error_details()
        result = await detector.detect_patterns(fd, ed)
        assert result == []


class TestExtractPatternsAI:
    async def test_success(self):
        detector = _make_detector()
        pat = _pattern()
        mock_result = MagicMock()
        mock_result.output = [pat]
        detector.pattern_extractor = MagicMock()
        detector.pattern_extractor.run = AsyncMock(return_value=mock_result)
        detector._enhance_pattern = AsyncMock(side_effect=lambda p, fd, ed: p)

        fd = _failure_data()
        ed = _error_details()
        result = await detector._extract_patterns_ai(fd, ed)
        assert len(result) == 1

    async def test_handles_exception(self):
        detector = _make_detector()
        detector.pattern_extractor = MagicMock()
        detector.pattern_extractor.run = AsyncMock(side_effect=RuntimeError("fail"))

        fd = _failure_data()
        ed = _error_details()
        result = await detector._extract_patterns_ai(fd, ed)
        assert result == []


class TestMatchKnownPatterns:
    async def test_exact_signature_match(self):
        detector = _make_detector()
        fd = _failure_data()
        ed = _error_details()
        sig = detector._create_failure_signature(fd, ed)

        pat = _pattern(pattern_signature=sig)
        detector._known_patterns[pat.pattern_id] = pat
        detector._pattern_signatures[sig] = pat.pattern_id

        # Mock similarity analyzer to avoid AI call
        detector._calculate_pattern_similarity = AsyncMock(return_value=0.0)

        result = await detector._match_known_patterns(fd, ed)
        assert len(result) >= 1
        assert result[0].pattern_id == pat.pattern_id

    async def test_similar_pattern_match(self):
        detector = _make_detector()
        fd = _failure_data()
        ed = _error_details()

        known = _pattern(pattern_id="known-1", pattern_signature="different_sig_12345")
        detector._known_patterns["known-1"] = known

        # Mock similarity to be above threshold
        detector._calculate_pattern_similarity = AsyncMock(return_value=0.85)

        result = await detector._match_known_patterns(fd, ed)
        assert len(result) >= 1
        assert "similar_" in result[0].pattern_id

    async def test_no_match_below_threshold(self):
        detector = _make_detector()
        fd = _failure_data()
        ed = _error_details()

        known = _pattern(pattern_id="known-1", pattern_signature="different_sig_12345")
        detector._known_patterns["known-1"] = known

        detector._calculate_pattern_similarity = AsyncMock(return_value=0.3)

        result = await detector._match_known_patterns(fd, ed)
        assert result == []

    async def test_handles_exception(self):
        detector = _make_detector()
        fd = _failure_data()
        ed = _error_details()
        detector._known_patterns["x"] = _pattern()
        detector._calculate_pattern_similarity = AsyncMock(side_effect=RuntimeError("fail"))

        result = await detector._match_known_patterns(fd, ed)
        assert result == []


class TestEnhancePattern:
    async def test_sets_pattern_id(self):
        detector = _make_detector()
        pat = _pattern()
        fd = _failure_data()
        ed = _error_details()
        detector._classify_pattern_type = AsyncMock(return_value="error_message")
        result = await detector._enhance_pattern(pat, fd, ed)
        assert result.pattern_id.startswith("pattern_")

    async def test_appends_failure_id(self):
        detector = _make_detector()
        pat = _pattern(associated_failures=None)
        fd = _failure_data(failure_id="fail-xyz")
        ed = _error_details()
        detector._classify_pattern_type = AsyncMock(return_value="error_message")
        result = await detector._enhance_pattern(pat, fd, ed)
        assert "fail-xyz" in result.associated_failures

    async def test_classifies_unknown_type(self):
        detector = _make_detector()
        pat = _pattern(pattern_type="unknown")
        fd = _failure_data()
        ed = _error_details()
        detector._classify_pattern_type = AsyncMock(return_value="stack_trace")
        result = await detector._enhance_pattern(pat, fd, ed)
        assert result.pattern_type == "stack_trace"

    async def test_handles_exception(self):
        detector = _make_detector()
        pat = _pattern()
        fd = _failure_data()
        ed = _error_details()
        detector._classify_pattern_type = AsyncMock(side_effect=RuntimeError("fail"))
        result = await detector._enhance_pattern(pat, fd, ed)
        # Returns original pattern on error
        assert result is pat


class TestClassifyPatternType:
    async def test_success(self):
        detector = _make_detector()
        mock_result = MagicMock()
        mock_result.output = "assertion_pattern"
        detector.pattern_classifier = MagicMock()
        detector.pattern_classifier.run = AsyncMock(return_value=mock_result)

        pat = _pattern()
        ed = _error_details()
        result = await detector._classify_pattern_type(pat, ed)
        assert result == "assertion_pattern"

    async def test_handles_exception(self):
        detector = _make_detector()
        detector.pattern_classifier = MagicMock()
        detector.pattern_classifier.run = AsyncMock(side_effect=RuntimeError("fail"))

        pat = _pattern()
        ed = _error_details()
        result = await detector._classify_pattern_type(pat, ed)
        assert result == "unknown"


class TestCalculatePatternSimilarity:
    async def test_success(self):
        detector = _make_detector()
        mock_result = MagicMock()
        mock_result.output = 0.85
        detector.similarity_analyzer = MagicMock()
        detector.similarity_analyzer.run = AsyncMock(return_value=mock_result)

        fd = _failure_data()
        ed = _error_details()
        kp = _pattern()
        result = await detector._calculate_pattern_similarity(fd, ed, kp)
        assert result == 0.85

    async def test_handles_exception(self):
        detector = _make_detector()
        detector.similarity_analyzer = MagicMock()
        detector.similarity_analyzer.run = AsyncMock(side_effect=RuntimeError("fail"))

        fd = _failure_data()
        ed = _error_details()
        kp = _pattern()
        result = await detector._calculate_pattern_similarity(fd, ed, kp)
        assert result == 0.0


class TestEnumFieldsAreStringsAfterValidation:
    """Pin the model contract that ``.value`` in source code violates.

    ``use_enum_values=True`` means a validated model holds the *string*, not the
    enum member. Source code that calls ``.value`` on these fields raises
    ``AttributeError`` in production. This test fails the moment someone drops
    ``use_enum_values`` or reintroduces the assumption, so the guard is explicit
    rather than folded into behavioural tests.
    """

    def test_error_details_error_type_is_plain_str(self):
        ed = ErrorDetails(error_type=FailureType.TIMEOUT_ERROR, error_message="boom")
        assert type(ed.error_type) is str
        assert ed.error_type == "timeout_error"
        assert not hasattr(ed.error_type, "value")

    def test_failure_context_framework_is_plain_str(self):
        fd = _failure_data(framework=TestFramework.PYTEST)
        assert type(fd.context.framework) is str
        assert fd.context.framework == "pytest"
        assert not hasattr(fd.context.framework, "value")

    def test_pattern_confidence_is_plain_str(self):
        pat = _pattern(confidence=ConfidenceLevel.HIGH)
        assert type(pat.confidence) is str
        assert pat.confidence == "high"
        assert not hasattr(pat.confidence, "value")
