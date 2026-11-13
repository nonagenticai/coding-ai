"""Tests for AIFailureAnalysisService covering uncovered lines.

Focus: analyze_test_failure flow, validation, retry, fallback, caching,
circuit breaker, metrics, cleanup, and analysis_context.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.ai_test_failure_models import (
    AIAnalysisResult,
    ConfidenceLevel,
    ErrorDetails,
    FailureAnalysisRequest,
    FailureAnalysisResponse,
    FailureType,
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


def _make_error_details() -> ErrorDetails:
    return ErrorDetails(
        error_type=FailureType.ASSERTION_ERROR,
        error_message="expected True got False",
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


# ===================================================================
# analyze_test_failure - success path
# ===================================================================


class TestAnalyzeTestFailureSuccess:
    @pytest.mark.asyncio
    async def test_analyze_success_path(self):
        """Full success path through analyze_test_failure."""
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
        assert response.analysis_result is not None
        assert response.request_id == "req-1"

    @pytest.mark.asyncio
    async def test_analyze_caches_result(self):
        """Result is cached after successful analysis."""
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

        # First call populates cache
        resp1 = await svc.analyze_test_failure(request)
        assert resp1.success is True

        # Second call should hit cache
        svc._active_analyses.clear()
        resp2 = await svc.analyze_test_failure(request)
        assert resp2.success is True
        assert svc._metrics["cache_hits"] >= 1

    @pytest.mark.asyncio
    async def test_degraded_result_propagates_and_is_not_cached(self):
        """LLM outage: analyzer returns degraded -> service response degraded, no cache."""
        svc = _build_service()
        request = _make_request()
        result = _make_analysis_result(request.failure_data)
        result.additional_metadata = {"degraded": True}

        svc.analyzer = MagicMock()
        svc.analyzer.analyze_failure = AsyncMock(
            return_value=FailureAnalysisResponse(
                request_id=request.request_id,
                success=True,
                degraded=True,
                analysis_result=result,
                processing_time=0.1,
            )
        )

        response = await svc.analyze_test_failure(request)
        assert response.success is True
        assert response.degraded is True
        # A degraded (outage) result must not be cached so recovery re-runs.
        assert svc._result_cache == {}

    @pytest.mark.asyncio
    async def test_genuine_empty_is_not_degraded(self):
        """LLM ran, found nothing: success + degraded=False (no metadata flag)."""
        svc = _build_service()
        request = _make_request()
        result = _make_analysis_result(request.failure_data)  # additional_metadata=None

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
        assert response.degraded is False


# ===================================================================
# analyze_test_failure - error paths
# ===================================================================


class TestAnalyzeTestFailureErrors:
    @pytest.mark.asyncio
    async def test_timeout_error_path(self):
        """AIAnalysisTimeoutError is caught and returns error response."""
        from services.ai_failure_analysis import AIAnalysisTimeoutError

        svc = _build_service()
        request = _make_request()

        # Must mock _perform_analysis_with_retry since _perform_core_analysis
        # catches all exceptions and falls back.
        with patch.object(
            svc,
            "_perform_analysis_with_retry",
            side_effect=AIAnalysisTimeoutError("timed out"),
        ):
            response = await svc.analyze_test_failure(request)
        assert response.success is False
        assert "timeout" in response.error_message.lower()
        assert svc._metrics["timeout_errors"] >= 1

    @pytest.mark.asyncio
    async def test_model_unavailable_circuit_breaker(self):
        """Circuit breaker open returns model unavailable error."""
        svc = _build_service()
        request = _make_request()

        # Force circuit breaker open
        svc._circuit_breaker_failures = 100
        svc._circuit_breaker_last_failure = datetime.now(UTC)

        response = await svc.analyze_test_failure(request)
        assert response.success is False
        assert "circuit breaker" in response.error_message.lower()

    @pytest.mark.asyncio
    async def test_validation_error_path(self):
        """AIAnalysisValidationError is caught and returns error response."""
        from services.ai_failure_analysis import AIAnalysisValidationError

        svc = _build_service()
        request = _make_request()

        with patch.object(
            svc, "_validate_request", side_effect=AIAnalysisValidationError("Invalid")
        ):
            response = await svc.analyze_test_failure(request)
        assert response.success is False
        assert "validation" in response.error_message.lower()

    @pytest.mark.asyncio
    async def test_unexpected_error_updates_circuit_breaker(self):
        """Unexpected errors update the circuit breaker."""
        svc = _build_service()
        request = _make_request()

        initial_failures = svc._circuit_breaker_failures

        # Must mock _perform_analysis_with_retry to raise a non-AI exception
        with patch.object(
            svc,
            "_perform_analysis_with_retry",
            side_effect=RuntimeError("unexpected boom"),
        ):
            response = await svc.analyze_test_failure(request)
        assert response.success is False
        assert svc._circuit_breaker_failures > initial_failures
        assert "unexpected boom" in response.error_message


# ===================================================================
# _validate_request
# ===================================================================


class TestValidateRequest:
    @pytest.mark.asyncio
    async def test_validate_missing_context(self):
        """Missing context raises validation error."""
        from services.ai_failure_analysis import AIAnalysisValidationError

        svc = _build_service()
        request = _make_request()
        request.failure_data.context = None  # type: ignore[assignment]

        with pytest.raises(AIAnalysisValidationError, match="Missing context"):
            await svc._validate_request(request)

    @pytest.mark.asyncio
    async def test_validate_raw_output_too_large(self):
        """Raw output exceeding 100KB raises validation error."""
        from services.ai_failure_analysis import AIAnalysisValidationError

        svc = _build_service()
        request = _make_request(raw_output="x" * 1000)
        # Bypass model validation to set oversized output
        object.__setattr__(request.failure_data, "raw_output", "x" * 100_001)

        with pytest.raises(AIAnalysisValidationError, match="size limit"):
            await svc._validate_request(request)

    @pytest.mark.asyncio
    async def test_validate_too_many_concurrent(self):
        """Too many concurrent analyses raises validation error."""
        from services.ai_failure_analysis import AIAnalysisValidationError

        svc = _build_service()
        for i in range(svc.max_concurrent_analyses):
            svc._active_analyses[f"req-{i}"] = datetime.now(UTC)

        request = _make_request()
        with pytest.raises(AIAnalysisValidationError, match="Too many concurrent"):
            await svc._validate_request(request)

    @pytest.mark.asyncio
    async def test_validate_success(self):
        """Valid request passes validation and is tracked."""
        svc = _build_service()
        request = _make_request()
        await svc._validate_request(request)
        assert request.request_id in svc._active_analyses


# ===================================================================
# _perform_analysis_with_retry
# ===================================================================


class TestPerformAnalysisWithRetry:
    @pytest.mark.asyncio
    async def test_retry_on_timeout(self):
        """Retry logic handles TimeoutError and eventually raises."""
        from services.ai_failure_analysis import AIAnalysisTimeoutError

        svc = _build_service()
        svc._max_retries = 2
        svc._retry_delay = 0.001
        svc.analysis_timeout = 0.001

        request = _make_request()

        async def slow_analysis(_req):
            await asyncio.sleep(10)

        with patch.object(svc, "_perform_core_analysis", side_effect=slow_analysis):
            with pytest.raises(AIAnalysisTimeoutError):
                await svc._perform_analysis_with_retry(request)

    @pytest.mark.asyncio
    async def test_retry_on_generic_error(self):
        """Retry logic handles generic errors and retries."""
        from services.ai_failure_analysis import AIFailureAnalysisError

        svc = _build_service()
        svc._max_retries = 2
        svc._retry_delay = 0.001

        request = _make_request()

        with patch.object(svc, "_perform_core_analysis", side_effect=RuntimeError("fail")):
            with pytest.raises(AIFailureAnalysisError, match="failed after"):
                await svc._perform_analysis_with_retry(request)

    @pytest.mark.asyncio
    async def test_no_result_raises_validation_error(self):
        """None result from core analysis raises validation error."""
        from services.ai_failure_analysis import AIAnalysisValidationError

        svc = _build_service()
        svc._max_retries = 1
        svc._retry_delay = 0.001

        request = _make_request()

        async def return_none(_req):
            return None

        with patch.object(svc, "_perform_core_analysis", side_effect=return_none):
            with pytest.raises(AIAnalysisValidationError, match="no results"):
                await svc._perform_analysis_with_retry(request)

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        """Successful analysis on first attempt."""
        svc = _build_service()
        request = _make_request()
        result = _make_analysis_result(request.failure_data)

        async def success(_req):
            return result

        with patch.object(svc, "_perform_core_analysis", side_effect=success):
            returned = await svc._perform_analysis_with_retry(request)
        assert returned is result


# ===================================================================
# _perform_core_analysis and _perform_fallback_analysis
# ===================================================================


class TestCoreAndFallbackAnalysis:
    @pytest.mark.asyncio
    async def test_core_analysis_no_analyzer(self):
        """If analyzer is None, fallback is used."""
        svc = _build_service()
        svc.analyzer = None

        request = _make_request()
        result = await svc._perform_core_analysis(request)
        assert result is not None
        assert result.additional_metadata is not None
        assert result.additional_metadata.get("fallback_analysis") is True

    @pytest.mark.asyncio
    async def test_core_analysis_failed_response(self):
        """If analyzer returns success=False, fallback is used."""
        svc = _build_service()
        request = _make_request()

        svc.analyzer = MagicMock()
        svc.analyzer.analyze_failure = AsyncMock(
            return_value=FailureAnalysisResponse(
                request_id="req-1",
                success=False,
                error_message="AI failed",
                processing_time=0.1,
            )
        )

        result = await svc._perform_core_analysis(request)
        assert result is not None
        assert result.additional_metadata is not None
        assert result.additional_metadata.get("fallback_analysis") is True

    @pytest.mark.asyncio
    async def test_core_analysis_no_result(self):
        """If analyzer returns success=True but no result, fallback is used."""
        svc = _build_service()
        request = _make_request()

        svc.analyzer = MagicMock()
        svc.analyzer.analyze_failure = AsyncMock(
            return_value=FailureAnalysisResponse(
                request_id="req-1",
                success=True,
                analysis_result=None,
                processing_time=0.1,
            )
        )

        result = await svc._perform_core_analysis(request)
        assert result is not None
        assert result.additional_metadata is not None
        assert result.additional_metadata.get("fallback_analysis") is True

    @pytest.mark.asyncio
    async def test_core_analysis_success(self):
        """If analyzer returns success with result, result is returned."""
        svc = _build_service()
        request = _make_request()
        expected = _make_analysis_result(request.failure_data)

        svc.analyzer = MagicMock()
        svc.analyzer.analyze_failure = AsyncMock(
            return_value=FailureAnalysisResponse(
                request_id="req-1",
                success=True,
                analysis_result=expected,
                processing_time=0.1,
            )
        )

        result = await svc._perform_core_analysis(request)
        assert result is expected

    @pytest.mark.asyncio
    async def test_fallback_analysis_failure(self):
        """Fallback analysis failure raises."""
        from services.ai_failure_analysis import AIFailureAnalysisError

        svc = _build_service()
        request = _make_request()

        with patch.object(svc, "_extract_basic_error_details", side_effect=RuntimeError("bad")):
            with pytest.raises(AIFailureAnalysisError, match="Both main and fallback"):
                await svc._perform_fallback_analysis(request)

    @pytest.mark.asyncio
    async def test_fallback_analysis_success(self):
        """Fallback analysis produces valid result."""
        svc = _build_service()
        request = _make_request()
        result = await svc._perform_fallback_analysis(request)
        assert result is not None
        assert result.analysis_confidence_score == 0.1
        assert result.additional_metadata.get("fallback_analysis") is True


# ===================================================================
# _extract_basic_error_details
# ===================================================================


class TestExtractBasicErrorDetails:
    @pytest.mark.asyncio
    async def test_assertion_error_detection(self):
        fd = _make_failure_data(raw_output="AssertionError: values differ")
        svc = _build_service()
        details = await svc._extract_basic_error_details(fd)
        assert details.error_type == FailureType.ASSERTION_ERROR

    @pytest.mark.asyncio
    async def test_import_error_detection(self):
        fd = _make_failure_data(raw_output="ImportError: No module named foo")
        svc = _build_service()
        details = await svc._extract_basic_error_details(fd)
        assert details.error_type == FailureType.IMPORT_ERROR

    @pytest.mark.asyncio
    async def test_module_not_found_detection(self):
        fd = _make_failure_data(raw_output="ModuleNotFoundError: No module named bar")
        svc = _build_service()
        details = await svc._extract_basic_error_details(fd)
        assert details.error_type == FailureType.IMPORT_ERROR

    @pytest.mark.asyncio
    async def test_syntax_error_detection(self):
        fd = _make_failure_data(raw_output="SyntaxError: invalid syntax")
        svc = _build_service()
        details = await svc._extract_basic_error_details(fd)
        assert details.error_type == FailureType.SYNTAX_ERROR

    @pytest.mark.asyncio
    async def test_timeout_error_detection(self):
        fd = _make_failure_data(raw_output="TimeoutError: operation timed out")
        svc = _build_service()
        details = await svc._extract_basic_error_details(fd)
        assert details.error_type == FailureType.TIMEOUT_ERROR

    @pytest.mark.asyncio
    async def test_timeout_lowercase_detection(self):
        fd = _make_failure_data(raw_output="test timeout during execution")
        svc = _build_service()
        details = await svc._extract_basic_error_details(fd)
        assert details.error_type == FailureType.TIMEOUT_ERROR

    @pytest.mark.asyncio
    async def test_runtime_error_fallback(self):
        fd = _make_failure_data(raw_output="ValueError: something bad happened")
        svc = _build_service()
        details = await svc._extract_basic_error_details(fd)
        assert details.error_type == FailureType.RUNTIME_ERROR

    @pytest.mark.asyncio
    async def test_error_message_extraction_from_lines(self):
        fd = _make_failure_data(raw_output="some setup\nValueError: bad value\nmore output")
        svc = _build_service()
        details = await svc._extract_basic_error_details(fd)
        assert "ValueError" in details.error_message or "bad value" in details.error_message

    @pytest.mark.asyncio
    async def test_unknown_error_message(self):
        fd = _make_failure_data(raw_output="all good no problems")
        svc = _build_service()
        details = await svc._extract_basic_error_details(fd)
        assert details.error_message == "Unknown error"

    @pytest.mark.asyncio
    async def test_traceback_truncation(self):
        fd = _make_failure_data(raw_output="x" * 2000)
        svc = _build_service()
        details = await svc._extract_basic_error_details(fd)
        assert details.traceback is not None
        assert len(details.traceback) <= 1000


# ===================================================================
# Caching
# ===================================================================


class TestCaching:
    def test_get_cached_result_miss(self):
        svc = _build_service()
        request = _make_request()
        result = svc._get_cached_result(request)
        assert result is None
        assert svc._metrics["cache_misses"] == 1

    def test_get_cached_result_hit(self):
        svc = _build_service()
        request = _make_request()
        analysis = _make_analysis_result(request.failure_data)

        svc._cache_result(request, analysis)
        result = svc._get_cached_result(request)
        assert result is not None
        assert svc._metrics["cache_hits"] == 1

    def test_get_cached_result_expired(self):
        svc = _build_service()
        svc.cache_ttl = 0
        request = _make_request()
        analysis = _make_analysis_result(request.failure_data)

        cache_key = svc._create_cache_key(request)
        svc._result_cache[cache_key] = (analysis, datetime.now(UTC) - timedelta(seconds=10))

        result = svc._get_cached_result(request)
        assert result is None
        assert cache_key not in svc._result_cache

    def test_cache_result_stores_and_cleans(self):
        svc = _build_service()
        request = _make_request()
        analysis = _make_analysis_result(request.failure_data)

        svc._cache_result(request, analysis)
        assert len(svc._result_cache) == 1

    def test_cleanup_cache_removes_expired(self):
        svc = _build_service()
        svc.cache_ttl = 0
        analysis = _make_analysis_result()

        svc._result_cache["old-key"] = (analysis, datetime.now(UTC) - timedelta(seconds=10))
        svc._cleanup_cache()
        assert len(svc._result_cache) == 0

    def test_create_cache_key_deterministic(self):
        svc = _build_service()
        request = _make_request()
        key1 = svc._create_cache_key(request)
        key2 = svc._create_cache_key(request)
        assert key1 == key2
        assert "|" in key1


# ===================================================================
# Circuit Breaker
# ===================================================================


class TestCircuitBreaker:
    def test_circuit_breaker_closed(self):
        svc = _build_service()
        assert svc._is_circuit_breaker_open() is False

    def test_circuit_breaker_opens_after_threshold(self):
        svc = _build_service()
        svc._circuit_breaker_failures = svc._circuit_breaker_threshold
        svc._circuit_breaker_last_failure = datetime.now(UTC)
        assert svc._is_circuit_breaker_open() is True

    def test_circuit_breaker_half_open_after_timeout(self):
        svc = _build_service()
        svc._circuit_breaker_failures = svc._circuit_breaker_threshold
        svc._circuit_breaker_last_failure = datetime.now(UTC) - timedelta(
            seconds=svc._circuit_breaker_timeout + 10
        )
        assert svc._is_circuit_breaker_open() is False

    def test_circuit_breaker_no_last_failure(self):
        svc = _build_service()
        svc._circuit_breaker_failures = 100
        svc._circuit_breaker_last_failure = None
        assert svc._is_circuit_breaker_open() is False

    def test_update_circuit_breaker(self):
        svc = _build_service()
        svc._update_circuit_breaker()
        assert svc._circuit_breaker_failures == 1
        assert svc._circuit_breaker_last_failure is not None

    def test_update_circuit_breaker_logs_warning_at_threshold(self):
        svc = _build_service()
        svc._circuit_breaker_failures = svc._circuit_breaker_threshold - 1
        svc._update_circuit_breaker()
        assert svc._circuit_breaker_failures == svc._circuit_breaker_threshold

    def test_reset_circuit_breaker(self):
        svc = _build_service()
        svc._circuit_breaker_failures = 5
        svc._circuit_breaker_last_failure = datetime.now(UTC)
        svc._reset_circuit_breaker()
        assert svc._circuit_breaker_failures == 0
        assert svc._circuit_breaker_last_failure is None

    def test_reset_circuit_breaker_noop_when_zero(self):
        svc = _build_service()
        svc._reset_circuit_breaker()
        assert svc._circuit_breaker_failures == 0


# ===================================================================
# Metrics
# ===================================================================


class TestMetrics:
    def test_update_metrics_disabled(self):
        svc = _build_service()
        svc.enable_metrics = False
        svc._update_metrics(success=True, duration=1.0)
        assert svc._metrics["total_analyses"] == 0

    def test_update_metrics_success(self):
        svc = _build_service()
        svc._update_metrics(success=True, duration=2.0)
        assert svc._metrics["total_analyses"] == 1
        assert svc._metrics["successful_analyses"] == 1
        assert svc._metrics["average_analysis_time"] == 2.0

    def test_update_metrics_failure_with_type(self):
        svc = _build_service()
        svc._update_metrics(success=False, duration=1.0, error_type="timeout")
        assert svc._metrics["failed_analyses"] == 1
        assert svc._metrics["timeout_errors"] == 1

    def test_update_metrics_failure_without_type(self):
        svc = _build_service()
        svc._update_metrics(success=False, duration=1.0)
        assert svc._metrics["failed_analyses"] == 1


# ===================================================================
# Response creation
# ===================================================================


class TestResponseCreation:
    def test_create_success_response(self):
        svc = _build_service()
        result = _make_analysis_result()
        response = svc._create_success_response("req-1", result, 1.5)
        assert response.success is True
        assert response.request_id == "req-1"
        assert response.analysis_result is not None
        assert response.processing_time == 1.5

    def test_create_error_response(self):
        svc = _build_service()
        response = svc._create_error_response("req-1", "something failed", 0.5)
        assert response.success is False
        assert response.error_message == "something failed"


# ===================================================================
# Cleanup and analysis_context
# ===================================================================


class TestCleanupAndContext:
    @pytest.mark.asyncio
    async def test_cleanup(self):
        svc = _build_service()
        svc._active_analyses["req-1"] = datetime.now(UTC)
        svc._result_cache["key"] = (_make_analysis_result(), datetime.now(UTC))

        await svc.cleanup()
        assert len(svc._active_analyses) == 0
        assert len(svc._result_cache) == 0

    @pytest.mark.asyncio
    async def test_analysis_context_manager(self):
        svc = _build_service()
        svc._active_analyses["req-1"] = datetime.now(UTC)

        async with svc.analysis_context("req-1"):
            assert "req-1" in svc._active_analyses

        assert "req-1" not in svc._active_analyses

    @pytest.mark.asyncio
    async def test_analysis_context_manager_missing_request(self):
        svc = _build_service()

        async with svc.analysis_context("nonexistent"):
            pass
        # Should not raise
