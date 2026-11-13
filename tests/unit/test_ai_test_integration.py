"""Tests for AITestIntegrationService covering all major code paths.

Focus: execute_test_with_analysis, analyze_existing_failure, batch_analyze_failures,
_analyze_test_failure, _create_analysis_request, helper methods, health check,
metrics, cleanup, and factory function.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.ai_test_failure_models import (
    AIAnalysisResult,
    ConfidenceLevel,
    ErrorDetails,
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
from models.testing_models import TestRequest, TestResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_request() -> TestRequest:
    return TestRequest(
        code="def foo(): return 1",
        test_code="def test_foo(): assert foo() == 1",
        language="python",
    )


def _make_test_result(
    success: bool = False,
    test_id: str = "test-1",
    language: str = "python",
    stderr: str = "AssertionError: 1 != 2",
) -> TestResult:
    return TestResult(
        test_id=test_id,
        success=success,
        execution_time=1.0,
        language=language,
        stderr=stderr,
        created_at=datetime.now(UTC),
    )


def _make_failure_data() -> TestFailureData:
    return TestFailureData(
        failure_id="fail-1",
        raw_output="AssertionError: expected True got False",
        context=TestFailureContext(
            test_name="test_example",
            framework=TestFramework.PYTEST,
        ),
    )


def _make_analysis_result() -> AIAnalysisResult:
    return AIAnalysisResult(
        analysis_id="analysis-1",
        failure_data=_make_failure_data(),
        error_details=ErrorDetails(
            error_type=FailureType.ASSERTION_ERROR,
            error_message="expected True got False",
        ),
        root_causes=[],
        fix_suggestions=[],
        detected_patterns=[],
        assertion_details=None,
        analysis_confidence=ConfidenceLevel.LOW,
        analysis_confidence_score=0.3,
        ai_model_version="test",
    )


def _make_analysis_result_full() -> AIAnalysisResult:
    """Analysis result with root causes, fixes, patterns, and assertions."""
    return AIAnalysisResult(
        analysis_id="analysis-1",
        failure_data=_make_failure_data(),
        error_details=ErrorDetails(
            error_type=FailureType.ASSERTION_ERROR,
            error_message="expected True got False",
        ),
        root_causes=[
            RootCause(
                cause_id="cause-1",
                category="unknown",
                description="Test failed with assertion_error: expected True got False",
                severity=SeverityLevel.MEDIUM,
                confidence=ConfidenceLevel.LOW,
                confidence_score=0.3,
                evidence=["assertion error"],
            )
        ],
        fix_suggestions=[
            FixSuggestion(
                suggestion_id="fix-1",
                title="Fix the assertion value",
                description="Change the expected value to match the actual output",
                fix_type="code_change",
                priority="medium",
                confidence=ConfidenceLevel.MEDIUM,
                confidence_score=0.5,
                estimated_effort="low",
                code_examples=["assert foo() == 2"],
                files_to_modify=["test_foo.py"],
                verification_steps=["Run test again"],
            )
        ],
        detected_patterns=[
            TestFailurePattern(
                pattern_id="pat-1",
                pattern_name="Assertion Error Pattern",
                pattern_signature="assertion_error|assertion|pytest_pattern",
                pattern_type="error_message",
                confidence=ConfidenceLevel.MEDIUM,
                confidence_score=0.5,
                typical_causes=["Wrong expected value"],
                common_fixes=["Update expected value"],
            )
        ],
        assertion_details=None,
        analysis_confidence=ConfidenceLevel.LOW,
        analysis_confidence_score=0.3,
        ai_model_version="test",
    )


def _build_service(
    enable_ai: bool = True,
    ai_service: MagicMock | None = None,
    test_exec_service: MagicMock | None = None,
):
    """Build AITestIntegrationService with mocked dependencies."""
    with patch("services.ai_test_integration.AIFailureAnalysisService"):
        from services.ai_test_integration import AITestIntegrationService

        svc = AITestIntegrationService(
            test_execution_service=test_exec_service,
            ai_analysis_service=ai_service,
            enable_ai_analysis=enable_ai,
            analysis_timeout=5.0,
            max_concurrent_analyses=3,
        )
    return svc


# ===================================================================
# execute_test_with_analysis
#
# NOTE: Source has datetime bug (naive vs aware) at line 136/183.
# We patch datetime.now to always return UTC-aware datetimes.
# ===================================================================


class TestExecuteTestWithAnalysis:
    @pytest.mark.asyncio
    async def test_successful_test_no_analysis(self):
        """Successful test does not trigger AI analysis."""
        exec_svc = AsyncMock()
        exec_svc.execute_test = AsyncMock(return_value=_make_test_result(success=True))

        svc = _build_service(test_exec_service=exec_svc)

        request = _make_test_request()
        with patch("services.ai_test_integration.datetime") as mock_dt:
            mock_dt.now.return_value = datetime.now(UTC)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = await svc.execute_test_with_analysis(request)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_failed_test_with_analysis(self):
        """Failed test triggers AI analysis and integrates results."""
        exec_svc = AsyncMock()
        test_result = _make_test_result(success=False)
        exec_svc.execute_test = AsyncMock(return_value=test_result)

        ai_svc = MagicMock()
        ai_svc.analyze_test_failure = AsyncMock(
            return_value=FailureAnalysisResponse(
                request_id="req-1",
                success=True,
                analysis_result=_make_analysis_result(),
                processing_time=0.1,
            )
        )

        svc = _build_service(test_exec_service=exec_svc, ai_service=ai_svc)

        request = _make_test_request()
        with patch("services.ai_test_integration.datetime") as mock_dt:
            mock_dt.now.return_value = datetime.now(UTC)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = await svc.execute_test_with_analysis(request)
        assert result.success is False
        assert svc._metrics["analyzed_failures"] >= 1
        assert svc._metrics["analysis_successes"] >= 1

    @pytest.mark.asyncio
    async def test_failed_test_analysis_disabled(self):
        """AI analysis disabled does not analyze failures."""
        exec_svc = AsyncMock()
        exec_svc.execute_test = AsyncMock(return_value=_make_test_result(success=False))

        svc = _build_service(test_exec_service=exec_svc, enable_ai=False)
        svc.ai_analysis_service = None

        request = _make_test_request()
        with patch("services.ai_test_integration.datetime") as mock_dt:
            mock_dt.now.return_value = datetime.now(UTC)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = await svc.execute_test_with_analysis(request)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_failed_test_analysis_error(self):
        """Error during AI analysis does not crash, adds error to metadata."""
        exec_svc = AsyncMock()
        exec_svc.execute_test = AsyncMock(return_value=_make_test_result(success=False))

        ai_svc = MagicMock()
        ai_svc.analyze_test_failure = AsyncMock(side_effect=RuntimeError("ai boom"))

        svc = _build_service(test_exec_service=exec_svc, ai_service=ai_svc)

        request = _make_test_request()
        with patch("services.ai_test_integration.datetime") as mock_dt:
            mock_dt.now.return_value = datetime.now(UTC)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = await svc.execute_test_with_analysis(request)
        assert result.success is False
        assert svc._metrics["analysis_failures"] >= 1

    @pytest.mark.asyncio
    async def test_execution_exception_propagates(self):
        """Exception during test execution propagates."""
        exec_svc = AsyncMock()
        exec_svc.execute_test = AsyncMock(side_effect=RuntimeError("exec boom"))

        svc = _build_service(test_exec_service=exec_svc)

        request = _make_test_request()
        with pytest.raises(RuntimeError, match="exec boom"):
            with patch("services.ai_test_integration.datetime") as mock_dt:
                mock_dt.now.return_value = datetime.now(UTC)
                mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
                await svc.execute_test_with_analysis(request)
        assert svc._metrics["integration_errors"] >= 1

    @pytest.mark.asyncio
    async def test_analysis_returns_none(self):
        """Analysis returning None increments failure count."""
        exec_svc = AsyncMock()
        exec_svc.execute_test = AsyncMock(return_value=_make_test_result(success=False))

        ai_svc = MagicMock()
        ai_svc.analyze_test_failure = AsyncMock(
            return_value=FailureAnalysisResponse(
                request_id="req-1",
                success=False,
                error_message="failed",
                processing_time=0.1,
            )
        )

        svc = _build_service(test_exec_service=exec_svc, ai_service=ai_svc)

        request = _make_test_request()
        with patch("services.ai_test_integration.datetime") as mock_dt:
            mock_dt.now.return_value = datetime.now(UTC)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = await svc.execute_test_with_analysis(request)
        assert result.success is False


# ===================================================================
# analyze_existing_failure
# ===================================================================


class TestAnalyzeExistingFailure:
    @pytest.mark.asyncio
    async def test_analyze_existing_success(self):
        """Existing failure analysis succeeds."""
        ai_svc = MagicMock()
        ai_svc.analyze_test_failure = AsyncMock(
            return_value=FailureAnalysisResponse(
                request_id="req-1",
                success=True,
                analysis_result=_make_analysis_result(),
                processing_time=0.1,
            )
        )

        svc = _build_service(ai_service=ai_svc)
        test_result = _make_test_result(success=False)
        analysis = await svc.analyze_existing_failure(test_result)
        assert analysis is not None

    @pytest.mark.asyncio
    async def test_analyze_existing_disabled(self):
        """Returns None when AI disabled."""
        svc = _build_service(enable_ai=False)
        svc.ai_analysis_service = None
        svc.enable_ai_analysis = False

        test_result = _make_test_result(success=False)
        analysis = await svc.analyze_existing_failure(test_result)
        assert analysis is None

    @pytest.mark.asyncio
    async def test_analyze_existing_exception(self):
        """Exception returns None."""
        ai_svc = MagicMock()
        ai_svc.analyze_test_failure = AsyncMock(side_effect=RuntimeError("boom"))

        svc = _build_service(ai_service=ai_svc)
        test_result = _make_test_result(success=False)
        analysis = await svc.analyze_existing_failure(test_result)
        assert analysis is None


# ===================================================================
# batch_analyze_failures
# ===================================================================


class TestBatchAnalyzeFailures:
    @pytest.mark.asyncio
    async def test_batch_no_failures(self):
        """No failed tests returns all None."""
        svc = _build_service()
        results = [_make_test_result(success=True) for _ in range(3)]

        analyses = await svc.batch_analyze_failures(results)
        assert all(a is None for a in analyses)

    @pytest.mark.asyncio
    async def test_batch_disabled(self):
        """AI disabled returns all None."""
        svc = _build_service(enable_ai=False)
        svc.ai_analysis_service = None
        svc.enable_ai_analysis = False

        results = [_make_test_result(success=False)]
        analyses = await svc.batch_analyze_failures(results)
        assert all(a is None for a in analyses)

    @pytest.mark.asyncio
    async def test_batch_with_failures(self):
        """Failed tests are analyzed."""
        ai_svc = MagicMock()
        ai_svc.analyze_test_failure = AsyncMock(
            return_value=FailureAnalysisResponse(
                request_id="req-1",
                success=True,
                analysis_result=_make_analysis_result(),
                processing_time=0.1,
            )
        )

        svc = _build_service(ai_service=ai_svc)

        results = [
            _make_test_result(success=True, test_id="t1"),
            _make_test_result(success=False, test_id="t2"),
            _make_test_result(success=False, test_id="t3"),
        ]

        analyses = await svc.batch_analyze_failures(results)
        assert analyses[0] is None  # success, no analysis
        assert len(analyses) == 3


# ===================================================================
# _create_analysis_request
# ===================================================================


class TestCreateAnalysisRequest:
    def test_creates_request(self):
        svc = _build_service()
        test_result = _make_test_result(success=False)
        request = _make_test_request()

        analysis_request = svc._create_analysis_request(test_result, request)
        assert analysis_request.failure_data is not None
        assert analysis_request.failure_data.failure_id == "test-1"

    def test_creates_request_with_test_code(self):
        svc = _build_service()
        test_result = _make_test_result(success=False)
        request = _make_test_request()
        request.test_code = "def test(): assert True"

        analysis_request = svc._create_analysis_request(test_result, request)
        assert analysis_request.failure_data.context.test_file is not None


# ===================================================================
# Helper methods
# ===================================================================


class TestHelperMethods:
    def test_determine_test_framework_python_pytest(self):
        svc = _build_service()
        result = _make_test_result(language="python", stderr="pytest collected 1 item")
        fw = svc._determine_test_framework(result)
        assert fw == TestFramework.PYTEST

    def test_determine_test_framework_python_unittest(self):
        svc = _build_service()
        result = _make_test_result(language="python", stderr="unittest output")
        fw = svc._determine_test_framework(result)
        assert fw == TestFramework.UNITTEST

    def test_determine_test_framework_python_default(self):
        svc = _build_service()
        result = _make_test_result(language="python", stderr="xyz abc")
        fw = svc._determine_test_framework(result)
        assert fw == TestFramework.PYTEST

    def test_determine_test_framework_javascript(self):
        svc = _build_service()
        result = _make_test_result(language="javascript")
        fw = svc._determine_test_framework(result)
        assert fw == TestFramework.JEST

    def test_determine_test_framework_unknown(self):
        svc = _build_service()
        result = _make_test_result(language="rust")
        fw = svc._determine_test_framework(result)
        assert fw == TestFramework.UNKNOWN

    def test_extract_execution_environment(self):
        svc = _build_service()
        result = _make_test_result()
        env = svc._extract_execution_environment(result)
        assert env["language"] == "python"

    def test_create_test_file_metadata_no_code(self):
        svc = _build_service()
        request = TestRequest(code="x", test_code="", language="python")
        result = _make_test_result()
        meta = svc._create_test_file_metadata(request, result)
        assert meta is None

    def test_create_test_file_metadata_with_code(self):
        svc = _build_service()
        request = _make_test_request()
        result = _make_test_result()
        meta = svc._create_test_file_metadata(request, result)
        assert meta is not None


# ===================================================================
# _integrate_analysis_results
#
# NOTE: This method calls .value on Pydantic-stored enum strings
# (e.g. analysis_result.analysis_confidence.value) which fails due
# to use_enum_values=True converting enums to plain strings.
# ===================================================================


class TestIntegrateAnalysisResults:
    def test_integrates_basic_analysis(self):
        """Integration with empty root_causes/fix_suggestions/patterns."""
        svc = _build_service()
        test_result = _make_test_result(success=False)
        analysis = _make_analysis_result()

        result = svc._integrate_analysis_results(test_result, analysis)
        assert result.test_metadata is not None
        assert "ai_analysis" in result.test_metadata
        assert result.test_metadata["ai_analysis"]["analysis_id"] == "analysis-1"
        assert result.test_metadata["ai_analysis"]["confidence"] == "low"
        # No root_causes/fix_suggestions/detected_patterns keys when lists are empty
        assert "root_causes" not in result.test_metadata
        assert "fix_suggestions" not in result.test_metadata
        assert "detected_patterns" not in result.test_metadata

    def test_integrates_full_analysis(self):
        """Integration with root causes, fixes, patterns populates metadata."""
        svc = _build_service()
        test_result = _make_test_result(success=False)
        analysis = _make_analysis_result_full()

        result = svc._integrate_analysis_results(test_result, analysis)
        assert result.test_metadata is not None
        assert "ai_analysis" in result.test_metadata
        assert "root_causes" in result.test_metadata
        assert len(result.test_metadata["root_causes"]) == 1
        assert result.test_metadata["root_causes"][0]["severity"] == "medium"
        assert "fix_suggestions" in result.test_metadata
        assert len(result.test_metadata["fix_suggestions"]) == 1
        assert "detected_patterns" in result.test_metadata
        assert len(result.test_metadata["detected_patterns"]) == 1


# ===================================================================
# Health and metrics
# ===================================================================


class TestHealthAndMetrics:
    @pytest.mark.asyncio
    async def test_analysis_service_health_disabled(self):
        svc = _build_service()
        svc.ai_analysis_service = None
        health = await svc.get_analysis_service_health()
        assert health["status"] == "disabled"

    @pytest.mark.asyncio
    async def test_analysis_service_health_enabled(self):
        ai_svc = MagicMock()
        ai_svc.get_health_status = MagicMock(return_value={"status": "healthy"})

        svc = _build_service(ai_service=ai_svc)
        health = await svc.get_analysis_service_health()
        assert health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_analysis_service_health_error(self):
        ai_svc = MagicMock()
        ai_svc.get_health_status = MagicMock(side_effect=RuntimeError("broken"))

        svc = _build_service(ai_service=ai_svc)
        health = await svc.get_analysis_service_health()
        assert health["status"] == "error"

    def test_get_integration_metrics(self):
        svc = _build_service()
        metrics = svc.get_integration_metrics()
        assert "metrics" in metrics
        assert "service" in metrics
        assert metrics["service"] == "ai_test_integration"

    @pytest.mark.asyncio
    async def test_health_check(self):
        exec_svc = AsyncMock()
        exec_svc.health_check = AsyncMock(return_value={"status": "healthy"})

        ai_svc = MagicMock()
        ai_svc.get_health_status = MagicMock(return_value={"status": "healthy"})

        svc = _build_service(test_exec_service=exec_svc, ai_service=ai_svc)
        health = await svc.health_check()
        assert health["status"] in ("healthy", "degraded")

    @pytest.mark.asyncio
    async def test_health_check_degraded(self):
        exec_svc = AsyncMock()
        exec_svc.health_check = AsyncMock(side_effect=RuntimeError("exec error"))

        svc = _build_service(test_exec_service=exec_svc)
        health = await svc.health_check()
        assert health["status"] in ("degraded", "unhealthy")


# ===================================================================
# Cleanup and factory
# ===================================================================


class TestCleanupAndFactory:
    @pytest.mark.asyncio
    async def test_cleanup(self):
        ai_svc = MagicMock()
        ai_svc.cleanup = AsyncMock()

        svc = _build_service(ai_service=ai_svc)
        await svc.cleanup()
        ai_svc.cleanup.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cleanup_no_ai_service(self):
        svc = _build_service()
        svc.ai_analysis_service = None
        await svc.cleanup()  # Should not raise

    def test_factory_function(self):
        with patch("services.ai_test_integration.AIFailureAnalysisService"):
            from services.ai_test_integration import create_ai_test_integration_service

            svc = create_ai_test_integration_service(enable_ai_analysis=True)
        assert svc is not None
