"""Tests targeting remaining coverage gaps across multiple modules.

Covers:
- shared.py: serializer None branches
- report_formatter.py: cache error paths, _serialize_for_cache, comprehensive cache
- ai_insights/base.py: _analyze_with_llm import error and exception paths
- ai_test_insights/config.py: get_health_grade, AnalysisResultHandler branches
- main.py: exception handlers
- ai_test_failure_analyzer.py: _create_failure_signature with file_path,
  _calculate_overall_confidence edge cases
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from services.ai_test_insights.config import (
    AnalysisResultHandler,
    get_health_grade,
)
from services.ai_test_insights.models import (
    TestCoverageInsights,
    TestQualityInsights,
    TestReliabilityMetrics,
)
from tests.conftest import AUTHED_TOKEN

# ===================================================================
# shared.py — serializer None branches
# ===================================================================


class TestSharedSerializerNoneBranches:
    """Cover the field_serializer methods that return None when v is None."""

    def test_execution_insights_response_none_generated_at(self):
        from routers.ai_test_insights.shared import ExecutionInsightsResponse

        resp = ExecutionInsightsResponse(
            status="success",
            message="ok",
            insights={"foo": "bar"},
        )
        # Serializer should handle actual datetime
        serialized = resp.serialize_datetime(resp.generated_at)
        assert isinstance(serialized, str)
        # Serializer returns None for None
        assert resp.serialize_datetime(None) is None

    def test_recommendations_response_none_generated_at(self):
        from routers.ai_test_insights.shared import RecommendationsResponse

        resp = RecommendationsResponse(
            status="success",
            message="ok",
            recommendations=[],
        )
        assert resp.serialize_datetime(None) is None
        assert isinstance(resp.serialize_datetime(resp.generated_at), str)

    def test_patterns_response_none_generated_at(self):
        from routers.ai_test_insights.shared import PatternsResponse

        resp = PatternsResponse(
            status="success",
            message="ok",
            patterns=[],
        )
        assert resp.serialize_datetime(None) is None

    def test_health_status_response_none_generated_at(self):
        from routers.ai_test_insights.shared import HealthStatusResponse

        resp = HealthStatusResponse(
            status="success",
            message="ok",
            health_status={"status": "healthy"},
        )
        assert resp.serialize_datetime(None) is None

    def test_service_settings_response_none_applied_at(self):
        from routers.ai_test_insights.shared import ServiceSettingsResponse

        resp = ServiceSettingsResponse(
            status="success",
            message="ok",
            settings={"key": "value"},
        )
        assert resp.serialize_datetime(None) is None


# ===================================================================
# report_formatter.py — cache error paths, _serialize_for_cache
# ===================================================================


class TestSerializeForCache:
    def test_serialize_pydantic_model(self):
        from services.ai_test_insights.models import TestPerformanceMetrics
        from services.ai_test_insights.report_formatter import _serialize_for_cache

        metrics = TestPerformanceMetrics(
            average_execution_time=1.0,
            median_execution_time=1.0,
            p95_execution_time=2.0,
            p99_execution_time=3.0,
            average_memory_usage=128.0,
            peak_memory_usage=256.0,
            average_cpu_usage=20.0,
            slow_tests_count=0,
            fast_tests_count=5,
            performance_trend="stable",
            bottleneck_categories=[],
        )
        result = _serialize_for_cache(metrics)
        assert isinstance(result, dict)
        assert result["average_execution_time"] == 1.0

    def test_serialize_list(self):
        from services.ai_test_insights.report_formatter import _serialize_for_cache

        result = _serialize_for_cache([1, "two", {"three": 3}])
        assert result == [1, "two", {"three": 3}]

    def test_serialize_dict(self):
        from services.ai_test_insights.report_formatter import _serialize_for_cache

        result = _serialize_for_cache({"key": "value", "nested": {"a": 1}})
        assert result == {"key": "value", "nested": {"a": 1}}

    def test_serialize_primitive(self):
        from services.ai_test_insights.report_formatter import _serialize_for_cache

        assert _serialize_for_cache(42) == 42
        assert _serialize_for_cache("hello") == "hello"
        assert _serialize_for_cache(None) is None


class TestReportFormatterCacheErrors:
    async def test_cache_analysis_results_error_is_logged(self):
        from services.ai_test_insights.report_formatter import TestReportFormatter

        cache_mgr = SimpleNamespace(
            set=AsyncMock(side_effect=RuntimeError("cache write fail")),
            get=AsyncMock(),
        )
        formatter = TestReportFormatter(cache_manager=cache_mgr)

        # Construct a minimal analysis
        from tests.unit.test_ai_test_insights_services import _analysis

        analysis = _analysis()

        # Should not raise
        await formatter._cache_analysis_results("test-id", analysis)
        cache_mgr.set.assert_awaited_once()

    async def test_cache_comprehensive_analysis_error_is_logged(self):
        from services.ai_test_insights.report_formatter import TestReportFormatter

        cache_mgr = SimpleNamespace(
            set=AsyncMock(side_effect=RuntimeError("cache write fail")),
            get=AsyncMock(),
        )
        formatter = TestReportFormatter(cache_manager=cache_mgr)

        await formatter._cache_comprehensive_analysis("test-id", {"data": "value"})
        cache_mgr.set.assert_awaited_once()

    async def test_get_cached_analysis_error_returns_none(self):
        from services.ai_test_insights.report_formatter import TestReportFormatter

        cache_mgr = SimpleNamespace(
            set=AsyncMock(),
            get=AsyncMock(side_effect=RuntimeError("cache read fail")),
        )
        formatter = TestReportFormatter(cache_manager=cache_mgr)

        result = await formatter.get_cached_analysis("test-id")
        assert result is None

    async def test_get_cached_comprehensive_analysis_error_returns_none(self):
        from services.ai_test_insights.report_formatter import TestReportFormatter

        cache_mgr = SimpleNamespace(
            set=AsyncMock(),
            get=AsyncMock(side_effect=RuntimeError("cache read fail")),
        )
        formatter = TestReportFormatter(cache_manager=cache_mgr)

        result = await formatter.get_cached_comprehensive_analysis("test-id")
        assert result is None

    async def test_get_cached_analysis_no_cache_returns_none(self):
        from services.ai_test_insights.report_formatter import TestReportFormatter

        formatter = TestReportFormatter(cache_manager=None)
        assert await formatter.get_cached_analysis("x") is None

    async def test_get_cached_comprehensive_no_cache_returns_none(self):
        from services.ai_test_insights.report_formatter import TestReportFormatter

        formatter = TestReportFormatter(cache_manager=None)
        assert await formatter.get_cached_comprehensive_analysis("x") is None

    async def test_cache_analysis_no_cache_manager_noop(self):
        from services.ai_test_insights.report_formatter import TestReportFormatter

        formatter = TestReportFormatter(cache_manager=None)
        from tests.unit.test_ai_test_insights_services import _analysis

        # Should not raise
        await formatter._cache_analysis_results("x", _analysis())

    async def test_cache_comprehensive_no_cache_manager_noop(self):
        from services.ai_test_insights.report_formatter import TestReportFormatter

        formatter = TestReportFormatter(cache_manager=None)
        await formatter._cache_comprehensive_analysis("x", {})

    async def test_create_comprehensive_analysis_exception_propagates(self):
        from services.ai_test_insights.report_formatter import TestReportFormatter

        formatter = TestReportFormatter()
        formatter.insight_generator.analyze_performance = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        formatter.insight_generator.analyze_reliability = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        formatter.insight_generator.analyze_coverage = AsyncMock(side_effect=RuntimeError("boom"))
        formatter.insight_generator.analyze_quality = AsyncMock(side_effect=RuntimeError("boom"))
        formatter.pattern_detector.detect_patterns = AsyncMock(side_effect=RuntimeError("boom"))
        formatter.insight_generator.generate_recommendations = AsyncMock(return_value=[])

        # The _handle_result returns defaults for exceptions, so analysis still succeeds
        from models.testing_models import TestResult

        tr = TestResult(
            test_id="t1",
            success=True,
            execution_time=1.0,
            language="python",
            created_at=datetime.now(UTC),
        )
        from models.testing_models import QualityResult

        qr = QualityResult(overall_score=80.0)

        result = await formatter.create_comprehensive_analysis([tr], [qr])
        assert result.analysis_id  # analysis still completed with defaults


# ===================================================================
# ai_insights/base.py — _analyze_with_llm
# ===================================================================


class TestBaseInsightGeneratorAnalyzeWithLLM:
    async def test_analyze_with_llm_disabled(self):
        from services.ai_insights.base import BaseInsightGenerator

        gen = BaseInsightGenerator(insight_rules={})
        gen._llm_enabled = False
        result = await gen._analyze_with_llm("content", "system prompt")
        assert result is None

    async def test_analyze_with_llm_import_error(self):
        from services.ai_insights.base import BaseInsightGenerator

        gen = BaseInsightGenerator(insight_rules={})
        gen._llm_enabled = True

        # services.llm_client doesn't exist so ImportError will be raised
        result = await gen._analyze_with_llm("content", "system prompt")
        assert result is None
        assert gen._llm_enabled is False  # Disabled after ImportError

    async def test_analyze_with_llm_runtime_error(self):
        from services.ai_insights.base import BaseInsightGenerator

        gen = BaseInsightGenerator(insight_rules={})
        gen._llm_enabled = True

        # Mock the import to succeed but the function to fail
        mock_module = MagicMock()
        mock_module.analyze_with_llm = AsyncMock(side_effect=RuntimeError("LLM down"))

        with patch.dict("sys.modules", {"services.llm_client": mock_module}):
            result = await gen._analyze_with_llm("content", "prompt")
        assert result is None

    async def test_generate_raises_not_implemented(self):
        from services.ai_insights.base import BaseInsightGenerator

        gen = BaseInsightGenerator(insight_rules={})
        with pytest.raises(NotImplementedError):
            await gen.generate({})


# ===================================================================
# ai_test_insights/config.py — get_health_grade
# ===================================================================


class TestGetHealthGrade:
    def test_grade_a(self):
        assert get_health_grade(95) == "A"
        assert get_health_grade(90) == "A"

    def test_grade_b(self):
        assert get_health_grade(85) == "B"
        assert get_health_grade(80) == "B"

    def test_grade_c(self):
        assert get_health_grade(75) == "C"
        assert get_health_grade(70) == "C"

    def test_grade_d(self):
        assert get_health_grade(65) == "D"
        assert get_health_grade(60) == "D"

    def test_grade_f(self):
        assert get_health_grade(50) == "F"
        assert get_health_grade(0) == "F"


class TestAnalysisResultHandlerBranches:
    def test_non_exception_result_returned_as_is(self):
        result = {"data": "value"}
        assert (
            AnalysisResultHandler.handle_analysis_result(result, "test", dict, logging.getLogger())
            == result
        )

    def test_reliability_default(self):
        default = AnalysisResultHandler._get_default_instance(TestReliabilityMetrics)
        assert isinstance(default, TestReliabilityMetrics)
        assert default.overall_success_rate == 0.0

    def test_coverage_default(self):
        default = AnalysisResultHandler._get_default_instance(TestCoverageInsights)
        assert isinstance(default, TestCoverageInsights)
        assert default.line_coverage == 0.0

    def test_quality_default(self):
        default = AnalysisResultHandler._get_default_instance(TestQualityInsights)
        assert isinstance(default, TestQualityInsights)
        assert default.test_quality_score == 0.0

    def test_unknown_type_returns_list(self):
        default = AnalysisResultHandler._get_default_instance(str)
        assert default == []


# ===================================================================
# main.py — exception handlers
# ===================================================================


class TestMainExceptionHandlers:
    @pytest.fixture(autouse=True)
    def _bypass_auth(self):
        """Neutralise auth so these tests exercise the EXCEPTION HANDLERS.

        Replaces `monkeypatch.setenv("SKIP_AUTH", "true")`; the hatch was
        removed from this service's fork and from coding-client 1.16.0.

        Only `test_validation_error_handler` actually needs this — it POSTs to a
        `require_tenant`-guarded route and asserts 422, which it can only reach
        once auth has passed. The other three hit unguarded paths.

        ⚠️ The replacement must set `_request_tenant_id_var`, not just return a
        string. This service's `require_tenant` stashes the resolved tenant in
        that contextvar, and the WM shim (`services.llm_agents`) forwards it to
        worker_management. A plain `lambda: "test-tenant"` leaves it unset, and
        the shim silently falls back to this service's own licence — so a test
        would go green while exercising the wrong branch.

        ⚠️ A dependency override is NOT enough on its own any more. coding-ai
        now registers `AuthMiddleware`, and a middleware runs BEFORE routing, so
        `app.dependency_overrides` cannot reach it — an anonymous request 401s
        without the overridden dependency ever being consulted. The introspection
        seams are patched here as well, and there are two of them: coding_client's
        (used by the middleware) and this service's fork (used by
        `require_tenant`). Patching only one leaves the other live.
        """
        from unittest.mock import patch as _patch

        from coding_client.testing import override_tenant_auth

        from auth import tenant as tenant_mod
        from main import app

        def _fake_require_tenant() -> str:
            tenant_mod._request_tenant_id_var.set("test-tenant")
            return "test-tenant"

        async def _introspect(token: str) -> dict:
            return {"active": True, "tenant_id": "test-tenant", "subject_id": "svc-test"}

        undo = override_tenant_auth(app, extra={tenant_mod.require_tenant: _fake_require_tenant})
        with (
            _patch("coding_client.auth.middleware._introspect_license", new=_introspect),
            _patch("auth.tenant._introspect_license", new=_introspect),
        ):
            yield
        undo()

    def test_health_endpoint(self):
        from main import app

        client = TestClient(
            app,
            raise_server_exceptions=False,
            headers={"Authorization": f"Bearer {AUTHED_TOKEN}"},
        )
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_root_endpoint(self):
        from main import app

        client = TestClient(
            app,
            raise_server_exceptions=False,
            headers={"Authorization": f"Bearer {AUTHED_TOKEN}"},
        )
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "service" in data
        assert "version" in data

    def test_http_exception_handler_string_detail(self):
        from main import app

        client = TestClient(
            app,
            raise_server_exceptions=False,
            headers={"Authorization": f"Bearer {AUTHED_TOKEN}"},
        )
        # 404 triggers HTTPException with string detail
        resp = client.get("/nonexistent-path")
        assert resp.status_code in (404, 405)

    def test_validation_error_handler(self):
        from main import app

        client = TestClient(
            app,
            raise_server_exceptions=False,
            headers={"Authorization": f"Bearer {AUTHED_TOKEN}"},
        )
        # POST to an endpoint that expects a body but send invalid data
        resp = client.post(
            "/api/ai/analysis/tests/failure-analysis",
            json={"invalid": "data"},
        )
        assert resp.status_code == 422
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == "HTTP_422"
        assert "details" in data["error"]

    def test_http_exception_handler_dict_detail(self):
        """Cover the dict branch of http_exception_handler."""
        from main import app

        client = TestClient(
            app,
            raise_server_exceptions=False,
            headers={"Authorization": f"Bearer {AUTHED_TOKEN}"},
        )
        # POST to analyze endpoint with empty code triggers 400 with dict detail
        resp = client.post(
            "/api/ai/insights/code/analyze",
            json={"code_content": "", "language": "python"},
        )
        # Should get 400/422/500 depending on validation
        assert resp.status_code in (400, 422, 500)


# ===================================================================
# ai_test_failure_analyzer.py — _create_failure_signature, _calculate_overall_confidence
# ===================================================================


class TestAITestFailureAnalyzerEdgeCases:
    def _build_analyzer(self):
        with patch("services.ai_test_failure_analyzer.make_agent_runner"):
            from services.ai_test_failure_analyzer import AITestFailureAnalyzer

            return AITestFailureAnalyzer()

    def test_create_failure_signature_with_file_path(self):
        analyzer = self._build_analyzer()
        from models.ai_test_failure_models import (
            ErrorDetails,
            FailureType,
            TestFailureContext,
            TestFailureData,
            TestFramework,
        )

        failure_data = TestFailureData(
            failure_id="f1",
            raw_output="error output",
            context=TestFailureContext(
                test_name="test_example",
                framework=TestFramework.PYTEST,
            ),
        )
        error_details = ErrorDetails(
            error_type=FailureType.ASSERTION_ERROR,
            error_message="expected True got False",
            file_path="/path/to/test_file.py",
        )
        sig = analyzer._create_failure_signature(failure_data, error_details)
        assert "test_file.py" in sig
        assert "assertion_error" in sig

    def test_create_failure_signature_without_file_path(self):
        analyzer = self._build_analyzer()
        from models.ai_test_failure_models import (
            ErrorDetails,
            FailureType,
            TestFailureContext,
            TestFailureData,
            TestFramework,
        )

        failure_data = TestFailureData(
            failure_id="f1",
            raw_output="error output",
            context=TestFailureContext(
                test_name="test_example",
                framework=TestFramework.PYTEST,
            ),
        )
        error_details = ErrorDetails(
            error_type=FailureType.ASSERTION_ERROR,
            error_message="expected True got False",
        )
        sig = analyzer._create_failure_signature(failure_data, error_details)
        assert "|" in sig
        assert "assertion_error" in sig

    def test_calculate_overall_confidence_empty(self):
        analyzer = self._build_analyzer()
        level, score = analyzer._calculate_overall_confidence([], [], [])
        assert level == "very_low"
        assert score == 0.0

    def test_calculate_overall_confidence_high(self):
        analyzer = self._build_analyzer()
        from models.ai_test_failure_models import (
            ConfidenceLevel,
            FixSuggestion,
            RootCause,
            SeverityLevel,
            TestFailurePattern,
        )

        root_causes = [
            RootCause(
                cause_id="c1",
                category="unknown",
                description="Root cause description for the test failure",
                severity=SeverityLevel.HIGH,
                confidence=ConfidenceLevel.VERY_HIGH,
                confidence_score=0.95,
            )
        ]
        fix_suggestions = [
            FixSuggestion(
                suggestion_id="s1",
                title="Fix the test assertion",
                description="Change the expected value to match the actual output of the function",
                fix_type="code_change",
                priority="high",
                confidence=ConfidenceLevel.VERY_HIGH,
                confidence_score=0.92,
                estimated_effort="low",
            )
        ]
        patterns = [
            TestFailurePattern(
                pattern_id="p1",
                pattern_name="Assertion Error Pattern",
                pattern_signature="assertion_error|expected|pytest_pattern",
                pattern_type="assertion_pattern",
                confidence=ConfidenceLevel.VERY_HIGH,
                confidence_score=0.91,
            )
        ]
        level, score = analyzer._calculate_overall_confidence(
            root_causes, fix_suggestions, patterns
        )
        assert level == "very_high"
        assert score >= 0.9

    def test_calculate_overall_confidence_medium(self):
        analyzer = self._build_analyzer()
        from models.ai_test_failure_models import (
            ConfidenceLevel,
            RootCause,
            SeverityLevel,
        )

        root_causes = [
            RootCause(
                cause_id="c1",
                category="unknown",
                description="Some root cause description for the test",
                severity=SeverityLevel.MEDIUM,
                confidence=ConfidenceLevel.MEDIUM,
                confidence_score=0.6,
            )
        ]
        level, score = analyzer._calculate_overall_confidence(root_causes, [], [])
        assert level == "medium"

    def test_calculate_overall_confidence_low(self):
        analyzer = self._build_analyzer()
        from models.ai_test_failure_models import (
            ConfidenceLevel,
            RootCause,
            SeverityLevel,
        )

        root_causes = [
            RootCause(
                cause_id="c1",
                category="unknown",
                description="Some root cause description for the test",
                severity=SeverityLevel.LOW,
                confidence=ConfidenceLevel.LOW,
                confidence_score=0.3,
            )
        ]
        level, score = analyzer._calculate_overall_confidence(root_causes, [], [])
        assert level == "low"

    def test_calculate_overall_confidence_very_low(self):
        analyzer = self._build_analyzer()
        from models.ai_test_failure_models import (
            ConfidenceLevel,
            RootCause,
            SeverityLevel,
        )

        root_causes = [
            RootCause(
                cause_id="c1",
                category="unknown",
                description="Some root cause description for the test",
                severity=SeverityLevel.INFO,
                confidence=ConfidenceLevel.VERY_LOW,
                confidence_score=0.1,
            )
        ]
        level, score = analyzer._calculate_overall_confidence(root_causes, [], [])
        assert level == "very_low"

    # NOTE: former tests for `_fallback_root_cause_analysis` /
    # `_fallback_fix_suggestions` removed — those AITestFailureAnalyzer helpers
    # fabricated confidence-0.3 content on an LLM outage and reported it as a
    # clean success. Outage is now surfaced honestly via degraded=True with
    # empty results (see test_ai_test_failure_analyzer TestAnalyzeFailureDegradedSignal).


# ===================================================================
# ai_test_integration.py — _integrate_analysis_results with assertion_details
# ===================================================================


class TestIntegrateWithAssertionDetails:
    def _build_service(self):
        with patch("services.ai_test_integration.AIFailureAnalysisService"):
            from services.ai_test_integration import AITestIntegrationService

            return AITestIntegrationService(
                ai_analysis_service=MagicMock(),
                enable_ai_analysis=True,
            )

    def test_assertion_details_populated(self):
        svc = self._build_service()
        from models.ai_test_failure_models import (
            AIAnalysisResult,
            AssertionDetails,
            ConfidenceLevel,
            ErrorDetails,
            FailureType,
            TestFailureContext,
            TestFailureData,
            TestFramework,
        )
        from models.testing_models import TestResult

        test_result = TestResult(
            test_id="t1",
            success=False,
            execution_time=1.0,
            language="python",
            created_at=datetime.now(UTC),
        )

        analysis = AIAnalysisResult(
            analysis_id="a1",
            failure_data=TestFailureData(
                failure_id="f1",
                raw_output="AssertionError",
                context=TestFailureContext(test_name="test_x", framework=TestFramework.PYTEST),
            ),
            error_details=ErrorDetails(
                error_type=FailureType.ASSERTION_ERROR,
                error_message="1 != 2",
            ),
            assertion_details=AssertionDetails(
                assertion_type="assertEqual",
                expected_value="1",
                actual_value="2",
                comparison_operator="==",
                assertion_message="values differ",
                value_difference="-1",
            ),
            analysis_confidence=ConfidenceLevel.LOW,
            analysis_confidence_score=0.3,
        )

        result = svc._integrate_analysis_results(test_result, analysis)
        assert "assertion_details" in result.test_metadata
        details = result.test_metadata["assertion_details"]
        assert details["assertion_type"] == "assertEqual"
        assert details["expected_value"] == "1"
        assert details["actual_value"] == "2"
