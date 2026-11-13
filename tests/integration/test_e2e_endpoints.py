"""End-to-end integration tests for coding-ai service.

Tests hit the LIVE service running at http://localhost:8690 with ENABLE_AUTH=false.
Run with: uv run pytest tests/integration -v -m integration
"""

from __future__ import annotations

import httpx
import pytest

BASE_URL = "http://localhost:8690"


@pytest.fixture(scope="module")
def client() -> httpx.Client:
    """Shared synchronous httpx client for the test module."""
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as c:
        yield c


# ---------------------------------------------------------------------------
# Infrastructure endpoints
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestHealthEndpoint:
    def test_health_returns_ok(self, client: httpx.Client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["service"] == "coding-ai"
        assert "version" in body
        assert "redis" in body

    def test_root_returns_service_info(self, client: httpx.Client):
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["service"] == "coding-ai"
        assert body["docs"] == "/docs"
        assert body["health"] == "/health"


@pytest.mark.integration
class TestOpenAPISpec:
    def test_openapi_json_available(self, client: httpx.Client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        spec = resp.json()
        assert spec["openapi"].startswith("3.")
        assert "paths" in spec
        assert "/health" in spec["paths"]


# ---------------------------------------------------------------------------
# AI Code Insights endpoints  (alias prefix /api/ai)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCapabilities:
    def test_capabilities_returns_200(self, client: httpx.Client):
        resp = client.get("/api/ai/capabilities")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert isinstance(body["insight_types"], list)
        assert len(body["insight_types"]) > 0
        assert "architecture" in body["insight_types"]

    def test_capabilities_has_priority_levels(self, client: httpx.Client):
        resp = client.get("/api/ai/capabilities")
        body = resp.json()
        assert isinstance(body["priority_levels"], list)
        assert "critical" in body["priority_levels"]
        assert "high" in body["priority_levels"]

    def test_capabilities_has_confidence_levels(self, client: httpx.Client):
        resp = client.get("/api/ai/capabilities")
        body = resp.json()
        assert isinstance(body["confidence_levels"], list)
        assert "high" in body["confidence_levels"]

    def test_capabilities_has_analysis_features(self, client: httpx.Client):
        resp = client.get("/api/ai/capabilities")
        body = resp.json()
        assert isinstance(body["analysis_features"], dict)
        assert "architecture_analysis" in body["analysis_features"]

    def test_capabilities_has_supported_languages(self, client: httpx.Client):
        resp = client.get("/api/ai/capabilities")
        body = resp.json()
        assert isinstance(body["supported_languages"], list)
        assert "python" in body["supported_languages"]
        assert "java" in body["supported_languages"]
        assert "rust" in body["supported_languages"]

    def test_capabilities_via_insights_code_prefix(self, client: httpx.Client):
        """Same router is mounted at /api/ai/insights/code as well."""
        resp = client.get("/api/ai/insights/code/capabilities")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert isinstance(body["insight_types"], list)


@pytest.mark.integration
class TestInsightTypes:
    def test_insight_types_returns_200(self, client: httpx.Client):
        resp = client.get("/api/ai/insight-types")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert isinstance(body["insight_types"], dict)
        assert body["total_types"] == 8

    def test_insight_types_contains_all_categories(self, client: httpx.Client):
        resp = client.get("/api/ai/insight-types")
        body = resp.json()
        expected = {
            "architecture",
            "refactoring",
            "security",
            "performance",
            "maintainability",
            "cross_language",
            "quality",
            "patterns",
        }
        assert set(body["insight_types"].keys()) == expected

    def test_insight_type_has_structure(self, client: httpx.Client):
        resp = client.get("/api/ai/insight-types")
        body = resp.json()
        arch = body["insight_types"]["architecture"]
        assert "name" in arch
        assert "description" in arch
        assert isinstance(arch["focus_areas"], list)
        assert isinstance(arch["typical_priorities"], list)
        assert isinstance(arch["example_insights"], list)

    def test_insight_types_has_ai_features(self, client: httpx.Client):
        resp = client.get("/api/ai/insight-types")
        body = resp.json()
        assert isinstance(body["ai_powered_features"], list)
        assert len(body["ai_powered_features"]) > 0

    def test_insight_types_via_insights_code_prefix(self, client: httpx.Client):
        resp = client.get("/api/ai/insights/code/insight-types")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_types"] == 8


# ---------------------------------------------------------------------------
# POST /api/ai/analyze — Code Insights Analysis
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAnalyzeEndpoint:
    def test_analyze_missing_body_returns_422(self, client: httpx.Client):
        resp = client.post("/api/ai/analyze", json={})
        assert resp.status_code == 422
        body = resp.json()
        assert body["error_code"] == "VALIDATION_ERROR"
        fields = [e["field"] for e in body["validation_errors"]]
        assert "body.container_name" in fields

    def test_analyze_with_nonexistent_container_returns_404(self, client: httpx.Client):
        """Container validation calls coding-containers which is not running,
        so we get a 404 for unknown containers."""
        resp = client.post(
            "/api/ai/analyze",
            json={"container_name": "nonexistent-container", "file_paths": ["src/main.py"]},
        )
        assert resp.status_code == 404
        body = resp.json()
        assert "not found" in body["message"].lower()

    def test_analyze_invalid_focus_areas_via_insights_code(self, client: httpx.Client):
        """The /api/ai/insights/code/analyze path also validates focus areas,
        but container validation runs first and will 404."""
        resp = client.post(
            "/api/ai/insights/code/analyze",
            json={
                "container_name": "test",
                "file_paths": ["x.py"],
                "focus_areas": ["not_a_real_area"],
            },
        )
        # Container validation fails first → 404
        assert resp.status_code == 404

    def test_analyze_max_insights_validation(self, client: httpx.Client):
        """max_insights must be 1-100."""
        resp = client.post(
            "/api/ai/analyze",
            json={"container_name": "c", "max_insights": 0},
        )
        assert resp.status_code == 422

        resp2 = client.post(
            "/api/ai/analyze",
            json={"container_name": "c", "max_insights": 101},
        )
        assert resp2.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/ai/failure-analysis — AI Test Failure Analysis
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFailureAnalysis:
    def test_failure_analysis_missing_fields_returns_422(self, client: httpx.Client):
        resp = client.post("/api/ai/failure-analysis", json={})
        assert resp.status_code == 422
        body = resp.json()
        assert body["error_code"] == "VALIDATION_ERROR"
        fields = [e["field"] for e in body["validation_errors"]]
        assert "body.container_name" in fields
        assert "body.test_output" in fields
        assert "body.test_name" in fields

    def test_failure_analysis_with_valid_payload(self, client: httpx.Client):
        """With no AI API key the service returns a degraded but successful response."""
        resp = client.post(
            "/api/ai/failure-analysis",
            json={
                "container_name": "test-container",
                "test_output": "AssertionError: assert 1 == 2\n  File test_math.py, line 5",
                "test_name": "test_basic_addition",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert "analysis_id" in body
        assert isinstance(body["root_causes"], list)
        assert len(body["root_causes"]) > 0
        assert isinstance(body["fix_suggestions"], list)
        assert isinstance(body["detected_patterns"], list)
        assert body["analysis_confidence"] is not None

    def test_failure_analysis_with_all_fields(self, client: httpx.Client):
        resp = client.post(
            "/api/ai/failure-analysis",
            json={
                "container_name": "test-container",
                "test_output": "TypeError: unsupported operand type(s)",
                "test_name": "test_type_error",
                "test_framework": "pytest",
                "language": "python",
                "source_code": "def add(a, b): return a + b",
                "test_code": "def test_type_error(): add('1', 2)",
                "file_path": "tests/test_math.py",
                "include_suggestions": True,
                "include_patterns": True,
                "max_root_causes": 3,
                "max_suggestions": 5,
                "minimum_confidence": 0.2,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert "processing_time" in body
        assert body["processing_time"] >= 0

    def test_failure_analysis_no_suggestions(self, client: httpx.Client):
        resp = client.post(
            "/api/ai/failure-analysis",
            json={
                "container_name": "c",
                "test_output": "Error",
                "test_name": "t",
                "include_suggestions": False,
                "include_patterns": False,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["fix_suggestions"] == []
        assert body["detected_patterns"] == []


# ---------------------------------------------------------------------------
# POST /api/ai/batch-analyze
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBatchAnalyze:
    def test_batch_analyze_missing_fields_returns_422(self, client: httpx.Client):
        resp = client.post("/api/ai/batch-analyze", json={})
        assert resp.status_code == 422
        body = resp.json()
        fields = [e["field"] for e in body["validation_errors"]]
        assert "body.container_name" in fields
        assert "body.test_results" in fields

    def test_batch_analyze_degraded_response(self, client: httpx.Client):
        """Without full AI backend the service returns 200 with zero analyzed tests."""
        resp = client.post(
            "/api/ai/batch-analyze",
            json={
                "container_name": "test-container",
                "test_results": [
                    {
                        "test_name": "test_a",
                        "status": "failed",
                        "error_message": "assert False",
                    }
                ],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "batch_id" in body
        assert body["total_tests"] == 1
        assert isinstance(body["results"], list)


# ---------------------------------------------------------------------------
# GET /api/ai/failure-analysis/health & /metrics
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFailureAnalysisHealth:
    def test_failure_analysis_health(self, client: httpx.Client):
        resp = client.get("/api/ai/failure-analysis/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["service"] == "ai_test_analysis"
        assert "service_status" in body
        assert "ai_analysis_available" in body
        assert "timestamp" in body

    def test_failure_analysis_metrics(self, client: httpx.Client):
        resp = client.get("/api/ai/failure-analysis/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert body["service"] == "ai_test_analysis"
        assert "metrics" in body
        metrics = body["metrics"]
        assert "total_analyses" in metrics
        assert "success_rate" in metrics
        assert "timestamp" in body


# ---------------------------------------------------------------------------
# GET /api/ai/patterns
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPatterns:
    def test_patterns_returns_200(self, client: httpx.Client):
        resp = client.get("/api/ai/patterns")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        data = body["data"]
        assert isinstance(data["patterns"], list)
        assert data["total_patterns"] > 0
        assert isinstance(data["pattern_types"], list)

    def test_patterns_structure(self, client: httpx.Client):
        resp = client.get("/api/ai/patterns")
        body = resp.json()
        pattern = body["data"]["patterns"][0]
        assert "id" in pattern
        assert "name" in pattern
        assert "type" in pattern
        assert "frequency" in pattern
        assert "confidence" in pattern
        assert isinstance(pattern["typical_causes"], list)
        assert isinstance(pattern["common_fixes"], list)

    def test_patterns_with_limit(self, client: httpx.Client):
        resp = client.get("/api/ai/patterns", params={"limit": 1})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["data"]["patterns"]) <= 1

    def test_patterns_via_analysis_tests_prefix(self, client: httpx.Client):
        resp = client.get("/api/ai/analysis/tests/patterns")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"


# ---------------------------------------------------------------------------
# Test Suite Insights endpoints  (/api/ai/insights/tests)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTestInsightsAnalyze:
    def test_analyze_missing_fields_returns_422(self, client: httpx.Client):
        resp = client.post("/api/ai/insights/tests/analyze", json={})
        assert resp.status_code == 422
        body = resp.json()
        fields = [e["field"] for e in body["validation_errors"]]
        assert "body.container_name" in fields
        assert "body.test_results" in fields

    def test_analyze_with_valid_payload(self, client: httpx.Client):
        resp = client.post(
            "/api/ai/insights/tests/analyze",
            json={
                "container_name": "test-container",
                "test_results": [
                    {
                        "test_name": "test_add",
                        "status": "failed",
                        "error_message": "AssertionError: assert 1 == 2",
                        "duration": 0.5,
                    }
                ],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "analysis_id" in body
        assert "performance_metrics" in body
        assert "reliability_metrics" in body
        assert "coverage_insights" in body


@pytest.mark.integration
class TestComprehensiveAnalysis:
    def test_comprehensive_missing_fields_returns_422(self, client: httpx.Client):
        resp = client.post("/api/ai/insights/tests/comprehensive-analysis", json={})
        assert resp.status_code == 422
        body = resp.json()
        fields = [e["field"] for e in body["validation_errors"]]
        assert "body.container_name" in fields
        assert "body.test_results" in fields


@pytest.mark.integration
class TestPredictFailures:
    def test_predict_failures_missing_fields_returns_422(self, client: httpx.Client):
        resp = client.post("/api/ai/insights/tests/predict-failures", json={})
        assert resp.status_code == 422
        body = resp.json()
        fields = [e["field"] for e in body["validation_errors"]]
        assert "body.container_name" in fields
        assert "body.test_results" in fields


@pytest.mark.integration
class TestRecommendations:
    def test_recommendations_missing_fields_returns_422(self, client: httpx.Client):
        resp = client.post("/api/ai/insights/tests/recommendations", json={})
        assert resp.status_code == 422
        body = resp.json()
        fields = [e["field"] for e in body["validation_errors"]]
        assert "body.container_name" in fields
        assert "body.analysis" in fields


@pytest.mark.integration
class TestAnalysisById:
    def test_analysis_not_found(self, client: httpx.Client):
        resp = client.get("/api/ai/insights/tests/analysis/nonexistent-id")
        assert resp.status_code == 404
        body = resp.json()
        assert body["error_code"] == "ANALYSIS_NOT_FOUND"


@pytest.mark.integration
class TestSettings:
    def test_settings_returns_403_without_admin(self, client: httpx.Client):
        resp = client.get("/api/ai/insights/tests/settings")
        assert resp.status_code == 403
        body = resp.json()
        assert body["error_code"] == "FORBIDDEN"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestErrorHandling:
    def test_404_for_unknown_route(self, client: httpx.Client):
        resp = client.get("/api/ai/nonexistent-endpoint")
        assert resp.status_code == 404

    def test_405_method_not_allowed(self, client: httpx.Client):
        resp = client.delete("/health")
        assert resp.status_code == 405

    def test_validation_error_shape(self, client: httpx.Client):
        """Validation errors follow the standard error envelope."""
        resp = client.post("/api/ai/failure-analysis", json={})
        assert resp.status_code == 422
        body = resp.json()
        assert body["status"] == "error"
        assert body["error_code"] == "VALIDATION_ERROR"
        assert body["path"] == "/api/ai/failure-analysis"
        assert "timestamp" in body
        assert "request_id" in body
        assert isinstance(body["validation_errors"], list)
        for err in body["validation_errors"]:
            assert "field" in err
            assert "message" in err
            assert "type" in err
