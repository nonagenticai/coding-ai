"""Tests for AI Code Insights router endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from coding_client.clients import ServiceError
from fastapi import HTTPException

from services.ai_insights.base import (
    CodebaseInsights,
    CodeInsight,
    InsightConfidence,
    InsightPriority,
    InsightType,
)

TENANT_HEADERS = {"X-Tenant-Id": "test-tenant"}


@pytest.fixture
def client(authed_client):
    """Authenticated: the subject of these tests is the HANDLER, not the gate.

    These routes used to answer 200 with no headers at all — that was the
    defect. They now carry Depends(require_tenant), and AuthMiddleware gates
    everything outside PUBLIC_PATHS besides. The 401 direction is asserted
    deliberately in tests/unit/test_auth_is_registered.py; here the credential
    is supplied so the response BODY stays under test.
    """
    return authed_client


# ---------------------------------------------------------------------------
# Existing tests
# ---------------------------------------------------------------------------
def test_capabilities_returns_200(client):
    response = client.get("/api/ai/insights/code/capabilities")
    assert response.status_code == 200
    data = response.json()
    assert "insight_types" in data
    assert "supported_languages" in data
    assert isinstance(data["insight_types"], list)
    assert len(data["insight_types"]) > 0


def test_insight_types_returns_200(client):
    response = client.get("/api/ai/insights/code/insight-types")
    assert response.status_code == 200
    data = response.json()
    assert "insight_types" in data
    assert "total_types" in data
    assert data["total_types"] == 8
    assert "ai_powered_features" in data


def test_analyze_missing_container_returns_422(client):
    response = client.post(
        "/api/ai/insights/code/analyze",
        json={"file_paths": ["src/main.py"]},
        headers=TENANT_HEADERS,
    )
    assert response.status_code == 422


def test_analyze_empty_body_returns_422(client):
    response = client.post(
        "/api/ai/insights/code/analyze",
        json={},
        headers=TENANT_HEADERS,
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# _convert_insight_to_model  (line 158)
# ---------------------------------------------------------------------------
def test_convert_insight_to_model():
    from routers.ai_insights import _convert_insight_to_model

    insight = CodeInsight(
        id="ins-1",
        insight_type=InsightType.SECURITY,
        priority=InsightPriority.HIGH,
        confidence=InsightConfidence.MEDIUM,
        title="Test",
        description="Desc",
        rationale="Reason",
        files=["a.py"],
        entities=["Foo"],
        lines={"a.py": [(1, 10)]},
        recommendations=["Fix it"],
        mitigation_suggestions=["Do X"],
        estimated_effort="low",
        evidence={"key": "val"},
        metrics={"score": 1.0},
        related_insights=["ins-2"],
        tags=["sec"],
    )

    model = _convert_insight_to_model(insight)
    assert model.id == "ins-1"
    assert model.insight_type == "security"
    assert model.priority == "high"
    assert model.confidence == "medium"
    assert model.files == ["a.py"]
    assert model.resolve_suggestions == ["Do X"]
    assert model.estimated_effort == "low"


# ---------------------------------------------------------------------------
# Typed-client seam
#
# These call sites moved from an inline httpx.AsyncClient to the typed
# CodingWorkspacesClient, so the mock target is the CLIENT METHOD, not httpx.
# That is the whole point of the migration: the route path and the query
# parameter names now belong to the client, and mocking one layer down (httpx)
# would keep asserting a URL this module no longer builds.
#
# The client raises ServiceError on any >=400 instead of returning a response,
# so every "status code" case below is expressed as a raised ServiceError.
# ---------------------------------------------------------------------------
def _service_error(status_code: int) -> ServiceError:
    """A ServiceError shaped exactly like the one ServiceClient._request raises."""
    return ServiceError(status_code, f"GET /api/workspaces/x returned {status_code}", body=None)


def _exec_result(stdout="", *, exit_code=0, stderr=""):
    """The shape coding-containers' /exec actually returns."""
    return {"exit_code": exit_code, "stdout": stdout, "stderr": stderr}


def _patch_workspaces_client(*, get_workspace=None, exec_=None):
    """Patch routers.ai_insights.CodingWorkspacesClient; return (patcher, instance).

    ``get_workspace`` / ``exec_`` are kwargs dicts forwarded to AsyncMock
    (``return_value=`` or ``side_effect=``).

    File discovery mocks ``exec``, not ``list_files``: ``/files`` runs a single
    non-recursive ``ls`` and keys its result under ``entries`` as a list of
    ``{permissions, size, modified, name, type}`` objects, so it can neither
    reach ``src/`` nor yield a path. Discovery runs a bounded ``find`` instead.
    """
    instance = MagicMock()
    instance.get_workspace = AsyncMock(**(get_workspace or {"return_value": {}}))
    instance.exec = AsyncMock(**(exec_ or {"return_value": _exec_result()}))
    instance.close = AsyncMock()
    patcher = patch("routers.ai_insights.CodingWorkspacesClient", return_value=instance)
    return patcher, instance


# ---------------------------------------------------------------------------
# _validate_container
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_validate_container_404():
    """Container returning 404 from coding-containers raises HTTPException 404."""
    from routers.ai_insights import _validate_container

    patcher, _ = _patch_workspaces_client(get_workspace={"side_effect": _service_error(404)})
    with patcher, pytest.raises(HTTPException) as exc_info:
        await _validate_container("nonexistent", "tenant1")
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Container 'nonexistent' not found"


@pytest.mark.asyncio
async def test_validate_container_success():
    """Container returning 200 passes validation."""
    from routers.ai_insights import _validate_container

    patcher, _ = _patch_workspaces_client(get_workspace={"return_value": {"container": {}}})
    with patcher:
        # Should not raise
        await _validate_container("existing", "tenant1")


@pytest.mark.asyncio
async def test_validate_container_http_error():
    """When coding-containers is unreachable, raises 404.

    The client retries ConnectError internally and then re-raises the ORIGINAL
    httpx exception, so this branch is still an httpx.HTTPError catch.
    """
    import httpx

    from routers.ai_insights import _validate_container

    patcher, _ = _patch_workspaces_client(get_workspace={"side_effect": httpx.ConnectError("down")})
    with patcher, pytest.raises(HTTPException) as exc_info:
        await _validate_container("container1", "tenant1")
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_validate_container_5xx_logs_warning():
    """5xx from coding-containers is logged but does not raise.

    Regression guard for the migration: ServiceError is raised for EVERY >=400,
    so a naive `except ServiceError: raise` would have turned this
    log-and-continue case into a hard failure.
    """
    from routers.ai_insights import _validate_container

    patcher, _ = _patch_workspaces_client(get_workspace={"side_effect": _service_error(500)})
    with patcher:
        # Should not raise for 5xx (just logs warning)
        await _validate_container("container1", "tenant1")


@pytest.mark.asyncio
async def test_validate_container_calls_get_workspace_with_tenant_headers():
    """Request shape: the container name is an argument, never an f-string URL."""
    from routers.ai_insights import _validate_container

    patcher, instance = _patch_workspaces_client()
    with patcher:
        await _validate_container("ctr-1", "tenant-a")

    assert instance.get_workspace.await_count == 1
    args, kwargs = instance.get_workspace.call_args
    assert args[0] == "ctr-1"
    assert kwargs["headers"]["X-Tenant-Id"] == "tenant-a"
    instance.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# _resolve_container_files
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resolve_container_files_with_paths():
    from routers.ai_insights import _resolve_container_files

    files = await _resolve_container_files("c1", "t1", ["a.py", "b.py"])
    assert files == ["a.py", "b.py"]


@pytest.mark.asyncio
async def test_resolve_container_files_no_paths():
    """An empty repository is an empty list — the one case the old code got right."""
    from routers.ai_insights import _resolve_container_files

    patcher, _ = _patch_workspaces_client(exec_={"return_value": _exec_result("")})
    with patcher:
        files = await _resolve_container_files("c1", "t1", None)
    assert files == []


@pytest.mark.asyncio
async def test_resolve_container_files_discovers_recursively():
    """Discovery reaches nested source files and returns real paths.

    The predecessor called ``list_files(path="/workspace")``, a single
    non-recursive ``ls``. A repository keeps its code under ``src/``, so that
    call could not have found ``src/main.py`` even with the response key read
    correctly — it returned a cheerful 200 and an empty analysis.
    """
    from routers.ai_insights import _resolve_container_files

    patcher, instance = _patch_workspaces_client(
        exec_={
            "return_value": _exec_result("/workspace/src/main.py\n/workspace/pkg/util/helper.go\n")
        }
    )
    with patcher:
        files = await _resolve_container_files("c1", "tenant-a", None)

    assert files == ["/workspace/src/main.py", "/workspace/pkg/util/helper.go"]

    args, kwargs = instance.exec.call_args
    assert args[0] == "c1"
    command = kwargs["command"]
    assert command[:4] == ["find", "/workspace", "-type", "f"]
    # argv, not a shell string: an interpolated path with a space or a quote
    # would otherwise re-split inside the container.
    assert isinstance(command, list)
    assert "-name" in command and "*.py" in command
    # Vendored trees are pruned container-side, not filtered afterwards.
    assert "*/node_modules/*" in command
    assert kwargs["headers"]["X-Tenant-Id"] == "tenant-a"


@pytest.mark.asyncio
async def test_resolve_container_files_404_returns_empty_list():
    """A 404 degrades to [] — it must NOT let ServiceError escape.

    Pre-migration this was `if resp.status_code != 200: return []`. If
    ServiceError escaped here, analyze_insights would turn it into a 500.
    """
    from routers.ai_insights import _resolve_container_files

    patcher, _ = _patch_workspaces_client(exec_={"side_effect": _service_error(404)})
    with patcher:
        files = await _resolve_container_files("c1", "t1", None)
    assert files == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [401, 403])
async def test_resolve_container_files_refusal_raises_403(status_code):
    """A permission refusal is surfaced, never flattened into "no files here".

    coding-ai is deliberately absent from coding-containers'
    CODING_ONBEHALF_SUBJECTS, so this was a PERMANENT 403 on every real customer
    workspace — and it was reported to the caller as a successful analysis of
    zero files. "You may not look" and "there is nothing here" are different
    answers and only one of them is actionable.
    """
    from routers.ai_insights import _resolve_container_files

    patcher, _ = _patch_workspaces_client(exec_={"side_effect": _service_error(status_code)})
    with patcher, pytest.raises(HTTPException) as exc_info:
        await _resolve_container_files("c1", "t1", None)
    assert exc_info.value.status_code == 403
    assert "Not permitted" in exc_info.value.detail


@pytest.mark.asyncio
async def test_resolve_container_files_partial_find_keeps_what_it_reached():
    """find exits non-zero on an unreadable subtree but still prints the rest."""
    from routers.ai_insights import _resolve_container_files

    patcher, _ = _patch_workspaces_client(
        exec_={
            "return_value": _exec_result(
                "/workspace/src/a.py\n",
                exit_code=1,
                stderr="find: '/workspace/x': Permission denied",
            )
        }
    )
    with patcher:
        files = await _resolve_container_files("c1", "t1", None)
    assert files == ["/workspace/src/a.py"]


@pytest.mark.asyncio
async def test_resolve_container_files_is_bounded():
    """A monorepo cannot turn one request into an unbounded one."""
    from routers.ai_insights import MAX_DISCOVERED_FILES, _resolve_container_files

    stdout = "".join(f"/workspace/f{i}.py\n" for i in range(MAX_DISCOVERED_FILES + 50))
    patcher, _ = _patch_workspaces_client(exec_={"return_value": _exec_result(stdout)})
    with patcher:
        files = await _resolve_container_files("c1", "t1", None)
    assert len(files) == MAX_DISCOVERED_FILES


@pytest.mark.asyncio
async def test_resolve_container_files_transport_error_returns_empty_list():
    import httpx

    from routers.ai_insights import _resolve_container_files

    patcher, _ = _patch_workspaces_client(exec_={"side_effect": httpx.ConnectError("down")})
    with patcher:
        files = await _resolve_container_files("c1", "t1", None)
    assert files == []


# ---------------------------------------------------------------------------
# _validate_focus_areas  (lines 247-260)
# ---------------------------------------------------------------------------
def test_validate_focus_areas_none():
    from routers.ai_insights import _validate_focus_areas

    assert _validate_focus_areas(None) is None


def test_validate_focus_areas_empty():
    from routers.ai_insights import _validate_focus_areas

    assert _validate_focus_areas([]) is None


def test_validate_focus_areas_valid():
    from routers.ai_insights import _validate_focus_areas

    result = _validate_focus_areas(["security", "architecture"])
    assert InsightType.SECURITY in result
    assert InsightType.ARCHITECTURE in result


def test_validate_focus_areas_invalid():
    from routers.ai_insights import _validate_focus_areas

    with pytest.raises(HTTPException) as exc_info:
        _validate_focus_areas(["invalid_area"])
    assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# _validate_file_paths  (lines 265-276)
# ---------------------------------------------------------------------------
def test_validate_file_paths_all_exist(tmp_path):
    from routers.ai_insights import _validate_file_paths

    f = tmp_path / "test.py"
    f.write_text("x=1")
    result = _validate_file_paths([str(f)])
    assert len(result) == 1


def test_validate_file_paths_none_exist():
    from routers.ai_insights import _validate_file_paths

    with pytest.raises(HTTPException) as exc_info:
        _validate_file_paths(["/nonexistent/file.py"])
    assert exc_info.value.status_code == 404


def test_validate_file_paths_mixed(tmp_path):
    from routers.ai_insights import _validate_file_paths

    f = tmp_path / "exists.py"
    f.write_text("x=1")
    result = _validate_file_paths([str(f), "/no/such/file.py"])
    assert len(result) == 1
    assert str(f) in result[0]


# ---------------------------------------------------------------------------
# _convert_insights_to_model  (lines 285-298)
# ---------------------------------------------------------------------------
def test_convert_insights_to_model():
    from routers.ai_insights import _convert_insights_to_model

    insight = CodeInsight(
        id="i1",
        insight_type=InsightType.ARCHITECTURE,
        priority=InsightPriority.HIGH,
        confidence=InsightConfidence.HIGH,
        title="T",
        description="D",
        rationale="R",
    )
    codebase = CodebaseInsights(
        total_files_analyzed=5,
        total_entities_analyzed=10,
        analysis_time_seconds=1.5,
    )
    codebase.insights_by_type[InsightType.ARCHITECTURE] = [insight]
    codebase.insights_by_priority[InsightPriority.HIGH] = [insight]
    codebase.high_impact_insights = [insight]

    model = _convert_insights_to_model(codebase)
    assert model.total_files_analyzed == 5
    assert "architecture" in model.insights_by_type
    assert "high" in model.insights_by_priority
    assert len(model.high_impact_insights) == 1


# ---------------------------------------------------------------------------
# analyze_insights endpoint  (lines 354-400)
# ---------------------------------------------------------------------------
def test_analyze_no_tenant_header(client):
    """analyze endpoint without tenant header requires authentication."""
    response = client.post(
        "/api/ai/insights/code/analyze",
        json={"container_name": "test-container"},
    )
    # No tenant header -> 401 Unauthorized
    assert response.status_code in (401, 404, 422, 500)


# ---------------------------------------------------------------------------
# capabilities endpoint content validation  (lines 537-539)
# ---------------------------------------------------------------------------
def test_capabilities_response_structure(client):
    response = client.get("/api/ai/insights/code/capabilities")
    assert response.status_code == 200
    data = response.json()
    assert "analysis_features" in data
    assert "ai_capabilities" in data
    assert "quality_metrics" in data
    assert "integration_sources" in data
    assert "output_categories" in data
    assert "effort_estimation" in data
    assert "ai_reasoning" in data
    assert "limitations" in data
    assert "capabilities" in data
    # Nested capabilities should have insight_types
    assert "insight_types" in data["capabilities"]


def test_capabilities_supported_languages(client):
    response = client.get("/api/ai/insights/code/capabilities")
    data = response.json()
    assert "python" in data["supported_languages"]
    assert "java" in data["supported_languages"]
    assert "rust" in data["supported_languages"]


def test_insight_types_content(client):
    response = client.get("/api/ai/insights/code/insight-types")
    data = response.json()
    types = data["insight_types"]
    assert "architecture" in types
    assert "security" in types
    assert "performance" in types
    assert "patterns" in types
    # Each type has expected fields
    for type_info in types.values():
        assert "name" in type_info
        assert "description" in type_info
        assert "focus_areas" in type_info
        assert "typical_priorities" in type_info
        assert "example_insights" in type_info
