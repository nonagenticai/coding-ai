"""Tenant-isolation tests for the shared container-validation helper.

The helper replaced four divergent copies (three of which were no-op stubs that
returned ``codebase_path="."`` with NO tenant check). These tests pin the
fail-closed behaviour: a container that does not belong to the caller's tenant
comes back 404 from coding-containers and must surface as 404 — never a silent
default path.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from coding_client.clients import ServiceError
from fastapi import HTTPException

from routers.ai_test_insights.shared import validate_and_get_container_for_tenant

MODULE = "routers.ai_test_insights.shared"


def _patch_settings(url: str):
    return patch(f"{MODULE}.get_settings", return_value=SimpleNamespace(coding_containers_url=url))


def _service_error(status_code: int) -> ServiceError:
    """Shaped exactly like the ServiceError that ServiceClient._request raises."""
    return ServiceError(status_code, f"GET /api/workspaces/ctr returned {status_code}", body=None)


def _patch_client(*, response=None, raise_exc: Exception | None = None):
    """Patch the typed CodingWorkspacesClient, not httpx.

    The helper no longer builds a URL or reads ``resp.status_code``: the client
    returns parsed JSON on success and RAISES ``ServiceError`` on any >=400.
    So the non-2xx cases below are ``side_effect=_service_error(...)`` rather
    than a response object carrying a status code.
    """
    client = MagicMock()
    client.get_workspace = (
        AsyncMock(side_effect=raise_exc) if raise_exc else AsyncMock(return_value=response)
    )
    client.close = AsyncMock()
    return patch(f"{MODULE}.CodingWorkspacesClient", return_value=client), client


async def test_no_url_degrades_without_raising():
    """No service configured (local/dev) -> degraded, but never raises.

    Also guards the migration: an empty URL must be caught BEFORE constructing
    CodingWorkspacesClient, which raises RuntimeError on a blank base_url.
    """
    with _patch_settings(""):
        info = await validate_and_get_container_for_tenant("ctr", "tenant-a")
    assert info.codebase_path == "."


async def test_returns_nested_codebase_path():
    """codebase_path lives under data['container'], not at the top level."""
    client_patch, client = _patch_client(
        response={"container": {"codebase_path": "/app/data/repo"}}
    )
    with _patch_settings("http://cc"), client_patch:
        info = await validate_and_get_container_for_tenant("ctr", "tenant-a")
    assert info.codebase_path == "/app/data/repo"
    # The caller's tenant is forwarded so coding-containers can enforce ownership.
    assert client.get_workspace.call_args.kwargs["headers"]["X-Tenant-Id"] == "tenant-a"


async def test_calls_get_workspace_with_the_container_name():
    """Request shape: the name is an argument to the typed method, not an f-string."""
    client_patch, client = _patch_client(response={"container": {"codebase_path": "/x"}})
    with _patch_settings("http://cc"), client_patch:
        await validate_and_get_container_for_tenant("ctr", "tenant-a")
    assert client.get_workspace.call_args.args[0] == "ctr"
    client.close.assert_awaited_once()


async def test_foreign_container_404_is_fail_closed():
    """A container of another tenant returns 404 -> raise, never fall back to '.'.

    The critical migration guard: ServiceError(404) must be mapped to 404 and
    must NOT fall through to the blanket `except Exception` -> 503, which would
    erase the difference between "not your container" and "service down".
    """
    client_patch, _ = _patch_client(raise_exc=_service_error(404))
    with _patch_settings("http://cc"), client_patch, pytest.raises(HTTPException) as exc:
        await validate_and_get_container_for_tenant("ctr", "tenant-a")
    assert exc.value.status_code == 404
    assert exc.value.detail == "Container 'ctr' not found"


async def test_auth_denied_is_fail_closed():
    client_patch, _ = _patch_client(raise_exc=_service_error(403))
    with _patch_settings("http://cc"), client_patch, pytest.raises(HTTPException) as exc:
        await validate_and_get_container_for_tenant("ctr", "tenant-a")
    assert exc.value.status_code == 403


async def test_401_is_fail_closed_as_403():
    client_patch, _ = _patch_client(raise_exc=_service_error(401))
    with _patch_settings("http://cc"), client_patch, pytest.raises(HTTPException) as exc:
        await validate_and_get_container_for_tenant("ctr", "tenant-a")
    assert exc.value.status_code == 403


async def test_other_upstream_error_is_502():
    """Any remaining >=400 stays a 502, distinct from the 503 unreachable case."""
    client_patch, _ = _patch_client(raise_exc=_service_error(500))
    with _patch_settings("http://cc"), client_patch, pytest.raises(HTTPException) as exc:
        await validate_and_get_container_for_tenant("ctr", "tenant-a")
    assert exc.value.status_code == 502


async def test_service_unreachable_fails_closed_503():
    """Configured-but-unreachable must NOT silently bypass the tenant check."""
    client_patch, _ = _patch_client(raise_exc=RuntimeError("connreset"))
    with _patch_settings("http://cc"), client_patch, pytest.raises(HTTPException) as exc:
        await validate_and_get_container_for_tenant("ctr", "tenant-a")
    assert exc.value.status_code == 503
