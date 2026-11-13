"""Tests for tenant authentication middleware and dependencies."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from src.auth.tenant import require_tenant


def _make_request(*, headers=None, state_attrs=None, client_host="127.0.0.1"):
    """Create a minimal mock request."""
    request = MagicMock()
    request.headers = headers or {}
    request.state = SimpleNamespace(**(state_attrs or {}))
    request.client = MagicMock()
    request.client.host = client_host
    return request


@pytest.mark.parametrize("value", ["true", "True", "TRUE", "1", "yes"])
async def test_skip_auth_no_longer_bypasses_anything(value):
    """SKIP_AUTH was deleted from this fork and from coding-client 1.16.0.

    Inverted rather than deleted: the old assertion (SKIP_AUTH=true yields
    "test-tenant") was the hatch itself, and without a test in its place nothing
    would notice it coming back. This service carried an INDEPENDENT copy of the
    branch, so removing it from the SDK alone would have left it live here.
    """
    request = _make_request()
    with patch.dict(os.environ, {"SKIP_AUTH": value}), pytest.raises(HTTPException) as exc:
        await require_tenant(request)
    assert exc.value.status_code == 401


async def test_tenant_from_header_via_state():
    request = _make_request(state_attrs={"tenant_id": "my-tenant"})
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("SKIP_AUTH", None)
        tenant = await require_tenant(request)
    assert tenant == "my-tenant"


async def test_trusted_network_returns_default_tenant_when_configured(monkeypatch):
    """The bypass still works — but only when an operator opts in TWICE.

    This used to pass with no configuration at all, because TRUSTED_CIDRS
    defaulted to a list containing 192.168.0.0/16 and 10.0.0.0/8. It then
    asserted the granted tenant was the literal ``"n8n-dev"``; now the operator
    must also name the tenant, and it is whatever they named.
    """
    import ipaddress

    from src.auth import _default_tenant as default_tenant_mod
    from src.auth import tenant as tenant_mod

    monkeypatch.setattr(tenant_mod, "TRUSTED_NETWORKS", [ipaddress.ip_network("192.168.0.0/16")])
    monkeypatch.setattr(default_tenant_mod, "DEFAULT_TENANT_ID", "tnt-configured")
    request = _make_request(client_host="192.168.1.100")
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("SKIP_AUTH", None)
        tenant = await require_tenant(request)
    assert tenant == "tnt-configured"


async def test_trusted_network_without_default_tenant_raises_401(monkeypatch):
    """An armed bypass with no tenant configured must REJECT, not invent one.

    Regression for the ``os.getenv("DEFAULT_TENANT_ID", "n8n-dev")`` literal:
    ``n8n-dev`` is the stale pre-uuid7 label of a real tenant, so substituting
    it wrote credential-less callers into a silo no live JWT can address. Fails
    against the old code, which returned "n8n-dev" here.
    """
    import ipaddress

    from src.auth import _default_tenant as default_tenant_mod
    from src.auth import tenant as tenant_mod

    monkeypatch.setattr(tenant_mod, "TRUSTED_NETWORKS", [ipaddress.ip_network("192.168.0.0/16")])
    monkeypatch.setattr(default_tenant_mod, "DEFAULT_TENANT_ID", None)
    request = _make_request(client_host="192.168.1.100")
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("SKIP_AUTH", None)
        with pytest.raises(HTTPException) as exc_info:
            await require_tenant(request)
    assert exc_info.value.status_code == 401
    assert "n8n-dev" not in str(exc_info.value.detail)


async def test_pod_cidr_is_not_trusted_by_default():
    """A k3s pod IP must NOT authenticate itself by being on the pod network.

    Regression for the live 2026-07-29 finding: TRUSTED_CIDRS defaulted to
    ``...,10.0.0.0/8``, which contains the k3s pod CIDR 10.42.0.0/16, so every
    pod in the cluster was granted DEFAULT_TENANT_ID instead of a 401.
    Fails against the old default — which is what makes it a real regression
    test rather than a restatement of current behaviour.
    """
    request = _make_request(client_host="10.42.0.17")
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("SKIP_AUTH", None)
        with pytest.raises(HTTPException) as exc_info:
            await require_tenant(request)
    assert exc_info.value.status_code == 401


async def test_missing_tenant_raises_401():
    request = _make_request(client_host="8.8.8.8")
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("SKIP_AUTH", None)
        with pytest.raises(HTTPException) as exc_info:
            await require_tenant(request)
    assert exc_info.value.status_code == 401
