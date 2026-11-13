"""Edge-case and error-path tests for tenant auth, middleware, and license introspection.

Complements test_tenant.py by covering:
- _is_trusted_network with IPv6, boundary IPs, edge-case inputs
- _introspect_license cache behaviour, error paths, inactive-not-cached
- TenantMiddleware dispatch branches (user+header, user-no-header, no-user)
- get_tenant_id with missing attribute
- get_current_user_with_tenant user-building path
- require_tenant with bearer tokens (active, inactive, missing tenant_id)
- require_tenant with no client object
- User model edge cases
"""

from __future__ import annotations

import ipaddress
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from coding_client.auth import User
from fastapi import HTTPException

from src.auth import _default_tenant as default_tenant_mod
from src.auth import tenant as tenant_mod
from src.auth.tenant import (
    TenantMiddleware,
    _introspect_license,
    _is_trusted_network,
    _license_cache,
    get_current_user_with_tenant,
    get_tenant_id,
    require_tenant,
)


def _make_request(*, headers=None, state_attrs=None, client_host="127.0.0.1"):
    """Create a minimal mock request."""
    request = MagicMock()
    request.headers = headers or {}
    request.state = SimpleNamespace(**(state_attrs or {}))
    request.client = MagicMock()
    request.client.host = client_host
    return request


# ---------------------------------------------------------------------------
# _is_trusted_network — parametrized
# ---------------------------------------------------------------------------


class TestIsTrustedNetwork:
    """``TRUSTED_CIDRS`` is empty by default, so nothing is trusted unless the
    operator configures it.

    These cases used to assert loopback/192.168/172.16/**10.0.0.0/8** were
    trusted with no configuration at all — a suite encoding the bug, not a
    contract worth keeping: 10/8 contains the k3s pod CIDR 10.42.0.0/16, which
    made every pod in the cluster an authenticated caller. They now assert the
    same addresses are trusted *when configured*; the capability is unchanged.
    """

    @pytest.fixture
    def _configured(self, monkeypatch):
        monkeypatch.setattr(
            tenant_mod,
            "TRUSTED_NETWORKS",
            [
                ipaddress.ip_network(c)
                for c in (
                    "127.0.0.1/32",
                    "::1/128",
                    "192.168.0.0/16",
                    "172.16.0.0/12",
                    "10.0.0.0/8",
                )
            ],
        )

    @pytest.mark.parametrize(
        "ip",
        [
            "127.0.0.1",
            "10.0.0.1",
            "10.255.255.255",
            "172.16.0.1",
            "172.31.255.255",
            "192.168.0.1",
            "192.168.255.254",
        ],
        ids=["loopback", "10-low", "10-high", "172-low", "172-high", "192-low", "192-high"],
    )
    def test_private_ipv4_trusted_when_configured(self, _configured, ip: str):
        assert _is_trusted_network(ip) is True

    @pytest.mark.parametrize(
        "ip",
        ["10.42.0.17", "10.0.0.1", "192.168.1.10", "172.16.0.9", "127.0.0.1", "::1"],
        ids=["k3s-pod", "10-any", "192-any", "172-any", "loopback", "ipv6-loopback"],
    )
    def test_nothing_is_trusted_by_default(self, ip: str):
        """Regression for the live 2026-07-29 finding: a k3s pod IP must not
        authenticate itself merely by being on the pod network.

        Fails against the old default — which is what makes it a real regression
        test rather than a restatement of current behaviour.
        """
        assert _is_trusted_network(ip) is False

    @pytest.mark.parametrize(
        "ip",
        ["8.8.8.8", "1.1.1.1", "203.0.113.5", "198.51.100.7"],
        ids=["google-dns", "cloudflare", "doc-range", "test-net"],
    )
    def test_public_ipv4_not_trusted(self, ip: str):
        assert _is_trusted_network(ip) is False

    @pytest.mark.parametrize(
        "val",
        [None, "", "not-an-ip", "999.999.999.999", "abc::xyz"],
        ids=["none", "empty", "hostname", "invalid-octets", "bad-ipv6"],
    )
    def test_invalid_inputs_not_trusted(self, val):
        assert _is_trusted_network(val) is False

    def test_ipv6_loopback_trusted_when_configured(self, _configured):
        assert _is_trusted_network("::1") is True


# ---------------------------------------------------------------------------
# _introspect_license
# ---------------------------------------------------------------------------


class TestIntrospectLicense:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        _license_cache.clear()
        yield
        _license_cache.clear()

    def _mock_client(self, response=None, side_effect=None):
        """Return a patched httpx.AsyncClient context manager."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        if side_effect:
            mock_client.post = AsyncMock(side_effect=side_effect)
        else:
            mock_client.post = AsyncMock(return_value=response)
        return mock_client

    async def test_active_license_returns_data(self):
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"active": True, "subject_id": "tenant-1"}

        with patch(
            "src.auth.tenant.httpx.AsyncClient", return_value=self._mock_client(response=resp)
        ):
            result = await _introspect_license("tok-active")

        assert result == {"active": True, "subject_id": "tenant-1"}

    async def test_active_license_cached(self):
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"active": True, "subject_id": "tenant-cache"}

        with patch(
            "src.auth.tenant.httpx.AsyncClient", return_value=self._mock_client(response=resp)
        ) as mock_cls:
            r1 = await _introspect_license("tok-cache-test")
            r2 = await _introspect_license("tok-cache-test")

        assert r1 == r2
        # AsyncClient was only constructed once because the second call hits cache
        assert mock_cls.call_count == 1

    async def test_inactive_license_not_cached(self):
        """Inactive licenses should NOT be cached (only active ones are cached)."""
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"active": False}

        with patch(
            "src.auth.tenant.httpx.AsyncClient", return_value=self._mock_client(response=resp)
        ) as mock_cls:
            r1 = await _introspect_license("tok-inactive-1")
            r2 = await _introspect_license("tok-inactive-1")

        assert r1 == {"active": False}
        assert r2 == {"active": False}
        # Called twice because inactive results are not cached
        assert mock_cls.call_count == 2

    async def test_non_200_returns_none(self):
        resp = MagicMock(status_code=500)

        with patch(
            "src.auth.tenant.httpx.AsyncClient", return_value=self._mock_client(response=resp)
        ):
            result = await _introspect_license("tok-500")

        assert result is None

    @pytest.mark.parametrize(
        "error",
        [
            httpx.ConnectError("refused"),
            httpx.TimeoutException("timeout"),
            httpx.ReadError("read error"),
        ],
        ids=["connect", "timeout", "read"],
    )
    async def test_network_errors_return_none(self, error):
        with patch(
            "src.auth.tenant.httpx.AsyncClient", return_value=self._mock_client(side_effect=error)
        ):
            result = await _introspect_license(f"tok-err-{type(error).__name__}")

        assert result is None


# ---------------------------------------------------------------------------
# TenantMiddleware dispatch
# ---------------------------------------------------------------------------


class TestTenantMiddleware:
    async def test_user_with_header_overrides_tenant(self):
        """X-Tenant-Id header overrides user's existing tenant_id."""
        middleware = TenantMiddleware(app=MagicMock())
        user = User(sub="u1", preferred_username="u1", tenant_id="original")
        request = _make_request(
            headers={"X-Tenant-Id": "override"},
            state_attrs={"user": user},
        )
        call_next = AsyncMock(return_value=MagicMock())

        await middleware.dispatch(request, call_next)

        assert user.tenant_id == "override"
        assert request.state.tenant_id == "override"

    async def test_user_without_tenant_falls_back_to_sub(self):
        """User with no tenant_id and no header gets sub as tenant_id."""
        middleware = TenantMiddleware(app=MagicMock())
        user = User(sub="user-sub-123", preferred_username="user-sub-123", tenant_id=None)
        request = _make_request(state_attrs={"user": user})
        call_next = AsyncMock(return_value=MagicMock())

        await middleware.dispatch(request, call_next)

        assert user.tenant_id == "user-sub-123"

    async def test_user_with_existing_tenant_no_header_unchanged(self):
        """User with tenant_id and no header keeps their tenant_id."""
        middleware = TenantMiddleware(app=MagicMock())
        user = User(sub="u1", preferred_username="u1", tenant_id="existing")
        request = _make_request(state_attrs={"user": user})
        call_next = AsyncMock(return_value=MagicMock())

        await middleware.dispatch(request, call_next)

        assert user.tenant_id == "existing"

    async def test_no_user_with_header(self):
        """Without user, X-Tenant-Id header sets state.tenant_id."""
        middleware = TenantMiddleware(app=MagicMock())
        request = _make_request(headers={"X-Tenant-Id": "header-tenant"})
        call_next = AsyncMock(return_value=MagicMock())

        await middleware.dispatch(request, call_next)

        assert request.state.tenant_id == "header-tenant"

    async def test_no_user_no_header_no_tenant_set(self):
        """Without user or header, no tenant_id is set on state."""
        middleware = TenantMiddleware(app=MagicMock())
        request = _make_request()
        call_next = AsyncMock(return_value=MagicMock())

        await middleware.dispatch(request, call_next)

        assert not hasattr(request.state, "tenant_id")

    async def test_call_next_invoked_and_response_returned(self):
        """Middleware always calls call_next and returns its response."""
        middleware = TenantMiddleware(app=MagicMock())
        request = _make_request()
        expected_response = MagicMock()
        call_next = AsyncMock(return_value=expected_response)

        result = await middleware.dispatch(request, call_next)

        call_next.assert_awaited_once_with(request)
        assert result is expected_response


# ---------------------------------------------------------------------------
# get_tenant_id
# ---------------------------------------------------------------------------


class TestGetTenantId:
    def test_returns_tenant_from_state(self):
        request = _make_request(state_attrs={"tenant_id": "t1"})
        assert get_tenant_id(request) == "t1"

    def test_returns_none_when_absent(self):
        request = _make_request()
        assert get_tenant_id(request) is None


# ---------------------------------------------------------------------------
# require_tenant — bearer token paths
# ---------------------------------------------------------------------------


class TestRequireTenantBearer:
    async def test_active_bearer_returns_tenant_id(self):
        request = _make_request(
            headers={"Authorization": "Bearer active-tok"},
            client_host="8.8.8.8",
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SKIP_AUTH", None)
            with patch(
                "src.auth.tenant._introspect_license",
                new_callable=AsyncMock,
                return_value={"active": True, "tenant_id": "lic-tenant"},
            ):
                tenant = await require_tenant(request)

        assert tenant == "lic-tenant"
        assert request.state.tenant_id == "lic-tenant"

    async def test_active_bearer_prefers_tenant_id_over_subject_id(self):
        """Regression: tenant is `tenant_id` (workspace), not `subject_id` (user).
        When the license returns both, scope on tenant_id."""
        request = _make_request(
            headers={"Authorization": "Bearer both-tok"},
            client_host="8.8.8.8",
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SKIP_AUTH", None)
            with patch(
                "src.auth.tenant._introspect_license",
                new_callable=AsyncMock,
                return_value={"active": True, "tenant_id": "the-tenant", "subject_id": "the-user"},
            ):
                tenant = await require_tenant(request)

        assert tenant == "the-tenant"
        assert tenant != "the-user"

    async def test_inactive_bearer_raises_401(self):
        request = _make_request(
            headers={"Authorization": "Bearer inactive-tok"},
            client_host="8.8.8.8",
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SKIP_AUTH", None)
            with patch(
                "src.auth.tenant._introspect_license",
                new_callable=AsyncMock,
                return_value={"active": False},
            ):
                with pytest.raises(HTTPException) as exc_info:
                    await require_tenant(request)

        assert exc_info.value.status_code == 401

    async def test_bearer_active_but_no_tenant_id_falls_through(self, monkeypatch):
        """Active license without tenant_id should fall through to next check.

        The fall-through *target* here is the trusted-network bypass, which is
        now opt-in, so the test configures it explicitly. What is under test is
        the fall-through, not the default.
        """
        monkeypatch.setattr(
            tenant_mod, "TRUSTED_NETWORKS", [ipaddress.ip_network("192.168.0.0/16")]
        )
        monkeypatch.setattr(default_tenant_mod, "DEFAULT_TENANT_ID", "tnt-configured")
        request = _make_request(
            headers={"Authorization": "Bearer no-tenant-tok"},
            client_host="192.168.1.1",  # trusted once configured above
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SKIP_AUTH", None)
            with patch(
                "src.auth.tenant._introspect_license",
                new_callable=AsyncMock,
                return_value={"active": True},  # no tenant_id
            ):
                tenant = await require_tenant(request)

        # Falls through to trusted network check
        assert tenant == "tnt-configured"

    async def test_introspect_failure_falls_through(self, monkeypatch):
        """If license introspection returns None, flow continues to next check."""
        monkeypatch.setattr(
            tenant_mod, "TRUSTED_NETWORKS", [ipaddress.ip_network("192.168.0.0/16")]
        )
        monkeypatch.setattr(default_tenant_mod, "DEFAULT_TENANT_ID", "tnt-configured")
        request = _make_request(
            headers={"Authorization": "Bearer fail-tok"},
            client_host="192.168.1.1",
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SKIP_AUTH", None)
            with patch(
                "src.auth.tenant._introspect_license",
                new_callable=AsyncMock,
                return_value=None,
            ):
                tenant = await require_tenant(request)

        assert tenant == "tnt-configured"


# ---------------------------------------------------------------------------
# require_tenant — no client object
# ---------------------------------------------------------------------------


class TestRequireTenantNoClient:
    async def test_no_client_raises_401(self):
        """request.client is None (e.g. websocket proxy) should raise 401."""
        request = _make_request(client_host="8.8.8.8")
        request.client = None
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SKIP_AUTH", None)
            with pytest.raises(HTTPException) as exc_info:
                await require_tenant(request)
        assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# get_current_user_with_tenant
# ---------------------------------------------------------------------------


class TestGetCurrentUserWithTenant:
    async def test_existing_user_reused(self):
        """Existing user on request.state is reused, not replaced."""
        user = User(sub="u1", preferred_username="u1", tenant_id="t1")
        request = _make_request(state_attrs={"user": user, "tenant_id": "t1"})
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SKIP_AUTH", None)
            result = await get_current_user_with_tenant(request)
        assert result is user
        assert result.tenant_id == "t1"

    async def test_no_user_creates_new(self):
        """When no user exists, a new User is created and assigned to state."""
        request = _make_request(state_attrs={"tenant_id": "assigned-t"})
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SKIP_AUTH", None)
            result = await get_current_user_with_tenant(request)
        assert isinstance(result, User)
        assert result.tenant_id == "assigned-t"
        assert request.state.user is result

    async def test_user_without_tenant_gets_assigned(self):
        """User with tenant_id=None gets tenant from require_tenant."""
        user = User(sub="u1", preferred_username="u1", tenant_id=None)
        request = _make_request(
            state_attrs={"user": user},
            client_host="192.168.1.1",
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SKIP_AUTH", None)
            result = await get_current_user_with_tenant(request)
        # user.effective_tenant_id returns sub "u1", which is truthy, so
        # require_tenant returns "u1" at Priority 3 — before the trusted-network
        # check is ever reached.
        assert result.tenant_id == "u1"
