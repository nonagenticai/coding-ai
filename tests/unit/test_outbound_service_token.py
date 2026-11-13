"""coding-ai must prove WHO it is on every call to a sibling coding-* service.

Until now each of these four call sites sent a bare ``X-Tenant-Id: <customer>``
and nothing else. That is an unverified claim, honoured only because the callees
still trust the header; the moment one flips ``TRUST_TENANT_HEADER=false`` the
call is anonymous and 401s. Worse, three of the four sites swallow an auth
refusal into a plausible-looking result — ``_validate_container`` turns it into
"container not found", ``_resolve_container_files`` into an empty file list,
``fetch_components`` into an empty ``proposals[]`` — so a dropped credential is
invisible in production and invisible to any behavioural test.

Hence these tests assert on the WIRE, via ``httpx.MockTransport``, and on both
halves at once:

* ``Authorization: Bearer <token>`` carries coding-ai's OWN service license;
* ``X-Tenant-Id`` still carries the CUSTOMER's tenant.

The second assertion is not redundant. The tempting shortcut — stuffing the
token into ``X-Tenant-Id``, which ``src/wm_client.py`` legitimately does for
worker_management's different protocol — would satisfy "the credential reached
the wire" while collapsing every customer into coding-ai's own tenant. A test
that only checked ``Authorization`` would pass on that mutant.

And the absence case: no token configured must mean NO ``Authorization`` header
at all. Never a fabricated one, never a bare ``Bearer ``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

TOKEN = "svc-coding-ai-token-abc"
TENANT = "019edc48-6060-76e9-8d8f-2e720e12c609"


def _capturing_transport(seen: list[httpx.Request], **response_kwargs) -> httpx.MockTransport:
    """A transport that records the real outbound request and answers 200."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, **(response_kwargs or {"json": {}}))

    return httpx.MockTransport(handler)


# ─────────────────────────────────────────────────────────────────────────────
# clients.outbound_identity — the shared builder
# ─────────────────────────────────────────────────────────────────────────────


class TestOutboundHeaders:
    def test_credential_and_tenant_are_separate_headers(self, monkeypatch):
        from clients.outbound_identity import outbound_headers

        monkeypatch.setenv("CODING_SERVICE_TOKEN", TOKEN)
        headers = outbound_headers(TENANT)

        assert headers["Authorization"] == f"Bearer {TOKEN}"
        assert headers["X-Tenant-Id"] == TENANT

    def test_no_token_means_no_authorization_header(self, monkeypatch):
        from clients.outbound_identity import outbound_headers

        monkeypatch.delenv("CODING_SERVICE_TOKEN", raising=False)
        headers = outbound_headers(TENANT)

        assert "Authorization" not in headers
        assert headers["X-Tenant-Id"] == TENANT

    def test_empty_token_is_not_sent_as_a_bare_bearer(self, monkeypatch):
        """A blank Secret key is the realistic misconfiguration, not an absent one."""
        from clients.outbound_identity import outbound_headers

        monkeypatch.setenv("CODING_SERVICE_TOKEN", "")

        assert "Authorization" not in outbound_headers(TENANT)

    def test_token_is_read_per_call_not_at_import(self, monkeypatch):
        """A rotated Secret must take effect without restarting the pod."""
        from clients.outbound_identity import outbound_headers

        monkeypatch.setenv("CODING_SERVICE_TOKEN", "first")
        assert outbound_headers(TENANT)["Authorization"] == "Bearer first"
        monkeypatch.setenv("CODING_SERVICE_TOKEN", "second")
        assert outbound_headers(TENANT)["Authorization"] == "Bearer second"

    def test_credential_is_sent_even_with_no_tenant(self, monkeypatch):
        """Proving who we are is not a claim about whose data we want."""
        from clients.outbound_identity import outbound_headers

        monkeypatch.setenv("CODING_SERVICE_TOKEN", TOKEN)
        headers = outbound_headers(None)

        assert headers["Authorization"] == f"Bearer {TOKEN}"
        assert "X-Tenant-Id" not in headers

    def test_missing_tenant_is_never_defaulted(self, monkeypatch):
        """An invented tenant would authorise us against data we do not own."""
        from clients.outbound_identity import outbound_headers

        monkeypatch.delenv("CODING_SERVICE_TOKEN", raising=False)

        assert outbound_headers(None) == {}


# ─────────────────────────────────────────────────────────────────────────────
# clients.coding_analysis_client.fetch_components  →  coding-analysis
# ─────────────────────────────────────────────────────────────────────────────


class TestAnalysisClientSendsTheServiceLicense:
    async def test_license_and_customer_tenant_both_reach_the_wire(self, monkeypatch):
        from clients import coding_analysis_client
        from config.settings import get_settings

        monkeypatch.setenv("CODING_SERVICE_TOKEN", TOKEN)
        monkeypatch.setenv("CODING_ANALYSIS_URL", "http://upstream.test")
        get_settings.cache_clear()

        seen: list[httpx.Request] = []
        client = httpx.AsyncClient(transport=_capturing_transport(seen, json=[]))
        monkeypatch.setattr(coding_analysis_client, "_get_client", lambda timeout=5.0: client)
        try:
            await coding_analysis_client.fetch_components("fr", tenant_id=TENANT)
        finally:
            await client.aclose()
            get_settings.cache_clear()

        assert seen[0].headers["authorization"] == f"Bearer {TOKEN}"
        # The tenant is the CUSTOMER's. Moving the token here instead would make
        # coding-analysis resolve every request as coding-ai's own tenant.
        assert seen[0].headers["x-tenant-id"] == TENANT

    async def test_no_authorization_header_when_no_token_configured(self, monkeypatch):
        from clients import coding_analysis_client
        from config.settings import get_settings

        monkeypatch.delenv("CODING_SERVICE_TOKEN", raising=False)
        monkeypatch.setenv("CODING_ANALYSIS_URL", "http://upstream.test")
        get_settings.cache_clear()

        seen: list[httpx.Request] = []
        client = httpx.AsyncClient(transport=_capturing_transport(seen, json=[]))
        monkeypatch.setattr(coding_analysis_client, "_get_client", lambda timeout=5.0: client)
        try:
            await coding_analysis_client.fetch_components("fr", tenant_id=TENANT)
        finally:
            await client.aclose()
            get_settings.cache_clear()

        assert "authorization" not in seen[0].headers
        assert seen[0].headers["x-tenant-id"] == TENANT


# ─────────────────────────────────────────────────────────────────────────────
# routers.ai_insights  →  coding-containers  (two call sites)
# ─────────────────────────────────────────────────────────────────────────────


def _patch_service_client_transport(monkeypatch, seen: list[httpx.Request], **response_kwargs):
    """Route the typed client's httpx.AsyncClient through MockTransport.

    These call sites no longer build an inline ``httpx.AsyncClient``; they use
    ``CodingWorkspacesClient``, which lazily constructs its httpx client in
    ``ServiceClient._ensure_client``. Patching the constructor THERE keeps these
    assertions genuinely wire-level — a mocked client method would only prove
    what we passed in, not what left the process. The credential reaching the
    wire is the whole claim of this file, so the seam moves down a layer rather
    than being replaced by a method mock.
    """
    from coding_client.clients import base as service_client_base

    transport = _capturing_transport(seen, **response_kwargs)
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real(*args, **kwargs)

    monkeypatch.setattr(service_client_base.httpx, "AsyncClient", factory)


class TestAiInsightsContainerCallsSendTheServiceLicense:
    async def test_validate_container_sends_license_and_tenant(self, monkeypatch):
        from routers import ai_insights

        monkeypatch.setenv("CODING_SERVICE_TOKEN", TOKEN)
        monkeypatch.setattr(
            ai_insights,
            "get_settings",
            lambda: SimpleNamespace(coding_containers_url="http://coding-containers:8691"),
        )
        seen: list[httpx.Request] = []
        _patch_service_client_transport(monkeypatch, seen, json={"container": {"name": "c1"}})

        await ai_insights._validate_container("c1", TENANT)

        assert seen[0].headers["authorization"] == f"Bearer {TOKEN}"
        assert seen[0].headers["x-tenant-id"] == TENANT

    async def test_validate_container_sends_no_authorization_when_unset(self, monkeypatch):
        from routers import ai_insights

        monkeypatch.delenv("CODING_SERVICE_TOKEN", raising=False)
        monkeypatch.setattr(
            ai_insights,
            "get_settings",
            lambda: SimpleNamespace(coding_containers_url="http://coding-containers:8691"),
        )
        seen: list[httpx.Request] = []
        _patch_service_client_transport(monkeypatch, seen, json={"container": {"name": "c1"}})

        await ai_insights._validate_container("c1", TENANT)

        assert "authorization" not in seen[0].headers
        assert seen[0].headers["x-tenant-id"] == TENANT

    async def test_resolve_container_files_sends_license_and_tenant(self, monkeypatch):
        from routers import ai_insights

        monkeypatch.setenv("CODING_SERVICE_TOKEN", TOKEN)
        monkeypatch.setattr(
            ai_insights,
            "get_settings",
            lambda: SimpleNamespace(coding_containers_url="http://coding-containers:8691"),
        )
        seen: list[httpx.Request] = []
        _patch_service_client_transport(
            monkeypatch,
            seen,
            json={"exit_code": 0, "stdout": "/workspace/a.py\n", "stderr": ""},
        )

        files = await ai_insights._resolve_container_files("c1", TENANT)

        assert files == ["/workspace/a.py"]
        # No caller credential in scope, so we fall back to our own license.
        assert seen[0].headers["authorization"] == f"Bearer {TOKEN}"
        assert seen[0].headers["x-tenant-id"] == TENANT


class TestContainerCallsPreferTheCallersCredential:
    """coding-ai acts AS the caller, not on the caller's behalf.

    It is deliberately absent from coding-containers' CODING_ONBEHALF_SUBJECTS,
    so presenting its own license while asserting a customer tenant is refused
    permanently with 403 TENANT_ASSERTION_NOT_PERMITTED. The SDK consults that
    allowlist only when the asserted tenant DISAGREES with the token's, so
    forwarding the caller's credential is served with no allowlist entry — and
    without granting this service tenant-impersonation rights it was denied on
    purpose.
    """

    async def test_caller_credential_is_forwarded_over_our_own(self, monkeypatch):
        from auth import tenant as tenant_mod
        from routers import ai_insights

        monkeypatch.setenv("CODING_SERVICE_TOKEN", TOKEN)
        monkeypatch.setattr(
            ai_insights,
            "get_settings",
            lambda: SimpleNamespace(coding_containers_url="http://coding-containers:8691"),
        )
        seen: list[httpx.Request] = []
        _patch_service_client_transport(monkeypatch, seen, json={"container": {"name": "c1"}})

        token = tenant_mod._request_credential_var.set("caller-license")
        try:
            await ai_insights._validate_container("c1", TENANT)
        finally:
            tenant_mod._request_credential_var.reset(token)

        assert seen[0].headers["authorization"] == "Bearer caller-license"
        # The tenant still rides along: the SDK compares the two and serves the
        # request when they agree. Dropping it would lose the disagreement check.
        assert seen[0].headers["x-tenant-id"] == TENANT

    async def test_falls_back_to_our_license_when_detached(self, monkeypatch):
        """Background work has no caller to borrow from and must still identify itself."""
        from auth import tenant as tenant_mod
        from routers import ai_insights

        monkeypatch.setenv("CODING_SERVICE_TOKEN", TOKEN)
        monkeypatch.setattr(
            ai_insights,
            "get_settings",
            lambda: SimpleNamespace(coding_containers_url="http://coding-containers:8691"),
        )
        seen: list[httpx.Request] = []
        _patch_service_client_transport(monkeypatch, seen, json={"container": {"name": "c1"}})

        assert tenant_mod.get_request_credential() is None
        await ai_insights._validate_container("c1", TENANT)

        assert seen[0].headers["authorization"] == f"Bearer {TOKEN}"
        assert seen[0].headers["x-tenant-id"] == TENANT
        assert seen[0].url.path == "/api/workspaces/c1"


# ─────────────────────────────────────────────────────────────────────────────
# routers.ai_test_insights.shared  →  coding-containers
# ─────────────────────────────────────────────────────────────────────────────


class TestSharedContainerValidationSendsTheServiceLicense:
    """This helper IS the tenant-isolation boundary for intelligent testing.

    It is the site where losing the credential is most expensive: a 401 becomes a
    403 "not authorized", which reads like a correct denial rather than a broken
    caller.
    """

    async def _call(self, monkeypatch, seen: list[httpx.Request]):
        from routers.ai_test_insights import shared

        # The helper now calls CodingWorkspacesClient, so the transport seam is
        # in coding_client.clients.base, not in this module's namespace.
        _patch_service_client_transport(
            monkeypatch, seen, json={"container": {"codebase_path": "/w"}}
        )

        with patch.object(
            shared,
            "get_settings",
            return_value=SimpleNamespace(coding_containers_url="http://cc:8691"),
        ):
            return await shared.validate_and_get_container_for_tenant("ctr", TENANT)

    async def test_sends_license_and_customer_tenant(self, monkeypatch):
        monkeypatch.setenv("CODING_SERVICE_TOKEN", TOKEN)
        seen: list[httpx.Request] = []

        info = await self._call(monkeypatch, seen)

        assert info.codebase_path == "/w"
        assert seen[0].headers["authorization"] == f"Bearer {TOKEN}"
        assert seen[0].headers["x-tenant-id"] == TENANT

    async def test_sends_no_authorization_when_unset(self, monkeypatch):
        monkeypatch.delenv("CODING_SERVICE_TOKEN", raising=False)
        seen: list[httpx.Request] = []

        await self._call(monkeypatch, seen)

        assert "authorization" not in seen[0].headers
        assert seen[0].headers["x-tenant-id"] == TENANT


# ─────────────────────────────────────────────────────────────────────────────
# Census guard
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("module_path", "needle"),
    [
        ("src/clients/coding_analysis_client.py", "outbound_headers(tenant_id)"),
        ("src/routers/ai_insights.py", "outbound_headers(tenant_id)"),
        ("src/routers/ai_test_insights/shared.py", "outbound_headers(tenant_id)"),
    ],
)
def test_no_sibling_call_site_hand_rolls_the_tenant_header(module_path, needle):
    """A new call site must not reintroduce a hand-built ``{"X-Tenant-Id": ...}``.

    Hand-rolling the dict is exactly how these four sites ended up credential-less
    in the first place: each was written independently and each forgot
    independently. ``src/wm_client.py`` and ``src/wm_bootstrap.py`` are excluded
    on purpose — worker_management's protocol wants the token IN ``X-Tenant-Id``.
    """
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2] / module_path).read_text()

    assert needle in source
    assert '"X-Tenant-Id":' not in source
    assert '"x-tenant-id":' not in source
