"""TRUST_TENANT_HEADER + no-fallback-tenant, driven through the REAL app.

Why this file exists separately from test_tenant.py / test_tenant_extended.py
----------------------------------------------------------------------------
Those files build a ``MagicMock()`` request and call ``require_tenant`` directly.
A mock request already satisfies whatever the code asks of it, which is the same
blind spot that let a live prod 500 ship green here twice on 2026-07-30 (see
``TestRealDependencyIsNotAnOpaque500`` in
``test_ai_test_insights_analysis_and_insights.py``). A gate that is only ever
exercised against a mock is not a gate.

So every test below goes through ``main.app`` with a real ``TestClient``: real
Starlette request, real ``TenantMiddleware`` in the real middleware stack, real
``Depends(require_tenant)`` on a real router, no ``dependency_overrides``. The
assertions are on the HTTP outcome the forged caller actually gets — 200 vs 401
— not on whether a flag was read.

``GET /api/ai/architecture/proposals`` is the probe endpoint: it takes
``Depends(require_tenant)`` and, with ``CODING_ANALYSIS_URL`` unset, degrades to
a 200 with ``upstream_available: false`` without touching the network. So a 200
proves the request got past auth AND reached the handler; a 401 proves it did
not.

⚠️ Module identity: pyproject sets ``pythonpath = ["src"]``, so ``auth.tenant``
(what ``main``/the routers import) and ``src.auth.tenant`` (what the sibling
test files import) are two DISTINCT module objects. App-level patching must
target the bare ``auth.*`` path. ``trust_tenant_header()`` reads os.environ at
call time, so ``monkeypatch.setenv`` correctly affects both copies.
"""

from __future__ import annotations

import ipaddress
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from main import app

PROBE = "/api/ai/architecture/proposals?container=fr"

#: A tenant this caller has no credential for. Any in-mesh workload can send it.
FORGED_TENANT = "0195f48b-d000-7000-8000-000000000002"

#: The escalation companion: x-user-roles feeds realm_access -> User.is_admin(),
#: which require_same_tenant honours as "may access ANY tenant".
FORGED_ADMIN_HEADERS = {
    "X-Tenant-Id": FORGED_TENANT,
    "X-User-Id": "attacker",
    "X-User-Roles": "admin,platform-admin",
}


@pytest.fixture
def client(monkeypatch):
    """TestClient with the upstream unset so the probe degrades offline."""
    from config.settings import get_settings

    monkeypatch.setenv("CODING_ANALYSIS_URL", "")
    get_settings.cache_clear()
    yield TestClient(app)
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _no_ambient_bypass(monkeypatch):
    """Neither bypass is armed unless a test arms it deliberately.

    TestClient's client host is "testclient", which is not an IP, so
    _is_trusted_network already returns False — but pinning both knobs keeps a
    401 assertion from passing for an unrelated reason.
    """
    monkeypatch.delenv("SKIP_AUTH", raising=False)
    monkeypatch.delenv("TRUST_TENANT_HEADER", raising=False)
    monkeypatch.delenv("DEFAULT_TENANT_ID", raising=False)
    # Third bypass: an ambient allowlist would let a forged assertion be vouched,
    # turning a 403 assertion below into a 200 for a reason no test named.
    monkeypatch.delenv("CODING_ONBEHALF_SUBJECTS", raising=False)


# ---------------------------------------------------------------------------
# Compatibility control: unset knob == today's behaviour. This is what makes
# merging a live no-op.
# ---------------------------------------------------------------------------


class TestKnobUnsetPreservesTodaysBehaviour:
    def test_bare_forged_header_is_still_honoured(self, client):
        resp = client.get(PROBE, headers={"X-Tenant-Id": FORGED_TENANT})
        assert resp.status_code == 200, resp.text
        assert resp.json()["upstream_available"] is False

    def test_explicit_true_is_honoured(self, client, monkeypatch):
        monkeypatch.setenv("TRUST_TENANT_HEADER", "true")
        assert client.get(PROBE, headers={"X-Tenant-Id": FORGED_TENANT}).status_code == 200

    def test_set_but_empty_is_treated_as_undecided(self, client, monkeypatch):
        """A Deployment that declares the var without a value has not decided."""
        monkeypatch.setenv("TRUST_TENANT_HEADER", "")
        assert client.get(PROBE, headers={"X-Tenant-Id": FORGED_TENANT}).status_code == 200

    def test_typo_does_not_silently_lock_the_service_out(self, client, monkeypatch):
        """An unparseable value keeps the trusting default, loudly.

        A typo turning the service MORE restrictive presents as an
        authentication outage with no obvious cause.
        """
        monkeypatch.setenv("TRUST_TENANT_HEADER", "flase")
        assert client.get(PROBE, headers={"X-Tenant-Id": FORGED_TENANT}).status_code == 200

    def test_no_header_and_no_credential_is_401_either_way(self, client):
        assert client.get(PROBE).status_code == 401


# ---------------------------------------------------------------------------
# Negative control: with the gate disabled the forged header must NOT be
# honoured — assert the outcome the attacker gets, not that a flag was read.
# ---------------------------------------------------------------------------


class TestGateDisabledRejectsForgedHeaders:
    @pytest.mark.parametrize("value", ["false", "False", "0", "no", "off", " FALSE "])
    def test_forged_tenant_header_gets_401(self, client, monkeypatch, value):
        monkeypatch.setenv("TRUST_TENANT_HEADER", value)
        resp = client.get(PROBE, headers={"X-Tenant-Id": FORGED_TENANT})
        assert resp.status_code == 401, resp.text
        assert FORGED_TENANT not in resp.text

    def test_forged_admin_roles_header_is_not_an_escalation(self, client, monkeypatch):
        """X-User-Roles: admin must not buy the caller anything.

        With the gate off the request never resolves a tenant at all, so it 401s
        before any is_admin()/require_same_tenant consideration.
        """
        monkeypatch.setenv("TRUST_TENANT_HEADER", "false")
        resp = client.get(PROBE, headers=FORGED_ADMIN_HEADERS)
        assert resp.status_code == 401, resp.text

    async def test_user_built_from_forged_headers_is_never_admin(self, monkeypatch):
        """Even with the gate ON, coding-ai never derives roles from a header.

        Locks in the divergence from coding_client.auth.tenant, whose
        get_current_user_with_tenant DOES build realm_access from x-user-roles.
        Copying that in without a gate would add a privilege escalation this
        service has never had.
        """
        from starlette.datastructures import Headers
        from starlette.requests import Request

        from auth.tenant import get_current_user_with_tenant

        monkeypatch.setenv("TRUST_TENANT_HEADER", "true")
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": Headers(FORGED_ADMIN_HEADERS).raw,
            "client": ("10.42.0.17", 1234),
            "query_string": b"",
        }
        request = Request(scope)
        request.state.tenant_id = FORGED_TENANT  # as TenantMiddleware would set it

        user = await get_current_user_with_tenant(request)

        assert user.is_admin() is False
        assert user.realm_access in (None, {}, {"roles": []})

    def test_header_is_ignored_not_downgraded_to_default_tenant(self, client, monkeypatch):
        """Disabling trust must not silently serve DEFAULT_TENANT_ID instead.

        The whole point of the knob is "treat the header as absent"; quietly
        substituting a fallback tenant would be a different, quieter data-
        isolation bug.
        """
        monkeypatch.setenv("TRUST_TENANT_HEADER", "false")
        monkeypatch.setenv("DEFAULT_TENANT_ID", "0195f48b-d000-7000-8000-000000000001")
        resp = client.get(PROBE, headers={"X-Tenant-Id": FORGED_TENANT})
        assert resp.status_code == 401, resp.text
        assert "0195f48b" not in resp.text

    def test_license_token_still_resolves_and_wins(self, client, monkeypatch):
        """Ignored, not rejected: resolution falls through to introspection.

        The header names the SAME tenant the license does — the shape
        ``coding-gateway`` puts on the wire when it re-injects the tenant it
        resolved from this very token. With the gate off the LICENSE tenant must
        still reach the handler, proving the header was skipped rather than the
        request short-circuited.

        ⚠️ This test used to send a *disagreeing* header here and assert 200.
        That assertion was the cross-tenant defect written down as a
        requirement: an unvouched caller naming a tenant its license does not
        name is a 403 on coding-github and coding-testing, and silently got the
        token's tenant here. The disagreeing case now lives in
        ``TestOnBehalfOfAssertion`` below, where it belongs; this test keeps the
        half it was actually built to prove.
        """
        monkeypatch.setenv("TRUST_TENANT_HEADER", "false")
        licensed = "0195f48b-d000-7000-8000-00000000000a"
        seen: dict[str, str] = {}

        async def _fake_fetch(container, *, tenant_id):
            seen["tenant_id"] = tenant_id
            return []

        # Both seams — see the note in TestOnBehalfOfAssertion._probe.
        _introspect = AsyncMock(return_value={"active": True, "tenant_id": licensed})
        with (
            patch("coding_client.auth.middleware._introspect_license", new=_introspect),
            patch("auth.tenant._introspect_license", new=_introspect),
        ):
            with patch("routers.ai_architecture.fetch_components", new=_fake_fetch):
                resp = client.get(
                    PROBE,
                    headers={
                        "X-Tenant-Id": licensed,
                        "Authorization": "Bearer lic-tok",
                    },
                )

        assert resp.status_code == 200, resp.text
        assert seen["tenant_id"] == licensed
        assert seen["tenant_id"] != FORGED_TENANT


# ---------------------------------------------------------------------------
# On-behalf-of: who may name a tenant their own license does not name.
# ---------------------------------------------------------------------------

#: The subject_id a vouching sibling's license introspects to. Shape matches
#: what licenses/scripts/provision_service_token.py prints.
VOUCHING_SUBJECT = "019ffa19-db80-700e-9840-8c17e23b35d5"

#: A real, live, *unallowlisted* subject. This is fr's, the one driven against
#: the live pod on 2026-08-14 that got 200 here and 403 on coding-github.
UNVOUCHED_SUBJECT = "019ffe9f-31fd-70da-b3c5-f921f7290b01"

#: The tenant the vouching caller's OWN license names — never the one asserted.
SERVICE_TENANT = "0195f48b-d000-7000-8000-0000000000ff"


class TestOnBehalfOfAssertion:
    """The rule coding-github and coding-testing enforce, and this fork did not.

    Every test drives the real app so the outcome asserted is the HTTP status a
    real caller receives, not a return value from a mocked resolver. The rule
    itself lives in ``coding_client.auth._onbehalf.resolve_tenant_assertion``;
    these tests pin that coding-ai actually *reaches* it, which is the thing that
    was missing — the SDK function was correct and simply never called.
    """

    @staticmethod
    def _probe(client, *, subject_id, asserted, token_tenant=SERVICE_TENANT):
        """Drive PROBE with a license introspecting to ``subject_id``/``token_tenant``.

        Returns ``(response, seen)`` where ``seen["tenant_id"]`` is the tenant
        that actually reached the route handler — absent if auth refused first.
        """
        seen: dict[str, str] = {}

        async def _fake_fetch(container, *, tenant_id):
            seen["tenant_id"] = tenant_id
            return []

        introspect = AsyncMock(
            return_value={
                "active": True,
                "tenant_id": token_tenant,
                "subject_id": subject_id,
            }
        )
        # BOTH introspection seams. coding-ai registers coding_client's
        # AuthMiddleware, which calls the SDK's `_introspect_license`, while
        # `require_tenant` calls this service's forked copy — two distinct
        # objects with two distinct caches. Patching only the fork leaves the
        # middleware introspecting for real, so every probe below 401s before
        # the on-behalf-of logic under test is ever reached.
        with (
            patch("coding_client.auth.middleware._introspect_license", new=introspect),
            patch("auth.tenant._introspect_license", new=introspect),
        ):
            with patch("routers.ai_architecture.fetch_components", new=_fake_fetch):
                resp = client.get(
                    PROBE,
                    headers={
                        "X-Tenant-Id": asserted,
                        "Authorization": "Bearer lic-tok",
                    },
                )
        return resp, seen

    def test_unallowlisted_subject_asserting_a_foreign_tenant_gets_403(self, client, monkeypatch):
        """The defect, driven live 2026-08-14 — 200 here, 403 on the siblings.

        fr's token is a live, active, perfectly valid license. It is simply not
        in CODING_ONBEHALF_SUBJECTS, so it may not speak for a tenant other than
        its own. 403 and not 401: the credential is authenticated, just not
        authorised to make this assertion, and a 401 would invite a pointless
        retry.
        """
        monkeypatch.setenv("TRUST_TENANT_HEADER", "false")
        monkeypatch.setenv("CODING_ONBEHALF_SUBJECTS", VOUCHING_SUBJECT)

        resp, seen = self._probe(client, subject_id=UNVOUCHED_SUBJECT, asserted=FORGED_TENANT)

        assert resp.status_code == 403, resp.text
        # Not a silent downgrade to the token's own tenant: the handler must not
        # have run at all. A 403 that still served data would be worse than a 200.
        assert seen == {}, f"handler ran despite refusal, saw {seen}"
        # And the refusal must not echo the tenant back to the caller.
        assert FORGED_TENANT not in resp.text

    def test_allowlisted_subject_gets_the_asserted_tenant_not_its_own(self, client, monkeypatch):
        """The grant half. Without it a 403-only change is indistinguishable from
        breaking service-to-service traffic outright.

        The tenant that reaches the handler must be the CUSTOMER's (asserted),
        never the vouching service's own — resolving to the service tenant is the
        data-scoping bug that makes worker_management check the wrong byo_mode.
        """
        monkeypatch.setenv("TRUST_TENANT_HEADER", "false")
        monkeypatch.setenv("CODING_ONBEHALF_SUBJECTS", VOUCHING_SUBJECT)

        resp, seen = self._probe(client, subject_id=VOUCHING_SUBJECT, asserted=FORGED_TENANT)

        assert resp.status_code == 200, resp.text
        assert seen["tenant_id"] == FORGED_TENANT
        assert seen["tenant_id"] != SERVICE_TENANT

    def test_empty_allowlist_vouches_for_nobody(self, client, monkeypatch):
        """Fail-closed on install: unset means refuse, never 'allow all'."""
        monkeypatch.setenv("TRUST_TENANT_HEADER", "false")
        monkeypatch.delenv("CODING_ONBEHALF_SUBJECTS", raising=False)

        resp, seen = self._probe(client, subject_id=VOUCHING_SUBJECT, asserted=FORGED_TENANT)

        assert resp.status_code == 403, resp.text
        assert seen == {}

    def test_agreeing_tenant_is_not_a_vouch_and_needs_no_allowlist(self, client, monkeypatch):
        """The north-south path must survive an empty allowlist.

        coding-gateway forwards the client's Authorization AND injects the tenant
        it resolved from that same token, so every browser-originated request
        already arrives in the vouching *shape*. Refusing on presence rather than
        on disagreement would 403 the entire front end.
        """
        monkeypatch.setenv("TRUST_TENANT_HEADER", "false")
        monkeypatch.delenv("CODING_ONBEHALF_SUBJECTS", raising=False)

        resp, seen = self._probe(client, subject_id=UNVOUCHED_SUBJECT, asserted=SERVICE_TENANT)

        assert resp.status_code == 200, resp.text
        assert seen["tenant_id"] == SERVICE_TENANT

    def test_subjectless_license_can_never_be_allowlisted(self, client, monkeypatch):
        """The ``license:<tenant>`` marker is shared by every subject-less license
        for a tenant, so honouring it would hand the vouch to a crowd. Even
        naming it explicitly must refuse.
        """
        monkeypatch.setenv("TRUST_TENANT_HEADER", "false")
        marker = f"license:{SERVICE_TENANT}"
        monkeypatch.setenv("CODING_ONBEHALF_SUBJECTS", marker)

        resp, seen = self._probe(client, subject_id=marker, asserted=FORGED_TENANT)

        assert resp.status_code == 403, resp.text
        assert seen == {}

    def test_vouched_tenant_is_what_the_wm_shim_forwards(self, client, monkeypatch):
        """The contextvar is the entire reason this fork exists — so the vouched
        tenant has to land in it, not the service's own.

        ``services.llm_agents`` reads ``get_request_tenant_id()`` and forwards it
        to worker_management, which enforces THAT tenant's byo_mode. Before this
        fix the value was whatever tenant the caller's license named — and now
        that coding-gateway substitutes its own license on a vouched call, that
        is svc-coding-gateway's tenant, so WM would check the gateway's byo_mode
        and let a byo_required customer reach the platform LLM.
        """
        monkeypatch.setenv("TRUST_TENANT_HEADER", "false")
        monkeypatch.setenv("CODING_ONBEHALF_SUBJECTS", VOUCHING_SUBJECT)

        from auth.tenant import get_request_tenant_id

        seen_ctx: dict[str, str | None] = {}

        async def _fake_fetch(container, *, tenant_id):
            # Read the contextvar from inside the endpoint's task, which is the
            # only place the shim ever reads it.
            seen_ctx["ctx"] = get_request_tenant_id()
            return []

        introspect = AsyncMock(
            return_value={
                "active": True,
                "tenant_id": SERVICE_TENANT,
                "subject_id": VOUCHING_SUBJECT,
            }
        )
        # BOTH introspection seams. coding-ai registers coding_client's
        # AuthMiddleware, which calls the SDK's `_introspect_license`, while
        # `require_tenant` calls this service's forked copy — two distinct
        # objects with two distinct caches. Patching only the fork leaves the
        # middleware introspecting for real, so every probe below 401s before
        # the on-behalf-of logic under test is ever reached.
        with (
            patch("coding_client.auth.middleware._introspect_license", new=introspect),
            patch("auth.tenant._introspect_license", new=introspect),
        ):
            with patch("routers.ai_architecture.fetch_components", new=_fake_fetch):
                resp = client.get(
                    PROBE,
                    headers={
                        "X-Tenant-Id": FORGED_TENANT,
                        "Authorization": "Bearer lic-tok",
                    },
                )

        assert resp.status_code == 200, resp.text
        assert seen_ctx["ctx"] == FORGED_TENANT
        assert seen_ctx["ctx"] != SERVICE_TENANT


# ---------------------------------------------------------------------------
# No-fallback-tenant: the boot gate.
# ---------------------------------------------------------------------------


class TestDefaultTenantBootGate:
    NETS = [ipaddress.ip_network("10.42.0.0/16")]

    def test_armed_bypass_without_a_tenant_refuses_to_boot(self, monkeypatch):
        from auth._default_tenant import enforce_default_tenant_configured

        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.delenv("DEFAULT_TENANT_ID", raising=False)

        with pytest.raises(RuntimeError) as exc_info:
            enforce_default_tenant_configured(self.NETS)

        assert "DEFAULT_TENANT_ID" in str(exc_info.value)
        # Must NOT be an ImportError: several coding-* services wrap their auth
        # import in `except ImportError: ... running WITHOUT authentication`, so
        # an ImportError here would be swallowed into the exact outcome the
        # check exists to prevent.
        assert not isinstance(exc_info.value, ImportError)

    def test_no_bypass_armed_is_a_no_op(self, monkeypatch):
        """Today's config: TRUSTED_CIDRS empty ⇒ this raises for nobody."""
        from auth._default_tenant import enforce_default_tenant_configured

        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.delenv("DEFAULT_TENANT_ID", raising=False)

        assert enforce_default_tenant_configured([]) is None

    def test_tenant_configured_is_a_no_op(self, monkeypatch):
        from auth._default_tenant import enforce_default_tenant_configured

        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("DEFAULT_TENANT_ID", "0195f48b-d000-7000-8000-000000000001")

        assert enforce_default_tenant_configured(self.NETS) is None

    def test_non_production_is_a_no_op(self, monkeypatch):
        from auth._default_tenant import enforce_default_tenant_configured

        monkeypatch.setenv("ENVIRONMENT", "development")
        monkeypatch.delenv("DEFAULT_TENANT_ID", raising=False)

        assert enforce_default_tenant_configured(self.NETS) is None

    @pytest.mark.parametrize("raw", ["", "   "])
    def test_blank_tenant_counts_as_unset(self, monkeypatch, raw):
        """A blank value must never pin a caller to an empty tenant."""
        from auth._default_tenant import enforce_default_tenant_configured, load_default_tenant_id

        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("DEFAULT_TENANT_ID", raw)

        assert load_default_tenant_id() is None
        with pytest.raises(RuntimeError):
            enforce_default_tenant_configured(self.NETS)


class TestNoInventedTenant:
    async def test_armed_bypass_with_unset_tenant_401s_at_request_time(self, monkeypatch):
        """Outcome-level proof, driven through the real ``require_tenant``.

        A real Starlette ``Request`` (not a MagicMock) from a k3s pod IP, with
        the bypass armed and ``DEFAULT_TENANT_ID`` unset, must get a 401 — not a
        made-up tenant. TestClient cannot express this case: its client host is
        the string "testclient", which is never in a CIDR.

        The env var is read through ``load_default_tenant_id()`` rather than
        stubbed, so restoring the ``"n8n-dev"`` default turns this green→red.
        """
        import ipaddress as _ipaddress

        from starlette.requests import Request

        from auth import _default_tenant as dt
        from auth import tenant as tenant_mod

        monkeypatch.delenv("DEFAULT_TENANT_ID", raising=False)
        monkeypatch.setattr(dt, "DEFAULT_TENANT_ID", dt.load_default_tenant_id())
        monkeypatch.setattr(tenant_mod, "TRUSTED_NETWORKS", [_ipaddress.ip_network("10.42.0.0/16")])

        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/",
                "headers": [],
                "client": ("10.42.0.17", 1234),
                "query_string": b"",
            }
        )

        with pytest.raises(HTTPException) as exc_info:
            await tenant_mod.require_tenant(request)

        assert exc_info.value.status_code == 401
        assert "n8n-dev" not in str(exc_info.value.detail)

    def test_resolve_returns_none_and_never_a_literal(self, monkeypatch):
        """Regression for ``os.getenv("DEFAULT_TENANT_ID", "n8n-dev")``.

        ``n8n-dev`` is the stale pre-uuid7 label of a real tenant
        (TENANT_UUIDS["n8n-dev"] = 0195f48b-d000-7000-8000-000000000001), so
        substituting it wrote credential-less callers into a silo no live JWT
        can address.
        """
        from auth import _default_tenant as mod

        monkeypatch.setattr(mod, "DEFAULT_TENANT_ID", None)
        assert mod.resolve_default_tenant("a test") is None

    def test_no_tenant_literal_survives_as_code_in_the_auth_package(self):
        """The literal must not creep back in as a *value*.

        Prose is allowed to name it — the modules explain at length why it was
        removed, and a guard that banned the word would just delete the
        explanation. So this parses each module and inspects only string
        constants that are NOT docstrings; comments are not AST nodes at all, so
        they fall out for free. An ``= "n8n-dev"`` or a ``getenv(..., "n8n-dev")``
        default is what gets caught.
        """
        import ast
        from pathlib import Path

        import auth

        offenders: list[str] = []
        for path in sorted(Path(auth.__file__).parent.glob("*.py")):
            tree = ast.parse(path.read_text())
            docstrings = {
                id(node.body[0].value)
                for node in ast.walk(tree)
                if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef)
                and node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            }
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and id(node) not in docstrings
                    and "n8n-dev" in node.value
                ):
                    offenders.append(f"{path.name}:{node.lineno}")

        assert offenders == []
