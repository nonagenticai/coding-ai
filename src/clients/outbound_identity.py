"""Identity headers coding-ai puts on its calls to sibling coding-* services.

Every outbound call in this service currently proves nothing. It sends a bare
``X-Tenant-Id: <customer>`` and is served because the callee still trusts that
header — a claim any pod in the mesh can make about any tenant. The moment a
callee flips ``TRUST_TENANT_HEADER=false`` (the whole point of that gate), every
one of those calls becomes an anonymous 401 and the feature above it dies.

The fix is the vouching arm added in coding-client 1.11.0: coding-ai presents
its OWN service license in ``Authorization: Bearer`` while the CUSTOMER tenant
stays in the separate ``X-Tenant-Id`` header. The callee honours the asserted
tenant only when this service's ``subject_id`` is in its
``CODING_ONBEHALF_SUBJECTS`` allowlist.

Keeping the two headers apart is the entire mechanism, not a style choice.
``src/wm_client.py`` shows the alternative — it overloads ``X-Tenant-Id`` with
``Bearer <token>`` — and that shape is precisely why it cannot act for a
customer: the callee resolves the tenant from the token, collapsing every
customer into coding-ai's own tenant. That client talks to worker_management,
whose protocol expects exactly that, so it is left alone; nothing here should be
copied back into it.

``Authorization`` prefers the CALLER's credential and falls back to coding-ai's
own. That order is load-bearing, and the reverse cost this service its whole
AI-insight feature:

coding-ai is deliberately absent from coding-containers'
``CODING_ONBEHALF_SUBJECTS`` — the allowlist's own comment names the seven
services that may speak for another tenant and coding-ai is not one of them.
Presenting our own license while asserting a customer tenant is therefore
refused, permanently, with ``403 TENANT_ASSERTION_NOT_PERMITTED``. Measured from
the live pod on 2026-08-25 against workspace ``tour-…-c0``:

    Bearer svc-coding-ai + X-Tenant-Id: <customer>  -> 403
    Bearer svc-coding-ai, no X-Tenant-Id           -> 404  (auth PASSED)
    X-Tenant-Id only, no credential                -> 401

The 404 is the tell: the credential authenticates perfectly well, it simply
resolves to coding-ai's own tenant, which owns no customer workspace. Widening
the allowlist would have made the 403 disappear by granting this service
tenant-impersonation rights it was deliberately denied.

Forwarding sidesteps the question. ``coding_client.auth._onbehalf`` consults
that allowlist ONLY when the asserted tenant disagrees with the token's; a
caller asking about their own tenant's workspace does not disagree and is served
with no allowlist entry at all. The caller's credential is always present on
these paths: ``require_tenant`` 401s without one whenever
``TRUST_TENANT_HEADER`` is false and ``TRUSTED_CIDRS`` is empty, which is this
deployment. coding-gateway forwards it verbatim, or substitutes its own
allowlisted license on a call it vouched for — either arrives able to reach the
workspace.

The fallback to ``service_token()`` remains for genuinely detached work with no
caller to borrow from. It authenticates; it just cannot reach another tenant's
resources, which is the correct outcome rather than a silent one.
"""

from __future__ import annotations

import os

from coding_client.auth.outbound import optional_tenant_headers

from auth.tenant import get_request_credential

__all__ = [
    "outbound_credential",
    "outbound_headers",
    "service_identity_token",
    "service_token",
]


def service_token() -> str | None:
    """This service's own license, read per call so a rotation needs no restart.

    Read from the environment rather than ``Settings`` deliberately: the pod
    carries ``CODING_SERVICE_TOKEN``, and a cached ``get_settings()`` would pin
    the value for the process lifetime, so a rotated secret would need a restart
    to take effect.

    The ``or None`` normalises an empty variable, which is the realistic
    misconfiguration (a Secret key present but blank). ``optional_tenant_headers``
    also tests for truthiness, so this is belt-and-braces rather than the load-
    bearing guard — but an empty ``Bearer `` reaching the wire would burn an
    introspection round-trip on the fleet's hottest auth path before being
    rejected.
    """
    return os.getenv("CODING_SERVICE_TOKEN") or None


def service_identity_token() -> str | None:
    """coding-ai's own service identity, for calls with no end-user behind them.

    Same credential as :func:`service_token`; a SECOND accessor exists only
    because a second env var was invented for it and never deployed.

    ``CODING_AI_LICENSE_TOKEN`` predates gitops #2603 (2026-08-13), which
    provisioned one ``provision_service_token.py`` licence per coding-* service
    and rolled it out fleet-wide as ``CODING_SERVICE_TOKEN``. coding-ai's copy
    introspects to subject ``019ff970-e327`` = ``svc-coding-ai`` (measured
    2026-08-14, recorded in coding-containers' configmap): it IS this service's
    licence, and it is the one the pod actually carries.
    ``CODING_AI_LICENSE_TOKEN`` appears in ZERO manifests, so
    ``settings.coding_ai_license_token`` resolved to ``""`` in every pod that
    ever ran, and ``wm_client``'s ``elif self._service_token:`` therefore emitted
    NO ``X-Tenant-Id`` at all on background and startup calls. Those calls were
    served anyway — worker_management's trusted-network bypass accepts an
    RFC1918 source without a tenant — so they were unattributed, not refused. A
    nil-tenant default is un-billing, not safe.

    The two vars differ in HEADER, never in credential: ``outbound_identity``
    puts it in ``Authorization: Bearer`` for sibling coding-* services, while
    ``wm_client`` puts it in ``X-Tenant-Id: Bearer`` because worker_management's
    protocol resolves the tenant from that header. The module docstring above
    says exactly that and tells you not to copy the header shape back — it makes
    no claim of a separate credential, and none was ever provisioned.

    ``CODING_AI_LICENSE_TOKEN`` still WINS when set: ``scripts/register_workers.py``
    documents it as its ``--license-token`` default, so an operator seeding
    workers against a different WM keeps that override. Read from the
    environment per call, like :func:`service_token`, so a rotation needs no
    restart — a cached ``get_settings()`` would pin the value for the process
    lifetime, which is the other half of why this defect survived.
    """
    return os.getenv("CODING_AI_LICENSE_TOKEN") or service_token()


def outbound_credential() -> str | None:
    """The credential to present outbound: the caller's, else our own.

    Split out from :func:`outbound_headers` so the choice is testable on its own
    and so a call site that needs to know *whose* identity it is using can ask.
    """
    return get_request_credential() or service_token()


def outbound_headers(tenant_id: str | None) -> dict[str, str]:
    """Build the identity headers for a call to a sibling coding-* service.

    ``tenant_id`` is the CUSTOMER's, already validated on the inbound request; it
    is passed through as-is and never defaulted. A missing tenant stays missing
    (``optional_tenant_headers``) rather than being substituted: every call site
    here treats an unscoped lookup as a legitimate degraded case and inventing a
    tenant would authorise the request against a tenant that may not own the
    resource.

    Sending the tenant alongside a forwarded caller credential is not redundant:
    the SDK compares the two and, when they agree, serves the request without
    consulting any on-behalf allowlist. Dropping the header would work equally
    well for the common case but would lose the disagreement check, which is the
    thing that catches a mis-scoped call.

    The credential is attached even with no tenant. Proving who we are is not a
    claim about whose data we want.
    """
    return optional_tenant_headers(tenant_id, service_token=outbound_credential())
