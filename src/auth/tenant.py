"""Tenant isolation dependencies for coding-ai.

Priority order for tenant extraction (``_resolve_tenant``):

1. ``request.state.tenant_id`` — written by ``TenantMiddleware`` from the
   ``X-Tenant-Id`` header
2. ``Authorization: Bearer <token>`` -> license introspection, then the
   on-behalf-of rule (:func:`coding_client.auth._onbehalf.resolve_tenant_assertion`)
3. ``request.state.user.effective_tenant_id`` (internal service-to-service)
4. Trusted-network bypass -> ``DEFAULT_TENANT_ID`` (``TRUSTED_CIDRS``, empty by
   default; no tenant configured means no grant)

Step 2 applies the SDK's on-behalf-of rule by CALLING it, not by restating it.
A caller presenting its own service license may name a different tenant in
``X-Tenant-Id``; that assertion is honoured only when the license introspects to
a ``subject_id`` in ``CODING_ONBEHALF_SUBJECTS``, and an unvouched disagreement
is a **403**. Until 2026-08-14 this fork had no such branch at all — it returned
the token's tenant and never read the asserted one — so this service's allowlist
was decorative while coding-github and coding-testing enforced it. Whatever else
diverges below, that rule must keep coming from the SDK: it is the one piece
three services have to agree on, and hand-copied auth rules in this codebase have
drifted apart every single time.

⚠️ Step 1 outranks step 2, and step 1 is an UNVERIFIED claim. A caller that can
reach :8690 pod-to-pod (which does not traverse the edge ext_authz filter that
would have stripped and re-injected the header) can assert any tenant and be
served it, without any license at all. ``TRUST_TENANT_HEADER=false`` turns step 1
off; see :mod:`._header_trust` for why the default leaves it on.

Two places in this service read an inbound identity header, and the distinction
between them is the whole design:

* ``TenantMiddleware.dispatch`` below reads ``X-Tenant-Id`` as an *unverified
  claim of who I am*, and is therefore GATED on ``trust_tenant_header()``.
* ``resolve_tenant_assertion`` (in the SDK) reads it as *whom I am asking to act
  for*, and is deliberately UNGATED — it is reached only after a license has
  already introspected active, and it grants nothing that the allowlist has not
  already permitted. Gating it would make vouching dead in exactly the
  ``TRUST_TENANT_HEADER=false`` posture it exists to serve. See that module's
  ``asserted_tenant`` docstring.

Any NEW ungated read that is not of the second kind is a forgery path.

Unlike ``coding_client.auth.tenant``, this fork still has no direct
``x-tenant-id`` read in ``require_tenant`` ("Priority 1b"), no
``optional_tenant``, no ``create_tenant_context``, and its
``get_current_user_with_tenant`` does NOT build a ``User`` from ``x-user-id`` /
``x-user-roles``. If any of those are ever added here, they must carry the same
``trust_tenant_header()`` gate or this service closes one route and leaves the
neighbouring one forgeable.

Why this is still a fork at all: ``require_tenant`` stashes the resolved tenant
in a request-scoped contextvar the WM shim needs, and the SDK's does not. That is
the ONLY behaviour worth forking for; see ``require_tenant`` below. Full
delegation to the SDK's ``require_tenant`` was considered and rejected — it would
also import Priority 1b, and it would move ``_introspect_license`` /
``TRUSTED_NETWORKS`` out from under this service's tests, which patch them by
name. The rule that must not drift is imported instead.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import os
from contextvars import ContextVar

import httpx
from cachetools import TTLCache
from coding_client.auth import User
from coding_client.auth._onbehalf import resolve_tenant_assertion
from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from config.settings import get_settings

from ._default_tenant import (
    DEFAULT_TENANT_ID,  # noqa: F401  (re-exported: importers and tests read it from here)
    enforce_default_tenant_configured,
    resolve_default_tenant,
)
from ._header_trust import trust_tenant_header

logger = logging.getLogger(__name__)

TENANT_ID_HEADER = "X-Tenant-Id"
_license_cache: TTLCache = TTLCache(maxsize=1000, ttl=300)

# Request-scoped tenant_id resolved by require_tenant, forwarded to WM by the
# LLM shim so worker_management enforces the END-USER tenant's byo_mode (not
# coding-ai's service tenant). None for background/startup calls (no request),
# where the shim falls back to the service token.
_request_tenant_id_var: ContextVar[str | None] = ContextVar(
    "coding_ai_request_tenant_id", default=None
)

# The caller's own credential, kept so outbound calls can act AS the caller
# rather than as this service on the caller's behalf.
#
# coding-ai is deliberately absent from coding-containers'
# ``CODING_ONBEHALF_SUBJECTS``, so presenting our own license while asserting a
# customer tenant is refused with 403 TENANT_ASSERTION_NOT_PERMITTED — by
# design, not by misconfiguration. Forwarding the caller's credential sidesteps
# the question entirely: the SDK consults that allowlist only when the asserted
# tenant DISAGREES with the token's (``coding_client.auth._onbehalf``), and a
# caller asking about their own tenant's workspace does not disagree.
#
# Set in ``require_tenant``, which every workspace-scoped route depends on, so
# by the time a handler runs the credential that authenticated it is available
# without threading ``Request`` through the call chain.
_request_credential_var: ContextVar[str | None] = ContextVar(
    "coding_ai_request_credential", default=None
)


def get_request_tenant_id() -> str | None:
    """Return the current request's resolved tenant_id, or None outside a request."""
    return _request_tenant_id_var.get()


def _bearer_token(request: Request) -> str | None:
    """Extract the bearer token from ``Authorization``, or None.

    The scheme test is case-INSENSITIVE and the token stripped, matching
    ``coding_client.auth.tenant`` and AuthMiddleware. A literal ``"Bearer "``
    test made a lowercase ``bearer`` invisible, which is why this is one helper
    and not two spellings of the same parse.
    """
    header = request.headers.get("Authorization", "")
    if not header.lower().startswith("bearer "):
        return None
    return header[7:].strip() or None


def get_request_credential() -> str | None:
    """Return the raw inbound bearer token, or None outside a credentialed request.

    None is a real case, not an error: background work and any path reached
    through the trusted-network bypass have no caller credential to borrow.
    Callers must fall back to this service's own identity rather than treating
    the absence as a failure.
    """
    return _request_credential_var.get()


# DEFAULT_TENANT_ID is re-exported above from ._default_tenant and is ``None``
# when unset — there is deliberately no literal default. It used to be
# ``os.getenv("DEFAULT_TENANT_ID", "n8n-dev")`` right here; see ._default_tenant
# for why that literal was a data-isolation bug rather than a harmless
# placeholder. Read it through ``resolve_default_tenant()``, never as a captured
# constant, so that "unset" means "grant nothing" at every site.

# ``TRUSTED_CIDRS`` nominates source networks whose requests are accepted with
# NO credential and granted DEFAULT_TENANT_ID. It is an authentication *bypass*,
# so it is empty by default: a caller is trusted only if someone says so.
#
# It previously defaulted to
# ``127.0.0.1/32,::1/128,192.168.0.0/16,172.16.0.0/12,10.0.0.0/8``. On this fleet
# every pod carries an address in the k3s pod CIDR ``10.42.0.0/16``, a subset of
# ``10.0.0.0/8`` — so **every pod in the cluster** satisfied the bypass and was
# handed DEFAULT_TENANT_ID instead of a 401. Demonstrated live on 2026-07-29: an
# unrelated pod in the ``artifacts`` namespace, with no Authorization header, no
# X-Tenant-Id and no license token, read a sibling service's API and got HTTP 200
# with data. "The caller's IP looks local" is not authentication in a cluster
# where every workload is local.
#
# NOTE: this file is a VENDORED copy of coding_client.auth.tenant. It exists on
# purpose — ``require_tenant`` below stashes the resolved id in a contextvar the
# WM shim needs, which the shared version does not do — but it means shared-lib
# fixes do NOT reach here. That is exactly how the on-behalf-of rule went missing
# from this service for three releases while its siblings enforced it, which is
# why that rule is now imported (see the module docstring) rather than copied.
#
# 🔧 REMAINING DEBT: this block is still the fourth copy of the TRUSTED_CIDRS
# parser. The pin is now >=1.14.0, which ships
# ``coding_client.auth._trusted_networks``, so the import is available — it is
# not done here only because this service's tests monkeypatch
# ``src.auth.tenant.TRUSTED_NETWORKS`` by name and would need to move with it.
# Deliberately left out of the security fix rather than bundled into it.
_TRUSTED_CIDRS_RAW = os.getenv("TRUSTED_CIDRS", "")
TRUSTED_NETWORKS: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
for _cidr in (c.strip() for c in _TRUSTED_CIDRS_RAW.split(",")):
    if not _cidr:
        continue
    try:
        TRUSTED_NETWORKS.append(ipaddress.ip_network(_cidr, strict=False))
    except ValueError:
        # Skip, don't crash: an empty list is MORE restrictive, so a typo must
        # not turn into an outage that reads like an auth bug.
        logger.error("TRUSTED_CIDRS: ignoring unparseable entry %r", _cidr)

if TRUSTED_NETWORKS:
    logger.warning(
        "TRUSTED_CIDRS is set to %s — requests from these networks are accepted "
        "with NO credential and granted DEFAULT_TENANT_ID. This is an "
        "authentication bypass; confirm it is intended.",
        [str(n) for n in TRUSTED_NETWORKS],
    )

# Boot gate: if the credential-less bypass is armed in production, DEFAULT_TENANT_ID
# must name the tenant it grants. Raises RuntimeError (never ImportError) — see
# ._default_tenant.enforce_default_tenant_configured. No-op while TRUSTED_CIDRS is
# empty, which is the default.
enforce_default_tenant_configured(TRUSTED_NETWORKS)


def _is_trusted_network(host: str | None) -> bool:
    if not host:
        return False
    try:
        addr = ipaddress.ip_address(host)
        return any(addr in net for net in TRUSTED_NETWORKS)
    except ValueError:
        return False


async def _introspect_license(token: str) -> dict | None:
    cache_key = f"lic:{hashlib.sha256(token.encode()).hexdigest()[:16]}"
    if cache_key in _license_cache:
        return _license_cache[cache_key]
    try:
        introspect_url = get_settings().license_introspect_url
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.post(introspect_url, json={"token": token})
            if resp.status_code == 200:
                data = resp.json()
                if data.get("active"):
                    _license_cache[cache_key] = data
                return data
    except Exception as e:
        logger.warning(f"License introspection failed: {e}")
    return None


class TenantMiddleware(BaseHTTPMiddleware):
    """Write the ``X-Tenant-Id`` claim into request state.

    This is the only place coding-ai reads ``X-Tenant-Id`` as a *claim of
    identity*, and therefore the only gate site for ``TRUST_TENANT_HEADER``. It
    feeds two branches — replacing an authenticated user's ``tenant_id``, and
    setting ``request.state.tenant_id`` for an unauthenticated caller — and both
    are closed by the single read below.

    It is no longer the only read of the header full stop: ``_resolve_tenant``
    now reaches ``resolve_tenant_assertion``, which reads it *ungated* as an
    on-behalf-of assertion. That is deliberate and is not a second gate site —
    see the module docstring for why the two reads answer different questions.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Gated on TRUST_TENANT_HEADER (see ._header_trust): when header trust is
        # off the header is treated as ABSENT, so request.state.tenant_id is never
        # populated from it and _resolve_tenant falls through to license
        # introspection instead of returning an unverified claim. It is NOT
        # downgraded to DEFAULT_TENANT_ID.
        header_tenant_id = request.headers.get(TENANT_ID_HEADER) if trust_tenant_header() else None
        if hasattr(request.state, "user") and request.state.user:
            user: User = request.state.user
            if header_tenant_id:
                user.tenant_id = header_tenant_id
            elif not user.tenant_id:
                user.tenant_id = user.sub
            request.state.tenant_id = user.effective_tenant_id
        elif header_tenant_id:
            request.state.tenant_id = header_tenant_id
        return await call_next(request)


def get_tenant_id(request: Request) -> str | None:
    return getattr(request.state, "tenant_id", None)


async def get_current_user_with_tenant(request: Request) -> User:
    """Get or build a User object with tenant context from the request.

    ⚠️ Deliberately narrower than ``coding_client.auth.tenant`` — do not "fix"
    this by copying the library version. The library builds a ``User`` from
    ``x-user-id`` / ``x-user-name`` / ``x-user-roles``, and ``x-user-roles``
    becomes ``realm_access["roles"]``, which feeds ``User.is_admin()``, which
    ``require_same_tenant`` honours as "may access ANY tenant". An unverified
    ``X-User-Roles: admin`` would then be a privilege escalation on top of the
    tenant forgery, not merely a second instance of it. coding-ai never reads
    those headers, so it has never had that escalation path, and the anonymous
    user built below carries no ``realm_access`` — ``is_admin()`` is always
    False here.
    """
    if hasattr(request.state, "user") and request.state.user:
        user: User = request.state.user
    else:
        # Anonymous fallback: coding-ai has no auth middleware populating
        # request.state.user, so this is the default path. SDK User requires
        # sub/preferred_username; empty values preserve the prior local-model
        # semantics (tenant_id is set below from the request).
        user = User(sub="", preferred_username="")

    tenant_id = await require_tenant(request)
    if not user.tenant_id:
        user.tenant_id = tenant_id
    request.state.user = user
    return user


async def require_tenant(request: Request) -> str:
    """Resolve the request's tenant_id and stash it for outbound WM calls.

    Wraps ``_resolve_tenant`` so the resolved id lands in a request-scoped
    contextvar that the WM shim (``services.llm_agents``) forwards to
    worker_management as a raw ``X-Tenant-Id``. Without this, the shim fell back
    to coding-ai's own service license token, so WM checked the SERVICE tenant's
    byo_mode and a byo_required end-user reached the platform LLM (Fix A is
    bypassed when a service Bearer token is sent). Set here (a dependency that
    runs in the endpoint's task) rather than in TenantMiddleware, whose
    BaseHTTPMiddleware contextvar propagation to the endpoint is unreliable.
    """
    tenant_id = await _resolve_tenant(request)
    _request_tenant_id_var.set(tenant_id)
    # Captured from the request rather than from _resolve_tenant's introspection
    # branch, because a tenant can also be resolved from request.state.user or
    # the trusted-network bypass, and this must be None in exactly those cases.
    _request_credential_var.set(_bearer_token(request))
    return tenant_id


async def _resolve_tenant(request: Request) -> str:
    tenant_id = get_tenant_id(request)
    if tenant_id:
        return tenant_id

    token = _bearer_token(request)
    if token:
        license_data = await _introspect_license(token)
        if license_data and license_data.get("active"):
            # `tenant_id` is the tenant claim; `subject_id` is the USER and must
            # not be used for tenant scoping (matches coding-client SDK).
            raw_tenant = license_data.get("tenant_id")
            tenant_id = raw_tenant.strip() if isinstance(raw_tenant, str) else ""
            if tenant_id:
                # On-behalf-of. The RULE is the SDK's — imported, never
                # re-derived here. This branch used to return the token's tenant
                # unconditionally, which made this service's
                # CODING_ONBEHALF_SUBJECTS allowlist decorative: a caller
                # presenting any live license could name any tenant in
                # X-Tenant-Id and coding-ai would neither honour it (so a
                # legitimately vouched call silently got the WRONG tenant) nor
                # refuse it (so nothing was ever a 403). Driven live 2026-08-14:
                # fr's token — a subject NOT on this service's allowlist —
                # asserting a foreign tenant got 200 here while coding-github and
                # coding-testing returned 403 on the byte-identical request.
                #
                # It matters twice over now that coding-gateway substitutes its
                # OWN license on a vouched call: without this, the tenant
                # ``require_tenant`` stashes in _request_tenant_id_var is
                # svc-coding-gateway's, so the WM shim below makes
                # worker_management enforce the GATEWAY's byo_mode instead of the
                # end user's — the exact BYO bypass the contextvar exists to
                # prevent.
                effective_tenant, refused = resolve_tenant_assertion(
                    request,
                    subject_id=(license_data.get("subject_id") or None),
                    token_tenant=tenant_id,
                )
                if refused:
                    # 403, not 401: the license is live and verified, it is
                    # simply not authorised to speak for that tenant. A 401 would
                    # invite a retry with different credentials.
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=(
                            "This credential may not assert a tenant other than the one "
                            "its license names."
                        ),
                    )
                request.state.tenant_id = effective_tenant
                return effective_tenant

    if hasattr(request.state, "user") and request.state.user:
        user: User = request.state.user
        tenant_id = user.effective_tenant_id
        if tenant_id:
            return tenant_id

    # Priority 4: trusted-network bypass. Usable only when an operator configured
    # BOTH the network (TRUSTED_CIDRS) and the tenant it grants
    # (DEFAULT_TENANT_ID). With no tenant configured we fall through to the 401
    # below rather than inventing one.
    client_host = request.client.host if request.client else None
    if _is_trusted_network(client_host):
        default_tenant = resolve_default_tenant("the trusted-network bypass in _resolve_tenant")
        if default_tenant is not None:
            request.state.tenant_id = default_tenant
            return default_tenant

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Tenant context required. Ensure X-Tenant-Id header is set or provide valid license token.",
    )
