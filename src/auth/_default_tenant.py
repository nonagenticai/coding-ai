"""Resolution of the fallback tenant handed to credential-less callers.

VENDORED MIRROR of ``coding_client.auth._default_tenant`` (coding-client 1.9.0),
for the same reason as :mod:`._header_trust`. One deliberate signature change:
:func:`enforce_default_tenant_configured` takes the already-parsed trusted
networks as an argument instead of re-reading ``TRUSTED_CIDRS`` itself, because
in this fork that list is owned by :mod:`.tenant` — re-parsing here would give
two sources of truth for "is the bypass armed", and importing it back would be
circular.

``DEFAULT_TENANT_ID`` is the tenant granted to a caller that presented **no
credential** and was admitted anyway — in coding-ai, only the ``TRUSTED_CIDRS``
bypass in :mod:`.tenant`. It is therefore a data-isolation key chosen by
configuration, not by the caller.

Why there is no default value
-----------------------------
This module-level constant used to live in ``tenant.py`` as::

    DEFAULT_TENANT_ID = os.getenv("DEFAULT_TENANT_ID", "n8n-dev")

so coding-ai, which sets no ``DEFAULT_TENANT_ID`` on its Deployment, would have
granted bypassed callers the literal string ``"n8n-dev"``. That literal is not
junk and it is not a real tenant_id either, which is exactly what made it
dangerous:

* ``n8n-dev`` **was** a real tenant — as a *text* tenant_id, until the
  2026-04-29 Keycloak migration moved tenant identity to uuid7 and demoted the
  label to a separate ``tenant_name`` attribute. The canonical id for that
  tenant is now the uuid7 recorded as ``TENANT_UUIDS["n8n-dev"]`` in
  ``gitops-production/clusters/strato/services/keycloak/realm-patch-script-configmap.yaml``.
* So the fallback pointed at the *stale label form* of a tenant that really
  exists. Rows written under it land in a silo that no live JWT can ever
  address, and reads under it silently cross out of whatever tenant the operator
  believed the service was serving.

There is no safe literal to pick. Any built-in default is a guess about someone
else's data isolation. A missing tenant is therefore an error: unset means the
credential-less path refuses to resolve a tenant at all (the caller gets the
401 at the end of ``_resolve_tenant``), and — when a bypass is actually armed in
production — the process refuses to boot.
"""

from __future__ import annotations

import ipaddress
import logging
import os
from collections.abc import Sequence

logger = logging.getLogger(__name__)

ENV_VAR = "DEFAULT_TENANT_ID"


def load_default_tenant_id() -> str | None:
    """Read ``DEFAULT_TENANT_ID`` from the environment. **No default value.**

    An empty or whitespace-only value is treated the same as unset, mirroring
    ``coding_client.auth.models.User.model_post_init`` — a blank tenant must
    never pin a caller to an empty tenant.
    """
    return os.getenv(ENV_VAR, "").strip() or None


DEFAULT_TENANT_ID: str | None = load_default_tenant_id()


def resolve_default_tenant(purpose: str) -> str | None:
    """Return the configured fallback tenant, or ``None`` — never a literal.

    Callers MUST treat ``None`` as "do not grant this caller a tenant".
    Substituting any made-up value here is the bug this module exists to
    prevent.

    Reads the module attribute at call time (not a captured copy) so tests can
    monkeypatch it.
    """
    tenant_id = DEFAULT_TENANT_ID
    if tenant_id is None:
        logger.error(
            "%s is not configured; refusing to invent a tenant for %s. This "
            "caller will be rejected. If this path is meant to be usable, set "
            "%s on the Deployment env block to the tenant it should act as "
            "(see gitops-production/docs/config-externalization-policy.md).",
            ENV_VAR,
            purpose,
            ENV_VAR,
        )
    return tenant_id


def enforce_default_tenant_configured(
    trusted_networks: Sequence[ipaddress.IPv4Network | ipaddress.IPv6Network],
) -> None:
    """Crash boot when a credential-less bypass is armed with no tenant to grant.

    Follows the fleet's fail-loud-in-production convention (see
    ``coding_client.config.base_settings`` and ``Settings._REQUIRED_IN_PRODUCTION``):
    a misconfiguration must be visible at deploy time, not at 3am on one unlucky
    request.

    The gate is deliberately narrow — it fires only when *all* of:

    1. ``ENVIRONMENT`` is ``production`` (the default), and
    2. ``trusted_networks`` is non-empty, i.e. an operator has armed the
       credential-less network bypass via ``TRUSTED_CIDRS``, and
    3. ``DEFAULT_TENANT_ID`` is unset.

    Condition 2 is what keeps this from being a breaking change: ``TRUSTED_CIDRS``
    has been empty by default in this service since PR #44, so today this raises
    for nobody. It exists so that *re-widening* ``TRUSTED_CIDRS`` without naming
    the tenant fails the rollout instead of silently reviving the literal.

    Raises:
        RuntimeError: never ``ImportError``. Several coding-* services wrap their
            auth-middleware import in ``except ImportError`` and log "running
            WITHOUT authentication", so an ``ImportError`` here could be
            swallowed into exactly the outcome this check is meant to prevent.
    """
    if os.getenv("ENVIRONMENT", "production") != "production":
        return
    if not trusted_networks:
        return
    if load_default_tenant_id() is not None:
        return
    raise RuntimeError(
        f"TRUSTED_CIDRS is set to {[str(n) for n in trusted_networks]} — requests "
        f"from those networks are admitted with NO credential — but {ENV_VAR} is "
        f"not set, so there is no tenant to grant them. Refusing to start rather "
        f"than serve a made-up tenant. Either set {ENV_VAR} on the Deployment env "
        f"block, or clear TRUSTED_CIDRS to disable the bypass."
    )
