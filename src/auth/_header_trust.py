"""Whether client-supplied identity headers may be believed.

VENDORED MIRROR of ``coding_client.auth._header_trust`` (coding-client 1.8.0).
It is duplicated rather than imported because ``src/auth/tenant.py`` is itself a
deliberate fork (see the NOTE there) and coding-ai's pin is ``coding-client
>=1.5.7``, which predates the module. Semantics — env var name, accepted
values, default, call-time read — are identical, so switching to
``from coding_client.auth._header_trust import trust_tenant_header`` once the
pin reaches 1.8.0+ is a one-line change with no behavioural delta.

``X-Tenant-Id`` (and its companions ``X-User-Id`` / ``X-User-Roles``) are
*claims*, not credentials. Nothing here can verify them. Whether they are
trustworthy is a property of the **network path**, not of the request:

* Arriving through the mesh edge, they are trustworthy. The EnvoyFilter
  ``ext-authz-enforce.yaml`` strips every client-supplied tenant header and
  re-injects the value resolved from the license token, so by the time the
  header reaches the pod it is an assertion by an authority.
* Arriving pod-to-pod, they are **not** trustworthy. In-cluster callers do not
  traverse the edge EnvoyFilter, so any workload that can open a socket to
  :8690 can send ``X-Tenant-Id: <victim-tenant>`` and be served that tenant's
  data. In coding-ai the header is consumed by ``TenantMiddleware``, which
  writes it into ``request.state.tenant_id`` — the *first* thing
  ``_resolve_tenant`` reads, i.e. **above** license introspection. A forged
  header therefore beats having no license at all.

``TRUSTED_CIDRS`` does not address this. That knob governs the *no-header* case;
emptying it does nothing to a request that carries a forged header, because such
a request never reaches the network check.

The knob
--------
``TRUST_TENANT_HEADER`` decides whether this process believes those headers.

**The default is ``true`` — i.e. today's behaviour, which is the insecure one.**
That is deliberate: coding-gateway forwards ``X-Tenant-Id`` to coding-ai, and
flipping the default would turn a routine merge into an outage. So merging this
changes nothing, and the security win lands when coding-ai's Deployment sets
``TRUST_TENANT_HEADER=false`` once it is known to have a verifiable alternative
(a license token, or the gateway in front).

Setting it to ``false`` does **not** downgrade the caller to
``DEFAULT_TENANT_ID``. The header is ignored as though absent and resolution
continues down the normal chain — license introspection next. A request with a
forged header and no valid license gets a 401, which is the point.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

#: Values that turn header trust OFF. Anything else leaves it ON.
_DISABLE_VALUES = frozenset({"0", "false", "no", "off"})

#: Values that explicitly turn header trust ON (accepted without complaint).
_ENABLE_VALUES = frozenset({"1", "true", "yes", "on"})

ENV_VAR = "TRUST_TENANT_HEADER"

#: Behaviour when the variable is unset. ``True`` == today's behaviour.
DEFAULT = True

_warned_unrecognised: set[str] = set()


def trust_tenant_header() -> bool:
    """Return True if client-supplied identity headers may be believed.

    Read from the environment at **call time**, not import time, so that a test
    can flip it with ``monkeypatch.setenv`` and so the behaviour of a running
    process is never pinned by import order.

    An unrecognised value keeps the default (trusting). A typo must not silently
    make the service *more* restrictive — that would present as an
    authentication outage with no obvious cause. It is logged at ERROR instead,
    once per distinct bad value.
    """
    raw = os.getenv(ENV_VAR)
    if raw is None:
        return DEFAULT

    normalised = raw.strip().lower()
    if not normalised:
        # Set-but-empty is how a Deployment expresses "I did not decide".
        # Treat it as unset rather than as a decision.
        return DEFAULT
    if normalised in _DISABLE_VALUES:
        return False
    if normalised in _ENABLE_VALUES:
        return True

    if raw not in _warned_unrecognised:
        _warned_unrecognised.add(raw)
        logger.error(
            "%s=%r is not a recognised boolean; falling back to the default (%s). "
            "Use one of %s to disable header trust.",
            ENV_VAR,
            raw,
            DEFAULT,
            sorted(_DISABLE_VALUES),
        )
    return DEFAULT
