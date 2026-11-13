"""coding-ai must not be the one service where a missing ``Depends`` means no auth.

WHAT WAS WRONG
--------------
This service had ``TenantMiddleware`` and **no** ``AuthMiddleware``. The forked
``TenantMiddleware`` writes ``request.state.tenant_id`` and never rejects, so
``Depends(require_tenant)`` was the *only* gate — and three handlers did not
declare it. Measured live against the running pod, with no headers at all::

    /api/ai/architecture/capabilities     -> 200
    /api/ai/insights/code/insight-types   -> 200
    /api/ai/insights/code/capabilities    -> 200
    /api/ai/capabilities                  -> 200   (alias mount)
    /api/ai/insight-types                 -> 200   (alias mount)
    /api/ai/analysis/tests/patterns       -> 401   (control)

Those three handlers return module constants only, so the *disclosure* was nil.
The defect is structural, and it had two halves:

1. every other service in the fleet has the blanket middleware — here, forgetting
   a ``Depends`` on the next route added is silently an open route;
2. the published OpenAPI marked the whole ``/api/ai`` prefix ``BearerAuth`` via
   ``configure_openapi(protected_prefixes={"/api/ai"})``, a pure string-prefix
   stamp that never inspects ``route.dependencies``. The contract advertised a
   gate that did not exist.

WHY THERE ARE THREE TESTS AND NOT ONE
-------------------------------------
Fixing one half while leaving the other leaves the contract lying in the
opposite direction, and **no test in this repo asserted either half** before
this file — which is exactly why the drift went unnoticed. So:

    A  the middleware is registered, and outermost   (the class fix)
    B  EVERY /api/ai route declares the dependency   (the instance fix, for all
                                                      current AND future routes)
    C  the published contract agrees with A and B    (the advertisement)

Test B is the load-bearing one: it is written over the live route table rather
than a hand-listed set of five paths, so a *new* undecorated route fails it. A
hand-listed assertion would pass forever after the next route is added, which is
the shape that produced this defect in the first place.

MUTANTS — each was run against this file, and the table records what was
MEASURED rather than what was expected
----------------------------------------------------------------------
    M1  drop the middleware, keep the Depends       -> both A tests fail
    M2  drop the Depends from ONE handler           -> B fails, naming BOTH
        mount paths of that handler; C fails too (the spec still advertises it)
    M3  drop both halves                            -> both A tests and B fail
    M4  register AuthMiddleware BEFORE Tenant       -> A fails (ordering: the
        last added runs first, so Auth must be added last to run outermost)
    M5  ``/health`` and ``/``                       -> stay reachable; standing
        false-positive control, and the crashloop guard, since PUBLIC_PATHS is
        exact-match and all three container probes hit /health

⚠️ C as first written ("the three paths carry a ``security`` block") survived
M1, M2 AND M3 — it asserted something the OpenAPI stamp makes unconditionally
true. It is now an agreement check between the spec and the resolved route
table, which is what can actually drift. Recorded because reading it did not
reveal that; only running the mutants did.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi.routing import APIRoute

import main
from auth.tenant import require_tenant

try:
    from fastapi.routing import iter_route_contexts
except ImportError:  # fastapi < 0.137.2: app.routes is already flat
    iter_route_contexts = None


@dataclass(frozen=True)
class _ResolvedRoute:
    """An ``APIRoute`` paired with its EFFECTIVE (fully-prefixed) path/methods.

    Needed on fastapi>=0.137: ``include_router()`` no longer clones routes, so
    the same ``APIRoute`` object is shared across every inclusion and its own
    ``.path``/``.methods`` are the route's un-prefixed declaration, not what is
    actually mounted (https://github.com/fastapi/fastapi/releases/tag/0.137.0).
    The effective values live on the ``RouteContext`` yielded by
    ``fastapi.routing.iter_route_contexts``, keyed back to this route's
    ``.dependant`` so ``_declares_require_tenant`` below still works unchanged.
    """

    path: str
    methods: object
    dependant: object


@pytest.fixture(scope="module")
def app():
    return main.app


def _api_routes(routes) -> list[APIRoute | _ResolvedRoute]:
    """Every route reachable from ``routes``, resolved to its EFFECTIVE path.

    ``app.routes`` mixes flat route objects with fastapi>=0.137's
    ``_IncludedRouter`` wrappers, which have no ``.path`` at all — the old
    ``isinstance(r, APIRoute)`` filter over ``app.routes`` directly silently
    drops every route reached through ``include_router()`` (this is the "0
    routes" failure mode, not a crash). ``iter_route_contexts`` is the public
    accessor that recurses into those wrappers.
    """
    if iter_route_contexts is None:
        return [r for r in routes if isinstance(r, APIRoute)]
    return [
        _ResolvedRoute(path=ctx.path, methods=ctx.methods, dependant=ctx.route.dependant)
        for ctx in iter_route_contexts(routes)
        if isinstance(ctx.route, APIRoute) and ctx.path is not None
    ]


@pytest.fixture(scope="module")
def api_routes(app) -> list[APIRoute | _ResolvedRoute]:
    """Every ``/api/ai`` route on the real application object."""
    return [r for r in _api_routes(app.routes) if r.path.startswith("/api/ai")]


def test_the_route_table_is_not_vacuous(api_routes):
    """Prove there are routes to check before trusting any pass below.

    Without this, a refactor that moved the prefix would turn tests B and C
    into greens that measured nothing.
    """
    assert len(api_routes) >= 15, f"only {len(api_routes)} /api/ai routes found"
    paths = {r.path for r in api_routes}
    for expected in (
        "/api/ai/architecture/capabilities",
        "/api/ai/insights/code/capabilities",
        "/api/ai/insights/code/insight-types",
    ):
        assert expected in paths, f"{expected} is not on the app — the census is stale"


# ── A: the middleware is registered, and outermost ──


def test_auth_middleware_is_registered(app):
    """The blanket gate exists at all.

    This is the half that protects routes nobody has written yet.
    """
    from coding_client.auth import AuthMiddleware

    installed = [m.cls for m in app.user_middleware]
    assert AuthMiddleware in installed, (
        "coding_client.auth.AuthMiddleware is not registered. Without it, "
        "Depends(require_tenant) is the only gate in this service and any route "
        "that omits it is unauthenticated."
    )


def test_auth_middleware_runs_outside_tenant_middleware(app):
    """Ordering, asserted rather than assumed.

    ``add_middleware`` PREPENDS, so ``user_middleware[0]`` runs first. This
    service's forked ``TenantMiddleware`` reads ``request.state.user``
    (auth/tenant.py:215), so ``AuthMiddleware`` must have run already and set
    it. Registering them the other way round leaves the fork reading a value
    that is never written.
    """
    from coding_client.auth import AuthMiddleware

    from auth.tenant import TenantMiddleware

    order = [m.cls for m in app.user_middleware]
    assert order.index(AuthMiddleware) < order.index(TenantMiddleware), (
        f"AuthMiddleware must run OUTSIDE TenantMiddleware; order was {order}"
    )


# ── B: every /api/ai route declares the dependency ──


def _declares_require_tenant(route: APIRoute) -> bool:
    """True when ``require_tenant`` is in this route's resolved dependency tree.

    Checked against the *dependant* graph rather than the function signature,
    so a dependency declared on the router or via ``dependencies=[...]`` also
    counts — a signature-only check would report a false failure for those.
    """
    seen = [route.dependant]
    while seen:
        d = seen.pop()
        if d.call is require_tenant:
            return True
        seen.extend(d.dependencies)
    return False


def test_every_api_ai_route_is_tenant_gated(api_routes):
    """No ``/api/ai`` route may be reachable without a resolved tenant.

    Written over the LIVE route table, not a hand-listed set of paths, so a
    newly added undecorated route fails this too. A hand-listed assertion would
    have passed forever after the next route was added — the shape that
    produced this defect.
    """
    ungated = sorted(
        f"{sorted(r.methods or [])} {r.path}" for r in api_routes if not _declares_require_tenant(r)
    )
    assert not ungated, (
        f"{len(ungated)} /api/ai route(s) do not depend on require_tenant: "
        f"{ungated}. This service publishes security: [{{BearerAuth: []}}] for "
        "the whole /api/ai prefix, so an ungated route makes the contract false."
    )


# ── C: the published contract agrees ──


def test_the_published_contract_agrees_with_the_route_table(app, api_routes):
    """Every path the spec advertises as protected must actually be gated.

    ⚠️ This test was first written as "the three paths carry a ``security``
    block", and it **could not fail**: ``configure_openapi(protected_prefixes=
    {"/api/ai"})`` stamps that block from the path STRING, without ever looking
    at ``route.dependencies``. It stayed green with the middleware deleted and
    with the dependencies deleted — asserting something unconditionally true,
    which is the definition of a decorative gate. It was caught by running the
    mutants rather than by reading it.

    What is actually worth pinning is the AGREEMENT between the advertisement
    and the code, because that is the pair that drifted: the stamp claimed
    ``BearerAuth`` for the whole prefix while three handlers had no gate at all.
    Both directions matter — a route gated but published as open would send
    integrators looking for a bug that is not there.
    """
    spec = app.openapi()
    gated = {r.path for r in api_routes if _declares_require_tenant(r)}

    advertised, lying = set(), []
    for path, operations in spec["paths"].items():
        if not path.startswith("/api/ai"):
            continue
        for method, operation in operations.items():
            if method not in {"get", "post", "put", "patch", "delete"}:
                continue
            if not operation.get("security"):
                continue
            advertised.add(path)
            if path not in gated:
                lying.append(f"{method.upper()} {path}")

    # Non-vacuity: if the stamp ever stops emitting `security` blocks, the loop
    # above finds nothing and the comparison below passes over an empty set.
    assert len(advertised) >= 15, (
        f"only {len(advertised)} /api/ai operations are published as protected — "
        "the OpenAPI stamp is not doing anything, so this check is vacuous"
    )
    assert not lying, (
        f"the published spec advertises security: [{{BearerAuth: []}}] for "
        f"{lying}, but those routes do not depend on require_tenant. The stamp "
        "is a string-prefix claim and cannot tell — that is exactly how three "
        "handlers came to answer 200 with no headers while /docs said otherwise."
    )


def test_health_and_root_stay_public(app):
    """False-positive control — and the crashloop guard.

    ``PUBLIC_PATHS`` is exact-match, and all three container probes
    (liveness/readiness/startup) hit ``/health``. If registering the middleware
    ever gated these, the pods would fail their probes and never become ready.
    """
    from coding_client.auth.middleware import PUBLIC_PATHS

    for path in ("/", "/health"):
        assert path in PUBLIC_PATHS, (
            f"{path} is not in PUBLIC_PATHS — the container probes would 401 "
            "and the deployment would never go ready"
        )
