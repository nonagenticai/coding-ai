"""CORS configuration invariants for coding-ai.

These tests assert the *invariant*, not the literal current values, so they
stay meaningful if the origin list is ever changed:

    a wildcard origin ("*") must NEVER be combined with allow_credentials=True

That combination is rejected by the CORS spec on the browser side, and
Starlette's ``CORSMiddleware`` implements it by *reflecting* the caller's
``Origin`` header back with ``Access-Control-Allow-Credentials: true`` --
i.e. it turns "allow everyone, anonymously" into "allow every origin to send
credentials", a genuine cross-origin credential leak.

If anyone reintroduces ``allow_origins=["*"]`` together with
``allow_credentials=True``, these tests go red.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.middleware.cors import CORSMiddleware

UNTRUSTED_ORIGIN = "https://evil.example"


def _assert_no_credentialed_access(response, origin: str) -> None:
    """Fail if a browser would let ``origin`` read a credentialed response.

    That requires BOTH an ``Access-Control-Allow-Origin`` covering the caller
    AND ``Access-Control-Allow-Credentials: true``. Starlette may emit a lone
    ``ACAC`` header for a rejected origin -- without a matching ``ACAO`` it is
    inert, so asserting on it alone would be testing framework internals
    rather than the security property.
    """
    acao = response.headers.get("access-control-allow-origin")
    acac = response.headers.get("access-control-allow-credentials")
    assert not (acao in (origin, "*") and acac == "true"), (
        f"{origin} would receive a credentialed cross-origin response "
        f"(Access-Control-Allow-Origin: {acao}, Access-Control-Allow-Credentials: {acac})."
    )


def _cors_options(app) -> dict:
    """Return the kwargs the CORSMiddleware was actually installed with."""
    for middleware in app.user_middleware:
        if middleware.cls is CORSMiddleware:
            options = dict(getattr(middleware, "kwargs", {}) or {})
            # Defensive: tolerate positional configuration too.
            args = list(getattr(middleware, "args", ()) or ())
            if args and "allow_origins" not in options:
                options["allow_origins"] = args[0]
            return options
    pytest.fail("CORSMiddleware is not installed on the app")


@pytest.fixture
def app():
    from src.main import app as fastapi_app

    return fastapi_app


def test_wildcard_origin_is_never_combined_with_credentials(app):
    """The installed middleware must not allow credentials on a wildcard origin."""
    options = _cors_options(app)
    origins = options.get("allow_origins", [])
    if isinstance(origins, str):
        origins = [origins]

    if "*" in origins:
        assert options.get("allow_credentials") is not True, (
            "CORS is configured with a wildcard origin AND allow_credentials=True. "
            "Starlette reflects the caller's Origin header in that case, so every "
            "origin on the network may make credentialed cross-origin calls."
        )


def test_untrusted_origin_is_not_granted_credentials(app):
    """A simple request from an unlisted origin must not be granted credentials."""
    client = TestClient(app)
    response = client.get("/health", headers={"Origin": UNTRUSTED_ORIGIN})

    _assert_no_credentialed_access(response, UNTRUSTED_ORIGIN)


def test_preflight_from_untrusted_origin_is_not_granted_credentials(app):
    """A preflight from an unlisted origin must not be granted credentials."""
    client = TestClient(app)
    response = client.options(
        "/health",
        headers={
            "Origin": UNTRUSTED_ORIGIN,
            "Access-Control-Request-Method": "GET",
        },
    )

    _assert_no_credentialed_access(response, UNTRUSTED_ORIGIN)


# ---------------------------------------------------------------------------
# resolve_cors_config -- the invariant must hold for every possible input,
# not just for whatever the default configuration happens to be today.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "configured",
    [
        ["*"],
        "*",
        ["*", "https://app.example"],
        "https://app.example, *",
        [],
        "",
        None,
    ],
)
def test_resolver_never_allows_credentials_when_wildcard_survives(configured):
    """Any input that resolves to a wildcard origin must disable credentials."""
    from src.main import resolve_cors_config

    settings = SimpleNamespace(cors_origins=configured)
    origins, allow_credentials = resolve_cors_config(settings)

    assert ("*" in origins) != allow_credentials, (
        f"{configured!r} resolved to origins={origins} credentials={allow_credentials}; "
        "credentials must be enabled if and only if the origin list is explicit."
    )


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("https://a.example,https://b.example", ["https://a.example", "https://b.example"]),
        (" https://a.example , https://b.example ", ["https://a.example", "https://b.example"]),
        (["https://a.example"], ["https://a.example"]),
        ("https://a.example,,", ["https://a.example"]),
    ],
)
def test_resolver_enables_credentials_for_explicit_origins(configured, expected):
    """An explicit origin list parses correctly and turns credentials on."""
    from src.main import resolve_cors_config

    origins, allow_credentials = resolve_cors_config(SimpleNamespace(cors_origins=configured))

    assert origins == expected
    assert allow_credentials is True


def test_resolver_defaults_to_wildcard_without_credentials_when_unset():
    """A settings object with no cors_origins attribute degrades safely."""
    from src.main import resolve_cors_config

    origins, allow_credentials = resolve_cors_config(SimpleNamespace())

    assert origins == ["*"]
    assert allow_credentials is False


# ---------------------------------------------------------------------------
# The setting must actually be reachable from the environment -- a knob that
# nothing can set is the same class of dead control as a hardcoded value.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        ("https://a.example,https://b.example", ["https://a.example", "https://b.example"]),
        ('["https://a.example","https://b.example"]', ["https://a.example", "https://b.example"]),
        ("https://a.example", ["https://a.example"]),
        ("*", ["*"]),
    ],
)
def test_cors_origins_is_settable_via_env(monkeypatch, env_value, expected):
    """CORS_ORIGINS is read from the environment in the fleet's comma form."""
    from src.config.settings import Settings

    monkeypatch.setenv("CORS_ORIGINS", env_value)
    assert Settings().cors_origins == expected


def test_cors_origins_defaults_to_wildcard(monkeypatch):
    """With CORS_ORIGINS unset the service keeps the permissive-but-anonymous default."""
    from src.config.settings import Settings

    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    assert Settings().cors_origins == ["*"]


def test_app_middleware_is_driven_by_the_resolver(app):
    """main.py must wire the middleware from resolve_cors_config, not literals.

    Without this, someone could hardcode the middleware again and the resolver
    unit tests above would still pass while the running app stayed vulnerable.
    """
    from src import main

    expected_origins, expected_credentials = main.resolve_cors_config(main.settings)
    options = _cors_options(app)

    assert options["allow_origins"] == expected_origins
    assert options["allow_credentials"] == expected_credentials


def test_explicit_env_origins_produce_a_scoped_credentialed_app(monkeypatch):
    """End-to-end: CORS_ORIGINS in the env -> Settings -> resolver -> live headers.

    Builds a throwaway app so the assertion exercises the real Starlette
    middleware rather than reloading ``src.main`` (whose module-level settings
    are cached at import time).
    """
    from fastapi import FastAPI

    from src.config.settings import Settings
    from src.main import resolve_cors_config

    monkeypatch.setenv("CORS_ORIGINS", "https://trusted.example")
    origins, allow_credentials = resolve_cors_config(Settings())

    assert origins == ["https://trusted.example"]
    assert allow_credentials is True

    probe = FastAPI()

    @probe.get("/probe")
    async def _probe() -> dict[str, bool]:
        return {"ok": True}

    probe.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    client = TestClient(probe)

    trusted = client.get("/probe", headers={"Origin": "https://trusted.example"})
    assert trusted.headers.get("access-control-allow-origin") == "https://trusted.example"
    assert trusted.headers.get("access-control-allow-credentials") == "true"

    untrusted = client.get("/probe", headers={"Origin": UNTRUSTED_ORIGIN})
    assert untrusted.headers.get("access-control-allow-origin") is None
    _assert_no_credentialed_access(untrusted, UNTRUSTED_ORIGIN)
