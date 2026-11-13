"""Shared test fixtures for coding-ai.

Uses real dependencies — no sys.modules stubs.
cachetools and coding_client are installed in the venv.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

# Inject default service URLs into the environment so the production-validator
# in src.config.settings.Settings.model_post_init does not raise during test
# collection. Real values come from gitops-production ConfigMap in deployed
# environments; tests use the legacy hardcoded values so existing substring
# assertions (e.g. ":8688", ":8689", ":8691") still hold.
_TEST_ENV_DEFAULTS = {
    "REDIS_URL": "redis://redis.redis.svc.cluster.local:6379/0",
    "CODING_CONTAINERS_URL": "http://coding-containers:8691",
    "INFERENCE_ROUTER_URL": "http://inference-router.inference-router.svc.cluster.local:8150/api",
    "LICENSE_INTROSPECT_URL": "http://licenses.licenses.svc.cluster.local:8544/license/introspect",
}
for _k, _v in _TEST_ENV_DEFAULTS.items():
    os.environ.setdefault(_k, _v)


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Reset the cached Settings between tests so per-test env tweaks apply."""
    from src.config.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _stub_container_validation():
    """Neutralize the cross-service tenant/container validation in unit tests.

    With CODING_CONTAINERS_URL set (above), the real fail-closed helper would
    make a network call to coding-containers and raise 503 in CI. Endpoint unit
    tests don't exercise that boundary — they test the handler logic — so we
    stub the helper to the prior no-op behaviour in every consumer module. The
    real fail-closed logic is covered directly in
    test_shared_container_validation.py (which calls the source helper and
    patches its own settings/httpx, so it is unaffected by this fixture).
    """
    _modules = (
        "routers.ai_test_analysis",
        "routers.ai_test_insights.test_pattern_recognition",
    )
    # Tests import endpoints via the ``src.`` package path; the app may load the
    # bare path too. Patch both so the consumer reference is always stubbed.
    targets = tuple(
        f"{prefix}{mod}.validate_and_get_container_for_tenant"
        for mod in _modules
        for prefix in ("", "src.")
    )
    started = []
    for target in targets:
        try:
            p = patch(target, new=AsyncMock(return_value=SimpleNamespace(codebase_path=".")))
            p.start()
            started.append(p)
        except (AttributeError, ModuleNotFoundError):
            # Module/name not importable in this test's context — skip it.
            pass
    yield
    for p in started:
        p.stop()


@pytest.fixture
def mock_db_session():
    """Mock database session."""
    session = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    return session


# ─────────────────────────────────────────────────────────────────────────────
# Authenticated test clients
#
# coding-ai registers coding_client's AuthMiddleware (src/main.py
# install_auth_middleware). A middleware is NOT a dependency, so
# `app.dependency_overrides` — including coding_client.testing's
# override_tenant_auth — cannot get past it: it runs before routing. Tests that
# need to reach a handler therefore have to present a credential the middleware
# accepts, which means patching the introspection seam it actually calls.
#
# That seam is `coding_client.auth.middleware._introspect_license`, and it is a
# DIFFERENT object from this service's forked `auth.tenant._introspect_license`.
# Patching only the fork leaves the middleware unpatched and every request 401s
# — which is exactly how the pre-existing suite failed when the middleware was
# first registered. Both are patched here.
#
# Deliberately NOT autouse: an autouse credential would make the 401 tests
# unable to fail, and the gate is the thing under test.
# ─────────────────────────────────────────────────────────────────────────────

AUTHED_TENANT = "0195f48b-d000-7000-8000-000000000001"
AUTHED_SUBJECT = "svc-test-caller"
AUTHED_TOKEN = "test-license-token"


@pytest.fixture
def authed_license_patch():
    """Patch BOTH introspection seams so ``AUTHED_TOKEN`` resolves.

    Yields the introspection payload so a test can assert on what the auth
    layer was handed, rather than only on the status code.
    """
    payload = {
        "active": True,
        "tenant_id": AUTHED_TENANT,
        "subject_id": AUTHED_SUBJECT,
    }

    async def _introspect(token: str) -> dict | None:
        return payload if token == AUTHED_TOKEN else None

    with (
        patch("coding_client.auth.middleware._introspect_license", new=_introspect),
        patch("auth.tenant._introspect_license", new=_introspect),
    ):
        yield payload


@pytest.fixture
def authed_client(authed_license_patch):
    """A ``TestClient`` carrying a licence the middleware accepts.

    Use for tests whose subject is the HANDLER. Tests whose subject is the
    GATE must build their own client so the 401 remains observable.
    """
    from fastapi.testclient import TestClient

    from src.main import app

    return TestClient(app, headers={"Authorization": f"Bearer {AUTHED_TOKEN}"})
