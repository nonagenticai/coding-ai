"""coding-ai must never call worker_management without identifying itself.

The measured defect: ``settings.coding_ai_license_token`` maps to
``CODING_AI_LICENSE_TOKEN``, which appears in ZERO manifests across
gitops-production. It was therefore ``""`` in every pod that ever ran, and
``WMClient``'s ``elif self._service_token:`` emitted no ``X-Tenant-Id`` header at
all on the background/startup path (``get_request_tenant_id()`` is ``None``
there). worker_management's trusted-network bypass accepts an RFC1918 source
with no tenant, so those calls were SERVED — and metered against nobody. A
nil-tenant default is un-billing, not safe.

The credential the pod actually carries is ``CODING_SERVICE_TOKEN`` (subject
``019ff970-e327`` = ``svc-coding-ai``), already present in the ``coding-ai-env``
Secret. These tests pin the reconciliation onto it.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from clients.outbound_identity import service_identity_token

_ENV_VARS = ("CODING_SERVICE_TOKEN", "CODING_AI_LICENSE_TOKEN")


@pytest.fixture(autouse=True)
def _clean_identity_env(monkeypatch: pytest.MonkeyPatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_identity_comes_from_the_var_the_pod_actually_carries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODING_SERVICE_TOKEN", "svc-coding-ai-token")
    assert service_identity_token() == "svc-coding-ai-token"


def test_the_legacy_var_still_wins_when_an_operator_sets_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """scripts/register_workers.py documents CODING_AI_LICENSE_TOKEN as an override."""
    monkeypatch.setenv("CODING_SERVICE_TOKEN", "svc-coding-ai-token")
    monkeypatch.setenv("CODING_AI_LICENSE_TOKEN", "explicit-override")
    assert service_identity_token() == "explicit-override"


def test_a_blank_value_is_not_an_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Secret key present but empty is the realistic misconfiguration."""
    monkeypatch.setenv("CODING_SERVICE_TOKEN", "")
    monkeypatch.setenv("CODING_AI_LICENSE_TOKEN", "")
    assert service_identity_token() is None


def test_unset_is_none_not_empty_string() -> None:
    assert service_identity_token() is None


def test_the_env_is_read_per_call_so_a_rotation_needs_no_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODING_SERVICE_TOKEN", "before")
    assert service_identity_token() == "before"
    monkeypatch.setenv("CODING_SERVICE_TOKEN", "after")
    assert service_identity_token() == "after"


def test_building_the_wm_client_with_no_identity_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The singleton refuses to exist rather than emitting header-less calls."""
    from src.config.settings import get_settings
    from src.services import llm_agents

    monkeypatch.setenv("WORKER_MANAGEMENT_URL", "http://wm.test:8887")
    get_settings.cache_clear()
    monkeypatch.setattr(llm_agents, "_wm_client", None)

    with pytest.raises(RuntimeError, match="CODING_SERVICE_TOKEN"):
        llm_agents._get_wm_client()


def test_the_wm_client_is_built_once_an_identity_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control arm: the guard must not block the correctly-configured case."""
    from src.config.settings import get_settings
    from src.services import llm_agents

    monkeypatch.setenv("WORKER_MANAGEMENT_URL", "http://wm.test:8887")
    monkeypatch.setenv("CODING_SERVICE_TOKEN", "svc-coding-ai-token")
    get_settings.cache_clear()
    monkeypatch.setattr(llm_agents, "_wm_client", None)
    try:
        assert llm_agents._get_wm_client() is not None
    finally:
        monkeypatch.setattr(llm_agents, "_wm_client", None)


def test_the_lifespan_refuses_to_boot_without_an_identity() -> None:
    """Startup must fail loudly, not wait for the first background call.

    Asserted over the source rather than by running the lifespan: the lifespan
    also opens Redis and seeds WM over the network, and a test that had to stub
    both would stop being a test of this guard. What must hold is structural —
    a ``raise`` guarded by ``service_identity_token()`` inside ``lifespan``.
    """
    src = pathlib.Path(__file__).resolve().parents[2] / "src" / "main.py"
    tree = ast.parse(src.read_text())
    lifespan = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == "lifespan"
    )
    guards = [
        n
        for n in ast.walk(lifespan)
        if isinstance(n, ast.If)
        and any(
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Name)
            and c.func.id == "service_identity_token"
            for c in ast.walk(n.test)
        )
        and any(isinstance(b, ast.Raise) for b in ast.walk(n))
    ]
    assert guards, "lifespan does not raise when the service identity is missing"


def test_nothing_identifies_the_service_from_cached_settings() -> None:
    """settings.coding_ai_license_token is a cached read of an undeployed var.

    Two runtime readers used it (llm_agents' WMClient construction and main's WM
    seed) and both were silently unattributed. Any new reader reintroduces the
    defect, so the count is pinned at zero.
    """
    root = pathlib.Path(__file__).resolve().parents[2] / "src"
    # Parsed, not grepped: this file and the ones it fixed both DISCUSS the old
    # attribute in prose, and a substring scan flags its own explanation. Only a
    # real ``<something>.coding_ai_license_token`` load counts as a reader.
    offenders = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "coding_ai_license_token"
                and isinstance(node.ctx, ast.Load)
            ):
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert offenders == [], (
        "resolve the service identity via clients.outbound_identity."
        f"service_identity_token(), not cached settings: {offenders}"
    )


@pytest.mark.parametrize(("raw", "expected"), [(None, False), ("off", False), ("on", True)])
def test_the_wm_client_singleton_carries_the_delegated_licence_flag(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: bool
) -> None:
    """WM_SEND_DELEGATED_LICENCE must actually reach the client llm_agents builds."""
    from src.config.settings import get_settings
    from src.services import llm_agents

    monkeypatch.setenv("WORKER_MANAGEMENT_URL", "http://wm.test:8887")
    monkeypatch.setenv("CODING_SERVICE_TOKEN", "svc-coding-ai-token")
    if raw is None:
        monkeypatch.delenv("WM_SEND_DELEGATED_LICENCE", raising=False)
    else:
        monkeypatch.setenv("WM_SEND_DELEGATED_LICENCE", raw)
    get_settings.cache_clear()
    monkeypatch.setattr(llm_agents, "_wm_client", None)
    try:
        assert llm_agents._get_wm_client()._send_delegated_licence is expected
    finally:
        monkeypatch.setattr(llm_agents, "_wm_client", None)


def test_the_wm_client_singleton_refuses_an_unknown_flag_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.config.settings import get_settings
    from src.services import llm_agents

    monkeypatch.setenv("WORKER_MANAGEMENT_URL", "http://wm.test:8887")
    monkeypatch.setenv("CODING_SERVICE_TOKEN", "svc-coding-ai-token")
    monkeypatch.setenv("WM_SEND_DELEGATED_LICENCE", "enabled")
    get_settings.cache_clear()
    monkeypatch.setattr(llm_agents, "_wm_client", None)
    try:
        with pytest.raises(ValueError, match="WM_SEND_DELEGATED_LICENCE"):
            llm_agents._get_wm_client()
    finally:
        monkeypatch.setattr(llm_agents, "_wm_client", None)


def test_the_lifespan_validates_the_delegated_licence_flag_at_boot() -> None:
    """An unknown WM_SEND_DELEGATED_LICENCE must crash startup, not the first user call.

    Structural, like the identity guard above: ``delegated_licence_enabled()``
    (which raises on unknown values) is called in ``lifespan`` OUTSIDE any
    ``try`` that could swallow it.
    """
    src = pathlib.Path(__file__).resolve().parents[2] / "src" / "main.py"
    tree = ast.parse(src.read_text())
    lifespan = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == "lifespan"
    )
    swallowed = {id(c) for t in ast.walk(lifespan) if isinstance(t, ast.Try) for c in ast.walk(t)}
    calls = [
        c
        for c in ast.walk(lifespan)
        if isinstance(c, ast.Call)
        and isinstance(c.func, ast.Name)
        and c.func.id == "delegated_licence_enabled"
        and id(c) not in swallowed
    ]
    assert calls, "lifespan does not validate WM_SEND_DELEGATED_LICENCE outside a try"
