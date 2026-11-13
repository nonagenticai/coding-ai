"""Unit tests for coding-ai WMClient (worker_management proxy)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, cast

import httpx
import pytest

from wm_client import (
    MessageResponse,
    WMBYONotConfiguredError,
    WMClient,
    WMClientError,
    WMServiceIdentityMissingError,
    WMValidationError,
    delegated_licence_enabled,
    parse_delegated_licence_flag,
)


def _make_client(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    service_token: str | Callable[[], str | None] = "svc-token",
    send_delegated_licence: bool = False,
) -> WMClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport)
    return WMClient(
        base_url="http://wm.test:8080",
        service_token=service_token,
        http=http,
        send_delegated_licence=send_delegated_licence,
    )


async def test_forwards_tenant_token_and_subject_id() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"output": "ok", "output_type": "string"})

    wm = _make_client(handler)
    # A bare tenant_id (UUID, not JWT-shaped) is sent UNPREFIXED so WM's
    # trusted-network path resolves byo_mode by tenant id and enforces
    # byo_required for the END-USER (Fix A) — not coding-ai's service tenant.
    res = await wm.send(
        "cai-root-cause-syntactic",
        "hello",
        tenant_token="019edc48-6060-76e9-8d8f-2e720e12c609",
        subject_id="user-42",
    )
    assert isinstance(res, MessageResponse)
    headers = cast(dict[str, str], captured["headers"])
    assert headers["x-tenant-id"] == "019edc48-6060-76e9-8d8f-2e720e12c609"
    assert headers["x-subject-id"] == "user-42"

    # A JWT-shaped token (>=2 dots) is Bearer-prefixed (WM's verified token path).
    await wm.send("cai-root-cause-syntactic", "hi", tenant_token="aa.bb.cc")
    headers = cast(dict[str, str], captured["headers"])
    assert headers["x-tenant-id"] == "Bearer aa.bb.cc"


async def test_falls_back_to_service_token() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={"output": []})

    wm = _make_client(handler)
    await wm.send("cai-pattern-detector-pattern-extractor", "hi")
    headers = cast(dict[str, str], captured["headers"])
    assert headers["x-tenant-id"] == "Bearer svc-token"


async def test_includes_output_schema_in_body() -> None:
    captured: dict[str, object] = {}
    schema = {"type": "array"}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"output": []})

    wm = _make_client(handler)
    await wm.send("w", "msg", output_schema=schema)
    body = cast(dict[str, Any], captured["body"])
    assert body["output_schema"] == schema


async def test_forwards_work_tier_when_set() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"output": "ok"})

    wm = _make_client(handler)
    await wm.send("w", "msg", work_tier="high")
    body = cast(dict[str, Any], captured["body"])
    assert body["work_tier"] == "high"


async def test_omits_work_tier_when_absent() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"output": "ok"})

    wm = _make_client(handler)
    await wm.send("w", "msg")
    body = cast(dict[str, Any], captured["body"])
    assert "work_tier" not in body


async def test_422_security_validation_raises_typed_error() -> None:
    body = {
        "error": {
            "code": "VALIDATION_ERROR",
            "message": "Validation failed: body -> message: contains potential XSS patterns",
            "status": 422,
        },
        "validation_errors": [
            {
                "field": "body -> message",
                "message": "contains potential XSS patterns",
                "type": "value_error",
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json=body)

    wm = _make_client(handler)
    with pytest.raises(WMValidationError) as excinfo:
        await wm.send("w", "<script>alert(1)</script>")
    assert excinfo.value.worker_name == "w"
    assert "XSS" in excinfo.value.message
    assert isinstance(excinfo.value, WMClientError)


async def test_402_byo_raises_typed_error() -> None:
    body = {
        "code": "byo_llm_not_configured",
        "message": "Configure your LLM provider in Settings before running agents.",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json=body)

    wm = _make_client(handler)
    with pytest.raises(WMBYONotConfiguredError) as excinfo:
        await wm.send("cai-root-cause-syntactic", "m")
    assert excinfo.value.worker_name == "cai-root-cause-syntactic"
    assert "provider" in excinfo.value.message.lower()
    assert isinstance(excinfo.value, WMClientError)


async def test_402_is_byo_regardless_of_body() -> None:
    # HTTP 402 is reserved fleet-wide for BYO-not-configured, so any 402 (even
    # without the body code at a known path) raises WMBYONotConfiguredError with
    # the default message — previously such a 402 leaked out as a generic error.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json={"detail": "other"})

    wm = _make_client(handler)
    with pytest.raises(WMBYONotConfiguredError) as excinfo:
        await wm.send("w", "m")
    assert "provider" in excinfo.value.message.lower()


async def test_retry_then_raise_on_transport_error() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("boom")

    wm = _make_client(handler)
    with pytest.raises(WMClientError, match="unreachable"):
        await wm.send("w", "m")
    assert calls["n"] == 2


async def test_5xx_does_not_retry() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, text="boom")

    wm = _make_client(handler)
    with pytest.raises(WMClientError, match="500"):
        await wm.send("w", "m")
    assert calls["n"] == 1


# --- service identity ------------------------------------------------------
#
# The defect these cover: settings.coding_ai_license_token reads
# CODING_AI_LICENSE_TOKEN, which is in no manifest, so the constructor got ""
# in every pod. The old `elif self._service_token:` then sent NO X-Tenant-Id,
# and WM's trusted-network bypass served the call and metered it against
# nobody. An unattributable 200 is the failure mode, so the assertion has to be
# that the request is never issued at all.


async def test_empty_service_token_refuses_the_call_and_sends_nothing() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"output": "ok"})

    wm = _make_client(handler, service_token="")
    with pytest.raises(WMServiceIdentityMissingError):
        await wm.send("cai-root-cause-syntactic", "hi")
    # The point of the test: not merely that it raised, but that no
    # unattributed request reached worker_management.
    assert calls == []


async def test_service_identity_missing_is_a_wm_client_error() -> None:
    """Existing `except WMClientError` handlers keep catching it."""
    assert issubclass(WMServiceIdentityMissingError, WMClientError)


async def test_a_callable_service_token_is_resolved_on_every_send() -> None:
    """A rotated secret must take effect without restarting the pod.

    llm_agents holds WMClient as a module-global singleton, so a str captured at
    construction would pin the value for the process lifetime — which is half of
    why the original defect survived undetected.
    """
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["x-tenant-id"])
        return httpx.Response(200, json={"output": "ok"})

    tokens = iter(["first", "second"])
    wm = _make_client(handler, service_token=lambda: next(tokens))
    await wm.send("w", "a")
    await wm.send("w", "b")
    assert seen == ["Bearer first", "Bearer second"]


async def test_a_callable_returning_none_refuses_the_call() -> None:
    """service_identity_token() returns None, not "", when nothing is set."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"output": "ok"})

    wm = _make_client(handler, service_token=lambda: None)
    with pytest.raises(WMServiceIdentityMissingError):
        await wm.send("w", "hi")
    assert calls == []


async def test_a_per_request_tenant_token_is_unaffected_by_a_missing_identity() -> None:
    """The end-user path must not be collateral damage of the new guard."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["x-tenant-id"])
        return httpx.Response(200, json={"output": "ok"})

    wm = _make_client(handler, service_token="")
    await wm.send("w", "hi", tenant_token="019edc48-6060-76e9-8d8f-2e720e12c609")
    assert seen == ["019edc48-6060-76e9-8d8f-2e720e12c609"]


# --- WM_SEND_DELEGATED_LICENCE: delegated licence for bare tenant ids ---------

_END_USER_TENANT = "019edc48-6060-76e9-8d8f-2e720e12c609"


def _capturing() -> tuple[list[dict[str, str]], Callable[[httpx.Request], httpx.Response]]:
    seen: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.headers))
        return httpx.Response(200, json={"output": "ok"})

    return seen, handler


async def test_default_is_off_and_a_bare_tenant_stays_tokenless() -> None:
    """The constructor default must be OFF: shipping this before WM lists
    coding-ai as a delegating subject would turn user calls into wrong-tenant 200s."""
    seen, handler = _capturing()
    transport = httpx.MockTransport(handler)
    wm = WMClient(
        base_url="http://wm.test",
        service_token="svc-token",
        http=httpx.AsyncClient(transport=transport),
    )
    await wm.send("w", "m", tenant_token=_END_USER_TENANT)
    assert seen[0]["x-tenant-id"] == _END_USER_TENANT
    assert "x-effective-tenant-id" not in seen[0]


async def test_on_a_bare_tenant_uuid_sends_service_licence_plus_effective_tenant() -> None:
    seen, handler = _capturing()
    wm = _make_client(handler, send_delegated_licence=True)
    await wm.send("w", "m", tenant_token=_END_USER_TENANT, subject_id="user-42")
    h = seen[0]
    assert h["x-tenant-id"] == "Bearer svc-token"
    assert h["x-effective-tenant-id"] == _END_USER_TENANT
    assert h["x-subject-id"] == "user-42"


async def test_on_the_service_licence_is_resolved_per_call() -> None:
    tokens = iter(["first", "second"])
    seen, handler = _capturing()
    wm = _make_client(handler, service_token=lambda: next(tokens), send_delegated_licence=True)
    await wm.send("w", "m", tenant_token=_END_USER_TENANT)
    await wm.send("w", "m", tenant_token=_END_USER_TENANT)
    assert [h["x-tenant-id"] for h in seen] == ["Bearer first", "Bearer second"]


async def test_on_a_jwt_tenant_token_keeps_todays_behaviour() -> None:
    seen, handler = _capturing()
    wm = _make_client(handler, send_delegated_licence=True)
    await wm.send("w", "m", tenant_token="aa.bb.cc")
    assert seen[0]["x-tenant-id"] == "Bearer aa.bb.cc"
    assert "x-effective-tenant-id" not in seen[0]


async def test_on_no_tenant_keeps_the_service_token_fallback_without_effective_header() -> None:
    seen, handler = _capturing()
    wm = _make_client(handler, send_delegated_licence=True)
    await wm.send("w", "m")
    assert seen[0]["x-tenant-id"] == "Bearer svc-token"
    assert "x-effective-tenant-id" not in seen[0]


async def test_on_a_non_uuid_bare_value_is_not_delegated() -> None:
    """Only a real UUID is delegated; a name/opaque value keeps the legacy
    unprefixed path (WM refuses unresolvable values there) rather than being
    adopted as an effective tenant on coding-ai's licence."""
    seen, handler = _capturing()
    wm = _make_client(handler, send_delegated_licence=True)
    await wm.send("w", "m", tenant_token="n8n-dev")
    assert seen[0]["x-tenant-id"] == "n8n-dev"
    assert "x-effective-tenant-id" not in seen[0]


async def test_on_a_missing_service_licence_refuses_instead_of_going_tokenless() -> None:
    seen, handler = _capturing()
    wm = _make_client(handler, service_token="", send_delegated_licence=True)
    with pytest.raises(WMServiceIdentityMissingError):
        await wm.send("w", "m", tenant_token=_END_USER_TENANT)
    assert seen == []


@pytest.mark.parametrize("raw", [None, "", "0", "false", "off", "no", " OFF ", "False"])
def test_flag_off_values(raw: str | None) -> None:
    assert parse_delegated_licence_flag(raw) is False


@pytest.mark.parametrize("raw", ["1", "true", "on", "yes", " ON ", "True"])
def test_flag_on_values(raw: str | None) -> None:
    assert parse_delegated_licence_flag(raw) is True


@pytest.mark.parametrize("raw", ["enabled", "2", "tru", "onn", "-"])
def test_flag_unknown_values_raise(raw: str) -> None:
    with pytest.raises(ValueError, match="WM_SEND_DELEGATED_LICENCE"):
        parse_delegated_licence_flag(raw)


def test_flag_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WM_SEND_DELEGATED_LICENCE", raising=False)
    assert delegated_licence_enabled() is False
    monkeypatch.setenv("WM_SEND_DELEGATED_LICENCE", "on")
    assert delegated_licence_enabled() is True
    monkeypatch.setenv("WM_SEND_DELEGATED_LICENCE", "bogus")
    with pytest.raises(ValueError):
        delegated_licence_enabled()
