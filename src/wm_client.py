"""worker_management proxy client for LLM calls.

coding-ai routes ALL LLM traffic through worker_management
(``worker-management:8887``). coding-ai never
speaks to inference-router or any provider SDK directly — provider choice
lives on each worker record in WM and is changed via ``PUT /workers/{name}``
with no coding-ai redeploy.

Plan: /Users/mini1/.claude/plans/karma-service-help-me-think-fancy-boot.md
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


# Flag that switches the bare-tenant-id branch of ``WMClient.send`` from the
# TOKENLESS assertion (``X-Tenant-Id: <uuid>``, no credential) to the DELEGATED
# one (``X-Tenant-Id: Bearer <CODING_SERVICE_TOKEN>`` +
# ``X-Effective-Tenant-Id: <uuid>``) — the shape fr already uses toward WM.
#
# ⛔ DEFAULT OFF, AND TURNING IT ON IS ORDERED. worker_management honours
# X-Effective-Tenant-Id ONLY for subjects listed in its AUTH_DELEGATING_SUBJECTS.
# coding-ai's licence subject (svc-coding-ai, 019ff970-e327-…) was NOT on that
# list when this was written. Flip this on before that gitops change lands and
# WM silently ignores the header and scopes every user call to coding-ai's OWN
# tenant: a wrong-tenant 200, with that tenant's BYO posture, not the user's.
# Order: (1) add the subject to AUTH_DELEGATING_SUBJECTS, verify on the live
# Deployment; (2) only then set WM_SEND_DELEGATED_LICENCE=on here.
DELEGATED_LICENCE_ENV = "WM_SEND_DELEGATED_LICENCE"
EFFECTIVE_TENANT_HEADER = "X-Effective-Tenant-Id"
_FLAG_ON = frozenset({"1", "true", "on", "yes"})
_FLAG_OFF = frozenset({"", "0", "false", "off", "no"})


def parse_delegated_licence_flag(raw: str | None) -> bool:
    """Parse ``WM_SEND_DELEGATED_LICENCE``; raise on anything unrecognised.

    An unknown value is a startup error, not a silent OFF: a typo such as
    ``WM_SEND_DELEGATED_LICENCE=enabled`` must not leave an operator believing
    the tokenless path is closed while it is still open.
    """
    value = (raw or "").strip().lower()
    if value in _FLAG_ON:
        return True
    if value in _FLAG_OFF:
        return False
    raise ValueError(
        f"{DELEGATED_LICENCE_ENV}={raw!r} is not recognised; use one of "
        f"{sorted(_FLAG_ON)} to enable or {sorted(_FLAG_OFF - {''})} (or unset) to disable"
    )


def delegated_licence_enabled() -> bool:
    """Read and validate ``WM_SEND_DELEGATED_LICENCE`` from the environment."""
    return parse_delegated_licence_flag(os.getenv(DELEGATED_LICENCE_ENV))


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


class MessageResponse(BaseModel):
    """Subset of worker_management's ``MessageResponse`` we consume."""

    model_config = ConfigDict(extra="ignore")

    output: Any
    output_type: str | None = None
    output_schema_used: dict[str, Any] | None = None


class WMClientError(RuntimeError):
    """Raised when the worker_management proxy call fails terminally."""


class WMServiceIdentityMissingError(WMClientError):
    """Raised when a call has no end-user tenant AND no service identity.

    The alternative — the behaviour this replaces — was to send the request with
    no ``X-Tenant-Id`` header at all. worker_management's trusted-network bypass
    accepts that from an in-cluster source, so the call SUCCEEDS and is metered
    against nobody. Failing here is the point: an unattributable call is a
    billing defect that no downstream error will ever report.
    """


class WMBYONotConfiguredError(WMClientError):
    """Raised when a byo_required tenant has no LLM provider configured (HTTP 402).

    worker_management returns 402 ``{"code": "byo_llm_not_configured", ...}`` when
    a commercial (byo_required) tenant runs an agent without configuring its own
    provider. Surfaced distinctly so the upstream API can tell the user to set up
    their LLM provider instead of returning a generic upstream failure.
    """

    def __init__(self, worker_name: str, message: str) -> None:
        super().__init__(message)
        self.worker_name = worker_name
        self.message = message


class WMValidationError(WMClientError):
    """Raised when WM rejects the request body (HTTP 422).

    The most common cause is WM's strict_mode security validation firing on
    the ``message`` payload — SQL-injection patterns, XSS, control
    characters, or path-traversal sequences detected in agent input.
    coding-ai keeps strict_mode on; this exception is the channel that lets
    the upstream API surface a precise rejection message instead of a
    generic 500.
    """

    def __init__(
        self,
        worker_name: str,
        message: str,
        validation_errors: list[dict[str, Any]],
    ) -> None:
        super().__init__(message)
        self.worker_name = worker_name
        self.message = message
        self.validation_errors = validation_errors


class WMClient:
    """HTTP client for ``POST /workers/{name}/message``.

    No client-side cache: every call hits WM so a ``PUT /workers/{name}``
    edit takes effect on the very next request by definition.
    """

    def __init__(
        self,
        base_url: str,
        service_token: str | Callable[[], str | None],
        http: httpx.AsyncClient,
        *,
        timeout: float = 60.0,
        send_delegated_licence: bool = False,
    ) -> None:
        if not base_url:
            raise ValueError("WMClient.base_url is required")
        self._base_url = base_url.rstrip("/")
        # A callable is resolved on EVERY send so a rotated secret takes effect
        # without a restart — this client is a module-global singleton in
        # llm_agents, so a str captured here would pin the value for the process
        # lifetime. A plain str stays accepted for tests and ad-hoc callers.
        #
        # An empty service_token is NOT fine, despite what this comment used to
        # claim. It is true that in-cluster traffic falls through WM's
        # trusted-network bypass (RFC1918 sources accepted without X-Tenant-Id)
        # — that is exactly the problem: the call is served and attributed to
        # nobody. send() now refuses instead. Per-request tenant_token still
        # wins when forwarded from an authenticated user request.
        self._service_token = service_token
        self._http = http
        self._timeout = timeout
        self._send_delegated_licence = send_delegated_licence

    def _resolve_service_token(self) -> str:
        """This service's identity at CALL time, never at construction time."""
        tok = self._service_token
        if not isinstance(tok, str):
            tok = tok()
        return tok or ""

    def _require_service_token(self, worker_name: str) -> str:
        service_token = self._resolve_service_token()
        if not service_token:
            raise WMServiceIdentityMissingError(
                "refusing to call worker_management with no service identity "
                f"(worker={worker_name}): CODING_SERVICE_TOKEN is unset or empty in "
                "this process, so the request would go out with no credential and "
                "be metered against nobody"
            )
        return service_token

    async def send(
        self,
        worker_name: str,
        message: str,
        *,
        output_schema: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        instruction: str | None = None,
        tenant_token: str | None = None,
        subject_id: str | None = None,
        work_tier: str | None = None,
    ) -> MessageResponse:
        """POST to ``/workers/{worker_name}/message`` and return the parsed result.

        ``tenant_token`` is the per-call end-user identity. A JWT-shaped value
        is forwarded as ``X-Tenant-Id: Bearer <token>``. A bare tenant UUID is
        sent unprefixed (tokenless) unless ``send_delegated_licence`` is set, in
        which case it goes out as ``X-Tenant-Id: Bearer <service token>`` plus
        ``X-Effective-Tenant-Id: <uuid>``. When omitted, falls
        back to the constructor's ``service_token`` (startup pre-warm,
        background jobs) — and raises if that is empty too, rather than
        emitting a header-less, unattributable call.

        ``subject_id`` is forwarded as ``X-Subject-Id`` for per-user metering.

        ``work_tier`` (small/medium/high) is forwarded so WM selects the
        tenant's per-tier BYO LLM model for this call; omit for the tenant
        default.

        Raises:
            WMServiceIdentityMissingError: no ``tenant_token`` and no service
                identity configured — the call is refused, not sent.
            WMValidationError: WM 422 (input rejection, security validation).
            WMClientError: transport failure (after one retry), other HTTP
                error, or unparseable response.
        """
        url = f"{self._base_url}/workers/{worker_name}/message"
        headers: dict[str, str] = {}
        # X-Tenant-Id ladder (mirrors kag):
        #   1. tenant_token — the per-request end-user identity. Bearer-prefix
        #      only if JWT-shaped (>=2 dots); a bare tenant_id UUID is sent
        #      UNPREFIXED so WM's trusted-network path resolves byo_mode by
        #      tenant id (Fix A) and enforces byo_required for the END-USER.
        #   2. service_token — coding-ai's own opaque license token; always
        #      Bearer (introspect succeeds on the opaque form). Used only for
        #      background/startup calls with no per-request tenant.
        #   1b. With WM_SEND_DELEGATED_LICENCE on, a bare tenant UUID is NOT sent
        #      tokenless: coding-ai presents its own licence and names the
        #      end-user tenant in X-Effective-Tenant-Id. WM (as a delegating
        #      subject) re-scopes to that tenant and re-resolves byo_mode for IT
        #      (worker_management src/api/auth_middleware.py, `if delegated:` ->
        #      _lookup_tenant_byo_mode), so byo_required still binds the END-USER.
        if tenant_token and tenant_token.count(".") >= 2:
            headers["X-Tenant-Id"] = f"Bearer {tenant_token}"
        elif tenant_token and self._send_delegated_licence and _is_uuid(tenant_token):
            service_token = self._require_service_token(worker_name)
            headers["X-Tenant-Id"] = f"Bearer {service_token}"
            headers[EFFECTIVE_TENANT_HEADER] = tenant_token
        elif tenant_token:
            headers["X-Tenant-Id"] = tenant_token
        else:
            headers["X-Tenant-Id"] = f"Bearer {self._require_service_token(worker_name)}"
        if subject_id:
            headers["X-Subject-Id"] = subject_id

        body: dict[str, Any] = {"message": message}
        if output_schema is not None:
            body["output_schema"] = output_schema
        if system_prompt is not None:
            body["system_prompt"] = system_prompt
        if instruction is not None:
            body["instruction"] = instruction
        if work_tier is not None:
            body["work_tier"] = work_tier

        resp = await self._post_with_one_retry(url, body, headers, worker_name)

        if resp.status_code == 422:
            self._raise_validation_error(resp, worker_name)
        if resp.status_code == 402:
            self._raise_byo_error(resp, worker_name)
        if resp.status_code >= 400:
            raise WMClientError(
                f"worker_management {resp.status_code} on {worker_name}: {resp.text[:500]}"
            )

        try:
            payload = resp.json()
        except ValueError as exc:
            raise WMClientError(
                f"worker_management response not valid JSON for {worker_name}: {exc}"
            ) from exc

        try:
            return MessageResponse.model_validate(payload)
        except Exception as exc:
            raise WMClientError(
                f"worker_management response not a MessageResponse for {worker_name}: {exc}"
            ) from exc

    def _raise_byo_error(self, resp: httpx.Response, worker_name: str) -> None:
        """Raise WMBYONotConfiguredError. Called only on HTTP 402.

        HTTP 402 is reserved fleet-wide for BYO-not-configured, so key on the
        STATUS, not a body code: WM wraps the detail as
        ``{"error":{"message":{"code":"byo_llm_not_configured",...}}}``, so a
        top-level ``payload.get("code")`` check silently missed it and the 402
        leaked out as a generic WMClientError. Dig the actionable message out
        best-effort across the known envelope shapes.
        """
        message = "Configure your LLM provider before running agents."
        try:
            payload = resp.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            error_node = payload.get("error")
            nested = error_node.get("message") if isinstance(error_node, dict) else None
            for node in (payload, payload.get("detail"), nested):
                if (
                    isinstance(node, dict)
                    and node.get("code") == "byo_llm_not_configured"
                    and isinstance(node.get("message"), str)
                ):
                    message = node["message"]
                    break
        raise WMBYONotConfiguredError(worker_name, message)

    def _raise_validation_error(self, resp: httpx.Response, worker_name: str) -> None:
        try:
            payload = resp.json()
        except ValueError:
            raise WMValidationError(
                worker_name=worker_name,
                message=f"worker_management 422 with non-JSON body: {resp.text[:500]}",
                validation_errors=[],
            ) from None

        error_block = payload.get("error") if isinstance(payload, dict) else None
        detail = (
            error_block.get("message") if isinstance(error_block, dict) else None
        ) or "input rejected by worker_management validation"
        raw_errors = payload.get("validation_errors") if isinstance(payload, dict) else None
        errors: list[dict[str, Any]] = (
            [e for e in raw_errors if isinstance(e, dict)] if isinstance(raw_errors, list) else []
        )
        raise WMValidationError(
            worker_name=worker_name,
            message=detail,
            validation_errors=errors,
        )

    async def _post_with_one_retry(
        self,
        url: str,
        body: dict[str, Any],
        headers: dict[str, str],
        worker_name: str,
    ) -> httpx.Response:
        try:
            return await self._http.post(url, json=body, headers=headers, timeout=self._timeout)
        except httpx.TransportError as first:
            logger.warning(
                "worker_management transport error on %s; retrying once: %s",
                worker_name,
                first,
            )
            try:
                return await self._http.post(url, json=body, headers=headers, timeout=self._timeout)
            except httpx.TransportError as second:
                raise WMClientError(
                    f"worker_management unreachable for {worker_name}: {second}"
                ) from second
