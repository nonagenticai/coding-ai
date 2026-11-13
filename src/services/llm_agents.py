"""LLM agents for coding-ai routed through worker_management.

All LLM traffic flows through ``worker_management`` (``POST
/workers/{name}/message``); coding-ai never speaks to inference-router or
any provider SDK directly. Provider choice + per-agent prompt /
hyperparameters live as worker records in WM, editable via ``PUT
/workers/{name}`` with no coding-ai redeploy.

The legacy ``make_agent_runner`` factory remains, returning a thin shim
that exposes the ``pydantic_ai.Agent``-style ``.run(prompt)`` interface so
the multi-agent fan-out engines (``ai_root_cause`` /
``ai_fix_suggestion`` / ``ai_pattern_detector`` /
``ai_test_failure_analyzer``) keep their current shape. Each call site
now identifies its worker by ``worker_name`` (matching one of the entries
in ``src/config/agents.py``); the seeded prompt + temp + max_tokens live
in WM and are no longer duplicated inline.

Plan: /Users/mini1/.claude/plans/karma-service-help-me-think-fancy-boot.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import TypeAdapter

from auth.tenant import get_request_tenant_id
from clients.outbound_identity import service_identity_token
from config.settings import get_settings
from wm_client import (
    WMBYONotConfiguredError,
    WMClient,
    WMClientError,
    delegated_licence_enabled,
)

logger = logging.getLogger(__name__)


# Registry-aligned schema ids. Kept here for any caller still importing
# them by name. They are NOT a model: every WM worker requests ``local-llm``
# (wm_bootstrap.WORKER_MODEL); the output contract travels per call as
# ``output_schema``.
SCHEMA_TEST_FAILURE = "test_failure_analysis"
SCHEMA_CODE_QUALITY = "code_quality_analysis"
SCHEMA_PERFORMANCE = "performance_analysis"


_wm_http_client: httpx.AsyncClient | None = None
_wm_client: WMClient | None = None


def _get_wm_http_client() -> httpx.AsyncClient:
    global _wm_http_client
    if _wm_http_client is None or _wm_http_client.is_closed:
        _wm_http_client = httpx.AsyncClient(timeout=60.0)
    return _wm_http_client


def _get_wm_client() -> WMClient:
    """Return the shared WMClient. Raises if WORKER_MANAGEMENT_URL is unset."""
    global _wm_client
    if _wm_client is None:
        settings = get_settings()
        if not settings.worker_management_url:
            raise RuntimeError(
                "WORKER_MANAGEMENT_URL is not configured; coding-ai cannot reach the LLM plane"
            )
        if not service_identity_token():
            raise RuntimeError(
                "CODING_SERVICE_TOKEN is not configured; coding-ai cannot identify "
                "itself to worker_management and background calls would be metered "
                "against nobody"
            )
        _wm_client = WMClient(
            base_url=settings.worker_management_url,
            # Callable, not settings.coding_ai_license_token: that field reads
            # CODING_AI_LICENSE_TOKEN, which is deployed in ZERO manifests, so it
            # was "" in every pod that ever ran and no X-Tenant-Id was sent at
            # all on background calls. service_identity_token() reads the var the
            # pod actually carries, per call so rotation needs no restart.
            service_token=service_identity_token,
            http=_get_wm_http_client(),
            send_delegated_licence=delegated_licence_enabled(),
        )
    return _wm_client


async def close_wm_clients() -> None:
    """Close the shared httpx client. Call during FastAPI lifespan shutdown."""
    global _wm_http_client, _wm_client
    if _wm_http_client is not None and not _wm_http_client.is_closed:
        await _wm_http_client.aclose()
    _wm_http_client = None
    _wm_client = None


@dataclass
class AgentRunResult:
    """Mimics pydantic-ai's ``AgentRunResult`` for legacy callers."""

    output: Any


def _schema_for_output_type(output_type: Any) -> dict[str, Any] | None:
    """Build the JSON Schema WM should enforce on the response.

    Returns ``None`` for output types where structured output isn't
    meaningful (``str``, primitives) — WM then returns the raw model
    response unmodified.
    """
    if output_type is None or output_type is str:
        return None
    try:
        return TypeAdapter(output_type).json_schema()
    except Exception:  # noqa: BLE001 — TypeAdapter raises a wide error set
        # Primitive types or unsupported generic shapes — let the model
        # return text and the caller coerce.
        return None


def _coerce_output(raw: Any, output_type: Any) -> Any:
    """Validate WM's parsed ``output`` against the caller's ``output_type``."""
    if output_type is None:
        return raw
    if output_type is str:
        return "" if raw is None else str(raw)
    try:
        return TypeAdapter(output_type).validate_python(raw)
    except Exception:  # noqa: BLE001 — fall back if pydantic can't shape it
        return raw


class _WMAgentShim:
    """Adapter exposing a ``pydantic_ai.Agent``-like interface backed by WM.

    Keeps the legacy ``.run(prompt)`` shape so the multi-agent fan-out
    services don't have to be rewritten yet. WM resolves the worker
    record by ``worker_name`` on every call, so a ``PUT /workers/{name}``
    edit takes effect on the very next call.
    """

    def __init__(self, *, worker_name: str, output_type: Any, work_tier: str | None = None) -> None:
        if not worker_name:
            raise ValueError("worker_name is required")
        self._worker_name = worker_name
        self._output_type = output_type
        self._output_schema = _schema_for_output_type(output_type)
        self._work_tier = work_tier

    async def run(self, prompt: str) -> AgentRunResult:
        wm = _get_wm_client()
        # Forward the END-USER tenant_id (resolved by require_tenant into a
        # request-scoped contextvar) so WM enforces that tenant's byo_mode.
        # None for background/startup calls → WM falls back to the service
        # token, which _get_wm_client has already asserted is present.
        tenant_token = get_request_tenant_id()
        try:
            result = await wm.send(
                worker_name=self._worker_name,
                message=prompt,
                output_schema=self._output_schema,
                work_tier=self._work_tier,
                tenant_token=tenant_token,
            )
        except WMBYONotConfiguredError:
            # Expected for byo_required tenants with no provider; surfaced as a
            # 402 by the app exception handler — don't log a stack trace.
            raise
        except WMClientError:
            logger.exception("worker_management call failed for worker %s", self._worker_name)
            raise

        return AgentRunResult(output=_coerce_output(result.output, self._output_type))


def make_agent_runner(
    *,
    worker_name: str,
    output_type: Any,
    work_tier: str | None = None,
    schema_id: str | None = None,  # accepted for backwards compat, ignored
    system_prompt: str | None = None,  # accepted for backwards compat, ignored
) -> _WMAgentShim:
    """Build a pydantic-ai-Agent-shaped runner backed by worker_management.

    ``worker_name`` MUST match one of the entries in
    ``src/config/agents.py`` (and a corresponding worker record in WM,
    seeded by ``scripts/register_workers.py`` or by the startup self-heal
    in ``src/main.py``).

    ``schema_id`` and ``system_prompt`` are accepted only so existing call
    sites stay compilable while they migrate to the worker-name shape;
    both are sourced from WM at runtime and the values passed here are
    ignored. Delete from call sites after migration.
    """
    if schema_id is not None or system_prompt is not None:
        logger.debug(
            "make_agent_runner: ignoring legacy schema_id/system_prompt for worker=%s "
            "(both now live on the WM worker record)",
            worker_name,
        )
    return _WMAgentShim(worker_name=worker_name, output_type=output_type, work_tier=work_tier)


__all__ = [
    "SCHEMA_CODE_QUALITY",
    "SCHEMA_PERFORMANCE",
    "SCHEMA_TEST_FAILURE",
    "AgentRunResult",
    "close_wm_clients",
    "make_agent_runner",
]
