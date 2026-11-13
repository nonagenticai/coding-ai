"""Idempotent coding-ai → worker_management worker-record seeding.

Every coding-ai agent's prompt + LLM hyperparameters live as worker records
in worker_management so they're editable via ``PUT /workers/{name}`` with
no coding-ai redeploy.

* On a clean WM database (or a new environment), call
  ``ensure_workers_registered`` once and every record from
  ``CodingAgentConfig`` entries is created.
* On every coding-ai startup, call it again. Existing records are left
  alone (operator edits preserved); only missing records are recreated.
  So flushing the WM database, moving to a new cluster, or losing a
  worker row to a bad migration self-heals on the next coding-ai boot.
* Operators bulk-overwrite by running ``scripts/register_workers.py --force``.

Plan: /Users/mini1/.claude/plans/karma-service-help-me-think-fancy-boot.md
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import httpx

from config.agents import AGENT_CONFIGS, CodingAgentConfig

logger = logging.getLogger(__name__)


# The ONE model every worker requests. The concrete vLLM model on pgx is
# defined once (pgx vllm.service) and served under this logical name; no
# worker record may carry a concrete model, a tier/placeholder, or a schema id.
WORKER_MODEL = "local-llm"


SeedAction = Literal["created", "skipped-exists", "updated", "would-create", "would-update"]


def build_create_payload(
    agent: CodingAgentConfig,
    *,
    base_url_override: str,
) -> dict[str, Any]:
    """Translate a ``CodingAgentConfig`` into a ``WorkerCreateRequest`` body.

    ``model_provider`` is fixed to ``inference_router`` at seed time; ops
    can flip it (or any other field) per-worker via ``PUT /workers/{name}``
    with no coding-ai redeploy.

    ``model`` is always ``WORKER_MODEL``. ``agent.schema_id`` is deliberately
    NOT sent: WM's ``WorkerCreateRequest`` has no schema-id field (it is
    ``extra="ignore"``), and the structured-output contract travels per call
    as ``output_schema`` on ``POST /workers/{name}/message`` (see
    ``services.llm_agents``). It used to be smuggled in as ``model``.
    """
    return {
        "worker_name": agent.worker_name,
        "agent_role": f"coding-ai agent ({agent.worker_name})",
        "agent_type": "pydantic_ai",
        "model_provider": "inference_router",
        "model": WORKER_MODEL,
        "temperature": agent.temperature,
        "max_tokens": agent.max_tokens,
        "system_prompt": agent.system_prompt,
        "base_url_override": base_url_override,
    }


async def ensure_workers_registered(
    http: httpx.AsyncClient,
    *,
    wm_base_url: str,
    license_token: str = "",
    base_url_override: str,
    overwrite: bool = False,
    dry_run: bool = False,
    agent_configs: dict[str, CodingAgentConfig] | None = None,
) -> dict[str, SeedAction]:
    """Ensure every coding-ai agent has a matching worker record in WM.

    ``license_token`` is optional: empty token relies on WM's RFC1918
    trusted-network bypass (which accepts in-cluster traffic). For seeding
    from outside the cluster, pass a Bearer-able license token.
    """
    if not wm_base_url:
        raise ValueError("ensure_workers_registered requires wm_base_url")
    if not base_url_override:
        raise ValueError(
            "ensure_workers_registered requires base_url_override "
            "(inference-router URL to put on each worker record)"
        )

    base = wm_base_url.rstrip("/")
    headers: dict[str, str] = {}
    if license_token:
        headers["X-Tenant-Id"] = f"Bearer {license_token}"
    configs = agent_configs if agent_configs is not None else AGENT_CONFIGS

    actions: dict[str, SeedAction] = {}
    for agent in configs.values():
        name = agent.worker_name
        get_resp = await http.get(f"{base}/workers/{name}", headers=headers, timeout=10.0)

        if get_resp.status_code == 200:
            if not overwrite:
                actions[name] = "skipped-exists"
                continue
            payload = build_create_payload(agent, base_url_override=base_url_override)
            if dry_run:
                actions[name] = "would-update"
                continue
            put_resp = await http.put(
                f"{base}/workers/{name}", json=payload, headers=headers, timeout=30.0
            )
            put_resp.raise_for_status()
            actions[name] = "updated"
            logger.info("worker %s updated in WM", name)
            continue

        if get_resp.status_code != 404:
            get_resp.raise_for_status()

        payload = build_create_payload(agent, base_url_override=base_url_override)
        if dry_run:
            actions[name] = "would-create"
            continue
        post_resp = await http.post(f"{base}/workers", json=payload, headers=headers, timeout=30.0)
        post_resp.raise_for_status()
        actions[name] = "created"
        logger.info("worker %s created in WM", name)

    return actions
