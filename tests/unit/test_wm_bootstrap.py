"""Unit tests for coding-ai wm_bootstrap.ensure_workers_registered."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from config.agents import CodingAgentConfig
from wm_bootstrap import build_create_payload, ensure_workers_registered


def _agent(name: str, prompt: str = "system prompt") -> CodingAgentConfig:
    return CodingAgentConfig(
        worker_name=name,
        schema_id="test_failure_analysis",
        system_prompt=prompt,
        temperature=0.2,
        max_tokens=3000,
    )


def _http(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_payload_contains_required_worker_create_fields() -> None:
    agent = _agent("cai-x", "do x")
    p = build_create_payload(agent, base_url_override="http://ir:8150/api")
    assert p == {
        "worker_name": "cai-x",
        "agent_role": "coding-ai agent (cai-x)",
        "agent_type": "pydantic_ai",
        "model_provider": "inference_router",
        "model": "local-llm",
        "temperature": 0.2,
        "max_tokens": 3000,
        "system_prompt": "do x",
        "base_url_override": "http://ir:8150/api",
    }


async def test_creates_missing_skips_existing() -> None:
    agents = {"a": _agent("cai-a"), "b": _agent("cai-b")}
    posts = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET" and req.url.path.endswith("/cai-a"):
            return httpx.Response(200, json={"worker_name": "cai-a"})
        if req.method == "GET" and req.url.path.endswith("/cai-b"):
            return httpx.Response(404, json={"error": {"code": "NOT_FOUND"}})
        if req.method == "POST":
            posts.append(req.content.decode())
            return httpx.Response(201, json={"worker_name": "cai-b"})
        raise AssertionError(f"unexpected {req.method} {req.url}")

    actions = await ensure_workers_registered(
        _http(handler),
        wm_base_url="http://wm:8080",
        license_token="t",
        base_url_override="http://ir:8150/api",
        agent_configs=agents,
    )
    assert actions == {"cai-a": "skipped-exists", "cai-b": "created"}
    assert len(posts) == 1


async def test_force_overwrites() -> None:
    agents = {"a": _agent("cai-a")}
    methods: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        methods.append(req.method)
        if req.method == "GET":
            return httpx.Response(200, json={"worker_name": "cai-a"})
        return httpx.Response(200, json={"worker_name": "cai-a"})

    actions = await ensure_workers_registered(
        _http(handler),
        wm_base_url="http://wm:8080",
        license_token="t",
        base_url_override="http://ir:8150/api",
        overwrite=True,
        agent_configs=agents,
    )
    assert actions == {"cai-a": "updated"}
    assert methods == ["GET", "PUT"]


async def test_dry_run_no_writes() -> None:
    agents = {"a": _agent("cai-a"), "b": _agent("cai-b")}
    methods: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        methods.append(req.method)
        if req.url.path.endswith("/cai-a"):
            return httpx.Response(200, json={"worker_name": "cai-a"})
        return httpx.Response(404, json={"error": {"code": "NOT_FOUND"}})

    actions = await ensure_workers_registered(
        _http(handler),
        wm_base_url="http://wm:8080",
        license_token="t",
        base_url_override="http://ir:8150/api",
        overwrite=True,
        dry_run=True,
        agent_configs=agents,
    )
    assert actions == {"cai-a": "would-update", "cai-b": "would-create"}
    assert all(m == "GET" for m in methods)


async def test_required_kwargs_validated() -> None:
    h = _http(lambda r: httpx.Response(200, json={}))
    with pytest.raises(ValueError, match="wm_base_url"):
        await ensure_workers_registered(
            h,
            wm_base_url="",
            license_token="t",
            base_url_override="http://ir:8150/api",
            agent_configs={},
        )
    with pytest.raises(ValueError, match="base_url_override"):
        await ensure_workers_registered(
            h,
            wm_base_url="http://wm:8080",
            license_token="t",
            base_url_override="",
            agent_configs={},
        )


async def test_empty_license_token_uses_trusted_network_bypass() -> None:
    agents = {"a": _agent("cai-a")}
    captured: dict[str, str] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(dict(req.headers))
        if req.method == "GET":
            return httpx.Response(404, json={"error": {"code": "NOT_FOUND"}})
        return httpx.Response(201, json={"worker_name": "cai-a"})

    actions = await ensure_workers_registered(
        _http(handler),
        wm_base_url="http://wm:8080",
        license_token="",
        base_url_override="http://ir:8150/api",
        agent_configs=agents,
    )
    assert actions == {"cai-a": "created"}
    assert "x-tenant-id" not in captured


def test_real_agent_configs_loads_all_agents() -> None:
    """Sanity-check the generated config at /src/config/agents.py.

    Each agent must be cai-prefixed, have a non-empty system prompt, and use
    one of the known schema_ids. Test-failure-analysis agents share a single
    schema_id; the architecture classifier has its own.
    """
    from config.agents import AGENT_CONFIGS

    assert len(AGENT_CONFIGS) >= 14
    assert all(name.startswith("cai-") for name in AGENT_CONFIGS)
    assert all(c.system_prompt for c in AGENT_CONFIGS.values())
    known_schemas = {"test_failure_analysis", "architecture_classification"}
    assert all(c.schema_id in known_schemas for c in AGENT_CONFIGS.values())
    # the 14 test-failure analyzers
    assert sum(1 for c in AGENT_CONFIGS.values() if c.schema_id == "test_failure_analysis") == 14


async def test_every_created_or_overwritten_worker_requests_local_llm() -> None:
    """Guard: ONE model, ``local-llm``, on every worker coding-ai writes to WM.

    Drives the REAL ``AGENT_CONFIGS`` through both write paths (POST for a
    missing row, PUT under ``overwrite``) and inspects the bytes on the wire.
    A schema id, a tier/placeholder or a concrete model name in ``model``
    fails here -- the schema id is not a model and WM has no field for it.
    """
    import json

    from config.agents import AGENT_CONFIGS

    schema_ids = {c.schema_id for c in AGENT_CONFIGS.values()}
    bodies: dict[str, list[dict]] = {"POST": [], "PUT": []}

    def handler_for(existing: bool) -> Callable[[httpx.Request], httpx.Response]:
        def handler(req: httpx.Request) -> httpx.Response:
            if req.method == "GET":
                return httpx.Response(200 if existing else 404, json={})
            bodies[req.method].append(json.loads(req.content))
            return httpx.Response(201 if req.method == "POST" else 200, json={})

        return handler

    for existing in (False, True):
        await ensure_workers_registered(
            _http(handler_for(existing)),
            wm_base_url="http://wm:8080",
            base_url_override="http://ir:8150/api",
            overwrite=existing,
        )

    assert len(bodies["POST"]) == len(AGENT_CONFIGS)
    assert len(bodies["PUT"]) == len(AGENT_CONFIGS)
    for body in bodies["POST"] + bodies["PUT"]:
        assert body["model"] == "local-llm", body["worker_name"]
        leaked = {k for k, v in body.items() if k != "system_prompt" and v in schema_ids}
        assert not leaked, (body["worker_name"], leaked)
