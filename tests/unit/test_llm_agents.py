"""Tests for the WM-routed LLM runners in coding-ai.

After migration to worker_management as the sole LLM gateway, every
``make_agent_runner`` returns a ``_WMAgentShim`` whose ``.run(prompt)``
posts to ``POST /workers/{name}/message`` on WM. The 14 multi-agent
fan-out engines (``ai_root_cause``, ``ai_fix_suggestion``,
``ai_pattern_detector``, ``ai_test_failure_analyzer``) all rely on this
shim, so we exercise:

* worker_name + output_type forwarded onto the WM request body,
* structured output round-trips via ``response_model.model_validate``,
* unstructured ``str`` output_type sends no schema and gets the raw text,
* WMClientError propagates so callers can decide how to degrade.

Plan: /Users/mini1/.claude/plans/karma-service-help-me-think-fancy-boot.md
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from pydantic import BaseModel

from services.llm_agents import (
    AgentRunResult,
    _WMAgentShim,
    make_agent_runner,
)
from wm_client import WMClient, WMClientError


def _wm_client(handler) -> WMClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport)
    # NOT service_token="": that was the shape of the production defect, and a
    # fixture carrying it meant every test here exercised the header-less path
    # while asserting nothing about it. WMClient now refuses an empty identity
    # on a call with no tenant_token — see test_wm_client.py.
    return WMClient(base_url="http://wm.test:8080", service_token="svc-token", http=http)


@pytest.fixture(autouse=True)
def _reset_wm_singletons():
    import services.llm_agents as mod

    mod._wm_client = None
    mod._wm_http_client = None
    yield
    mod._wm_client = None
    mod._wm_http_client = None


class _Out(BaseModel):
    answer: str
    confidence: float


class TestMakeAgentRunner:
    def test_requires_worker_name(self) -> None:
        with pytest.raises(ValueError, match="worker_name"):
            make_agent_runner(worker_name="", output_type=str)

    def test_returns_wm_agent_shim(self) -> None:
        runner = make_agent_runner(worker_name="cai-x", output_type=str)
        assert isinstance(runner, _WMAgentShim)

    def test_legacy_schema_id_and_system_prompt_args_are_accepted(self) -> None:
        """Existing call sites still pass schema_id / system_prompt — must not raise."""
        runner = make_agent_runner(
            worker_name="cai-x",
            output_type=str,
            schema_id="anything",
            system_prompt="ignored at runtime",
        )
        assert runner._worker_name == "cai-x"


class TestRun:
    """``.run(prompt)`` posts to WM and validates the structured output."""

    @pytest.mark.asyncio
    async def test_structured_output_sends_schema_and_validates(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"output": {"answer": "yes", "confidence": 0.9}})

        wm = _wm_client(handler)
        with patch("services.llm_agents._get_wm_client", return_value=wm):
            runner = make_agent_runner(worker_name="cai-x", output_type=_Out)
            result = await runner.run("why?")

        assert isinstance(result, AgentRunResult)
        assert isinstance(result.output, _Out)
        assert result.output.answer == "yes"
        assert captured["url"] == "http://wm.test:8080/workers/cai-x/message"
        assert captured["body"]["message"] == "why?"
        assert captured["body"]["output_schema"]["properties"]["answer"]["type"] == "string"

    @pytest.mark.asyncio
    async def test_list_of_basemodel_output(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "output": [
                        {"answer": "first", "confidence": 0.5},
                        {"answer": "second", "confidence": 0.7},
                    ]
                },
            )

        wm = _wm_client(handler)
        with patch("services.llm_agents._get_wm_client", return_value=wm):
            runner = make_agent_runner(worker_name="cai-x", output_type=list[_Out])
            result = await runner.run("p")

        assert isinstance(result.output, list)
        assert len(result.output) == 2
        assert all(isinstance(o, _Out) for o in result.output)
        # list[_Out] gets a JSON Schema
        assert captured["body"]["output_schema"] is not None

    @pytest.mark.asyncio
    async def test_str_output_skips_schema_and_returns_raw_text(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"output": "free-form answer"})

        wm = _wm_client(handler)
        with patch("services.llm_agents._get_wm_client", return_value=wm):
            runner = make_agent_runner(worker_name="cai-x", output_type=str)
            result = await runner.run("p")

        assert result.output == "free-form answer"
        assert "output_schema" not in captured["body"]

    @pytest.mark.asyncio
    async def test_wm_client_error_propagates(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="provider unavailable")

        wm = _wm_client(handler)
        with patch("services.llm_agents._get_wm_client", return_value=wm):
            runner = make_agent_runner(worker_name="cai-x", output_type=str)
            with pytest.raises(WMClientError, match="500"):
                await runner.run("p")
