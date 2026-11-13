"""Tests for the ArchiMate-type classifier orchestrator + router."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from models.architecture_models import (
    ArchimateElementType,
    ArchimateLayer,
    ArchimateProposal,
    CodeUnit,
)
from services.architecture_classifier import (
    ArchitectureClassifier,
    _coerce_proposal,
    _normalize_proposals,
)
from services.llm_agents import AgentRunResult

TENANT_HEADERS = {"X-Tenant-Id": "test-tenant"}


def _unit(
    uid: str = "u1",
    container: str = "ctr-1",
    path: str = "src/foo/bar.py",
    kind: str = "module",
    imports: list[str] | None = None,
    public_symbols: list[str] | None = None,
    doc_excerpt: str = "",
) -> CodeUnit:
    return CodeUnit(
        id=uid,
        container=container,
        path=path,
        kind=kind,
        imports=imports or [],
        public_symbols=public_symbols or [],
        doc_excerpt=doc_excerpt,
    )


def _proposal(
    archimate_type: ArchimateElementType = ArchimateElementType.APPLICATION_COMPONENT,
    archimate_layer: ArchimateLayer = ArchimateLayer.APPLICATION,
    confidence: float = 0.8,
    rationale: str = "looks like an app component",
) -> ArchimateProposal:
    return ArchimateProposal(
        archimate_type=archimate_type,
        archimate_layer=archimate_layer,
        confidence=confidence,
        rationale=rationale,
    )


def _agent_returning(outputs_per_call: list[Any]) -> AsyncMock:
    """AsyncMock whose .run returns AgentRunResult(output=...) per call."""
    mock = AsyncMock()
    mock.run = AsyncMock(side_effect=[AgentRunResult(output=o) for o in outputs_per_call])
    return mock


# ---------------------------------------------------------------------------
# Orchestrator unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_returns_one_result_per_unit_in_input_order():
    units = [_unit("u1"), _unit("u2"), _unit("u3")]
    agent = _agent_returning(
        [
            [_proposal(confidence=0.9)],
            [_proposal(confidence=0.7)],
            [_proposal(confidence=0.6)],
        ]
    )
    with patch.object(ArchitectureClassifier, "__init__", lambda self: None):
        classifier = ArchitectureClassifier()
        classifier._agent = agent
        results = await classifier.classify(units, top_n=3)

    assert [r.code_unit_id for r in results] == ["u1", "u2", "u3"]
    assert agent.run.await_count == 3
    assert all(len(r.ranked) == 1 for r in results)


@pytest.mark.asyncio
async def test_classify_sorts_proposals_by_descending_confidence():
    agent = _agent_returning(
        [
            [
                _proposal(confidence=0.3),
                _proposal(
                    archimate_type=ArchimateElementType.ARTIFACT,
                    archimate_layer=ArchimateLayer.TECHNOLOGY,
                    confidence=0.95,
                ),
                _proposal(
                    archimate_type=ArchimateElementType.APPLICATION_SERVICE,
                    confidence=0.6,
                ),
            ]
        ]
    )
    with patch.object(ArchitectureClassifier, "__init__", lambda self: None):
        classifier = ArchitectureClassifier()
        classifier._agent = agent
        results = await classifier.classify([_unit()], top_n=3)

    confidences = [p.confidence for p in results[0].ranked]
    assert confidences == sorted(confidences, reverse=True)
    assert confidences[0] == 0.95


@pytest.mark.asyncio
async def test_classify_caps_at_top_n():
    agent = _agent_returning(
        [
            [
                _proposal(confidence=0.9),
                _proposal(
                    archimate_type=ArchimateElementType.ARTIFACT,
                    archimate_layer=ArchimateLayer.TECHNOLOGY,
                    confidence=0.8,
                ),
                _proposal(
                    archimate_type=ArchimateElementType.APPLICATION_SERVICE,
                    confidence=0.7,
                ),
                _proposal(
                    archimate_type=ArchimateElementType.DATA_OBJECT,
                    confidence=0.6,
                ),
            ]
        ]
    )
    with patch.object(ArchitectureClassifier, "__init__", lambda self: None):
        classifier = ArchitectureClassifier()
        classifier._agent = agent
        results = await classifier.classify([_unit()], top_n=2)

    assert len(results[0].ranked) == 2


@pytest.mark.asyncio
async def test_classify_drops_out_of_closed_set_proposals():
    """Defense in depth: a stray free-text type from a misbehaving worker is dropped."""
    agent = _agent_returning(
        [
            [
                {
                    "archimate_type": "not_a_real_type",
                    "archimate_layer": "application",
                    "confidence": 0.99,
                    "rationale": "garbage",
                },
                {
                    "archimate_type": "application_component",
                    "archimate_layer": "application",
                    "confidence": 0.7,
                    "rationale": "fine",
                },
            ]
        ]
    )
    with patch.object(ArchitectureClassifier, "__init__", lambda self: None):
        classifier = ArchitectureClassifier()
        classifier._agent = agent
        results = await classifier.classify([_unit()], top_n=3)

    types = [p.archimate_type for p in results[0].ranked]
    assert "not_a_real_type" not in types
    assert "application_component" in types


@pytest.mark.asyncio
async def test_classify_corrects_wrong_layer_for_chosen_type():
    """If the model emits artifact with layer=application, correct to technology."""
    agent = _agent_returning(
        [
            [
                {
                    "archimate_type": "artifact",
                    "archimate_layer": "application",  # wrong, should be technology
                    "confidence": 0.9,
                    "rationale": "deployable shell",
                }
            ]
        ]
    )
    with patch.object(ArchitectureClassifier, "__init__", lambda self: None):
        classifier = ArchitectureClassifier()
        classifier._agent = agent
        results = await classifier.classify([_unit()], top_n=3)

    p = results[0].ranked[0]
    assert p.archimate_type == "artifact"
    assert p.archimate_layer == "technology"


@pytest.mark.asyncio
async def test_classify_returns_empty_ranked_when_agent_raises():
    agent = AsyncMock()
    agent.run = AsyncMock(side_effect=RuntimeError("WM blew up"))
    with patch.object(ArchitectureClassifier, "__init__", lambda self: None):
        classifier = ArchitectureClassifier()
        classifier._agent = agent
        results = await classifier.classify([_unit("u1")], top_n=3)

    assert len(results) == 1
    assert results[0].code_unit_id == "u1"
    assert results[0].ranked == []


@pytest.mark.asyncio
async def test_classify_handles_non_list_output():
    agent = _agent_returning(["surprise string instead of list"])
    with patch.object(ArchitectureClassifier, "__init__", lambda self: None):
        classifier = ArchitectureClassifier()
        classifier._agent = agent
        results = await classifier.classify([_unit("u1")], top_n=3)
    assert results[0].ranked == []


@pytest.mark.asyncio
async def test_classify_echoes_container_back():
    agent = _agent_returning([[_proposal()]])
    with patch.object(ArchitectureClassifier, "__init__", lambda self: None):
        classifier = ArchitectureClassifier()
        classifier._agent = agent
        results = await classifier.classify([_unit("u9", container="ctr-special")], top_n=3)
    assert results[0].container == "ctr-special"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_coerce_proposal_clamps_confidence_above_one():
    p = _coerce_proposal(
        {
            "archimate_type": "application_component",
            "archimate_layer": "application",
            "confidence": 1.5,
            "rationale": "over",
        }
    )
    assert p is not None
    assert p.confidence == 1.0


def test_coerce_proposal_clamps_negative_confidence_to_zero():
    p = _coerce_proposal(
        {
            "archimate_type": "application_component",
            "archimate_layer": "application",
            "confidence": -0.5,
            "rationale": "under",
        }
    )
    assert p is not None
    assert p.confidence == 0.0


def test_coerce_proposal_handles_non_numeric_confidence():
    p = _coerce_proposal(
        {
            "archimate_type": "artifact",
            "archimate_layer": "technology",
            "confidence": "not a number",
            "rationale": "garbage conf",
        }
    )
    assert p is not None
    assert p.confidence == 0.0


def test_coerce_proposal_rejects_non_dict_non_proposal():
    assert _coerce_proposal("just a string") is None
    assert _coerce_proposal(42) is None


def test_coerce_proposal_passes_through_proposal_instance():
    src = _proposal()
    out = _coerce_proposal(src)
    assert isinstance(out, ArchimateProposal)
    assert out.archimate_type == src.archimate_type


def test_normalize_proposals_sorts_and_caps():
    raw = [
        _proposal(confidence=0.2),
        _proposal(confidence=0.9),
        _proposal(confidence=0.5),
        _proposal(confidence=0.7),
    ]
    out = _normalize_proposals(raw, top_n=2)
    assert len(out) == 2
    assert out[0].confidence == 0.9
    assert out[1].confidence == 0.7


# ---------------------------------------------------------------------------
# Router smoke / contract tests
# ---------------------------------------------------------------------------


@pytest.fixture
def client(authed_client):
    """Authenticated: the subject of these tests is the HANDLER, not the gate.

    These routes used to answer 200 with no headers at all — that was the
    defect. They now carry Depends(require_tenant), and AuthMiddleware gates
    everything outside PUBLIC_PATHS besides. The 401 direction is asserted
    deliberately in tests/unit/test_auth_is_registered.py; here the credential
    is supplied so the response BODY stays under test.
    """
    return authed_client


def test_capabilities_endpoint_advertises_closed_set(client):
    response = client.get("/api/ai/architecture/capabilities")
    assert response.status_code == 200
    data = response.json()
    assert "application_component" in data["supported_element_types"]
    assert "artifact" in data["supported_element_types"]
    assert "technology" in data["supported_layers"]
    assert "application" in data["supported_layers"]
    # closed-set count matches the Enum (whole ArchiMate 3.2 universe in use)
    assert len(data["supported_element_types"]) == len(list(ArchimateElementType))
    # 7 layers
    assert len(data["supported_layers"]) == len(list(ArchimateLayer)) == 7
    assert "artifact" in data["typical_code_binding_types"]


def test_classify_returns_200_on_three_unit_payload(client):
    fake_outputs = [
        [_proposal(confidence=0.9)],
        [
            _proposal(
                archimate_type=ArchimateElementType.ARTIFACT,
                archimate_layer=ArchimateLayer.TECHNOLOGY,
                confidence=0.85,
            )
        ],
        [
            _proposal(
                archimate_type=ArchimateElementType.APPLICATION_SERVICE,
                confidence=0.75,
            )
        ],
    ]
    with patch("services.architecture_classifier.make_agent_runner") as factory:
        factory.return_value = _agent_returning(fake_outputs)
        # Reset cached singleton so the patched factory is used
        import services.architecture_classifier as svc

        svc._classifier_singleton = None

        payload = {
            "code_units": [
                {"id": "u1", "container": "c", "path": "src/a.py", "kind": "module"},
                {"id": "u2", "container": "c", "path": "src/b.py", "kind": "module"},
                {"id": "u3", "container": "c", "path": "src/c.py", "kind": "module"},
            ],
            "top_n": 1,
        }
        response = client.post(
            "/api/ai/architecture/tools/classify-code-units",
            json=payload,
            headers=TENANT_HEADERS,
        )

    assert response.status_code == 200, response.text
    data = response.json()
    assert len(data["classifications"]) == 3
    assert [c["code_unit_id"] for c in data["classifications"]] == ["u1", "u2", "u3"]
    # Every proposal type must be in the closed set
    valid_types = {t.value for t in ArchimateElementType}
    for c in data["classifications"]:
        for p in c["ranked"]:
            assert p["archimate_type"] in valid_types


def test_classify_rejects_empty_code_units(client):
    response = client.post(
        "/api/ai/architecture/tools/classify-code-units",
        json={"code_units": [], "top_n": 3},
        headers=TENANT_HEADERS,
    )
    assert response.status_code == 422


def test_classify_rejects_top_n_out_of_range(client):
    response = client.post(
        "/api/ai/architecture/tools/classify-code-units",
        json={"code_units": [{"id": "u1"}], "top_n": 99},
        headers=TENANT_HEADERS,
    )
    assert response.status_code == 422


def test_classify_rejects_more_than_200_units(client):
    units = [{"id": f"u{i}"} for i in range(201)]
    response = client.post(
        "/api/ai/architecture/tools/classify-code-units",
        json={"code_units": units, "top_n": 3},
        headers=TENANT_HEADERS,
    )
    assert response.status_code == 422


def test_agent_config_seeds_into_wm_bootstrap():
    """The classifier worker must appear in AGENT_CONFIGS so wm_bootstrap seeds it."""
    from config.agents import AGENT_CONFIGS

    assert "cai-architecture-classifier" in AGENT_CONFIGS
    cfg = AGENT_CONFIGS["cai-architecture-classifier"]
    assert cfg.schema_id == "architecture_classification"
    assert cfg.temperature <= 0.2  # classification benefits from low randomness
    # System prompt must enumerate (or strongly imply) the closed set, plus the
    # bridge-relevant types the prompt biases toward.
    assert "application_component" in cfg.system_prompt
    assert "artifact" in cfg.system_prompt
