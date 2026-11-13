"""Tests for the P4.1 ``GET /api/ai/architecture/proposals?container=<name>`` endpoint.

Three slices:
- Pure mapping (Component → CodeUnit, classification → ranked_candidates, assemble).
- coding-analysis client error paths (missing setting, 404, transient).
- Endpoint integration with the classifier patched + a httpx mock for upstream.

Contract: ``requirements_management/docs/ARCHITECTURE_PROPOSALS_CONTRACT.md``.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from clients.coding_analysis_client import CodingAnalysisUnavailableError
from main import app
from models.architecture_models import (
    ArchimateElementType,
    ArchimateLayer,
    ArchimateProposal,
    CodeUnitClassification,
)
from services.architecture_proposals import (
    archimate_proposal_to_ranked_candidate,
    assemble_proposal_items,
    component_to_code_unit,
    component_to_proposal_code_unit,
)

TENANT_HEADERS = {"X-Tenant-Id": "test-tenant"}


# ─────────────────────────────────────────────────────────────────────────────
# Pure mapping
# ─────────────────────────────────────────────────────────────────────────────


class TestComponentToCodeUnit:
    def test_extracts_id_and_container(self):
        unit = component_to_code_unit(
            {"id": "fr.composer", "name": "Composer", "files": [], "dependencies": [], "tags": []},
            container="fr",
        )
        assert unit.id == "fr.composer"
        assert unit.container == "fr"

    def test_uses_first_file_as_path_when_present(self):
        unit = component_to_code_unit(
            {"id": "x", "name": "X", "files": ["a/b.py", "a/c.py"]},
            container="ctr",
        )
        assert unit.path == "a/b.py"

    def test_uses_first_tag_as_kind_else_component(self):
        with_tag = component_to_code_unit(
            {"id": "x", "name": "X", "files": [], "tags": ["service"]},
            container="ctr",
        )
        without_tag = component_to_code_unit(
            {"id": "x", "name": "X", "files": []},
            container="ctr",
        )
        assert with_tag.kind == "service"
        assert without_tag.kind == "component"

    def test_pipes_dependencies_as_imports(self):
        unit = component_to_code_unit(
            {"id": "x", "name": "X", "files": [], "dependencies": ["a", "b", "c"]},
            container="ctr",
        )
        assert unit.imports == ["a", "b", "c"]


class TestComponentToProposalCodeUnit:
    def test_component_id_is_primary_key(self):
        pu = component_to_proposal_code_unit(
            {"id": "fr.composer", "name": "Composer", "files": ["a/b.ts"], "layer": "presentation"},
            container="fr",
        )
        # The contract's primary binding key, NOT path:
        assert pu.component_id == "fr.composer"
        assert pu.container == "fr"
        assert pu.layer == "presentation"
        assert pu.name == "Composer"
        # path is descriptive, not the key
        assert pu.path == "a/b.ts"

    def test_node_id_is_null_when_unavailable(self):
        # coding-analysis Component schema doesn't expose graph node id.
        pu = component_to_proposal_code_unit({"id": "x", "name": "X", "files": []}, container="ctr")
        assert pu.node_id is None

    def test_handles_missing_layer(self):
        pu = component_to_proposal_code_unit(
            {"id": "x", "name": "X", "files": [], "layer": None},
            container="ctr",
        )
        assert pu.layer == ""


class TestArchimateProposalToRankedCandidate:
    def test_flattens_to_element_type_and_score(self):
        proposal = ArchimateProposal(
            archimate_type=ArchimateElementType.APPLICATION_SERVICE,
            archimate_layer=ArchimateLayer.APPLICATION,
            confidence=0.95,
            rationale="long explanation",
        )
        rc = archimate_proposal_to_ranked_candidate(proposal)
        assert rc.element_type == "application_service"
        assert rc.score == 0.95


class TestAssembleProposalItems:
    def test_one_item_per_component_in_order(self):
        components = [
            {"id": "a", "name": "A", "files": [], "dependencies": [], "tags": []},
            {"id": "b", "name": "B", "files": [], "dependencies": [], "tags": []},
        ]
        classifications = [
            CodeUnitClassification(
                code_unit_id="a",
                container="ctr",
                ranked=[
                    ArchimateProposal(
                        archimate_type=ArchimateElementType.APPLICATION_SERVICE,
                        archimate_layer=ArchimateLayer.APPLICATION,
                        confidence=0.9,
                        rationale="r",
                    )
                ],
            ),
            CodeUnitClassification(code_unit_id="b", container="ctr", ranked=[]),
        ]
        items = assemble_proposal_items(components, classifications, container="ctr")
        assert [i.code_unit.component_id for i in items] == ["a", "b"]
        assert [c.element_type for c in items[0].ranked_candidates] == ["application_service"]
        assert items[1].ranked_candidates == []

    def test_unit_with_no_classification_gets_empty_ranked(self):
        # Degrade-per-unit: a component the classifier dropped still
        # shows up with code_unit populated and empty ranked.
        components = [{"id": "x", "name": "X", "files": [], "dependencies": [], "tags": []}]
        items = assemble_proposal_items(components, [], container="ctr")
        assert len(items) == 1
        assert items[0].code_unit.component_id == "x"
        assert items[0].ranked_candidates == []


# ─────────────────────────────────────────────────────────────────────────────
# coding-analysis client error paths
# ─────────────────────────────────────────────────────────────────────────────


class TestCodingAnalysisClient:
    @pytest.mark.asyncio
    async def test_raises_unavailable_when_url_unset(self, monkeypatch):
        from clients import coding_analysis_client
        from config.settings import get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("CODING_ANALYSIS_URL", "")
        get_settings.cache_clear()
        with pytest.raises(CodingAnalysisUnavailableError):
            await coding_analysis_client.fetch_components("fr", tenant_id="test")
        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_404_returns_empty_not_exception(self, monkeypatch):
        from clients import coding_analysis_client
        from config.settings import get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("CODING_ANALYSIS_URL", "http://upstream.test")
        get_settings.cache_clear()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"detail": "not found"})

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        monkeypatch.setattr(coding_analysis_client, "_get_client", lambda timeout=5.0: client)

        result = await coding_analysis_client.fetch_components("fr", tenant_id="test")
        assert result == []
        await client.aclose()
        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_500_raises_unavailable(self, monkeypatch):
        from clients import coding_analysis_client
        from config.settings import get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("CODING_ANALYSIS_URL", "http://upstream.test")
        get_settings.cache_clear()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        monkeypatch.setattr(coding_analysis_client, "_get_client", lambda timeout=5.0: client)

        with pytest.raises(CodingAnalysisUnavailableError):
            await coding_analysis_client.fetch_components("fr", tenant_id="test")
        await client.aclose()
        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_returns_components_on_success(self, monkeypatch):
        from clients import coding_analysis_client
        from config.settings import get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("CODING_ANALYSIS_URL", "http://upstream.test")
        get_settings.cache_clear()

        captured_headers: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_headers.update(dict(request.headers))
            return httpx.Response(
                200,
                json=[
                    {"id": "a", "name": "A", "files": ["x.py"]},
                    {"id": "b", "name": "B", "files": []},
                ],
            )

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        monkeypatch.setattr(coding_analysis_client, "_get_client", lambda timeout=5.0: client)

        result = await coding_analysis_client.fetch_components("fr", tenant_id="tnt-1")
        assert [c["id"] for c in result] == ["a", "b"]
        # Tenant header forwarded:
        assert captured_headers.get("x-tenant-id") == "tnt-1"
        await client.aclose()
        get_settings.cache_clear()


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint integration
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def client():
    return TestClient(app)


class TestProposalsEndpoint:
    def test_returns_200_with_upstream_unavailable_when_url_unset(self, client, monkeypatch):
        from config.settings import get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("CODING_ANALYSIS_URL", "")
        get_settings.cache_clear()

        resp = client.get("/api/ai/architecture/proposals?container=fr", headers=TENANT_HEADERS)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["container"] == "fr"
        assert body["proposals"] == []
        assert body["upstream_available"] is False
        get_settings.cache_clear()

    def test_returns_empty_when_container_has_no_components(self, client, monkeypatch):
        from clients import coding_analysis_client
        from config.settings import get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("CODING_ANALYSIS_URL", "http://upstream.test")
        get_settings.cache_clear()

        async def empty(container, *, tenant_id):
            return []

        monkeypatch.setattr(coding_analysis_client, "fetch_components", empty)
        # The router imports the symbol; patch it there too so the call site
        # resolves the mock and not the module-level original.
        import routers.ai_architecture as router_mod

        monkeypatch.setattr(router_mod, "fetch_components", empty)

        resp = client.get("/api/ai/architecture/proposals?container=fr", headers=TENANT_HEADERS)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["proposals"] == []
        assert body["upstream_available"] is True
        get_settings.cache_clear()

    def test_returns_items_with_ranked_candidates_when_classifier_returns(
        self, client, monkeypatch
    ):
        import routers.ai_architecture as router_mod
        from config.settings import get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("CODING_ANALYSIS_URL", "http://upstream.test")
        get_settings.cache_clear()

        components = [
            {
                "id": "fr.composer",
                "name": "Composer",
                "files": ["frontend/src/lib/composer.ts"],
                "dependencies": [],
                "layer": "presentation",
                "tags": ["service"],
            }
        ]

        async def fake_fetch(container, *, tenant_id):
            return components

        monkeypatch.setattr(router_mod, "fetch_components", fake_fetch)

        class _FakeClassifier:
            async def classify(self, *, units, top_n):
                # Return one ranked proposal for fr.composer.
                return [
                    CodeUnitClassification(
                        code_unit_id="fr.composer",
                        container="fr",
                        ranked=[
                            ArchimateProposal(
                                archimate_type=ArchimateElementType.APPLICATION_SERVICE,
                                archimate_layer=ArchimateLayer.APPLICATION,
                                confidence=0.91,
                                rationale="composes ui panels",
                            )
                        ],
                    )
                ]

        monkeypatch.setattr(router_mod, "get_architecture_classifier", lambda: _FakeClassifier())

        resp = client.get("/api/ai/architecture/proposals?container=fr", headers=TENANT_HEADERS)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["container"] == "fr"
        assert len(body["proposals"]) == 1
        item = body["proposals"][0]
        assert item["code_unit"]["component_id"] == "fr.composer"
        assert item["code_unit"]["container"] == "fr"
        assert item["ranked_candidates"] == [{"element_type": "application_service", "score": 0.91}]
        get_settings.cache_clear()

    def test_degrades_to_empty_ranked_when_classifier_raises(self, client, monkeypatch):
        import routers.ai_architecture as router_mod
        from config.settings import get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("CODING_ANALYSIS_URL", "http://upstream.test")
        get_settings.cache_clear()

        components = [{"id": "x", "name": "X", "files": [], "dependencies": [], "tags": []}]

        async def fake_fetch(container, *, tenant_id):
            return components

        class _RaisingClassifier:
            async def classify(self, *, units, top_n):
                raise RuntimeError("worker_management is on fire")

        monkeypatch.setattr(router_mod, "fetch_components", fake_fetch)
        monkeypatch.setattr(router_mod, "get_architecture_classifier", lambda: _RaisingClassifier())

        resp = client.get("/api/ai/architecture/proposals?container=x", headers=TENANT_HEADERS)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # Code-unit shell present, ranked empty (degrade-per-unit), upstream_available stays True.
        assert len(body["proposals"]) == 1
        assert body["proposals"][0]["code_unit"]["component_id"] == "x"
        assert body["proposals"][0]["ranked_candidates"] == []
        assert body["upstream_available"] is True
        get_settings.cache_clear()

    def test_limit_offset_bounds_classifier_work_and_reports_total(self, client, monkeypatch):
        # The classifier runs one LLM call per unit, so a large container must be
        # paged. Assert only the requested page is classified and `total` reflects
        # the full component count (so the consumer can page the review queue).
        import routers.ai_architecture as router_mod
        from config.settings import get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("CODING_ANALYSIS_URL", "http://upstream.test")
        get_settings.cache_clear()

        components = [
            {"id": f"c{i}", "name": f"C{i}", "files": [], "dependencies": [], "tags": []}
            for i in range(5)
        ]

        async def fake_fetch(container, *, tenant_id):
            return components

        seen: dict[str, list[str]] = {}

        class _RecordingClassifier:
            async def classify(self, *, units, top_n):
                seen["ids"] = [u.id for u in units]
                return []

        monkeypatch.setattr(router_mod, "fetch_components", fake_fetch)
        monkeypatch.setattr(
            router_mod, "get_architecture_classifier", lambda: _RecordingClassifier()
        )

        resp = client.get(
            "/api/ai/architecture/proposals?container=fr&limit=2&offset=1",
            headers=TENANT_HEADERS,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["total"] == 5
        assert body["limit"] == 2
        assert body["offset"] == 1
        assert len(body["proposals"]) == 2
        # Only the requested page (offset 1, limit 2) reached the classifier.
        assert seen["ids"] == ["c1", "c2"]
        get_settings.cache_clear()

    def test_400_on_missing_container_query_param(self, client):
        # The Query is required (min_length=1); FastAPI returns 422.
        resp = client.get("/api/ai/architecture/proposals", headers=TENANT_HEADERS)
        assert resp.status_code == 422

    def test_requires_tenant(self, client):
        # No X-Tenant-Id header → 401/403 per the standard auth dependency.
        resp = client.get("/api/ai/architecture/proposals?container=fr")
        assert resp.status_code in (401, 403)
