"""Mapping layer for P4.1 — coding-analysis Component → contract code_unit
and classifier ranked output → contract ranked_candidates.

Pure functions only. The orchestration (fetch + classify + assemble) lives in
the router; this module exists so the mapping can be unit-tested without
standing up httpx or worker_management.

Contract: ``requirements_management/docs/ARCHITECTURE_PROPOSALS_CONTRACT.md``.
"""

from __future__ import annotations

from typing import Any

from models.architecture_models import (
    ArchimateProposal,
    ArchitectureProposalItem,
    CodeUnit,
    CodeUnitClassification,
    ProposalCodeUnit,
    RankedCandidate,
)


def component_to_code_unit(component: dict[str, Any], *, container: str) -> CodeUnit:
    """Map a coding-analysis ``Component`` dict to the classifier's ``CodeUnit``.

    The classifier reads ``imports`` / ``public_symbols`` / ``doc_excerpt``
    as evidence; coding-analysis exposes ``dependencies`` and ``files`` /
    ``tags`` so we plumb what's available:

    - ``id`` ← ``Component.id`` (the primary binding key the contract pins on)
    - ``container`` ← request param
    - ``path`` ← first file (descriptive, classifier uses as hint)
    - ``kind`` ← first tag, or ``"component"``
    - ``imports`` ← ``Component.dependencies`` (component-ids this one depends on)
    - ``public_symbols`` ← `[]` (not available from coding-analysis)
    - ``doc_excerpt`` ← `""` (not available from coding-analysis)
    """
    files = component.get("files") or []
    tags = component.get("tags") or []
    dependencies = component.get("dependencies") or []
    return CodeUnit(
        id=str(component.get("id", "")),
        container=container,
        path=str(files[0]) if files else "",
        kind=str(tags[0]) if tags else "component",
        imports=[str(d) for d in dependencies if d],
        public_symbols=[],
        doc_excerpt="",
    )


def component_to_proposal_code_unit(
    component: dict[str, Any], *, container: str
) -> ProposalCodeUnit:
    """Map a Component dict to the contract ``code_unit`` shape.

    Kept separate from ``component_to_code_unit`` because the contract block
    carries ``component_id`` + ``name`` + ``layer`` (coding-analysis layer)
    that the classifier input doesn't need but the downstream UI does.
    """
    files = component.get("files") or []
    tags = component.get("tags") or []
    return ProposalCodeUnit(
        container=container,
        component_id=str(component.get("id", "")),
        name=str(component.get("name", "")),
        layer=str(component.get("layer") or ""),
        kind=str(tags[0]) if tags else "component",
        path=str(files[0]) if files else "",
        node_id=None,
    )


def archimate_proposal_to_ranked_candidate(p: ArchimateProposal) -> RankedCandidate:
    """Flatten ``{archimate_type, archimate_layer, confidence, rationale}``
    to the cross-service ``{element_type, score}``. Rationale + layer are
    not part of the slim contract; the downstream UI shows ``element_type``
    + confidence next to the human-readable code-unit row.
    """
    return RankedCandidate(
        element_type=str(p.archimate_type),
        score=float(p.confidence),
    )


def assemble_proposal_items(
    components: list[dict[str, Any]],
    classifications: list[CodeUnitClassification],
    *,
    container: str,
) -> list[ArchitectureProposalItem]:
    """Join a Component list with the classifier's per-unit ranked output.

    Order follows the Component list (i.e. coding-analysis's order). When a
    Component has no classification (e.g. the classifier dropped it on a
    transient error), ``ranked_candidates`` is empty for that item — never
    a hard failure for the whole batch. Mirrors the contract's
    "errors degrade per unit" stance.
    """
    by_unit_id: dict[str, list[ArchimateProposal]] = {
        c.code_unit_id: list(c.ranked) for c in classifications
    }
    items: list[ArchitectureProposalItem] = []
    for component in components:
        unit_id = str(component.get("id", ""))
        ranked = by_unit_id.get(unit_id, [])
        items.append(
            ArchitectureProposalItem(
                code_unit=component_to_proposal_code_unit(component, container=container),
                ranked_candidates=[archimate_proposal_to_ranked_candidate(p) for p in ranked],
            )
        )
    return items
