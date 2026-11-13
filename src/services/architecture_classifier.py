"""ArchiMate-type proposal orchestrator for code units.

Pure orchestrator: builds a per-unit prompt, calls the
``cai-architecture-classifier`` agent through worker_management via the
``make_agent_runner`` shim, and returns ranked proposals constrained to the
ArchiMate closed set.

No DB, no cache writes. Stateless. Storage of accepted mappings is the
downstream consumer's job (requirements_management ``architecture_relationships``).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from models.architecture_models import (
    ELEMENT_LAYER,
    ArchimateElementType,
    ArchimateLayer,
    ArchimateProposal,
    CodeUnit,
    CodeUnitClassification,
)
from services.llm_agents import make_agent_runner

logger = logging.getLogger(__name__)

ARCHITECTURE_CLASSIFIER_WORKER = "cai-architecture-classifier"
ARCHITECTURE_CLASSIFIER_MODEL = "architecture_classification"

# Max concurrent per-unit LLM classifications. Bounds latency for a page of
# units while staying gentle on the provider's rate limits.
_CLASSIFY_CONCURRENCY = 6


class ArchitectureClassifier:
    """Stateless orchestrator over the ArchiMate-classifier WM worker."""

    def __init__(self) -> None:
        self._agent = make_agent_runner(
            worker_name=ARCHITECTURE_CLASSIFIER_WORKER,
            output_type=list[ArchimateProposal],
        )

    async def classify(self, units: list[CodeUnit], top_n: int = 3) -> list[CodeUnitClassification]:
        # One LLM call per unit. Run them concurrently under a small semaphore so
        # a page of units returns in ~ceil(n/CONCURRENCY) waves instead of n
        # sequential round-trips (which read-timed-out downstream). The cap keeps
        # us gentle on the LLM provider; per-unit failures already degrade to an
        # empty ranked list inside `_classify_one`, so gather never raises here.
        sem = asyncio.Semaphore(_CLASSIFY_CONCURRENCY)

        async def _one(unit: CodeUnit) -> CodeUnitClassification:
            async with sem:
                ranked = await self._classify_one(unit, top_n)
            return CodeUnitClassification(
                code_unit_id=unit.id,
                container=unit.container,
                ranked=ranked,
            )

        # gather preserves input order, so the response stays stable for paging.
        return list(await asyncio.gather(*(_one(u) for u in units)))

    async def _classify_one(self, unit: CodeUnit, top_n: int) -> list[ArchimateProposal]:
        prompt = _build_prompt(unit, top_n)
        try:
            run_result = await self._agent.run(prompt)
        except Exception:
            logger.exception(
                "architecture classifier failed for code unit %s (container=%s)",
                unit.id,
                unit.container,
            )
            return []

        raw = run_result.output
        if not isinstance(raw, list):
            logger.warning(
                "architecture classifier returned non-list output for code unit %s: %r",
                unit.id,
                type(raw).__name__,
            )
            return []

        cleaned = _normalize_proposals(raw, top_n)
        return cleaned


def _build_prompt(unit: CodeUnit, top_n: int) -> str:
    parts: list[str] = [
        f"Classify this code unit. Return up to top_n={top_n} ranked proposals "
        "sorted by descending confidence. Use ONLY archimate_type values from the closed set "
        "stated in your system prompt; archimate_layer MUST match.",
        "",
        "Code unit:",
        f"  id: {unit.id}",
    ]
    if unit.container:
        parts.append(f"  container: {unit.container}")
    if unit.path:
        parts.append(f"  path: {unit.path}")
    if unit.kind:
        parts.append(f"  kind: {unit.kind}")
    if unit.imports:
        # Cap evidence so a noisy unit can't blow the prompt budget.
        imps = ", ".join(unit.imports[:30])
        parts.append(f"  imports: {imps}")
    if unit.public_symbols:
        syms = ", ".join(unit.public_symbols[:30])
        parts.append(f"  public_symbols: {syms}")
    if unit.doc_excerpt:
        parts.append(f"  doc_excerpt: {unit.doc_excerpt[:500]}")
    return "\n".join(parts)


def _normalize_proposals(raw: list[Any], top_n: int) -> list[ArchimateProposal]:
    """Defense-in-depth: drop anything outside the closed set, fix layer drift, sort, cap."""
    cleaned: list[ArchimateProposal] = []
    for item in raw:
        proposal = _coerce_proposal(item)
        if proposal is None:
            continue
        cleaned.append(proposal)

    cleaned.sort(key=lambda p: p.confidence, reverse=True)
    return cleaned[:top_n]


def _coerce_proposal(item: Any) -> ArchimateProposal | None:
    """Accept either an ArchimateProposal instance or a dict-like payload.

    The WM JSON-schema enforcement already validates output_type, but defense
    in depth: if a future WM build returns a dict (or a list shape changes),
    coerce and validate explicitly. Drops items with an out-of-closed-set
    archimate_type rather than raising — one bad proposal does not poison
    the whole batch.
    """
    if isinstance(item, ArchimateProposal):
        return _maybe_fix_layer(item)

    if not isinstance(item, dict):
        return None

    raw_type = item.get("archimate_type")
    raw_layer = item.get("archimate_layer")
    raw_conf = item.get("confidence")
    raw_rationale = item.get("rationale", "")

    try:
        elem_type = ArchimateElementType(raw_type)
    except (ValueError, TypeError):
        logger.warning("classifier emitted out-of-closed-set archimate_type=%r; dropping", raw_type)
        return None

    expected_layer = ELEMENT_LAYER[elem_type]
    try:
        layer = ArchimateLayer(raw_layer) if raw_layer else expected_layer
    except (ValueError, TypeError):
        layer = expected_layer
    if layer != expected_layer:
        logger.warning(
            "classifier emitted layer=%s but %s belongs to %s; correcting",
            layer.value,
            elem_type.value,
            expected_layer.value,
        )
        layer = expected_layer

    try:
        confidence = float(raw_conf) if raw_conf is not None else 0.0
    except (ValueError, TypeError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    try:
        return ArchimateProposal(
            archimate_type=elem_type,
            archimate_layer=layer,
            confidence=confidence,
            rationale=str(raw_rationale or ""),
        )
    except Exception:
        logger.exception("failed to construct ArchimateProposal from %r", item)
        return None


def _maybe_fix_layer(p: ArchimateProposal) -> ArchimateProposal:
    """If the model emitted the wrong layer for the chosen type, correct it."""
    elem_type = ArchimateElementType(p.archimate_type)
    expected_layer = ELEMENT_LAYER[elem_type]
    if ArchimateLayer(p.archimate_layer) == expected_layer:
        return p
    return ArchimateProposal(
        archimate_type=elem_type,
        archimate_layer=expected_layer,
        confidence=p.confidence,
        rationale=p.rationale,
    )


_classifier_singleton: ArchitectureClassifier | None = None


def get_architecture_classifier() -> ArchitectureClassifier:
    global _classifier_singleton
    if _classifier_singleton is None:
        _classifier_singleton = ArchitectureClassifier()
    return _classifier_singleton
