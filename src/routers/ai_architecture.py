"""Architecture-governance producer endpoints.

Upstream half of the code↔architecture bridge:
``POST /architecture/tools/classify-code-units`` returns ranked ArchiMate
proposals so the downstream ``requirements_management`` conformance loop
(`POST /architecture/tools/conformance`) has something to gate.

Stateless. Storage of accepted mappings lives in requirements_management.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Body, Depends, Query

from auth.tenant import require_tenant
from clients.coding_analysis_client import (
    CodingAnalysisUnavailableError,
    fetch_components,
)
from models.architecture_models import (
    TYPICAL_CODE_BINDING_TYPES,
    ArchimateElementType,
    ArchimateLayer,
    ArchitectureCapabilitiesResponse,
    ArchitectureProposalsResponse,
    ClassifyCodeUnitsRequest,
    ClassifyCodeUnitsResponse,
)
from models.error_models import (
    Error400Response,
    Error500Response,
    ValidationErrorResponse,
)
from services.architecture_classifier import (
    ARCHITECTURE_CLASSIFIER_MODEL,
    get_architecture_classifier,
)
from services.architecture_proposals import (
    assemble_proposal_items,
    component_to_code_unit,
)
from utils.responses import internal_error

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/tools/classify-code-units",
    operation_id="classify_code_units",
    response_model=ClassifyCodeUnitsResponse,
    summary="Propose ArchiMate-type bindings for code units",
    description=(
        "Takes a batch of code units (passed inline) and returns ranked ArchiMate-type "
        "proposals per unit. Proposals are constrained to the ArchiMate 3.2 closed set "
        "used by requirements_management's conformance metamodel. "
        "All proposals are human-gated downstream; confidence is informational, not a threshold."
    ),
    responses={
        400: {"model": Error400Response, "description": "Bad request"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": Error500Response, "description": "Internal server error"},
    },
)
async def classify_code_units(
    request: ClassifyCodeUnitsRequest = Body(..., description="Inline classification request"),
    tenant_id: str = Depends(require_tenant),
) -> ClassifyCodeUnitsResponse:
    try:
        classifier = get_architecture_classifier()
        classifications = await classifier.classify(
            units=request.code_units,
            top_n=request.top_n,
        )
        logger.info(
            "architecture classification ok: tenant=%s units=%d top_n=%d",
            tenant_id,
            len(request.code_units),
            request.top_n,
        )
        return ClassifyCodeUnitsResponse(
            message=f"Classified {len(classifications)} code units",
            classifications=classifications,
            model=ARCHITECTURE_CLASSIFIER_MODEL,
        )
    except Exception as e:
        logger.exception("architecture classification failed: %s", e)
        raise internal_error(f"architecture classification failed: {e}") from e


@router.get(
    "/proposals",
    operation_id="list_architecture_proposals",
    response_model=ArchitectureProposalsResponse,
    summary="Ranked ArchiMate-type proposals for one container",
    description=(
        "Pulls the per-container component list from coding-analysis "
        "(`GET /api/analysis/architecture/components/{container}`) and classifies "
        "each one through `cai-architecture-classifier`. Returns one item per "
        "detected component (`code_unit` + `ranked_candidates`). The downstream "
        "join (requirements_management P4.3) adds `proposed_mapping`. "
        "Stateless; recomputed per request. When coding-analysis is unset or "
        "unreachable the endpoint still returns 200 with `upstream_available=false` "
        "and an empty `proposals[]` — matches the contract's "
        "'errors degrade per item' stance, applied here to the bulk fetch."
    ),
    responses={
        500: {"model": Error500Response, "description": "Internal server error"},
    },
)
async def list_architecture_proposals(
    container: str = Query(..., min_length=1, description="Container name"),
    limit: int = Query(
        25,
        ge=1,
        le=200,
        description=(
            "Max components to classify in this page. The classifier runs ONE LLM "
            "call per unit, so an unbounded container (100s of units) read-timed-out "
            "downstream; this bounds the page. Page with `offset`."
        ),
    ),
    offset: int = Query(0, ge=0, description="Component offset for the page within `total`."),
    tenant_id: str = Depends(require_tenant),
) -> ArchitectureProposalsResponse:
    try:
        components = await fetch_components(container, tenant_id=tenant_id)
    except CodingAnalysisUnavailableError as exc:
        logger.warning(
            "coding-analysis unavailable for proposals: tenant=%s container=%s err=%s",
            tenant_id,
            container,
            exc,
        )
        return ArchitectureProposalsResponse(
            message=f"coding-analysis unavailable: {exc}",
            container=container,
            proposals=[],
            upstream_available=False,
        )

    total = len(components)
    if not components:
        logger.info(
            "architecture proposals: tenant=%s container=%s components=0", tenant_id, container
        )
        return ArchitectureProposalsResponse(
            message="No components for this container",
            container=container,
            proposals=[],
            total=0,
            limit=limit,
            offset=offset,
        )

    # Bound the classifier work to one page — see `limit` above.
    page = components[offset : offset + limit]
    units = [component_to_code_unit(c, container=container) for c in page]
    try:
        classifier = get_architecture_classifier()
        classifications = await classifier.classify(units=units, top_n=3)
    except Exception as exc:
        # Whole-batch classifier failure → degrade to "no proposals" but
        # keep the code_unit shells so the downstream UI can still render
        # the container's structure and a "(classification unavailable)"
        # state per unit. Do NOT 5xx — that takes the dashboard down.
        logger.exception(
            "architecture classifier failed for proposals: tenant=%s container=%s err=%s",
            tenant_id,
            container,
            exc,
        )
        classifications = []

    items = assemble_proposal_items(page, classifications, container=container)
    logger.info(
        "architecture proposals ok: tenant=%s container=%s total=%d page=%d offset=%d items=%d",
        tenant_id,
        container,
        total,
        len(page),
        offset,
        len(items),
    )
    return ArchitectureProposalsResponse(
        message=f"Returned {len(items)} of {total} proposal items (offset {offset})",
        container=container,
        proposals=items,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/capabilities",
    operation_id="get_architecture_capabilities",
    response_model=ArchitectureCapabilitiesResponse,
    summary="Architecture classifier capabilities (closed set + limits)",
    description=(
        "Self-describing advert for downstream UIs: the ArchiMate element-type "
        "closed set this service can emit, the layer set, the code-binding "
        "subset the prompt biases toward, and per-request limits."
    ),
)
async def get_architecture_capabilities(
    tenant_id: str = Depends(require_tenant),
) -> ArchitectureCapabilitiesResponse:
    # tenant_id is unused by the body -- this route returns module constants
    # only. The dependency is the point: it made the route match the
    # `security: [{BearerAuth: []}]` that `configure_openapi` has always
    # published for the whole /api/ai prefix. Measured live before this change,
    # with NO headers, it answered 200.
    del tenant_id
    return ArchitectureCapabilitiesResponse(
        message="Architecture classifier capabilities",
        supported_element_types=[t.value for t in ArchimateElementType],
        supported_layers=[layer.value for layer in ArchimateLayer],
        typical_code_binding_types=[t.value for t in TYPICAL_CODE_BINDING_TYPES],
        max_units_per_request=200,
        max_top_n=5,
    )
