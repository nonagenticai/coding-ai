"""Architecture-governance models for coding-ai.

Schemas for the ``/architecture/tools/classify-code-units`` endpoint, which
proposes ArchiMate-type bindings for code units so requirements_management's
``POST /architecture/tools/conformance`` has something to gate.

The ``ArchimateElementType`` and ``ArchimateLayer`` enums are byte-identical
to the closed sets in ``requirements_management/src/tools/architecture/metamodel.py``.
If those change upstream, this enum must change in lockstep — the bridge is
silently broken otherwise.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from .common_api_models import BaseResponse, ResponseStatus


class ArchimateLayer(StrEnum):
    MOTIVATION = "motivation"
    STRATEGY = "strategy"
    BUSINESS = "business"
    APPLICATION = "application"
    TECHNOLOGY = "technology"
    PHYSICAL = "physical"
    IMPLEMENTATION_MIGRATION = "implementation_migration"


class ArchimateElementType(StrEnum):
    # business
    BUSINESS_ACTOR = "business_actor"
    BUSINESS_ROLE = "business_role"
    BUSINESS_COLLABORATION = "business_collaboration"
    BUSINESS_INTERFACE = "business_interface"
    BUSINESS_PROCESS = "business_process"
    BUSINESS_FUNCTION = "business_function"
    BUSINESS_INTERACTION = "business_interaction"
    BUSINESS_EVENT = "business_event"
    BUSINESS_SERVICE = "business_service"
    BUSINESS_OBJECT = "business_object"
    CONTRACT = "contract"
    REPRESENTATION = "representation"
    PRODUCT = "product"
    # application
    APPLICATION_COMPONENT = "application_component"
    APPLICATION_COLLABORATION = "application_collaboration"
    APPLICATION_INTERFACE = "application_interface"
    APPLICATION_FUNCTION = "application_function"
    APPLICATION_INTERACTION = "application_interaction"
    APPLICATION_PROCESS = "application_process"
    APPLICATION_EVENT = "application_event"
    APPLICATION_SERVICE = "application_service"
    DATA_OBJECT = "data_object"
    # technology
    NODE = "node"
    DEVICE = "device"
    SYSTEM_SOFTWARE = "system_software"
    TECHNOLOGY_COLLABORATION = "technology_collaboration"
    TECHNOLOGY_INTERFACE = "technology_interface"
    TECHNOLOGY_FUNCTION = "technology_function"
    TECHNOLOGY_PROCESS = "technology_process"
    TECHNOLOGY_INTERACTION = "technology_interaction"
    TECHNOLOGY_EVENT = "technology_event"
    TECHNOLOGY_SERVICE = "technology_service"
    ARTIFACT = "artifact"
    COMMUNICATION_NETWORK = "communication_network"
    PATH = "path"
    # physical
    EQUIPMENT = "equipment"
    FACILITY = "facility"
    DISTRIBUTION_NETWORK = "distribution_network"
    MATERIAL = "material"
    # motivation
    STAKEHOLDER = "stakeholder"
    DRIVER = "driver"
    ASSESSMENT = "assessment"
    GOAL = "goal"
    OUTCOME = "outcome"
    PRINCIPLE = "principle"
    CONSTRAINT = "constraint"
    MEANING = "meaning"
    VALUE = "value"
    # strategy
    RESOURCE = "resource"
    CAPABILITY = "capability"
    VALUE_STREAM = "value_stream"
    COURSE_OF_ACTION = "course_of_action"
    # implementation & migration
    WORK_PACKAGE = "work_package"
    DELIVERABLE = "deliverable"
    IMPLEMENTATION_EVENT = "implementation_event"
    PLATEAU = "plateau"
    GAP = "gap"


# element_type → layer, byte-identical to _LAYER_ELEMENT_MAP in metamodel.py.
ELEMENT_LAYER: dict[ArchimateElementType, ArchimateLayer] = {
    # business
    ArchimateElementType.BUSINESS_ACTOR: ArchimateLayer.BUSINESS,
    ArchimateElementType.BUSINESS_ROLE: ArchimateLayer.BUSINESS,
    ArchimateElementType.BUSINESS_COLLABORATION: ArchimateLayer.BUSINESS,
    ArchimateElementType.BUSINESS_INTERFACE: ArchimateLayer.BUSINESS,
    ArchimateElementType.BUSINESS_PROCESS: ArchimateLayer.BUSINESS,
    ArchimateElementType.BUSINESS_FUNCTION: ArchimateLayer.BUSINESS,
    ArchimateElementType.BUSINESS_INTERACTION: ArchimateLayer.BUSINESS,
    ArchimateElementType.BUSINESS_EVENT: ArchimateLayer.BUSINESS,
    ArchimateElementType.BUSINESS_SERVICE: ArchimateLayer.BUSINESS,
    ArchimateElementType.BUSINESS_OBJECT: ArchimateLayer.BUSINESS,
    ArchimateElementType.CONTRACT: ArchimateLayer.BUSINESS,
    ArchimateElementType.REPRESENTATION: ArchimateLayer.BUSINESS,
    ArchimateElementType.PRODUCT: ArchimateLayer.BUSINESS,
    # application
    ArchimateElementType.APPLICATION_COMPONENT: ArchimateLayer.APPLICATION,
    ArchimateElementType.APPLICATION_COLLABORATION: ArchimateLayer.APPLICATION,
    ArchimateElementType.APPLICATION_INTERFACE: ArchimateLayer.APPLICATION,
    ArchimateElementType.APPLICATION_FUNCTION: ArchimateLayer.APPLICATION,
    ArchimateElementType.APPLICATION_INTERACTION: ArchimateLayer.APPLICATION,
    ArchimateElementType.APPLICATION_PROCESS: ArchimateLayer.APPLICATION,
    ArchimateElementType.APPLICATION_EVENT: ArchimateLayer.APPLICATION,
    ArchimateElementType.APPLICATION_SERVICE: ArchimateLayer.APPLICATION,
    ArchimateElementType.DATA_OBJECT: ArchimateLayer.APPLICATION,
    # technology
    ArchimateElementType.NODE: ArchimateLayer.TECHNOLOGY,
    ArchimateElementType.DEVICE: ArchimateLayer.TECHNOLOGY,
    ArchimateElementType.SYSTEM_SOFTWARE: ArchimateLayer.TECHNOLOGY,
    ArchimateElementType.TECHNOLOGY_COLLABORATION: ArchimateLayer.TECHNOLOGY,
    ArchimateElementType.TECHNOLOGY_INTERFACE: ArchimateLayer.TECHNOLOGY,
    ArchimateElementType.TECHNOLOGY_FUNCTION: ArchimateLayer.TECHNOLOGY,
    ArchimateElementType.TECHNOLOGY_PROCESS: ArchimateLayer.TECHNOLOGY,
    ArchimateElementType.TECHNOLOGY_INTERACTION: ArchimateLayer.TECHNOLOGY,
    ArchimateElementType.TECHNOLOGY_EVENT: ArchimateLayer.TECHNOLOGY,
    ArchimateElementType.TECHNOLOGY_SERVICE: ArchimateLayer.TECHNOLOGY,
    ArchimateElementType.ARTIFACT: ArchimateLayer.TECHNOLOGY,
    ArchimateElementType.COMMUNICATION_NETWORK: ArchimateLayer.TECHNOLOGY,
    ArchimateElementType.PATH: ArchimateLayer.TECHNOLOGY,
    # physical
    ArchimateElementType.EQUIPMENT: ArchimateLayer.PHYSICAL,
    ArchimateElementType.FACILITY: ArchimateLayer.PHYSICAL,
    ArchimateElementType.DISTRIBUTION_NETWORK: ArchimateLayer.PHYSICAL,
    ArchimateElementType.MATERIAL: ArchimateLayer.PHYSICAL,
    # motivation
    ArchimateElementType.STAKEHOLDER: ArchimateLayer.MOTIVATION,
    ArchimateElementType.DRIVER: ArchimateLayer.MOTIVATION,
    ArchimateElementType.ASSESSMENT: ArchimateLayer.MOTIVATION,
    ArchimateElementType.GOAL: ArchimateLayer.MOTIVATION,
    ArchimateElementType.OUTCOME: ArchimateLayer.MOTIVATION,
    ArchimateElementType.PRINCIPLE: ArchimateLayer.MOTIVATION,
    ArchimateElementType.CONSTRAINT: ArchimateLayer.MOTIVATION,
    ArchimateElementType.MEANING: ArchimateLayer.MOTIVATION,
    ArchimateElementType.VALUE: ArchimateLayer.MOTIVATION,
    # strategy
    ArchimateElementType.RESOURCE: ArchimateLayer.STRATEGY,
    ArchimateElementType.CAPABILITY: ArchimateLayer.STRATEGY,
    ArchimateElementType.VALUE_STREAM: ArchimateLayer.STRATEGY,
    ArchimateElementType.COURSE_OF_ACTION: ArchimateLayer.STRATEGY,
    # implementation & migration
    ArchimateElementType.WORK_PACKAGE: ArchimateLayer.IMPLEMENTATION_MIGRATION,
    ArchimateElementType.DELIVERABLE: ArchimateLayer.IMPLEMENTATION_MIGRATION,
    ArchimateElementType.IMPLEMENTATION_EVENT: ArchimateLayer.IMPLEMENTATION_MIGRATION,
    ArchimateElementType.PLATEAU: ArchimateLayer.IMPLEMENTATION_MIGRATION,
    ArchimateElementType.GAP: ArchimateLayer.IMPLEMENTATION_MIGRATION,
}


# Code-relevant subset the prompt biases toward. The classifier may still
# return any value in the closed set; this is documentation + the value the
# capabilities endpoint advertises as "typical".
TYPICAL_CODE_BINDING_TYPES: list[ArchimateElementType] = [
    ArchimateElementType.ARTIFACT,
    ArchimateElementType.APPLICATION_COMPONENT,
    ArchimateElementType.APPLICATION_SERVICE,
    ArchimateElementType.APPLICATION_INTERFACE,
    ArchimateElementType.APPLICATION_FUNCTION,
    ArchimateElementType.DATA_OBJECT,
]


class CodeUnit(BaseModel):
    """A code-graph unit the caller wants classified.

    Mirrors the shape requirements_management's conformance pull expects
    (``container`` + stable ``id`` + structural metadata). The classifier
    reads ``path``/``kind``/``imports``/``public_symbols``/``doc_excerpt``
    as evidence; ``container`` and ``id`` are echoed back so the caller can
    correlate proposals with the code unit they describe.
    """

    id: str = Field(..., description="Stable code-unit id (from the code graph)")
    container: str = Field("", description="Container / workspace the unit belongs to")
    path: str = Field("", description="Source path or qualified name")
    kind: str = Field(
        "",
        description=(
            "What the unit is (e.g. module, class, package, file). "
            "Free-form: the LLM uses it as a hint, never as a hard constraint."
        ),
    )
    imports: list[str] = Field(default_factory=list, description="Imported modules / dependencies")
    public_symbols: list[str] = Field(
        default_factory=list,
        description="Exported names (functions/classes) — bias toward the public API surface",
    )
    doc_excerpt: str = Field(
        "",
        description="Short docstring or comment excerpt (kept brief to control prompt size)",
    )


class ArchimateProposal(BaseModel):
    """One ranked proposal for binding a code unit to an ArchiMate element.

    The closed ``archimate_type`` enum matches the upstream metamodel
    1:1. ``archimate_layer`` is derived from the type but emitted for
    convenience so the downstream UI doesn't need a second lookup.
    """

    model_config = ConfigDict(use_enum_values=True)

    archimate_type: ArchimateElementType = Field(
        ...,
        description="Proposed ArchiMate element type (must be in the closed set)",
    )
    archimate_layer: ArchimateLayer = Field(
        ...,
        description="ArchiMate layer derived from archimate_type",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence 0..1")
    rationale: str = Field(
        ..., description="Short justification grounded in the code-unit evidence"
    )


class CodeUnitClassification(BaseModel):
    """Top-N ranked proposals for one code unit."""

    code_unit_id: str = Field(..., description="Echoed back from the input")
    container: str = Field("", description="Echoed back from the input")
    ranked: list[ArchimateProposal] = Field(
        default_factory=list,
        description="Proposals sorted by descending confidence; empty if classification failed",
    )


class ClassifyCodeUnitsRequest(BaseModel):
    """Inline classification request.

    v1 accepts code units inline only — the caller (requirements_management)
    already fetches them from coding-analysis for its conformance pull, so a
    second pull from here would be wasted. A pull-by-id path can be added
    later if a use case appears that doesn't already have the units in hand.
    """

    code_units: list[CodeUnit] = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Code units to classify (1..200). Cap protects against runaway fan-out.",
    )
    top_n: int = Field(
        3,
        ge=1,
        le=5,
        description="Max proposals per code unit (capped to keep prompts focused).",
    )


class ClassifyCodeUnitsResponse(BaseResponse):
    """Per-unit ranked proposals + classification metadata."""

    status: ResponseStatus = ResponseStatus.SUCCESS
    classifications: list[CodeUnitClassification] = Field(default_factory=list)
    model: str = Field("", description="Logical model / worker that produced the proposals")
    generated_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO 8601 UTC timestamp",
    )


class ArchitectureCapabilitiesResponse(BaseResponse):
    """Self-describing capability advert for the downstream UI."""

    status: ResponseStatus = ResponseStatus.SUCCESS
    supported_element_types: list[str] = Field(default_factory=list)
    supported_layers: list[str] = Field(default_factory=list)
    typical_code_binding_types: list[str] = Field(default_factory=list)
    max_units_per_request: int = Field(default=200)
    max_top_n: int = Field(default=5)


# ── P4.1: GET /architecture/proposals?container=<name> ─────────────────────
# Wire shape pinned in
# ``requirements_management/docs/ARCHITECTURE_PROPOSALS_CONTRACT.md``. coding-ai
# returns ``code_unit`` + ``ranked_candidates`` per detected unit; the
# downstream join (P4.3 in requirements_management) adds ``proposed_mapping``
# and the ``already_bound`` flag.


class ProposalCodeUnit(BaseModel):
    """The ``code_unit`` block of one proposal item.

    Same fields the conformance tool already consumes for ``code_units``:
    ``(container, component_id)`` is the primary binding key — never key
    on ``path``. ``node_id`` is null today (coding-analysis's Component
    schema doesn't expose a graph-node id).
    """

    container: str = Field(..., description="Container the unit belongs to (== request param)")
    component_id: str = Field(..., description="coding-analysis Component id — PRIMARY binding key")
    name: str = Field("", description="Human-readable label")
    layer: str = Field("", description="coding-analysis layer (not the ArchiMate layer)")
    kind: str = Field("", description="Free-form unit kind (component | module | service | ...)")
    path: str = Field("", description="Source path for one representative file — descriptive only")
    node_id: str | None = Field(default=None, description="Optional coding-analysis graph node id")


class RankedCandidate(BaseModel):
    """One element of ``ranked_candidates`` — slim ``{element_type, score}``.

    A flattened view of ``ArchimateProposal`` for cross-service transport:
    ``element_type`` is the closed-set string value, ``score`` is the
    confidence in 0..1. Sorted desc by score by the producer.
    """

    element_type: str = Field(..., description="ArchiMate element_type from the closed set")
    score: float = Field(..., ge=0.0, le=1.0, description="Classifier confidence 0..1")


class ArchitectureProposalItem(BaseModel):
    """One proposal item — ``code_unit`` + its ``ranked_candidates``."""

    code_unit: ProposalCodeUnit
    ranked_candidates: list[RankedCandidate] = Field(
        default_factory=list,
        description="Sorted desc by score. Empty when classification failed for this unit.",
    )


class ArchitectureProposalsResponse(BaseResponse):
    """Response for ``GET /architecture/proposals?container=<name>``.

    Stateless. Recomputed per request; downstream WM endpoint (P4.3) caches
    on its side keyed by ``(tenant_id, project_id, container, content_hash)``.
    """

    status: ResponseStatus = ResponseStatus.SUCCESS
    container: str = Field(..., description="Container name from the request")
    proposals: list[ArchitectureProposalItem] = Field(default_factory=list)
    generated_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO 8601 UTC timestamp",
    )
    upstream_available: bool = Field(
        default=True,
        description=(
            "False when coding-analysis was unreachable or not configured. "
            "Caller can surface this as 'no data available yet' instead of "
            "'no proposals'."
        ),
    )
    total: int = Field(
        default=0,
        description=(
            "Total detected components for the container (before limit/offset). "
            "Lets the consumer page the review queue without re-counting."
        ),
    )
    limit: int = Field(
        default=0,
        description="Max components classified in THIS page (0 = no cap applied).",
    )
    offset: int = Field(
        default=0,
        description="Component offset of this page within `total`.",
    )
