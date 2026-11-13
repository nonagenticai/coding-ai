"""AI Code Insights API Endpoints.

API endpoints for AI-powered code analysis and intelligent insights generation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx
from coding_client.clients import CodingWorkspacesClient, ServiceError
from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from auth.tenant import require_tenant
from clients.outbound_identity import outbound_headers
from config.settings import get_settings
from models.ai_insights_api_models import (
    AICapabilitiesResponse,
    InsightTypesResponse,
)
from models.common_api_models import BaseResponse, ResponseStatus
from models.error_models import (
    Error400Response,
    Error404Response,
    Error500Response,
    ValidationErrorResponse,
)
from services.ai_insights.base import (
    CodebaseInsights,
    CodeInsight,
    InsightConfidence,
    InsightPriority,
    InsightType,
)

# Container validation and file discovery use coding-containers HTTP API.
from services.ai_insights.engine import (
    get_ai_insights_engine,
)
from utils.responses import (
    bad_request,
    internal_error,
    not_found,
)

logger = logging.getLogger(__name__)

# Where every workspace checkout is mounted; see coding-containers'
# kubernetes_container_provisioner (repo-data -> /workspace).
WORKSPACE_ROOT = "/workspace"

# Extensions the insights engine can actually parse. Kept as a set so the find
# expression below is generated from one list rather than restated.
SOURCE_EXTENSIONS = frozenset(
    {
        ".c",
        ".cpp",
        ".cs",
        ".go",
        ".java",
        ".js",
        ".jsx",
        ".py",
        ".rb",
        ".rs",
        ".ts",
        ".tsx",
    }
)

# Vendored trees and build output dwarf the repository's own code and would
# spend the whole file budget before reaching src/. Pruned in `find` rather than
# filtered afterwards so the cost is paid container-side.
EXCLUDED_PATH_GLOBS = (
    "*/.*",
    "*/node_modules/*",
    "*/__pycache__/*",
    "*/venv/*",
    "*/.venv/*",
    "*/dist/*",
    "*/build/*",
    "*/target/*",
    "*/vendor/*",
)

# Container-side budget for the discovery command, in seconds. Distinct from the
# client's HTTP timeout: this one bounds `find`, not the round trip.
FILE_DISCOVERY_TIMEOUT_SECONDS = 60

# A monorepo can hold six figures of source files. The engine reads every file
# it is handed, so an unbounded list turns one request into an unbounded one.
MAX_DISCOVERED_FILES = 2000

router = APIRouter()


class InsightsAnalysisRequest(BaseModel):
    """Request for AI insights analysis."""

    container_name: str = Field(
        ...,
        description="Container name to analyze (required for container scoping)",
    )
    file_paths: list[str] | None = Field(
        None,
        description="Specific file paths within container to analyze (optional)",
    )
    include_experimental: bool = Field(
        False, description="Include experimental/low-confidence insights"
    )
    focus_areas: list[str] | None = Field(None, description="Specific insight types to focus on")
    max_insights: int | None = Field(
        None,
        ge=1,
        le=100,
        description="Maximum number of insights to generate",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "container_name": "fastapi-backend-abc123",
                "file_paths": [
                    "src/main.py",
                    "lib/utils.java",
                    "core/engine.rs",
                ],
                "include_experimental": False,
                "focus_areas": ["security", "refactoring", "architecture"],
                "max_insights": 20,
            },
        }
    )


class CodeInsightModel(BaseModel):
    """Code insight response model."""

    id: str
    insight_type: str
    priority: str
    confidence: str

    title: str
    description: str
    rationale: str

    files: list[str]
    entities: list[str]
    lines: dict[str, list[tuple[int, int]]]

    recommendations: list[str]
    resolve_suggestions: list[str]
    estimated_effort: str

    evidence: dict[str, Any]
    metrics: dict[str, float]
    related_insights: list[str]

    generated_at: str
    tags: list[str]
    impact_score: float


class CodebaseInsightsModel(BaseModel):
    """Codebase insights response model."""

    total_files_analyzed: int
    total_entities_analyzed: int
    analysis_time_seconds: float

    insights_by_type: dict[str, list[CodeInsightModel]]
    insights_by_priority: dict[str, list[CodeInsightModel]]
    high_impact_insights: list[CodeInsightModel]

    overall_quality_score: float
    technical_debt_ratio: float
    maintainability_score: float
    security_risk_score: float

    language_distribution: dict[str, int]
    cross_language_issues: list[CodeInsightModel]

    quick_wins: list[CodeInsightModel]
    critical_issues: list[CodeInsightModel]
    refactoring_candidates: list[CodeInsightModel]

    common_patterns: list[dict[str, Any]]
    anti_patterns: list[dict[str, Any]]


class InsightsAnalysisResponse(BaseResponse):
    """Response from AI insights analysis."""

    status: ResponseStatus = ResponseStatus.SUCCESS
    result: CodebaseInsightsModel = Field(..., description="AI insights analysis result")


def _convert_insight_to_model(insight: CodeInsight) -> CodeInsightModel:
    """Convert CodeInsight to response model."""
    return CodeInsightModel(
        id=insight.id,
        insight_type=insight.insight_type.value,
        priority=insight.priority.value,
        confidence=insight.confidence.value,
        title=insight.title,
        description=insight.description,
        rationale=insight.rationale,
        files=insight.files,
        entities=insight.entities,
        lines=insight.lines,
        recommendations=insight.recommendations,
        resolve_suggestions=insight.mitigation_suggestions,
        estimated_effort=insight.estimated_effort,
        evidence=insight.evidence,
        metrics=insight.metrics,
        related_insights=insight.related_insights,
        generated_at=insight.generated_at.isoformat(),
        tags=insight.tags,
        impact_score=insight.impact_score,
    )


async def _validate_container(container_name: str, tenant_id: str) -> None:
    """Validate container exists via coding-containers service.

    Matches monolith behaviour: raises HTTPException with a plain string detail
    for unknown containers so parity tests see the same error shape.

    Note the failure mode this shares with :func:`_resolve_container_files`: an
    auth refusal from coding-containers is swallowed into "container not found"
    / an empty file list. That makes a dropped credential invisible in
    production, which is why ``outbound_headers`` is asserted directly in tests
    rather than inferred from behaviour here.

    Uses the typed ``CodingWorkspacesClient`` rather than a hand-rolled httpx
    call: the route path and its query parameter names are the client's problem,
    not this module's. (coding-realtime shipped a silent 200-with-null-content
    bug by sending ``file_path=`` to a route whose parameter is ``path``.)

    The client RAISES ``ServiceError`` on any >=400 instead of returning the
    response, so every status branch below is expressed as an exception check.
    The mapping is deliberately identical to the pre-migration httpx version:
    404 -> HTTPException 404, any other >=400 -> log and continue (return None),
    transport failure -> HTTPException 404.
    """
    containers_url = get_settings().coding_containers_url
    if not containers_url:
        # Preserved behaviour: with no URL the old code built the relative URL
        # "/api/workspaces/..." and httpx raised UnsupportedProtocol (an
        # httpx.HTTPError), landing in the 404 branch below. Guarded explicitly
        # because CodingWorkspacesClient raises RuntimeError for an empty
        # base_url, and RuntimeError would escape as a 500 from analyze_insights.
        logger.warning(
            "coding_containers_url not set; cannot validate '%s'",
            container_name,
        )
        raise HTTPException(
            status_code=404,
            detail=f"Container '{container_name}' not found",
        )

    # max_retries=1 keeps the old single-attempt semantics: the client would
    # otherwise retry ConnectError/ReadTimeout 3x behind a request-path 404.
    client = CodingWorkspacesClient(base_url=containers_url, timeout=5.0, max_retries=1)
    try:
        await client.get_workspace(container_name, headers=outbound_headers(tenant_id))
    except ServiceError as exc:
        if exc.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail=f"Container '{container_name}' not found",
            ) from exc
        if exc.status_code in (401, 403):
            # Surfaced, not swallowed. This used to fall into the log-and-continue
            # branch below, so a permission refusal was reported to the caller as
            # an empty analysis: a 200 saying "no insights" for a workspace we
            # were never allowed to read. "Nothing here" and "you may not look"
            # are different answers and only one of them is actionable.
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Not permitted to read workspace '{container_name}'. "
                    "The credential presented does not grant access to this tenant's "
                    "workspaces."
                ),
            ) from exc
        # Genuinely transient (5xx) — log and continue, as before.
        logger.warning(
            "Container validation returned %s for '%s'",
            exc.status_code,
            container_name,
        )
    except httpx.HTTPError as exc:
        logger.warning(
            "Could not reach coding-containers for validation of '%s': %s",
            container_name,
            exc,
        )
        # Service unreachable — raise 404 matching monolith format
        # (in practice containers are always validated; unreachable = treat as not found)
        raise HTTPException(
            status_code=404,
            detail=f"Container '{container_name}' not found",
        ) from exc
    finally:
        await client.close()


async def _resolve_container_files(
    container_name: str, tenant_id: str, file_paths: list[str] | None = None
) -> list[str]:
    """Resolve the source files to analyse, discovering them if not supplied.

    Returns ``file_paths`` unchanged when the caller named them. Otherwise walks
    the workspace checkout and returns absolute paths to its source files.

    Discovery used to call ``list_files(path="/workspace")`` and was broken three
    ways at once, each of which produced an empty list and a cheerful 200:

    1. It read ``data.get("files", data)``. The route keys a directory listing
       under ``entries`` (``ContainerFilesResponse`` in coding-containers) and
       has never emitted ``files``, so the fallback handed back the whole
       response dict, which is not a list, so the ``isinstance`` guard below
       returned ``[]``.
    2. Entries are objects — ``{permissions, size, modified, name, type}`` — not
       path strings, so ``f.endswith(ext)`` would have raised on a dict even
       once the key was right, landing in the blanket ``except Exception`` that
       also returned ``[]``.
    3. ``/files`` runs a single non-recursive ``ls``. A repository keeps its code
       under ``src/``, so even a fully working version of the above would have
       found nothing but top-level config.

    It now runs one bounded ``find`` through ``/exec`` — the same shape
    coding-security's scanner uses — which is recursive, returns real paths, and
    prunes vendored trees container-side. A permission refusal raises 403 instead
    of being flattened into "no files here"; genuinely transient failures still
    degrade to an empty list.
    """
    if file_paths:
        return file_paths

    containers_url = get_settings().coding_containers_url
    if not containers_url:
        # A missing URL is a deployment fault, not a caller fault, and it is not
        # transient — so it stays a logged empty list rather than a 403, which
        # would tell the caller something false about their permissions.
        logger.warning(
            "Could not discover files for '%s': coding_containers_url not set",
            container_name,
        )
        return []

    client = CodingWorkspacesClient(base_url=containers_url, timeout=15.0, max_retries=1)
    try:
        find_argv = ["find", WORKSPACE_ROOT, "-type", "f", "("]
        for index, extension in enumerate(sorted(SOURCE_EXTENSIONS)):
            if index:
                find_argv.append("-o")
            find_argv += ["-name", f"*{extension}"]
        find_argv += [")"]
        for pruned in EXCLUDED_PATH_GLOBS:
            find_argv += ["-not", "-path", pruned]

        payload = await client.exec(
            container_name,
            headers=outbound_headers(tenant_id),
            command=find_argv,
            timeout=FILE_DISCOVERY_TIMEOUT_SECONDS,
        )
    except ServiceError as exc:
        if exc.status_code in (401, 403):
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Not permitted to read workspace '{container_name}'. "
                    "The credential presented does not grant access to this tenant's "
                    "workspaces."
                ),
            ) from exc
        logger.warning(
            "Failed to discover files in '%s': HTTP %s",
            container_name,
            exc.status_code,
        )
        return []
    except httpx.HTTPError as exc:
        logger.warning(
            "Could not reach coding-containers to discover files in '%s': %s", container_name, exc
        )
        return []
    finally:
        await client.close()

    if not isinstance(payload, dict):
        logger.warning("Unexpected exec payload discovering files in '%s'", container_name)
        return []
    if payload.get("exit_code", 1) != 0:
        # find exits non-zero on an unreadable subtree while still printing what
        # it could reach, so this is logged and the partial list kept rather than
        # discarded — but it is never reported as a clean empty repository.
        logger.warning(
            "find over %s in '%s' exited %s: %s",
            WORKSPACE_ROOT,
            container_name,
            payload.get("exit_code"),
            (payload.get("stderr") or "").strip()[:500],
        )
    discovered = [
        line.strip() for line in (payload.get("stdout") or "").splitlines() if line.strip()
    ]
    if not discovered:
        logger.info("No source files found under %s in '%s'", WORKSPACE_ROOT, container_name)
    return discovered[:MAX_DISCOVERED_FILES]


def _validate_focus_areas(focus_areas: list[str] | None) -> list[Any] | None:
    """Validate and convert focus areas."""
    if not focus_areas:
        return None

    valid_focus_areas = [t.value for t in InsightType]
    invalid_areas = [area for area in focus_areas if area not in valid_focus_areas]
    if invalid_areas:
        raise bad_request(
            f"Invalid focus areas: {invalid_areas}. Valid options: {valid_focus_areas}",
            details={
                "invalid_areas": invalid_areas,
                "valid_options": valid_focus_areas,
            },
        )
    return [InsightType(area) for area in focus_areas]


def _validate_file_paths(file_paths: list[str]) -> list[str]:
    """Validate that file paths exist."""
    valid_files = []
    for file_path_str in file_paths:
        file_path_obj = Path(file_path_str)
        if file_path_obj.exists():
            valid_files.append(str(file_path_obj))
        else:
            logger.warning("File not found: %s", file_path_str)

    if not valid_files:
        raise not_found("Files", "analysis targets")

    return valid_files


def _convert_insights_to_model(
    insights: CodebaseInsights,
) -> CodebaseInsightsModel:
    """Convert CodebaseInsights to response model."""

    # Convert insight lists by type
    insights_by_type = {}
    for insight_type, insight_list in insights.insights_by_type.items():
        insights_by_type[insight_type.value] = [
            _convert_insight_to_model(insight) for insight in insight_list
        ]

    # Convert insight lists by priority
    insights_by_priority = {}
    for priority, insight_list in insights.insights_by_priority.items():
        insights_by_priority[priority.value] = [
            _convert_insight_to_model(insight) for insight in insight_list
        ]

    return CodebaseInsightsModel(
        total_files_analyzed=insights.total_files_analyzed,
        total_entities_analyzed=insights.total_entities_analyzed,
        analysis_time_seconds=insights.analysis_time_seconds,
        insights_by_type=insights_by_type,
        insights_by_priority=insights_by_priority,
        high_impact_insights=[
            _convert_insight_to_model(insight) for insight in insights.high_impact_insights
        ],
        overall_quality_score=insights.overall_quality_score,
        technical_debt_ratio=insights.technical_debt_ratio,
        maintainability_score=insights.maintainability_score,
        security_risk_score=insights.security_risk_score,
        language_distribution=insights.language_distribution,
        cross_language_issues=[
            _convert_insight_to_model(insight) for insight in insights.cross_language_issues
        ],
        quick_wins=[_convert_insight_to_model(insight) for insight in insights.quick_wins],
        critical_issues=[
            _convert_insight_to_model(insight) for insight in insights.critical_issues
        ],
        refactoring_candidates=[
            _convert_insight_to_model(insight) for insight in insights.refactoring_candidates
        ],
        common_patterns=insights.common_patterns,
        anti_patterns=insights.anti_patterns,
    )


@router.post(
    "/analyze",
    operation_id="analyze_code_insights",
    response_model=InsightsAnalysisResponse,
    summary="Generate AI Code Insights",
    description="""
    **AI-Powered Code Intelligence** - Generate comprehensive insights
    and recommendations for your codebase.

    Leverages advanced AI analysis to provide intelligent insights about code
    quality, security, architecture, and optimization opportunities across
    multiple programming languages.

    All AI insights analysis must be scoped to a container. Provide the container_name
    to analyze all files in the container, or specify specific file_paths within the container.
    """,
    responses={
        400: {"model": Error400Response, "description": "Bad request"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": Error500Response, "description": "Internal server error"},
    },
)
async def analyze_insights(
    request: InsightsAnalysisRequest = Body(..., description="AI insights analysis request"),
    tenant_id: str = Depends(require_tenant),
) -> InsightsAnalysisResponse:
    """Generate comprehensive AI insights for code analysis."""

    try:
        # Container validation via lightweight HTTP call to coding-containers.
        # Matches monolith behaviour: 404 with string detail for unknown containers.
        await _validate_container(request.container_name, tenant_id)

        # Resolve file paths from container
        file_paths = await _resolve_container_files(
            request.container_name, tenant_id, request.file_paths
        )

        # Validate focus areas
        focus_areas_enum = _validate_focus_areas(request.focus_areas)

        # Validate file paths exist
        valid_files = _validate_file_paths(file_paths)

        # Get AI insights engine
        engine = get_ai_insights_engine()

        # Generate insights
        result = await engine.generate_insights(
            file_paths=valid_files,
            include_experimental=request.include_experimental,
            focus_areas=focus_areas_enum,
            max_insights=request.max_insights,
        )

        # Convert to response model
        response_model = _convert_insights_to_model(result)

        total_insights = sum(len(insights) for insights in result.insights_by_type.values())

        logger.info("AI insights analysis completed: %d insights generated", total_insights)

        return InsightsAnalysisResponse(
            message=(
                f"Generated {total_insights} insights for {result.total_files_analyzed} files"
            ),
            result=response_model,
            request_id=str(hash(f"insights_{len(request.file_paths or [])}_{total_insights}")),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("AI insights analysis failed: %s", e)
        raise internal_error(f"AI insights analysis failed: {str(e)}") from e


@router.get(
    "/capabilities",
    operation_id="get_insights_capabilities",
    response_model=AICapabilitiesResponse,
    summary="AI Insights Engine Capabilities",
    description="""
    **AI Insights System Capabilities** - Discover available AI
    analysis types and features.

    Provides comprehensive information about the AI insights engine's capabilities,
    including supported analysis types, insight categories, and AI reasoning approaches.
    """,
    responses={
        400: {"model": Error400Response, "description": "Bad request"},
        404: {"model": Error404Response, "description": "Resource not found"},
        422: {
            "model": ValidationErrorResponse,
            "description": "Validation error",
        },
        500: {
            "model": Error500Response,
            "description": "Internal server error",
        },
    },
)
async def get_insights_capabilities(
    tenant_id: str = Depends(require_tenant),
) -> AICapabilitiesResponse:
    """Get AI insights engine capabilities and features.

    ``tenant_id`` is unused by the body -- this route returns module constants
    only. The dependency is the point: see ``get_architecture_capabilities``.
    """
    del tenant_id

    try:
        # Build capabilities structure
        insight_types = [t.value for t in InsightType]
        priority_levels = [p.value for p in InsightPriority]
        confidence_levels = [c.value for c in InsightConfidence]

        analysis_features = {
            "architecture_analysis": "High-level structural recommendations",
            "refactoring_intelligence": ("Code consolidation and cleanup opportunities"),
            "security_intelligence": ("Vulnerability analysis and security recommendations"),
            "performance_optimization": "Performance update suggestions",
            "maintainability_enhancement": "Code maintainability updates",
            "cross_language_intelligence": "Multi-language integration insights",
            "quality_assessment": "Comprehensive quality scoring",
            "pattern_recognition": "Design pattern analysis and recommendations",
        }

        ai_capabilities = {
            "multi_dimensional_analysis": (
                "Combines multiple analysis types for comprehensive insights"
            ),
            "impact_scoring": ("Calculates impact scores based on priority and confidence"),
            "evidence_gathering": ("Provides supporting evidence for each recommendation"),
            "relationship_analysis": "Identifies connections between insights",
            "trend_detection": ("Recognizes patterns across codebase"),
            "progressive_recommendations": ("Suggests implementation order for updates"),
        }

        quality_metrics = {
            "overall_quality_score": "Comprehensive quality assessment (0.0 - 1.0)",
            "technical_debt_ratio": "Proportion of code needing refactoring",
            "maintainability_score": "Code maintainability assessment",
            "security_risk_score": "Overall security risk evaluation",
        }

        integration_sources = {
            "similarity_analysis": "Code similarity and duplication detection",
            "security_analysis": "Vulnerability scanning and security assessment",
            "complexity_analysis": "Code complexity measurement",
            "relationship_analysis": "Entity and dependency relationships",
        }

        output_categories = {
            "quick_wins": "Low effort, high impact updates",
            "critical_issues": "Urgent problems requiring immediate attention",
            "refactoring_candidates": "Code consolidation opportunities",
            "high_impact": "Most valuable updates to prioritize",
            "cross_language_issues": "Multi-language integration opportunities",
        }

        effort_estimation = {
            "low": "1-2 hours of development time",
            "medium": "1-3 days of development time",
            "high": "1+ weeks of development time",
        }

        supported_languages = [
            "java",
            "rust",
            "go",
            "python",
            "javascript",
            "typescript",
            "csharp",
            "cpp",
            "c",
        ]

        ai_reasoning = {
            "evidence_based": ("All insights backed by concrete code analysis evidence"),
            "confidence_scoring": ("AI confidence levels for each recommendation"),
            "impact_calculation": ("Automated impact assessment based on multiple factors"),
            "context_aware": "Considers codebase context and patterns",
            "cross_domain": "Integrates insights from multiple analysis domains",
        }

        limitations = {
            "max_recommended_files": 1000,
            "max_insights_per_request": 100,
            "requires_similarity_analysis": True,
            "requires_security_analysis": True,
        }

        # Also include nested structure for backward compatibility
        capabilities_nested = {
            "insight_types": insight_types,
            "priority_levels": priority_levels,
            "confidence_levels": confidence_levels,
        }

        return AICapabilitiesResponse(
            message="AI insights capabilities retrieved successfully",
            insight_types=insight_types,
            priority_levels=priority_levels,
            confidence_levels=confidence_levels,
            analysis_features=analysis_features,
            ai_capabilities=ai_capabilities,
            quality_metrics=quality_metrics,
            integration_sources=integration_sources,
            output_categories=output_categories,
            effort_estimation=effort_estimation,
            supported_languages=supported_languages,
            ai_reasoning=ai_reasoning,
            limitations=limitations,
            capabilities=capabilities_nested,
        )

    except Exception as e:
        logger.error("Error retrieving AI insights capabilities: %s", e)
        raise internal_error(f"Error retrieving capabilities: {str(e)}") from e


@router.get(
    "/insight-types",
    operation_id="list_insight_types",
    response_model=InsightTypesResponse,
    summary="Available Insight Types",
    description="""
    **Insight Type Definitions** - Detailed information about each AI insight category.

    Provides detailed descriptions of each insight type, including:
    - Purpose and focus area
    - Example insights generated
    - Typical priority levels
    - Integration with other analysis types
    """,
    responses={
        400: {"model": Error400Response, "description": "Bad request"},
        404: {"model": Error404Response, "description": "Resource not found"},
        422: {
            "model": ValidationErrorResponse,
            "description": "Validation error",
        },
        500: {
            "model": Error500Response,
            "description": "Internal server error",
        },
    },
)
async def get_insight_types(
    tenant_id: str = Depends(require_tenant),
) -> InsightTypesResponse:
    """Get detailed information about available insight types.

    ``tenant_id`` is unused by the body -- this route returns module constants
    only. The dependency is the point: see ``get_architecture_capabilities``.
    """
    del tenant_id

    insight_type_info = {
        "architecture": {
            "name": "Architecture Analysis",
            "description": ("High-level structural recommendations for better system design"),
            "focus_areas": [
                "Language dominance detection",
                "Structural pattern recognition",
                "Modularization opportunities",
                "Dependency architecture analysis",
            ],
            "typical_priorities": ["medium", "low"],
            "example_insights": [
                "Language dominance suggests microservice opportunities",
                "Consistent structural patterns detected",
                "Cyclic dependency detected between modules",
            ],
        },
        "refactoring": {
            "name": "Refactoring Intelligence",
            "description": "Code consolidation and cleanup opportunities",
            "focus_areas": [
                "Duplication detection",
                "Similarity-based refactoring",
                "Extract method candidates",
                "DRY principle violations",
            ],
            "typical_priorities": ["high", "medium"],
            "example_insights": [
                "High similarity code suitable for refactoring",
                "Duplicate code blocks detected",
                "Long method extraction opportunities",
            ],
        },
        "security": {
            "name": "Security Intelligence",
            "description": "Vulnerability analysis and security update recommendations",
            "focus_areas": [
                "Critical vulnerability analysis",
                "Security hotspot detection",
                "Pattern-based security analysis",
                "Security trend analysis",
            ],
            "typical_priorities": ["critical", "high"],
            "example_insights": [
                "Critical security vulnerabilities detected",
                "Security hotspots requiring attention",
                "Common vulnerability patterns identified",
            ],
        },
        "performance": {
            "name": "Performance Optimization",
            "description": "Performance update suggestions",
            "focus_areas": [
                "Algorithm optimization via refactoring",
                "Memory usage optimization",
                "Cache locality updates",
                "Lazy loading opportunities",
            ],
            "typical_priorities": ["medium", "low"],
            "example_insights": [
                "Performance optimization via refactoring",
                "Memory usage optimization opportunities",
                "Algorithm efficiency updates",
            ],
        },
        "maintainability": {
            "name": "Maintainability Augmentation",
            "description": "Code maintainability and readability updates",
            "focus_areas": [
                "Code complexity analysis",
                "Language diversity assessment",
                "Technical debt measurement",
                "Coding standard compliance",
            ],
            "typical_priorities": ["high", "medium"],
            "example_insights": [
                "High code duplication affecting maintainability",
                "Language diversity impact on maintenance",
                "Technical debt accumulation detected",
            ],
        },
        "cross_language": {
            "name": "Cross-Language Intelligence",
            "description": "Multi-language integration patterns and opportunities",
            "focus_areas": [
                "API unification opportunities",
                "Code generation candidates",
                "Microservice boundaries",
                "Cross-language pattern consistency",
            ],
            "typical_priorities": ["medium", "low"],
            "example_insights": [
                "Cross-language similarity patterns detected",
                "API unification opportunities identified",
                "Consistent patterns across languages",
            ],
        },
        "quality": {
            "name": "Quality Assessment",
            "description": "Overall code quality updates",
            "focus_areas": [
                "Overall quality scoring",
                "Quality gate recommendations",
                "Code review priority",
                "Best practice adherence",
            ],
            "typical_priorities": ["high", "medium"],
            "example_insights": [
                "Overall code quality below threshold",
                "Quality gate recommendations",
                "Code review priority areas",
            ],
        },
        "patterns": {
            "name": "Design Pattern Recognition",
            "description": "Design pattern recommendations and consistency analysis",
            "focus_areas": [
                "Pattern consistency detection",
                "Anti-pattern identification",
                "Pattern library opportunities",
                "Architectural pattern compliance",
            ],
            "typical_priorities": ["low", "info"],
            "example_insights": [
                "Consistent design patterns detected",
                "Anti-pattern usage identified",
                "Pattern library opportunities",
            ],
        },
    }

    ai_powered_features = [
        "Evidence-based recommendations",
        "Confidence scoring",
        "Impact assessment",
        "Context-aware analysis",
        "Cross-domain integration",
    ]

    return InsightTypesResponse(
        message="Insight types retrieved successfully",
        insight_types=insight_type_info,
        total_types=8,
        ai_powered_features=ai_powered_features,
    )
