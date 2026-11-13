"""
Shared models and dependencies for intelligent testing modules.
This module prevents circular imports by centralizing common dependencies.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from coding_client.clients import CodingWorkspacesClient, ServiceError
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from clients.outbound_identity import outbound_headers
from config.settings import get_settings

# TODO: get_docker_service - dependency not yet extracted
# from core.dependencies import get_docker_service
from models.common_responses_models import LegacyBaseResponse
from models.testing_models import TestSecurityLevel

# TODO: TestConfig - dependency not yet extracted
# from python_test_suite.src.config import TestConfig
# TODO: DockerService - dependency not yet extracted
# from services.docker_service import DockerService
# TODO: IntelligentTestExecutionService, TestExecutionSettings, create_intelligent_test_service - not yet extracted
# from services.test_execution.intelligent_test_integration import (
#     IntelligentTestExecutionService,
#     TestExecutionSettings,
#     create_intelligent_test_service,
# )
# TODO: IntelligentTestExecutionResult - not yet extracted
# from services.test_execution.intelligent_test_orchestrator import (
#     IntelligentTestExecutionResult,
# )
# TODO: ExecutionPattern, TestRecommendation - not yet extracted
# from services.test_execution.pydantic_ai_agents import (
#     ExecutionPattern,
#     TestRecommendation,
# )
from .utilities_and_helpers import TestRequestValidator

logger = logging.getLogger(__name__)


async def validate_and_get_container_for_tenant(
    container_name: str, tenant_id: str | None = None
) -> SimpleNamespace:
    """Validate that ``container_name`` belongs to ``tenant_id`` and return its info.

    The single, fail-closed implementation shared by every intelligent-testing
    router (replacing the prior no-op stubs that returned ``codebase_path="."``
    without any tenant check). It asks coding-containers, which enforces tenant
    ownership: a container belonging to a different tenant comes back 404, which
    we surface as 404 — never a silent default path.

    - ``tenant_id`` is forwarded as the ``X-Tenant-Id`` header (the caller's
      already-validated tenant); coding-containers filters ``get_container`` by it.
      coding-ai's own service license goes alongside it in ``Authorization`` so
      the assertion survives ``TRUST_TENANT_HEADER=false`` — a bare tenant header
      proves nothing, and this check is the tenant-isolation boundary for every
      intelligent-testing router. See ``clients.outbound_identity``.
    - 404 -> not found (or not this tenant's); 401/403 -> not authorized; both
      raise instead of falling back.
    - When the container service is not configured (no URL — local/dev), we
      degrade with a clear WARNING rather than silently pretending to validate.
    - When it is configured but unreachable, we fail closed (503) so a service
      outage cannot be used to bypass the tenant check.

    The call goes through the typed ``CodingWorkspacesClient``. That client
    raises ``ServiceError`` for every >=400 rather than returning a response, so
    the status mapping below MUST catch ``ServiceError`` ahead of the blanket
    ``except Exception``: letting a 404 fall into the unreachable handler would
    turn "not this tenant's container" into a 503 and lose the fail-closed
    distinction this helper exists to make.
    """
    containers_url = get_settings().coding_containers_url
    if not containers_url:
        logger.warning(
            "coding_containers_url not set; skipping container validation for '%s' "
            "(degraded mode — tenant isolation NOT enforced).",
            container_name,
        )
        return SimpleNamespace(codebase_path=".")

    # max_retries=1 preserves the previous single-attempt behaviour; the client
    # would otherwise retry transport errors 3x before this 503.
    client = CodingWorkspacesClient(base_url=containers_url, timeout=5.0, max_retries=1)
    try:
        data = await client.get_workspace(container_name, headers=outbound_headers(tenant_id))
    except ServiceError as exc:
        # Status mapping is byte-for-byte the pre-migration one.
        if exc.status_code == 404:
            raise HTTPException(
                status_code=404, detail=f"Container '{container_name}' not found"
            ) from exc
        if exc.status_code in (401, 403):
            raise HTTPException(
                status_code=403, detail="Not authorized for this container"
            ) from exc
        raise HTTPException(status_code=502, detail="Container validation failed") from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Container service unreachable while validating '{container_name}'",
        ) from exc
    finally:
        await client.close()

    # ContainerDetailResponse nests the container under "container"; codebase_path
    # lives there (NOT at the top level — the old `data.get("codebase_path")` was
    # always a miss, so even the non-stub copy silently returned ".").
    container = (data or {}).get("container") or {}
    codebase_path = container.get("codebase_path") or "/app/data/repo"
    return SimpleNamespace(codebase_path=codebase_path)


# Request/Response Models (shared across modules)
class IntelligentTestRequest(BaseModel):
    """Request model for intelligent test execution.

    Attributes:
        code (str): Class attribute code.
        language (str): Class attribute language.
        test_code (Optional[str]): Class attribute test_code.
        timeout (int): Class attribute timeout.
        security_level (TestSecurityLevel): Class attribute security_level.
        project_name (Optional[str]): Class attribute project_name.
        enable_ai_optimization (bool): Class attribute enable_ai_optimization.
        enable_pattern_analysis (bool): Class attribute enable_pattern_analysis.
        enable_failure_analysis (bool): Class attribute enable_failure_analysis.
        enable_scheduling_optimization (bool): Class attribute enable_scheduling_optimization.
        priority_level (int): Class attribute priority_level.
        expected_execution_time_ms (Optional[float]): Class attribute expected_execution_time_ms.
        resource_weight (float): Class attribute resource_weight.
        test_type (Optional[str]): Class attribute test_type.
        dependencies (Dict[str, str]): Class attribute dependencies.
        resource_limits (Optional[Dict[str, Any]]): Class attribute resource_limits.

    Example:
        >>> obj = IntelligentTestRequest()"""

    code: str = Field(..., description="Source code to test")
    language: str = Field(
        ...,
        description="Programming language",
        json_schema_extra={
            "enum": [
                "python",
                "javascript",
                "typescript",
                "java",
                "go",
                "rust",
                "csharp",
                "cpp",
                "nodejs",
            ]
        },
    )
    container_name: str = Field(
        ..., description="Container name for container-scoped test execution"
    )
    test_code: str | None = Field(None, description="Optional test code")
    timeout: int = Field(300, ge=30, le=3600, description="Execution timeout in seconds")
    security_level: TestSecurityLevel = Field(
        TestSecurityLevel.MEDIUM, description="Security level"
    )
    project_name: str | None = Field(None, description="Optional project name")

    # AI-specific settings
    enable_ai_optimization: bool = Field(True, description="Enable AI optimization")
    enable_pattern_analysis: bool = Field(True, description="Enable pattern analysis")
    enable_failure_analysis: bool = Field(True, description="Enable failure analysis")
    enable_scheduling_optimization: bool = Field(True, description="Enable scheduling optimization")
    priority_level: int = Field(5, ge=1, le=10, description="Priority level (1=lowest, 10=highest)")
    expected_execution_time_ms: float | None = Field(
        None, ge=0, description="Expected execution time in milliseconds"
    )
    resource_weight: float = Field(1.0, ge=0.1, le=10.0, description="Resource weight factor")

    # Test configuration
    test_type: str | None = Field(
        "unit", description="Type of test (unit, integration, benchmark, etc.)"
    )
    dependencies: dict[str, str] = Field(default_factory=dict, description="Test dependencies")
    resource_limits: dict[str, Any] | None = Field(None, description="Resource limits")

    @field_validator("language")
    @classmethod
    def validate_language(cls, v):
        """Validate supported languages using the modular validator."""
        return TestRequestValidator.validate_language(v)


class BatchTestRequest(BaseModel):
    """Request model for batch test execution."""

    tests: list[IntelligentTestRequest] = Field(..., description="List of test requests")
    enable_retry: bool = Field(True, description="Enable retry logic")
    max_concurrent: int | None = Field(None, ge=1, le=20, description="Maximum concurrent tests")


class TestExecutionResponse(LegacyBaseResponse):
    """Response model for test execution."""

    result: Any | None = Field(None, description="Test execution result")
    execution_id: str = Field(..., description="Unique execution identifier")

    model_config = ConfigDict()


class BatchTestExecutionResponse(LegacyBaseResponse):
    """Response model for batch test execution."""

    results: list[Any] = Field(..., description="Test execution results")
    execution_id: str = Field(..., description="Unique batch execution identifier")
    summary: dict[str, Any] = Field(..., description="Execution summary")

    model_config = ConfigDict()


class ExecutionInsightsResponse(LegacyBaseResponse):
    """Response model for execution insights."""

    insights: dict[str, Any] = Field(..., description="Comprehensive execution insights")
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of insights generation",
    )

    model_config = ConfigDict()

    @field_serializer("generated_at")
    def serialize_datetime(self, v: datetime | None) -> str | None:
        if v is None:
            return None
        return v.isoformat()


class RecommendationsResponse(LegacyBaseResponse):
    """Response model for recommendations."""

    recommendations: list[Any] = Field(..., description="AI-generated recommendations")
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of recommendations generation",
    )

    model_config = ConfigDict()

    @field_serializer("generated_at")
    def serialize_datetime(self, v: datetime | None) -> str | None:
        if v is None:
            return None
        return v.isoformat()


class PatternsResponse(LegacyBaseResponse):
    """Response model for patterns."""

    patterns: list[Any] = Field(..., description="Identified execution patterns")
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of patterns generation",
    )

    model_config = ConfigDict()

    @field_serializer("generated_at")
    def serialize_datetime(self, v: datetime | None) -> str | None:
        if v is None:
            return None
        return v.isoformat()


class HealthStatusResponse(LegacyBaseResponse):
    """Response model for health status."""

    health_status: dict[str, Any] = Field(..., description="Comprehensive health status")
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of health check",
    )

    model_config = ConfigDict()

    @field_serializer("generated_at")
    def serialize_datetime(self, v: datetime | None) -> str | None:
        if v is None:
            return None
        return v.isoformat()


class ServiceSettingsResponse(LegacyBaseResponse):
    """Response model for service settings."""

    settings: Any = Field(..., description="Current service settings")
    applied_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of settings application",
    )

    model_config = ConfigDict()

    @field_serializer("applied_at")
    def serialize_datetime(self, v: datetime | None) -> str | None:
        if v is None:
            return None
        return v.isoformat()
