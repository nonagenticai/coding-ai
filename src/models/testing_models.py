"""Stub models for testing types that belong to coding-testing.

These are minimal stubs to satisfy imports in AI services.
The real TestRequest/TestResult models live in coding-testing.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TestType(StrEnum):
    """Test execution types."""

    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    PERFORMANCE = "performance"
    SECURITY = "security"
    BENCHMARK = "benchmark"


class TestSecurityLevel(StrEnum):
    """Security levels for test execution."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TestRequest(BaseModel):
    __test__ = False
    """Stub for TestRequest - real model lives in coding-testing."""

    model_config = ConfigDict(extra="allow")

    container_name: str = ""
    test_commands: list[str] = Field(default_factory=list)
    code: str = ""
    test_code: str = ""
    language: str = "python"
    security_level: str | None = None
    timeout: int | None = None
    dependencies: dict[str, str] = Field(default_factory=dict)


class TestResult(BaseModel):
    __test__ = False
    """Stub for TestResult - real model lives in coding-testing."""

    model_config = ConfigDict(extra="allow")

    test_id: str = Field(default="")
    language: str = Field(default="python")
    test_type: str | None = Field(default=None)
    success: bool = Field(default=True)
    execution_time: float = Field(default=0.0)
    stderr: str = Field(default="")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    test_metadata: dict[str, Any] = Field(default_factory=dict)


class QualityIssue(BaseModel):
    """Individual quality issue found by analysis tools."""

    model_config = ConfigDict(extra="allow")

    message: str = Field(default="")
    severity: str | None = Field(default=None)
    category: str | None = Field(default=None)


class QualityResult(BaseModel):
    """Result model for quality analysis."""

    model_config = ConfigDict(extra="allow")

    overall_score: float = Field(default=0.0)
    issues: list[QualityIssue] = Field(default_factory=list)
