"""Pydantic v2 models for AI-powered test failure analysis.

Supports test failure input data, AI analysis results, root cause identification,
fix suggestions, pattern detection, and confidence scoring.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TestFramework(StrEnum):
    __test__ = False
    PYTEST = "pytest"
    UNITTEST = "unittest"
    JEST = "jest"
    MOCHA = "mocha"
    JUNIT = "junit"
    XUNIT = "xunit"
    RSPEC = "rspec"
    UNKNOWN = "unknown"


class FailureType(StrEnum):
    ASSERTION_ERROR = "assertion_error"
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT_ERROR = "timeout_error"
    MEMORY_ERROR = "memory_error"
    NETWORK_ERROR = "network_error"
    DEPENDENCY_ERROR = "dependency_error"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN_ERROR = "unknown_error"


class SeverityLevel(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ConfidenceLevel(StrEnum):
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class TestFileMetadata(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
        extra="forbid",
    )

    file_path: str = Field(..., min_length=1, max_length=1000)
    file_size: int | None = Field(default=None, ge=0)
    line_count: int | None = Field(default=None, ge=0)
    last_modified: datetime | None = Field(default=None)
    framework: TestFramework = Field(default=TestFramework.UNKNOWN)
    test_count: int | None = Field(default=None, ge=0)


class TestFailureContext(BaseModel):
    __test__ = False
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
        extra="forbid",
    )

    test_name: str = Field(..., min_length=1, max_length=200)
    test_class: str | None = Field(default=None, max_length=200)
    test_method: str | None = Field(default=None, max_length=200)
    test_file: TestFileMetadata | None = Field(default=None)
    source_file: str | None = Field(default=None, max_length=1000)
    test_line_number: int | None = Field(default=None, ge=1)
    framework: TestFramework = Field(default=TestFramework.UNKNOWN)
    execution_environment: dict[str, Any] | None = Field(default=None)


class TestFailureData(BaseModel):
    __test__ = False
    model_config = ConfigDict(
        str_strip_whitespace=False,
        validate_default=True,
        use_enum_values=True,
        extra="forbid",
    )

    failure_id: str = Field(..., min_length=1, max_length=100)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    raw_output: str = Field(..., max_length=100_000)
    exit_code: int | None = Field(default=None)
    execution_time: float | None = Field(default=None, ge=0)
    memory_usage: int | None = Field(default=None, ge=0)
    context: TestFailureContext = Field(...)

    @field_validator("failure_id")
    @classmethod
    def validate_failure_id(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "failure_id must contain only alphanumeric characters, hyphens, and underscores"
            )
        return v


class ErrorDetails(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
        extra="forbid",
    )

    error_type: FailureType = Field(...)
    error_message: str = Field(..., max_length=5000)
    traceback: str | None = Field(default=None, max_length=50_000)
    file_path: str | None = Field(default=None, max_length=1000)
    line_number: int | None = Field(default=None, ge=1)
    column_number: int | None = Field(default=None, ge=1)
    stack_trace: list[str] | None = Field(default=None)


class AssertionDetails(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
        extra="forbid",
    )

    assertion_type: Literal[
        "assertEqual",
        "assertNotEqual",
        "assertTrue",
        "assertFalse",
        "assertIs",
        "assertIsNot",
        "assertIsNone",
        "assertIsNotNone",
        "assertIn",
        "assertNotIn",
        "assertRaises",
        "assertRaisesRegex",
        "assertAlmostEqual",
        "assertNotAlmostEqual",
        "assertGreater",
        "assertGreaterEqual",
        "assertLess",
        "assertLessEqual",
        "assertRegex",
        "assertNotRegex",
        "assertCountEqual",
        "assertMultiLineEqual",
        "assertSequenceEqual",
        "assertListEqual",
        "assertTupleEqual",
        "assertSetEqual",
        "assertDictEqual",
        "custom",
        "unknown",
    ] = Field(default="unknown")
    expected_value: str | None = Field(default=None, max_length=10_000)
    actual_value: str | None = Field(default=None, max_length=10_000)
    comparison_operator: str | None = Field(default=None, max_length=10)
    assertion_message: str | None = Field(default=None, max_length=1000)
    value_difference: str | None = Field(default=None, max_length=10_000)


class RootCause(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
        extra="forbid",
    )

    cause_id: str = Field(..., min_length=1, max_length=100)
    category: Literal[
        "logic_error",
        "type_mismatch",
        "null_reference",
        "index_out_of_bounds",
        "configuration_issue",
        "dependency_missing",
        "environment_issue",
        "data_validation",
        "race_condition",
        "resource_exhaustion",
        "network_connectivity",
        "permission_denied",
        "timeout",
        "incompatible_versions",
        "missing_initialization",
        "state_corruption",
        "external_service_failure",
        "file_system_error",
        "unknown",
    ] = Field(...)
    description: str = Field(..., min_length=10, max_length=1000)
    severity: SeverityLevel = Field(...)
    confidence: ConfidenceLevel = Field(...)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    related_patterns: list[str] | None = Field(default=None)

    @field_validator("cause_id")
    @classmethod
    def validate_cause_id(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "cause_id must contain only alphanumeric characters, hyphens, and underscores"
            )
        return v

    @model_validator(mode="after")
    def validate_confidence_consistency(self) -> RootCause:
        score = self.confidence_score
        level = self.confidence
        if level == ConfidenceLevel.VERY_HIGH and not (0.9 <= score <= 1.0):
            raise ValueError("VERY_HIGH confidence must have score 0.9-1.0")
        elif level == ConfidenceLevel.HIGH and not (0.75 <= score < 0.9):
            raise ValueError("HIGH confidence must have score 0.75-0.89")
        elif level == ConfidenceLevel.MEDIUM and not (0.5 <= score < 0.75):
            raise ValueError("MEDIUM confidence must have score 0.5-0.74")
        elif level == ConfidenceLevel.LOW and not (0.25 <= score < 0.5):
            raise ValueError("LOW confidence must have score 0.25-0.49")
        elif level == ConfidenceLevel.VERY_LOW and not (0.0 <= score < 0.25):
            raise ValueError("VERY_LOW confidence must have score 0.0-0.24")
        return self


class FixSuggestion(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
        extra="forbid",
    )

    suggestion_id: str = Field(..., min_length=1, max_length=100)
    title: str = Field(..., min_length=5, max_length=200)
    description: str = Field(..., min_length=20, max_length=2000)
    fix_type: Literal[
        "code_change",
        "configuration_change",
        "dependency_update",
        "environment_setup",
        "test_modification",
        "data_correction",
        "infrastructure_change",
        "documentation_update",
        "process_change",
        "investigation_needed",
        "unknown",
    ] = Field(...)
    priority: Literal["critical", "high", "medium", "low"] = Field(...)
    confidence: ConfidenceLevel = Field(...)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    estimated_effort: Literal["trivial", "low", "medium", "high", "very_high"] = Field(...)
    code_examples: list[str] | None = Field(default=None)
    files_to_modify: list[str] | None = Field(default=None)
    prerequisites: list[str] | None = Field(default=None)
    potential_side_effects: list[str] | None = Field(default=None)
    verification_steps: list[str] | None = Field(default=None)

    @field_validator("suggestion_id")
    @classmethod
    def validate_suggestion_id(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "suggestion_id must contain only alphanumeric characters, hyphens, and underscores"
            )
        return v


class TestFailurePattern(BaseModel):
    __test__ = False
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
        extra="forbid",
    )

    pattern_id: str = Field(..., min_length=1, max_length=100)
    pattern_name: str = Field(..., min_length=5, max_length=200)
    pattern_signature: str = Field(..., min_length=10, max_length=500)
    pattern_type: Literal[
        "error_message",
        "stack_trace",
        "assertion_pattern",
        "execution_flow",
        "timing_pattern",
        "resource_usage",
        "configuration_pattern",
        "dependency_pattern",
        "unknown",
    ] = Field(...)
    frequency: int = Field(default=1, ge=1)
    confidence: ConfidenceLevel = Field(...)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    associated_failures: list[str] | None = Field(default=None)
    typical_causes: list[str] | None = Field(default=None)
    common_fixes: list[str] | None = Field(default=None)


class AIAnalysisResult(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
        extra="forbid",
    )

    analysis_id: str = Field(..., min_length=1, max_length=100)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    failure_data: TestFailureData = Field(...)
    error_details: ErrorDetails = Field(...)
    assertion_details: AssertionDetails | None = Field(default=None)
    root_causes: list[RootCause] = Field(default_factory=list)
    fix_suggestions: list[FixSuggestion] = Field(default_factory=list)
    detected_patterns: list[TestFailurePattern] = Field(default_factory=list)
    analysis_confidence: ConfidenceLevel = Field(...)
    analysis_confidence_score: float = Field(..., ge=0.0, le=1.0)
    processing_time: float | None = Field(default=None, ge=0)
    ai_model_version: str | None = Field(default=None, max_length=100)
    additional_metadata: dict[str, Any] | None = Field(default=None)

    @field_validator("analysis_id")
    @classmethod
    def validate_analysis_id(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "analysis_id must contain only alphanumeric characters, hyphens, and underscores"
            )
        return v

    @field_validator("root_causes")
    @classmethod
    def validate_root_causes_sorted(cls, v: list[RootCause]) -> list[RootCause]:
        if len(v) > 1:
            return sorted(v, key=lambda x: x.confidence_score, reverse=True)
        return v

    @field_validator("fix_suggestions")
    @classmethod
    def validate_fix_suggestions_sorted(cls, v: list[FixSuggestion]) -> list[FixSuggestion]:
        if len(v) > 1:
            return sorted(v, key=lambda x: x.confidence_score, reverse=True)
        return v


class FailureAnalysisRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
        extra="forbid",
    )

    request_id: str = Field(..., min_length=1, max_length=100)
    failure_data: TestFailureData = Field(...)
    analysis_options: dict[str, Any] | None = Field(default=None)
    include_patterns: bool = Field(default=True)
    include_suggestions: bool = Field(default=True)
    max_root_causes: int = Field(default=5, ge=1, le=20)
    max_suggestions: int = Field(default=10, ge=1, le=50)
    minimum_confidence: float = Field(default=0.1, ge=0.0, le=1.0)

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "request_id must contain only alphanumeric characters, hyphens, and underscores"
            )
        return v


class FailureAnalysisResponse(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
        extra="forbid",
    )

    request_id: str = Field(..., min_length=1, max_length=100)
    success: bool = Field(...)
    degraded: bool = Field(
        default=False,
        description=(
            "True when the analysis completed on a fallback/degraded path — the "
            "LLM plane was unreachable or one or more analysis agents failed, so "
            "results are basic/non-AI (typically empty). A successful analysis "
            "that genuinely found nothing keeps degraded=False."
        ),
    )
    analysis_result: AIAnalysisResult | None = Field(default=None)
    error_message: str | None = Field(default=None, max_length=1000)
    warnings: list[str] | None = Field(default=None)
    processing_time: float = Field(..., ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
