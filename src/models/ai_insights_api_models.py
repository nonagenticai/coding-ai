"""AI Insights API response models."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .common_api_models import BaseResponse, ResponseStatus


class AICapabilitiesResponse(BaseResponse):
    status: ResponseStatus = ResponseStatus.SUCCESS
    insight_types: list[str] = Field(default_factory=list)
    priority_levels: list[str] = Field(default_factory=list)
    confidence_levels: list[str] = Field(default_factory=list)
    analysis_features: dict[str, str] = Field(default_factory=dict)
    ai_capabilities: dict[str, str] = Field(default_factory=dict)
    quality_metrics: dict[str, str] = Field(default_factory=dict)
    integration_sources: dict[str, str] = Field(default_factory=dict)
    output_categories: dict[str, str] = Field(default_factory=dict)
    effort_estimation: dict[str, str] = Field(default_factory=dict)
    supported_languages: list[str] = Field(default_factory=list)
    ai_reasoning: dict[str, str] = Field(default_factory=dict)
    limitations: dict[str, Any] = Field(default_factory=dict)
    capabilities: dict[str, Any] = Field(default_factory=dict)


class InsightTypesResponse(BaseResponse):
    status: ResponseStatus = ResponseStatus.SUCCESS
    insight_types: dict[str, dict[str, Any]] = Field(default_factory=dict)
    total_types: int = Field(...)
    ai_powered_features: list[str] = Field(default_factory=list)
