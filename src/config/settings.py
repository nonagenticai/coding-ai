"""Runtime settings for coding-ai."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Annotated, ClassVar

from coding_client.config import BaseMicroserviceSettings
from pydantic import Field, field_validator
from pydantic_settings import NoDecode


def _split_origin_string(raw: str) -> list[object]:
    """Split a CORS_ORIGINS env value into individual origins.

    Accepts the fleet's comma-separated form ("https://a,https://b") and, for
    compatibility with the ConfigMaps that use it, a JSON array.
    """
    text = raw.strip()
    if text.startswith("["):
        try:
            decoded = json.loads(text)
        except ValueError:
            return list(text.split(","))
        if isinstance(decoded, list):
            return list(decoded)
        return [decoded]
    return list(text.split(","))


class Settings(BaseMicroserviceSettings):
    """coding-ai configuration loaded from environment variables.

    Service URLs default to empty string; production deployments must inject
    them via the gitops ConfigMap. ``model_post_init`` enforces this when
    ``environment == "production"``. See
    ``gitops-production/docs/config-externalization-policy.md`` and the
    canonical pattern in ``inference-router/src/settings.py``.
    """

    model_config = {"env_prefix": "", "case_sensitive": False, "extra": "ignore"}

    service_name: str = Field(default="coding-ai")
    version: str = Field(default="1.0.0")
    environment: str = Field(default="production")
    enable_docs: bool = Field(default=False)
    enable_auth: bool = Field(default=True)
    # Tenant granted to a credential-less caller admitted by the TRUSTED_CIDRS
    # bypass. NO literal default — this used to be "n8n-dev", the stale
    # pre-uuid7 label of a real tenant, so an unset value silently wrote into a
    # silo no live JWT can address. See src/auth/_default_tenant.py. The
    # authoritative read is auth._default_tenant.resolve_default_tenant(); this
    # field only mirrors the env var for logging/introspection.
    default_tenant_id: str = Field(default="")

    # Browser origins permitted to make cross-origin calls to this service.
    # Comma-separated env var, matching the fleet convention used by
    # tts-backend / chat-bridge / book-fulfillment ConfigMaps:
    #     CORS_ORIGINS="https://one.example,https://two.example"
    #
    # The wildcard default keeps in-cluster and local callers working out of
    # the box. It is SAFE only because ``main.resolve_cors_config`` turns
    # ``allow_credentials`` OFF whenever "*" is in the list -- a wildcard origin
    # combined with credentials makes Starlette reflect the caller's Origin
    # header back with ``Access-Control-Allow-Credentials: true``, which lets
    # any origin make credentialed cross-origin requests. Set an explicit
    # origin list here (or via the ConfigMap) to enable credentialed CORS.
    #
    # ``NoDecode`` suppresses pydantic-settings' JSON decoding for this complex
    # field so the plain comma-separated form above is accepted; the validator
    # below still accepts a real list (or a JSON array) unchanged.
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["*"])

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: object) -> list[str]:
        """Accept a comma-separated string, a JSON array, or a real list."""
        items: list[object]
        if value is None:
            items = []
        elif isinstance(value, str):
            items = _split_origin_string(value)
        elif isinstance(value, list | tuple | set):
            items = list(value)
        else:
            items = [value]

        origins = [str(item).strip() for item in items]
        return [origin for origin in origins if origin] or ["*"]

    redis_url: str = Field(default="")
    coding_containers_url: str = Field(default="")
    # coding-analysis (the as-built code-graph) — pulled by the P4.1
    # ``GET /api/ai/architecture/proposals`` endpoint to fetch the
    # per-container ``Component`` list before classifying. Empty default
    # so non-prod can run without it; the endpoint degrades to an empty
    # ``proposals[]`` (200) when unset or unreachable rather than 5xx.
    coding_analysis_url: str = Field(default="")

    # worker_management proxy. Once the make_agent_runner sites are migrated
    # this is the only LLM destination coding-ai needs to know about; the
    # provider (inference_router today, anthropic / vllm / etc tomorrow) lives
    # on each WM worker record, swappable via PUT /workers/{name} with no
    # redeploy. See /Users/mini1/.claude/plans/karma-service-help-me-think-fancy-boot.md
    worker_management_url: str = Field(default="")
    # LEGACY OVERRIDE ONLY — do not add new readers, and do not read this
    # field to identify the service. It maps to CODING_AI_LICENSE_TOKEN, which
    # is deployed in ZERO manifests fleet-wide, so it resolves to "" in every
    # pod; wm_client's `elif self._service_token:` then emitted no X-Tenant-Id
    # at all and WM's trusted-network bypass served the call unattributed.
    # The service identity coding-ai actually carries is CODING_SERVICE_TOKEN
    # (subject 019ff970-e327 = svc-coding-ai). Resolve it through
    # clients.outbound_identity.service_identity_token(), which prefers this
    # var when an operator sets it and reads the env per call so a rotation
    # needs no restart. Kept as a field only because scripts/register_workers.py
    # documents CODING_AI_LICENSE_TOKEN as its --license-token default.
    coding_ai_license_token: str = Field(default="")
    # URL stored as ``base_url_override`` on each seeded worker record. Lets
    # coding-ai bootstrap workers pointing at inference-router today; ops
    # can PUT a different value per worker afterwards.
    inference_router_url: str = Field(default="")

    # License introspection endpoint (licenses service). Required in
    # production; coding-ai calls it from the tenant middleware to validate
    # inbound Bearer tokens.
    license_introspect_url: str = Field(default="")

    # Tracing config is deliberately NOT mirrored here. coding_client.telemetry
    # reads OTLP_TRACES_ENDPOINT and OTEL_SDK_DISABLED from the environment
    # directly, so a field here would have zero readers — and this fleet has
    # already paid twice for zero-reader settings fields that looked authoritative.
    # Note the old default was otel_sdk_disabled=True, i.e. tracing off unless the
    # manifest said otherwise; the manifest does set OTEL_SDK_DISABLED=false on
    # all eight coding workloads, verified in the live pods.

    _SERVICE_NAME: ClassVar[str] = "coding-ai"
    _REQUIRED_IN_PRODUCTION: ClassVar[tuple[str, ...]] = (
        "redis_url",
        "coding_containers_url",
        "inference_router_url",
        "license_introspect_url",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached coding-ai settings instance."""
    return Settings()
