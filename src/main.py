"""FastAPI application entry point for coding-ai."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from coding_client.api.exception_handlers import register_exception_handlers
from coding_client.api.logging_setup import configure_logging
from coding_client.api.openapi_config import configure_openapi
from coding_client.api.request_logging import HealthCheckFilter, RequestLoggingMiddleware
from coding_client.telemetry import setup_tracing
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from auth.tenant import TenantMiddleware
from clients.outbound_identity import service_identity_token
from config.settings import get_settings
from routers.ai_architecture import router as ai_architecture_router
from routers.ai_insights import router as ai_insights_router
from routers.ai_test_analysis import router as ai_test_analysis_router
from routers.ai_test_insights import router as ai_test_insights_router
from wm_bootstrap import ensure_workers_registered
from wm_client import DELEGATED_LICENCE_ENV, WMBYONotConfiguredError, delegated_licence_enabled

configure_logging()

logger = logging.getLogger(__name__)

settings = get_settings()

SERVICE_NAME = settings.service_name
VERSION = settings.version
ENVIRONMENT = settings.environment
ENABLE_DOCS = settings.enable_docs
# No module-level DEFAULT_TENANT_ID here. It had no readers, and re-exporting
# it would hand an importer a captured snapshot (now "" rather than the old
# "n8n-dev" literal) instead of the authoritative call-time read. The single
# source of truth is auth._default_tenant.resolve_default_tenant().
# No module-level ENABLE_AUTH either. It had zero readers, and the variable is
# absent from coding-ai's manifests, so nothing could switch it. Leaving it next
# to install_auth_middleware() below would have invited someone to connect the
# two — creating an env var that turns this service's only authentication off,
# the same shape as the SKIP_AUTH hatch deleted fleet-wide in coding-client
# 1.16.0. Removed rather than left as a loaded gun with the safety on.

REDIS_URL = settings.redis_url
_redis: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis | None:
    """Return the Redis client instance, or None if not connected."""
    return _redis


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan events."""
    global _redis
    logger.info(f"Starting {SERVICE_NAME} v{VERSION} ({ENVIRONMENT})")

    # --- Redis init ---
    try:
        _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        await _redis.ping()  # type: ignore[invalid-await]
        logger.info("Redis connected at %s", REDIS_URL)
    except Exception:
        logger.warning(
            "Redis unavailable at %s – continuing without cache", REDIS_URL, exc_info=True
        )
        _redis = None

    # Validate WM_SEND_DELEGATED_LICENCE at boot: an unrecognised value raises
    # here (crashloop naming the key) instead of on the first user LLM call.
    _delegated = delegated_licence_enabled()
    logger.info(
        "%s=%s (bare-tenant WM calls are %s)",
        DELEGATED_LICENCE_ENV,
        "on" if _delegated else "off",
        "delegated: service licence + X-Effective-Tenant-Id"
        if _delegated
        else "tokenless: X-Tenant-Id only",
    )

    # --- worker_management self-heal ---
    # If WM's database was flushed or coding-ai is starting in a fresh
    # environment, missing workers are recreated from src/config/agents.py.
    # Existing records are left alone (operator edits preserved). Failures
    # log but don't block boot — the per-request path will surface a 422/500
    # if a worker is genuinely missing when traffic arrives.
    if settings.worker_management_url:
        # Fail the boot, loudly, rather than run header-less. Every WM call with
        # no end-user behind it presents this token as X-Tenant-Id; with it
        # unset, WM's trusted-network bypass serves the call anyway and meters it
        # against nobody. A crashloop naming the missing key is recoverable in
        # one commit; months of silently unattributed LLM spend is not.
        if not service_identity_token():
            raise RuntimeError(
                "CODING_SERVICE_TOKEN is unset but WORKER_MANAGEMENT_URL is set: "
                "coding-ai cannot identify itself to worker_management, so every "
                "background and startup LLM call would go out with no X-Tenant-Id "
                "and be metered against nobody. Set CODING_SERVICE_TOKEN in the "
                "coding-ai-env Secret (it is already there in gitops-production) "
                "or unset WORKER_MANAGEMENT_URL to disable the LLM plane."
            )
        try:
            import httpx as _httpx

            async with _httpx.AsyncClient() as _http:
                _actions = await ensure_workers_registered(
                    _http,
                    wm_base_url=settings.worker_management_url,
                    license_token=service_identity_token() or "",
                    base_url_override=settings.inference_router_url,
                    overwrite=False,
                )
            _created = sum(1 for a in _actions.values() if a == "created")
            _existed = sum(1 for a in _actions.values() if a == "skipped-exists")
            logger.info(
                "worker_management seed: %d created, %d already existed (total %d)",
                _created,
                _existed,
                len(_actions),
            )
        except Exception:
            logger.warning("worker_management seed at startup failed", exc_info=True)
    else:
        logger.info("skipping worker_management seed: WORKER_MANAGEMENT_URL not set")

    # Tracing is NOT set up here. It used to be, and it had three independent
    # reasons to export nothing:
    #   1. it imported the .grpc exporter while this image ships .http, so every
    #      startup raised ModuleNotFoundError into the `except` below and logged
    #      "continuing without tracing" -- verified live in the running pod;
    #   2. it never called trace.set_tracer_provider(), so even the import fix
    #      would have left FastAPIInstrumentor resolving a non-recording proxy;
    #   3. instrument_app() inside a lifespan is a silent no-op -- Starlette has
    #      already built the middleware stack -- so it would still have exported
    #      zero spans while logging that tracing was enabled.
    # setup_tracing() is called at module scope below and raises if anyone moves
    # it back in here.

    yield

    # --- Shutdown ---
    if _redis:
        await _redis.aclose()
        logger.info("Redis connection closed")
    logger.info(f"Shutting down {SERVICE_NAME}")


app = FastAPI(
    title=SERVICE_NAME,
    description="""AI-powered code analysis and test intelligence service for the code_graph_helper platform.

**Capabilities:**
- Static code insight generation (architecture, security, performance, maintainability, cross-language)
- Test failure root-cause analysis with fix suggestions (PydanticAI agents)
- Test suite health analysis with failure prediction and pattern detection

**Auth:** Tenant isolation via `X-Tenant-Id` header (an unverified claim — set
`TRUST_TENANT_HEADER=false` to ignore it and require a real credential),
license token introspection (`Bearer` token → licenses service), or the
trusted-network bypass (`TRUSTED_CIDRS`, empty by default, and it grants
`DEFAULT_TENANT_ID` only when that is configured — there is no built-in default
tenant). There is no environment variable that bypasses authentication.

| Endpoint Group | Prefix | Routes |
|---|---|---|
| Code Insights | `/api/ai/insights/code` | 3 (analyze, capabilities, insight-types) |
| Test Failure Analysis | `/api/ai/analysis/tests` | 5 (failure-analysis, batch-analyze, health, metrics, patterns) |
| Test Suite Insights | `/api/ai/insights/tests` | 8 (analyze, comprehensive-analysis, predict-failures, recommendations, analysis/{id}, settings) |
| Infrastructure | `/` | 2 (root, health) |
""",
    version=VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


def resolve_cors_config(app_settings: object) -> tuple[list[str], bool]:
    """Return ``(allow_origins, allow_credentials)`` for the CORS middleware.

    Enforces the CORS invariant in code rather than by convention: a wildcard
    origin is NEVER combined with credentials. Starlette implements that
    combination by reflecting the caller's ``Origin`` header back alongside
    ``Access-Control-Allow-Credentials: true``, which would let any origin make
    credentialed cross-origin requests to this service.

    Origins come from ``settings.cors_origins`` (env ``CORS_ORIGINS``,
    comma-separated). Set an explicit origin list to enable credentialed CORS;
    leave it at the ``*`` default and credentials stay off.

    Mirrors the in-fleet pattern in ``coding-gateway/src/app_factory.py``.
    """
    origins = getattr(app_settings, "cors_origins", None) or ["*"]
    if isinstance(origins, str):
        origins = origins.split(",")
    origins = [str(origin).strip() for origin in origins]
    origins = [origin for origin in origins if origin]
    if not origins:
        origins = ["*"]
    allow_credentials = "*" not in origins
    return origins, allow_credentials


_cors_origins, _cors_allow_credentials = resolve_cors_config(settings)
logger.info(
    "CORS configured: origins=%s allow_credentials=%s",
    _cors_origins,
    _cors_allow_credentials,
)

# allow_methods / allow_headers stay permissive on purpose. Starlette expands
# ``["*"]`` methods to a fixed verb list and reflects only the *requested*
# headers, so neither leaks anything beyond the origins allowed above -- and
# with credentials off under the wildcard default there is no credentialed
# access to widen in the first place.
app.add_middleware(
    CORSMiddleware,  # type: ignore[invalid-argument-type]
    allow_origins=_cors_origins,
    allow_credentials=_cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())
app.add_middleware(RequestLoggingMiddleware)  # type: ignore[invalid-argument-type]


def install_auth_middleware(target_app: FastAPI) -> None:
    """Attach the auth + tenant middleware, or refuse to start.

    coding-ai was the only service in the fleet with **no** ``AuthMiddleware``.
    It had ``TenantMiddleware`` alone, which writes ``request.state.tenant_id``
    and never rejects, so ``Depends(require_tenant)`` was the *only* gate —
    and three handlers did not declare it. Measured live against the running
    pod, with no headers at all::

        /api/ai/architecture/capabilities     -> 200
        /api/ai/insights/code/insight-types   -> 200
        /api/ai/insights/code/capabilities    -> 200
        /api/ai/capabilities                  -> 200   (alias mount)
        /api/ai/insight-types                 -> 200   (alias mount)
        /api/ai/analysis/tests/patterns       -> 401   (control)

    Those three handlers return only module constants, so the *disclosure* was
    nil. The defect is structural: every other service has the blanket
    middleware, and here "forgot a ``Depends``" means "no auth". The published
    OpenAPI meanwhile marked the whole ``/api/ai`` prefix ``BearerAuth`` — a
    pure string-prefix stamp (``configure_openapi``) that never inspects
    ``route.dependencies`` — so the contract was advertising a gate that did
    not exist.

    Fails CLOSED, following ``coding-analysis``'s ``install_auth_middleware``.
    The alternative — log a warning and serve every route unauthenticated — is
    the loudest possible security failure reduced to a line nobody reads, and
    it is not hypothetical: ``auth.tenant`` re-exports names from
    coding_client, so a coding-client version that moves one would take the
    ``except ImportError`` branch.

    Ordering matters. ``add_middleware`` prepends, so the LAST added runs
    FIRST: ``AuthMiddleware`` must go on after ``TenantMiddleware`` so it runs
    outermost and has set ``request.state.user`` by the time this service's
    forked ``TenantMiddleware`` reads it (``auth/tenant.py:215``).

    ``PUBLIC_PATHS`` is exact-match, so this was checked rather than assumed:
    coding-ai's served ``openapi.json`` has 21 paths, of which only ``/`` and
    ``/health`` fall outside ``/api/ai``, and all three container probes use
    ``/health``. Nothing that must stay reachable becomes gated.

    ⚠️ Deliberately NOT gated on ``ENABLE_AUTH``, which is where this differs
    from coding-analysis's otherwise-identical helper. In that repo the flag is
    load-bearing; here ``settings.enable_auth`` had **zero readers** and the
    variable is absent from coding-ai's manifests entirely. Wiring it up now
    would not preserve an existing switch — it would *create* an env var that
    turns this service's only authentication off, which is the same shape as
    the ``SKIP_AUTH`` hatch deleted from the whole fleet in 1.16.0. The dead
    ``ENABLE_AUTH`` constant is removed in the same commit so nobody connects
    the two later.
    """
    try:
        from coding_client.auth import AuthMiddleware
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "the auth middleware could not be imported; refusing to start unauthenticated"
        ) from exc

    target_app.add_middleware(TenantMiddleware)  # type: ignore[invalid-argument-type]
    target_app.add_middleware(AuthMiddleware)  # type: ignore[invalid-argument-type]
    logger.info("Auth + Tenant middleware enabled")


install_auth_middleware(app)

app.include_router(ai_insights_router, prefix="/api/ai/insights/code", tags=["AI Code Insights"])
app.include_router(
    ai_test_analysis_router, prefix="/api/ai/analysis/tests", tags=["AI Test Analysis"]
)
app.include_router(
    ai_test_insights_router, prefix="/api/ai/insights/tests", tags=["AI Test Insights"]
)
app.include_router(
    ai_architecture_router,
    prefix="/api/ai/architecture",
    tags=["AI Architecture Governance"],
)

# Alias mounts at shorter prefixes so the gateway can proxy /api/ai/* without 404s.
# The gateway forwards the full path, so /api/ai/capabilities must resolve here.
# The deeper prefixes above remain functional for any direct callers.
app.include_router(
    ai_insights_router, prefix="/api/ai", tags=["AI Code Insights (alias)"], include_in_schema=False
)
app.include_router(
    ai_test_analysis_router,
    prefix="/api/ai",
    tags=["AI Test Analysis (alias)"],
    include_in_schema=False,
)
app.include_router(
    ai_architecture_router,
    prefix="/api/ai",
    tags=["AI Architecture Governance (alias)"],
    include_in_schema=False,
)

register_exception_handlers(app, debug=getattr(settings, "DEBUG", False))


@app.exception_handler(WMBYONotConfiguredError)
async def _byo_not_configured_handler(
    request: Request, exc: WMBYONotConfiguredError
) -> JSONResponse:
    """A byo_required tenant ran an agent without configuring a provider.

    Surface worker_management's hard-fail as HTTP 402 with the structured
    ``byo_llm_not_configured`` code (matches the shared error envelope) so the
    n8n node / caller points the user at fr → Settings → LLM Provider. Registered
    after the shared handlers; class-specific dispatch wins over the generic one.
    """
    return JSONResponse(
        status_code=402,
        content={"error": {"code": "byo_llm_not_configured", "message": str(exc)}},
    )


configure_openapi(
    app,
    protected_prefixes={"/api/ai"},
    public_endpoints={"/", "/health", "/docs", "/redoc", "/openapi.json"},
)

# Tracing, at module scope — before the app serves anything. setup_tracing()
# raises if it is ever called after startup, which is what the old lifespan block
# did silently. It also honours OTEL_SDK_DISABLED and OTLP_TRACES_ENDPOINT, so
# settings.otel_sdk_disabled / settings.otlp_traces_endpoint keep their meaning.
setup_tracing(app, SERVICE_NAME)


@app.get("/health", operation_id="health_check")
async def health():
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": VERSION,
        "redis": "connected" if _redis else "disconnected",
    }


@app.get("/", operation_id="get_root")
async def root():
    return {"service": SERVICE_NAME, "version": VERSION, "docs": "/docs", "health": "/health"}
