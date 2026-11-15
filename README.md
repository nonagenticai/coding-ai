# coding-ai

[![CI](https://github.com/nonagenticai/coding-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/nonagenticai/coding-ai/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/nonagenticai/coding-ai)](https://github.com/nonagenticai/coding-ai/releases)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

AI-powered code analysis and test intelligence microservice within the code_graph_helper platform. Provides three core analysis capabilities: static code insight generation (architecture, security, performance, maintainability), test failure root-cause analysis with fix suggestions, and test suite health analysis with failure prediction. A fourth router group classifies code units into ArchiMate 3.2 types and proposes architecture mappings for a container. All operations are tenant-scoped and container-scoped. LLM calls are routed through the platform's `worker_management` service, which owns provider and model selection per agent; coding-ai never calls a provider API directly.

Port **8690**.

| | |
|---|---|
| **Runtime** | Python 3.14, FastAPI |
| **Depends on** | `worker_management` (all LLM traffic), coding-containers, coding-analysis, Redis, licenses |
| **Runs on** | Container / Kubernetes; listens on port 8690 |

## Architecture

```
coding-ai (FastAPI, port 8690)
├── routers/
│   ├── ai_insights         POST /analyze, GET /capabilities, GET /insight-types
│   ├── ai_test_analysis    POST /failure-analysis, POST /batch-analyze, GET health, metrics, patterns
│   ├── ai_test_insights/   POST /analyze, /comprehensive-analysis, /predict-failures,
│   │                        /recommendations, GET /analysis/{id}, /comprehensive-analysis/{id},
│   │                        GET /service-health, /metrics
│   └── ai_architecture     POST /tools/classify-code-units, GET /proposals, GET /capabilities
├── services/
│   ├── ai_insights/        Rule-based code insight engine (8 generators)
│   ├── ai_failure_analysis.py  AIFailureAnalysisService (circuit breaker, caching, graceful degradation)
│   ├── ai_test_insights/   AITestInsightsService (statistical performance/reliability/coverage/quality/pattern analyzers)
│   ├── llm_agents.py       make_agent_runner shim: every LLM call goes to worker_management
│   └── ai_*                Root cause engine, fix suggestion engine, pattern detector
├── wm_client.py            worker_management HTTP client (POST /workers/{name}/message)
├── wm_bootstrap.py         Idempotent seeding of the 15 agent worker records
├── clients/                coding_analysis_client.py; outbound_identity.py (caller and service credentials)
├── auth/                   tenant.py, _header_trust.py, _default_tenant.py
└── config/                 settings.py (environment), agents.py (15 agent definitions)
```

Four router groups are mounted at canonical prefixes; three of them are also aliased at shorter
`/api/ai/*` paths for gateway compatibility:

| Canonical prefix | Alias prefix | Purpose |
|---|---|---|
| `/api/ai/insights/code` | `/api/ai` | Code insight generation |
| `/api/ai/analysis/tests` | `/api/ai` | Test failure analysis |
| `/api/ai/insights/tests` | (none) | Test suite insights |
| `/api/ai/architecture` | `/api/ai` | ArchiMate-type classification, architecture proposals, classifier capabilities |

## Auth

Tenant isolation via middleware (`TenantMiddleware`). Tenant resolved in this order:

1. `X-Tenant-Id` header. ⚠️ This is an **unverified claim**, not a credential, and it is read *above* license introspection — an in-mesh caller that reaches `:8690` pod-to-pod does not traverse the edge ext_authz filter that would have stripped and re-injected it, so it can assert any tenant. Set `TRUST_TENANT_HEADER=false` to ignore the header entirely; resolution then falls through to step 2 and a forged header with no valid license gets a 401. Default (unset) is `true` — today's behaviour.
2. `Authorization: Bearer <license-token>` -- introspected against the licences service configured in `LICENSE_INTROSPECT_URL` (path `/license/introspect`). Active tokens provide `tenant_id` as tenant (`subject_id` is the USER and must not be used for tenant scoping).
3. User object from upstream middleware (if present)
4. **Trusted network bypass** -- source networks named by `TRUSTED_CIDRS` are admitted with NO credential and granted `DEFAULT_TENANT_ID`. **Empty by default** (since PR #44): it used to default to RFC1918, and since every pod carries an address in the k3s pod CIDR `10.42.0.0/16`, that meant *every workload in the cluster* satisfied the bypass. Both knobs are required — there is **no built-in default tenant**, so an armed bypass with `DEFAULT_TENANT_ID` unset rejects the caller (and, in production, refuses to boot) rather than inventing one.
5. Otherwise: 401.

There is no environment variable that bypasses this. `SKIP_AUTH` was removed
(it returned a hardcoded `test-tenant` before any credential was examined). To
bypass auth in a test suite, use `coding_client.testing.override_tenant_auth`,
which is per-app and cannot be switched on from the environment. When overriding
this service's `require_tenant`, set `_request_tenant_id_var` too — the WM shim
forwards it, and a plain string leaves the shim on its service-token fallback.

Some endpoints require user identity (`get_current_user_with_tenant`) or admin role checks.

## API Routes

### Code Insights (`/api/ai/insights/code`)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/analyze` | tenant | Generate AI code insights for files within a container. Request body: `container_name`, `file_paths`, `focus_areas`, `max_insights`, `include_experimental`. |
| `GET` | `/capabilities` | tenant | List supported insight types, priority/confidence levels, analysis features, supported languages (9), quality metrics, limitations. |
| `GET` | `/insight-types` | tenant | Detailed documentation of each insight category (architecture, refactoring, security, performance, maintainability, cross_language, quality, patterns). |

### Test Failure Analysis (`/api/ai/analysis/tests`)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/failure-analysis` | tenant | Analyze a single test failure. Request body: `container_name`, `test_output`, `test_name`, `test_framework`, `language`, `source_code`, `test_code`, `file_path`, `include_suggestions`, `include_patterns`, `max_root_causes`, `max_suggestions`, `minimum_confidence`. Degrades gracefully when AI unavailable. |
| `POST` | `/batch-analyze` | tenant | Batch-analyze up to 50 test failures concurrently. Request body: `container_name`, `test_results`, `max_concurrent` (up to 10). |
| `GET` | `/failure-analysis/health` | tenant | Health status of AI analysis service (model availability, circuit breaker state). |
| `GET` | `/failure-analysis/metrics` | tenant | Performance metrics: total analyses, success rate, avg response time, cache stats, token usage. |
| `GET` | `/patterns` | tenant | Known failure pattern statistics. Query: `limit` (default 50, max 200). |

### Test Suite Insights (`/api/ai/insights/tests`)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/analyze` | tenant | AI-powered analysis of test execution patterns and quality indicators. Request body: `container_name`, `test_results`, `quality_results`, `time_period`, `include_predictions`. |
| `POST` | `/comprehensive-analysis` | tenant | Full analysis using all specialized analyzers. Request body: `container_name`, `test_results`, `quality_results`, `code_changes`, `time_period`. |
| `POST` | `/predict-failures` | tenant+user | Predict test failure likelihood from historical data and code changes. Request body: `container_name`, `test_results`, `code_changes`. |
| `POST` | `/recommendations` | tenant+user | Generate recommendations from a previous analysis. Request body: `container_name`, `analysis`, `focus_areas`. |
| `GET` | `/analysis/{analysis_id}` | tenant+user | Retrieve cached analysis results by ID. |
| `GET` | `/comprehensive-analysis/{analysis_id}` | tenant+user | Retrieve cached comprehensive analysis results. |
| `GET` | `/service-health` | tenant+user | Health check for the AI test insights service. |
| `GET` | `/metrics` | tenant+user | Service metrics and feature/agent availability. |

### Infrastructure

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | none | Top-level health check (service name, version, Redis status). |
| `GET` | `/` | none | Service info with links to `/docs` and `/health`. |

## AI Providers

coding-ai does not call an LLM provider directly. Every LLM request goes through the platform's
`worker_management` service (`WORKER_MANAGEMENT_URL`), which owns provider selection and routing:

```
coding-ai --POST /workers/{name}/message--> worker_management --> upstream model
```

Each coding-ai agent is a **worker record** in `worker_management`. The 15 definitions in
`src/config/agents.py` supply the worker name, system prompt, temperature (default `0.2`; `0.1`
for the architecture classifier) and `max_tokens` (`3000`). At startup `src/wm_bootstrap.py`
creates any missing records and leaves existing records untouched, preserving operator edits;
`scripts/register_workers.py` does the same from the command line (`--dry-run`, `--force`).
Seeded records request the logical model `local-llm` from provider `inference_router` and carry
`INFERENCE_ROUTER_URL` as their `base_url_override`. An operator can change any worker's
provider, model, prompt or hyperparameters with `PUT /workers/{name}` — no coding-ai redeploy.

| Worker group | Worker names |
|---|---|
| Root-cause analyzers | `cai-root-cause-syntactic`, `cai-root-cause-semantic`, `cai-root-cause-environmental`, `cai-root-cause-integration`, `cai-root-cause-performance` |
| Pattern detection | `cai-pattern-detector-pattern-extractor`, `cai-pattern-detector-pattern-classifier`, `cai-pattern-detector-similarity-analyzer` |
| Fix suggestions | `cai-fix-suggestion-immediate-fix`, `cai-fix-suggestion-preventive-fix`, `cai-fix-suggestion-best-practice` |
| Failure analysis | `cai-test-failure-analyzer-root-cause`, `cai-test-failure-analyzer-fix-suggestion`, `cai-test-failure-analyzer-pattern` |
| Architecture | `cai-architecture-classifier` |

Structured output travels per call as a JSON Schema in `output_schema`; it is not stored on the
worker record. Requests forward the end-user tenant (resolved by `require_tenant`) so
`worker_management` enforces that tenant's BYO-LLM posture; a `byo_required` tenant without a
provider receives HTTP 402 `byo_llm_not_configured`. `CODING_SERVICE_TOKEN` identifies coding-ai
on background and startup calls and is required whenever `WORKER_MANAGEMENT_URL` is set: the
service refuses to start without it rather than emit unattributable calls.

No provider SDK or provider API key is used directly. `pydantic-ai` and the OpenAI SDK remain
runtime dependencies (the shared client library talks to an OpenAI-compatible endpoint), but
coding-ai itself only speaks to `worker_management`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVICE_NAME` | `coding-ai` | Service identifier |
| `VERSION` | `1.0.0` | Service version |
| `ENVIRONMENT` | `production` | Environment name. In production, missing required URLs fail boot. |
| `LOG_LEVEL` | `INFO` | Root log level, read by `coding_client` at startup. |
| `CORS_ORIGINS` | `*` | Comma-separated browser origins. Credentials are enabled only for an explicit list; `*` forces `allow_credentials=false`. |
| `ENABLE_DOCS` | `false` | Read but not enforced -- `/docs` and `/redoc` always mount. |
| `DEFAULT_TENANT_ID` | *(unset)* | Tenant granted to a credential-less caller admitted by `TRUSTED_CIDRS`. **No default** -- unset means such callers are rejected, never handed an invented tenant. It used to default to `n8n-dev`, the stale pre-uuid7 *label* of a real tenant, so rows written under it landed in a silo no live JWT can address. |
| `TRUST_TENANT_HEADER` | `true` | Whether the unverified `X-Tenant-Id` header may be believed. `false` ignores it (falls through to license introspection -- it is **not** downgraded to `DEFAULT_TENANT_ID`). |
| `REDIS_URL` | *(empty)* | Redis connection for caching. No code-side default; production injects it (required in production). |
| `CODING_CONTAINERS_URL` | *(empty)* | coding-containers service URL (container validation). No code-side default; required in production. |
| `CODING_ANALYSIS_URL` | *(empty)* | coding-analysis service URL. Used by the architecture proposals route, which degrades to an empty `proposals[]` when unset or unreachable. Not required in production. |
| `WORKER_MANAGEMENT_URL` | *(empty)* | worker_management base URL. When unset, worker seeding is skipped and LLM calls fail fast; it is not in the production required-set. |
| `INFERENCE_ROUTER_URL` | *(empty)* | `base_url_override` stamped on every seeded worker record. Required in production. |
| `CODING_SERVICE_TOKEN` | *(empty)* | coding-ai's service identity for worker_management. Required at boot whenever `WORKER_MANAGEMENT_URL` is set; read per call so a rotation needs no restart. |
| `CODING_AI_LICENSE_TOKEN` | *(empty)* | Legacy override for the service identity; wins when set (and is `scripts/register_workers.py`'s `--license-token` default). |
| `WM_SEND_DELEGATED_LICENCE` | *(off)* | When on, bare-tenant WM calls send `X-Tenant-Id: Bearer <service token>` plus `X-Effective-Tenant-Id: <tenant>` instead of a tokenless tenant header. Unrecognised values fail boot. |
| `LICENSE_INTROSPECT_URL` | *(empty)* | Full licence-introspection URL (licenses service path `/license/introspect`). Required in production; unset means `Bearer` licences cannot be introspected. |
| `TRUSTED_CIDRS` | *(empty)* | CIDRs admitted with NO credential and granted `DEFAULT_TENANT_ID`. An authentication bypass, so empty by default. |
| `OTEL_SDK_DISABLED` | `false` | `true` disables tracing. Read directly by `coding_client.telemetry`, not by this service's `Settings`. |
| `OTLP_TRACES_ENDPOINT` | *(unset)* | Full OTLP/HTTP endpoint including `/v1/traces`; unset disables span export. |

## Service Dependencies

| Service | Port | Used for |
|---------|------|----------|
| coding-containers | 8691 | Workspace validation and file discovery (`CodingWorkspacesClient`) |
| coding-analysis | 8688 | Per-container component list for architecture proposals (`src/clients/coding_analysis_client.py`) |
| worker_management | 8887 | All LLM calls (`POST /workers/{name}/message`) and worker-record seeding at startup |
| Redis | 6379 | Caching (optional, degrades gracefully) |
| licenses (in-cluster) | 8544 | Licence-token introspection for tenant auth (via `LICENSE_INTROSPECT_URL`) |

coding-ai does not call inference-router directly: `worker_management` dispatches each worker
record to the upstream in its `base_url_override`, which is seeded from `INFERENCE_ROUTER_URL`.

## Project Structure

```
coding-ai/
├── src/
│   ├── main.py                           FastAPI app, lifespan, auth/CORS/tracing wiring
│   ├── type_patterns.py                  Shared input-validation patterns
│   ├── wm_client.py                      worker_management HTTP client
│   ├── wm_bootstrap.py                   Idempotent seeding of the 15 agent worker records
│   ├── auth/
│   │   ├── tenant.py                     TenantMiddleware, require_tenant, licence introspection, contextvars
│   │   ├── _header_trust.py              TRUST_TENANT_HEADER gate for the X-Tenant-Id claim
│   │   └── _default_tenant.py            DEFAULT_TENANT_ID resolution for the TRUSTED_CIDRS bypass
│   ├── clients/
│   │   ├── coding_analysis_client.py     Outbound client to coding-analysis components API
│   │   └── outbound_identity.py          Caller-forwarded credential / service-token headers
│   ├── config/
│   │   ├── settings.py                   Environment-backed Settings, production fail-fast
│   │   └── agents.py                     The 15 agent definitions (prompt, temperature, max_tokens)
│   ├── models/
│   │   ├── ai_insights_api_models.py
│   │   ├── ai_test_failure_models.py
│   │   ├── architecture_models.py
│   │   ├── common_api_models.py
│   │   ├── common_responses_models.py
│   │   ├── error_models.py
│   │   └── testing_models.py             Stub test-execution types owned by coding-testing
│   ├── routers/
│   │   ├── ai_insights.py                Code insights (3 routes)
│   │   ├── ai_test_analysis.py           Test failure analysis (5 routes)
│   │   ├── ai_architecture.py            ArchiMate classification and proposals (3 routes)
│   │   └── ai_test_insights/             Test suite insights package (8 routes)
│   │       ├── ai_analysis_endpoints.py
│   │       ├── ai_test_insights_config.py
│   │       ├── insight_generation_services.py
│   │       ├── metrics_and_calculations.py
│   │       ├── report_formatting_endpoints.py
│   │       ├── shared.py
│   │       ├── test_pattern_recognition.py
│   │       └── utilities_and_helpers.py
│   ├── services/
│   │   ├── ai_insights/                  Rule-based insight engine and 8 generators
│   │   ├── ai_test_insights/             Test-suite analyzers, pattern detection, report formatting
│   │   ├── llm_agents.py                 WM-backed make_agent_runner shim (pydantic-ai-shaped)
│   │   ├── ai_failure_analysis.py        AIFailureAnalysisService (circuit breaker, caching)
│   │   ├── ai_root_cause.py              Multi-agent root-cause engine
│   │   ├── ai_fix_suggestion.py          Multi-agent fix-suggestion engine
│   │   ├── ai_pattern_detector.py        Multi-agent failure-pattern detector
│   │   ├── ai_test_failure_analyzer.py   End-to-end failure analysis agent group
│   │   ├── ai_test_integration.py        Stubs for execution owned by coding-testing
│   │   ├── architecture_classifier.py    ArchiMate 3.2 classifier (cai-architecture-classifier)
│   │   ├── architecture_proposals.py     coding-analysis Component -> proposal contract mapping
│   │   ├── intelligent_test_recommendation.py
│   │   ├── performance_insight_analyzer.py
│   │   ├── quality_metrics_analyzer.py   Stub quality-metrics model
│   │   └── test_pattern_analyzer.py
│   └── utils/
│       ├── response_builder.py
│       └── responses.py
├── tests/
├── scripts/
│   ├── register_workers.py               CLI to seed/refresh WM worker records
│   ├── pip-audit.sh
│   └── pip-audit-autofix.sh
├── Dockerfile                            Multi-stage build (python:3.14-slim + uv)
├── docker-compose.yml
├── pyproject.toml
└── uv.lock
```

## Running

```bash
# Docker
docker compose up -d

# Local development
uv sync
uv run uvicorn src.main:app --host 0.0.0.0 --port 8690 --reload
```

## Dev Tools

- **Lint + format:** `ruff` (`ruff check src/`, `ruff format src/`)
- **Type checking:** `ty` (`ty check src/`)
- **Tests:** `uv run pytest` (unit), `uv run pytest -m integration` (integration)

## CI/CD

`.github/workflows/ci.yml` calls the public shared workflow
`nonagenticai/ci-workflows/.github/workflows/python-service.yml` on GitHub-hosted runners:
secret scan, private-reference scan, lint/format/type checks, unit tests, Docker build and smoke
test. On `main` it publishes `ghcr.io/nonagenticai/coding-ai:sha-<7-char-commit>` and `:latest`.
There is no deploy job and no cluster credential in this repository: Flux in `gitops-production`
watches the GHCR ImagePolicy and rolls the workload when the image changes.

## Security

Report vulnerabilities privately through the repository's Security tab. Do not open a public
issue for a security problem.

## Contributing

Issues and pull requests are welcome. See
[`CONTRIBUTING.md`](https://github.com/nonagenticai/.github/blob/main/CONTRIBUTING.md) for the
organisation-wide rules; setup, test and lint commands are in this README. Commits carry DCO
sign-off.

## Licensing

The code in this repository is Apache-2.0 and needs no licence key to build, test or run.
Licence tokens are an **optional** identity source: with `LICENSE_INTROSPECT_URL` unset (the
default), a `Bearer` licence cannot be introspected, and callers authenticate through the
platform's mesh-forwarded identity instead. With the URL set, every licence token is
introspected against the licences service, and an unreachable service yields no identity
(401) — never an admission.

The licensing stack is public too — [`licenses`](https://github.com/nonagenticai/licenses)
(entitlements, plans and tokens) and
[`license-authz`](https://github.com/nonagenticai/license-authz) (the Envoy `ext_authz`
enforcement point). Neither is required to run this code.

## License

Apache-2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
