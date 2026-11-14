# coding-ai

[![CI](https://github.com/nonagenticai/coding-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/nonagenticai/coding-ai/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/nonagenticai/coding-ai)](https://github.com/nonagenticai/coding-ai/releases)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

AI-powered code analysis and test intelligence microservice within the code_graph_helper platform. Provides three core capabilities: static code insight generation (architecture, security, performance, maintainability), test failure root-cause analysis with fix suggestions, and test suite health analysis with failure prediction. All operations are tenant-scoped and container-scoped. Uses PydanticAI agents with configurable provider chains (OpenAI, Anthropic, Groq).

Port **8690**.

## Architecture

```
coding-ai (FastAPI, port 8690)
├── routers/
│   ├── ai_insights         POST /analyze, GET /capabilities, GET /insight-types
│   ├── ai_test_analysis    POST /failure-analysis, POST /batch-analyze, GET health, metrics, patterns
│   └── ai_test_insights/   POST /analyze, /comprehensive-analysis, /predict-failures,
│                            /recommendations, GET /analysis/{id}, /comprehensive-analysis/{id},
│                            GET|PUT /settings
├── services/
│   ├── ai_insights/        Rule-based + optional LLM code insight engine (8 generators)
│   ├── ai_failure_analysis AIFailureAnalysisService (PydanticAI, circuit breaker, caching)
│   ├── ai_test_insights/   AITestInsightsService (PydanticAI agents for perf/reliability/coverage/quality/patterns)
│   └── ai_*                Root cause engine, fix suggestion engine, pattern detector
├── clients/                HTTP clients for coding-analysis (:8688) and coding-testing (:8689)
├── auth/                   Tenant middleware, license introspection
└── config/                 PydanticAI provider/agent configuration
```

Three router groups are mounted at canonical prefixes and aliased at shorter `/api/ai/*` paths for gateway compatibility:

| Canonical prefix | Alias prefix | Purpose |
|---|---|---|
| `/api/ai/insights/code` | `/api/ai` | Code insight generation |
| `/api/ai/analysis/tests` | `/api/ai` | Test failure analysis |
| `/api/ai/insights/tests` | (none) | Test suite insights |

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
| `GET` | `/capabilities` | none | List supported insight types, priority/confidence levels, analysis features, supported languages (9), quality metrics, limitations. |
| `GET` | `/insight-types` | none | Detailed documentation of each insight category (architecture, refactoring, security, performance, maintainability, cross_language, quality, patterns). |

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
| `GET` | `/settings` | admin | Get service configuration. |
| `PUT` | `/settings` | admin | Update service configuration. Validates concurrent limits, timeouts, thresholds. |

### Infrastructure

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | none | Top-level health check (service name, version, Redis status). |
| `GET` | `/` | none | Service info with links to `/docs` and `/health`. |

## AI Providers

Configured via PydanticAI with primary/fallback provider chain:

| Provider | Default model | Env vars |
|----------|--------------|----------|
| OpenAI | gpt-4o | `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL` |
| Anthropic | claude-3-5-sonnet-20241022 | `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `ANTHROPIC_BASE_URL` |
| Groq | llama-3.3-70b-versatile | `GROQ_API_KEY`, `GROQ_MODEL`, `GROQ_BASE_URL` |

Three PydanticAI agents are configured by default: `test_failure_analyzer` (primary provider), `code_quality_analyzer` (fallback provider), `performance_analyzer` (primary provider). Each agent's provider, model, temperature, max_tokens, and enabled state can be overridden via `AGENT_<NAME>_<PARAM>` env vars (e.g. `AGENT_TEST_FAILURE_ANALYZER_MODEL`).

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVICE_NAME` | `coding-ai` | Service identifier |
| `VERSION` | `1.0.0` | Service version |
| `ENVIRONMENT` | `production` | Environment name |
| `ENABLE_DOCS` | `false` | Enable /docs and /redoc (read but not enforced -- docs always mount) |
| `ENABLE_AUTH` | `true` | Enable tenant authentication (read in main.py, not enforced in middleware) |
| `DEFAULT_TENANT_ID` | *(unset)* | Tenant granted to a credential-less caller admitted by `TRUSTED_CIDRS`. **No default** -- unset means such callers are rejected, never handed an invented tenant. It used to default to `n8n-dev`, the stale pre-uuid7 *label* of a real tenant, so rows written under it landed in a silo no live JWT can address. |
| `TRUST_TENANT_HEADER` | `true` | Whether the unverified `X-Tenant-Id` header may be believed. `false` ignores it (falls through to license introspection -- it is **not** downgraded to `DEFAULT_TENANT_ID`). |
| `REDIS_URL` | `redis://code-graph-redis:6379/0` | Redis connection for caching |
| `CODING_CONTAINERS_URL` | `http://coding-containers:8691` | coding-containers service URL (container validation) |
| `HTTP_CLIENT_TIMEOUT` | `30.0` | Default timeout for inter-service HTTP calls |
| `TRUSTED_CIDRS` | *(empty)* | CIDRs admitted with NO credential and granted `DEFAULT_TENANT_ID`. An authentication bypass, so empty by default. |
| `OTEL_SDK_DISABLED` | `true` | Set to `false` to enable OpenTelemetry tracing |
| `OTLP_TRACES_ENDPOINT` | -- | OTLP gRPC endpoint for traces (disabled unless explicitly configured) |
| `PYDANTIC_AI_ENABLED` | `true` | Master switch for PydanticAI features |
| `PYDANTIC_AI_PRIMARY_PROVIDER` | `openai` | Primary AI provider |
| `PYDANTIC_AI_FALLBACK_PROVIDER` | `anthropic` | Fallback AI provider |
| `PYDANTIC_AI_CACHE_TTL_SECONDS` | `3600` | AI result cache TTL |
| `PYDANTIC_AI_MAX_CONCURRENT_ANALYSES` | `5` | Max concurrent AI analysis requests |
| `OPENAI_API_KEY` | (none) | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model name |
| `OPENAI_BASE_URL` | (none) | OpenAI-compatible base URL |
| `OPENAI_MAX_TOKENS` | `32000` | OpenAI max tokens |
| `OPENAI_TEMPERATURE` | `0.1` | OpenAI temperature |
| `ANTHROPIC_API_KEY` | (none) | Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-3-5-sonnet-20241022` | Anthropic model name |
| `ANTHROPIC_BASE_URL` | (none) | Anthropic base URL |
| `ANTHROPIC_MAX_TOKENS` | `32000` | Anthropic max tokens |
| `ANTHROPIC_TEMPERATURE` | `0.1` | Anthropic temperature |
| `GROQ_API_KEY` | (none) | Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `GROQ_BASE_URL` | (none) | Groq base URL |
| `GROQ_MAX_TOKENS` | `32000` | Groq max tokens |
| `GROQ_TEMPERATURE` | `0.1` | Groq temperature |

## Service Dependencies

| Service | Port | Used for |
|---------|------|----------|
| coding-containers | 8691 | Workspace existence validation before analysis (`/api/workspaces/{name}`) |
| coding-analysis | 8688 | Fetch code analysis results (`CodingAnalysisClient`) |
| coding-testing | 8689 | Fetch test execution results (`CodingTestingClient`) |
| Redis | 6379 | Caching (optional, degrades gracefully) |
| licenses (in-cluster) | 8544 | License token introspection for tenant auth |

## Project Structure

```
coding-ai/
├── src/
│   ├── main.py                           FastAPI app, lifespan, error handlers
│   ├── auth/
│   │   ├── models.py                     User model (sub, tenant_id, roles)
│   │   └── tenant.py                     TenantMiddleware, require_tenant, license introspection
│   ├── clients/
│   │   ├── base.py                       BaseServiceClient (httpx)
│   │   ├── coding_analysis.py            CodingAnalysisClient
│   │   └── coding_testing.py             CodingTestingClient
│   ├── config/
│   │   └── pydantic_ai.py                PydanticAIConfig, provider/agent settings
│   ├── models/                           Pydantic request/response models
│   ├── routers/
│   │   ├── ai_insights.py                Code insights endpoints (3 routes)
│   │   ├── ai_test_analysis.py           Test failure analysis endpoints (5 routes)
│   │   └── ai_test_insights/             Test suite insights package (8 routes)
│   ├── services/                         Business logic, AI engines
│   └── utils/                            Response builders, helpers
├── tests/
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

GitHub Actions workflow: `ruff + ty → build amd64 → smoke test → push`

Image pushed to `192.168.1.114:5000/coding-ai:sha-<commit>`, promoted to `:latest` on success. Flux auto-deploys to K3s.

## Security

Report vulnerabilities privately through the repository's Security tab — see
[`SECURITY.md`](SECURITY.md). Do not open a public issue for a security problem.

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
