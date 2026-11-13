# Operation ID Plan: coding-ai

## Files to Edit

- `src/main.py` (app-level routes: `/health`, `/`)
- `src/routers/ai_insights.py` (prefix `/api/ai/insights/code` AND alias `/api/ai`)
- `src/routers/ai_test_analysis.py` (prefix `/api/ai/analysis/tests` AND alias `/api/ai`)
- `src/routers/ai_test_insights/ai_analysis_endpoints.py` (prefix `/api/ai/insights/tests`)
- `src/routers/ai_test_insights/test_pattern_recognition.py` (prefix `/api/ai/insights/tests`)
- `src/routers/ai_test_insights/insight_generation_services.py` (prefix `/api/ai/insights/tests`)
- `src/routers/ai_test_insights/report_formatting_endpoints.py` (prefix `/api/ai/insights/tests`)
- `src/routers/ai_test_insights/configuration_management.py` (prefix `/api/ai/insights/tests`)

## Mapping Table

### main.py (app-level, no prefix)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| GET | `/health` | `health` | `health_check` |
| GET | `/` | `root` | `get_root` |

### routers/ai_insights.py (prefix `/api/ai/insights/code`)

Note: This router is mounted twice — at `/api/ai/insights/code` and `/api/ai` (alias).
Setting `operation_id` on the decorators will cause duplicate operation_id errors.

**Fix:** Add `include_in_schema=False` to the alias mount in `src/main.py`:
```python
app.include_router(ai_insights_router, prefix="/api/ai", tags=["AI Code Insights (alias)"],
                   include_in_schema=False)
```

| Method | Effective Path (primary) | Function | operation_id |
|--------|--------------------------|----------|--------------|
| POST | `/api/ai/insights/code/analyze` | `analyze_insights` | `analyze_code_insights` |
| GET | `/api/ai/insights/code/capabilities` | `get_insights_capabilities` | `get_insights_capabilities` |
| GET | `/api/ai/insights/code/insight-types` | `get_insight_types` | `list_insight_types` |

### routers/ai_test_analysis.py (prefix `/api/ai/analysis/tests`)

Note: This router is also mounted twice — at `/api/ai/analysis/tests` and `/api/ai` (alias).

**Fix:** Add `include_in_schema=False` to the alias mount in `src/main.py`:
```python
app.include_router(ai_test_analysis_router, prefix="/api/ai", tags=["AI Test Analysis (alias)"],
                   include_in_schema=False)
```

| Method | Effective Path (primary) | Function | operation_id |
|--------|--------------------------|----------|--------------|
| POST | `/api/ai/analysis/tests/failure-analysis` | `analyze_test_failure` | `analyze_test_failure` |
| POST | `/api/ai/analysis/tests/batch-analyze` | `batch_analyze_failures` | `batch_analyze_test_failures` |
| GET | `/api/ai/analysis/tests/failure-analysis/health` | `health_check` | `ai_test_analysis_health_check` |
| GET | `/api/ai/analysis/tests/failure-analysis/metrics` | `get_metrics` | `get_test_analysis_metrics` |
| GET | `/api/ai/analysis/tests/patterns` | `get_patterns` | `list_test_failure_patterns` |

Note: The health_check route already has `operation_id="ai_test_analysis_health_check"` set.

### routers/ai_test_insights/ai_analysis_endpoints.py (prefix `/api/ai/insights/tests`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| POST | `/api/ai/insights/tests/analyze` | `analyze_test_insights` | `analyze_test_insights` |
| POST | `/api/ai/insights/tests/comprehensive-analysis` | `comprehensive_test_analysis` | `comprehensive_test_analysis` |

### routers/ai_test_insights/test_pattern_recognition.py (prefix `/api/ai/insights/tests`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| POST | `/api/ai/insights/tests/predict-failures` | `predict_test_failures` | `predict_test_failures` |

### routers/ai_test_insights/insight_generation_services.py (prefix `/api/ai/insights/tests`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| POST | `/api/ai/insights/tests/recommendations` | `generate_test_recommendations` | `generate_test_recommendations` |

### routers/ai_test_insights/report_formatting_endpoints.py (prefix `/api/ai/insights/tests`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| GET | `/api/ai/insights/tests/analysis/{analysis_id}` | `get_cached_analysis` | `get_cached_test_analysis` |
| GET | `/api/ai/insights/tests/comprehensive-analysis/{analysis_id}` | `get_cached_comprehensive_analysis` | `get_cached_comprehensive_analysis` |

### routers/ai_test_insights/configuration_management.py (prefix `/api/ai/insights/tests`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| GET | `/api/ai/insights/tests/settings` | `get_service_settings` | `get_test_insights_settings` |
| PUT | `/api/ai/insights/tests/settings` | `update_service_settings` | `update_test_insights_settings` |

## Total: 19 routes (2 app-level + 3 ai-insights + 5 ai-test-analysis + 2 ai-analysis + 1 pattern + 1 recommendations + 2 reports + 2 config + 1 predict-failures)

## Duplicate Mount Issues

Two routers are mounted at alias prefixes for gateway compatibility:

```python
# In src/main.py — these alias mounts will cause duplicate operation_id errors:
app.include_router(ai_insights_router, prefix="/api/ai", tags=["AI Code Insights (alias)"])
app.include_router(ai_test_analysis_router, prefix="/api/ai", tags=["AI Test Analysis (alias)"])
```

**Fix:** Add `include_in_schema=False` to both alias mounts:
```python
app.include_router(ai_insights_router, prefix="/api/ai", tags=["AI Code Insights (alias)"],
                   include_in_schema=False)
app.include_router(ai_test_analysis_router, prefix="/api/ai", tags=["AI Test Analysis (alias)"],
                   include_in_schema=False)
```

This preserves the runtime routing (gateway requests still work) while preventing OpenAPI schema conflicts.

## Execution Steps

```bash
cd /Users/mini1/Movies/code_graph_helper/coding-ai && git pull --rebase
```

1. Fix the duplicate mount issues in `src/main.py` (add `include_in_schema=False` to 2 alias mounts)
2. Add `operation_id=` to the 2 app-level routes in `src/main.py`
3. Add `operation_id=` to the 3 routes in `src/routers/ai_insights.py`
4. Add `operation_id=` to the 5 routes in `src/routers/ai_test_analysis.py` (1 already has it)
5. Add `operation_id=` to the 2 routes in `src/routers/ai_test_insights/ai_analysis_endpoints.py`
6. Add `operation_id=` to the 1 route in `src/routers/ai_test_insights/test_pattern_recognition.py`
7. Add `operation_id=` to the 1 route in `src/routers/ai_test_insights/insight_generation_services.py`
8. Add `operation_id=` to the 2 routes in `src/routers/ai_test_insights/report_formatting_endpoints.py`
9. Add `operation_id=` to the 2 routes in `src/routers/ai_test_insights/configuration_management.py`
10. Validate uniqueness: `grep -r 'operation_id=' src/ | sort | uniq -d` (should be empty)
11. `pre-commit run --all-files`
12. `pytest`
13. `git commit` and `git push`
