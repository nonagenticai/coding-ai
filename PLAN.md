# coding-ai — Extraction Plan

## Purpose
AI-powered insights service. Owns all AI/LLM-based analysis: code insights, test failure analysis, pattern detection, fix suggestions, root cause analysis.

## Port: 8690

---

## Endpoints

| Route Prefix | Source Router | Source Endpoint File |
|---|---|---|
| `/api/ai/insights/code` | `ai_insights` | `src/app/api/endpoints/ai_insights.py` |
| `/api/ai/analysis/tests` | `ai_test_analysis` | `src/app/api/endpoints/ai_test_analysis.py` |
| `/api/ai/insights/tests` | `ai_test_insights` | `src/app/api/endpoints/ai_test_insights.py` |

## Services to Extract

| Monolith Service | Target Location |
|---|---|
| `services/ai_failure_analysis_service.py` | `src/services/ai_failure_analysis.py` |
| `services/ai_fix_suggestion_engine.py` | `src/services/ai_fix_suggestion.py` |
| `services/ai_pattern_detector.py` | `src/services/ai_pattern_detector.py` |
| `services/ai_root_cause_engine.py` | `src/services/ai_root_cause.py` |
| `services/ai_test_failure_analyzer.py` | `src/services/ai_test_failure_analyzer.py` |
| `services/ai_test_integration_service.py` | `src/services/ai_test_integration.py` |
| `services/intelligent_test_recommendation_engine.py` | `src/services/intelligent_test_recommendation.py` |
| `services/test_pattern_analyzer.py` | `src/services/test_pattern_analyzer.py` |
| `services/performance_insight_analyzer.py` | `src/services/performance_insight_analyzer.py` |
| `ai_insights/*` | `src/services/ai_insights/` |
| `ai_test_insights/*` | `src/services/ai_test_insights/` |

## Dependencies
- coding-client (shared models, auth, DB)
- pydantic-ai (agent framework)
- openai, anthropic, groq SDKs (LLM providers)
- coding-analysis client (for code context)
- coding-testing client (for test data)
