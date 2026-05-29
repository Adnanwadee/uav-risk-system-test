# Smart Skies

Smart Skies is a hybrid AI decision framework for UAV flight risk management.
It combines deterministic core validation, Stage1 ML + SHAP, Stage2 RAG with citations, an operational agent with public-safe reasoning artifacts, and optional Groq-backed LLM synthesis.

## Active Architecture

The current production path is:

`POST /users/{user_id}/profiles/{profile_id}/assessments/stage2`
-> `Stage2AssessmentInput`
-> `build_runtime_rag_adapter_if_available()`
-> `OperationalAgentV2`
-> `Stage2PipelineV2`
-> `decision_engine.py`
-> optional `llm/orchestrator.py`
-> `reporting.py`
-> `Stage2AssessmentResponse`

Key rules:

- Frontend sends only raw profile data (via profile endpoints), raw scenario fields, raw secondary overrides, and operator notes.
- Frontend must not send processed feature vectors, one-hot/model columns, or model predictions.
- `decision_engine.py` owns the final decision and decision score.
- LLM synthesis is optional and does not override the decision engine, evidence support status, or citations.
- RAG provides evidence bundles, citations, and provenance; it is not final decision authority.
- `OperationalAgentV2` exposes findings, action items, working memory summary artifacts, and tool trace summaries.
- System Work Trace (Transparency Trace) is structured public-safe metadata and must not include hidden chain-of-thought, raw prompts, or raw completions.
- Legacy ACE stack is not the canonical runtime path for Stage2 v2.

## Persistence Limitation

Assessment persistence is not implemented yet.
`assessment_id` is generated for traceability in API responses, but it is not stored for later retrieval.

## Canonical Validation Commands

Run these checks for readiness:

```bash
python scripts/validate_stage2_rag_index.py
python scripts/run_stage2_rag_diagnostic.py --run-quality
python scripts/run_stage2_pipeline_v2_smoke.py --use-real-rag
python scripts/check_groq_provider_reachability.py
```

Use `check_groq_provider_reachability.py` only when the environment is configured for Groq/LLM access.

## Manual-Only Commands

Do not use these as casual readiness checks:

```bash
python scripts/rebuild_stage2_rag_index.py --force
python src/uav_risk/stage2/rag/build_index.py --force
python src/uav_risk/stage2/rag/force_download.py
python src/uav_risk/stage2/knowledge/models/embedding/train_script.py
python scripts/simulate_agent_live.py
```

These commands can overwrite canonical RAG outputs, download models, train embeddings, or run legacy/demo code paths.

## Documentation

- [Frontend Handoff](docs/FRONTEND_HANDOFF.md)
- [API Examples](docs/API_EXAMPLES.md)
- [Stage2 RAG Readiness](docs/STAGE2_RAG_READINESS.md)

## Notes For Reviewers

- RAG citations are evidence support, not legal guarantees.
- The Stage2 response includes a typed public contract for frontend consumption.
- The current frontend contract is documented in the docs listed above.