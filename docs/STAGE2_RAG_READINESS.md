# Stage2 RAG Readiness

## Current Readiness Status

- RAG source coverage status: **9 expected / 9 indexed**.
- RAG quality is **proven by diagnostics**.
- Real RAG smoke completed independently.
- Groq reachability completed independently.
- Before Stage 5 runtime reuse, combined real-RAG + env-LLM smoke could terminate from runtime memory/resource pressure.

## Stage 5 Runtime Reuse Update

Stage 5 adds safe process-local reuse for expensive runtime initialization:
- cached runtime RAG adapter construction
- env-keyed cached LLM orchestrator construction

This reduces repeated heavy loading in a single process and improves combined-smoke stability expectations.

## Runtime Metadata Exposed in API/Smoke

Stage2 response and smoke output expose:
- `rag_quality_is_proven`, `quality_is_proven`
- `scenario_evidence_status`, `scenario_evidence_complete`
- `corpus_coverage_status`
- `expected_source_count`, `indexed_source_count`, `missing_sources_count`
- `reranker_configured`, `reranker_available`, `reranker_used`, `reranker_reason`
- `retrieval_origins`
- `synthetic_bundle_count`
- `agent_requested_query_count`
- `system_work_trace_entry_count`, `system_work_trace_stages`
- `llm_synthesis_status`, `llm_provider`, `llm_model_name`, `external_llm_provider_used`

## Synthetic Evidence Policy

- Synthetic/HyDE-like text is never promoted to grounded citation evidence.
- Synthetic-only outcomes must stay marked synthetic and be treated as non-grounded support.

## FAISS Secret Warning

- Default FAISS secret is insecure for production/release.
- Set `UAV_FAISS_SECRET` outside git with a strong secret before production use.
- Diagnostics may warn `default_faiss_secret_in_use` until configured.

## Combined RAG + LLM Limitation

- Independent checks succeeded:
  - real RAG smoke
  - Groq provider reachability
- Combined real-RAG + env-LLM smoke may still terminate under constrained resources.
- Treat this as runtime efficiency/loading limitation, not confirmed backend decision-logic failure.

## Manual Validation Commands

Do not run these automatically in this stage unless explicitly instructed.

```bash
python scripts/validate_stage2_rag_index.py
python scripts/run_stage2_rag_diagnostic.py --run-quality
python scripts/run_stage2_pipeline_v2_smoke.py --use-real-rag
python scripts/check_groq_provider_reachability.py
python scripts/run_stage2_pipeline_v2_smoke.py --use-real-rag --use-env-llm
python scripts/run_backend_trace_validation.py
```

The new `run_backend_trace_validation.py` command emits concise JSON across 15 backend phases:
- request contract -> core validation -> raw feature assembly -> Stage1/SHAP -> Stage2/RAG/agent/decision/LLM -> report -> trace -> persistence -> final API contract.
