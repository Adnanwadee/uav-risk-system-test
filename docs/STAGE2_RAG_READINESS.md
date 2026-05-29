# Stage2 RAG Readiness

## Current Readiness Status

- RAG source coverage status: **9 expected / 9 indexed**.
- RAG quality is **proven by diagnostics**.
- Real RAG smoke completed independently.
- Groq reachability completed independently.
- Combined real-RAG + env-LLM smoke may terminate from runtime memory/resource pressure.

## Runtime Metadata Exposed in API

Stage2 response/diagnostics expose:
- `rag_quality_is_proven`
- `scenario_evidence_status`
- `corpus_coverage_status`
- `expected_source_count`
- `indexed_source_count`
- `missing_sources`
- `source_ids`, `source_titles`
- reranker status fields:
  - `reranker_configured`
  - `reranker_available`
  - `reranker_used`
  - `reranker_reason`

## Synthetic Evidence Policy

- Synthetic/HyDE-like text is never promoted to grounded citation evidence.
- Synthetic-only outcomes must stay marked synthetic and be treated as non-grounded support.

## FAISS Secret Warning

- Default FAISS secret is insecure for production/release.
- Set `UAV_FAISS_SECRET` outside git with a strong secret before production use.

## Combined RAG + LLM Limitation

- Independent checks succeeded:
  - real RAG smoke
  - Groq provider reachability
- Combined real-RAG + env-LLM smoke may terminate due to resource pressure.
- Treat this as runtime efficiency/loading limitation, not confirmed backend decision-logic failure.

## Manual Validation Commands

Do not run these automatically in this stage unless explicitly instructed.

```bash
python scripts/validate_stage2_rag_index.py
python scripts/run_stage2_rag_diagnostic.py --run-quality
python scripts/run_stage2_pipeline_v2_smoke.py --use-real-rag
python scripts/check_groq_provider_reachability.py
python scripts/run_stage2_pipeline_v2_smoke.py --use-real-rag --use-env-llm
```

Note: The combined `--use-real-rag --use-env-llm` command may terminate until runtime caching/reuse is improved.
