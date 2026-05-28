# Stage2 RAG Readiness

## Canonical Paths
- Docs corpus: `src/uav_risk/stage2/docs/`
- Models: `src/uav_risk/stage2/knowledge/models/`
- Canonical vector DB: `src/uav_risk/stage2/knowledge/vectdb/`

## Canonical Artifacts
- `dense_index.faiss`
- `dense_index.faiss.sig` (if enabled)
- `dense_mapping.json`
- `sparse_index.pkl`
- `metadata.json`

## Validation Commands
- `python scripts/validate_stage2_rag_index.py`
- `python scripts/run_stage2_rag_diagnostic.py --run-quality`
- `python scripts/run_stage2_pipeline_v2_smoke.py --use-real-rag`

## Metric Semantics
- `rag_quality_is_proven`: global RAG quality harness status across fixed expected/unsupported cases.
- `scenario_evidence_complete`: scenario-level completeness for current pipeline smoke run.
- `insufficient_evidence_count`: count of scenario bundles returned with `insufficient_evidence`.

## Current Evidence Safety Rules
- LLM output is not evidence in current retrieval path.
- HyDE-generated text is not evidence.
- Unsupported/out-of-domain queries must return `insufficient_evidence`.
- Citation provenance must come from retrieved chunk metadata (`source_id`, source filename/title, page/chunk where available).

## Active Runtime Files
- `src/uav_risk/stage2/rag/config_v3.py`
- `src/uav_risk/stage2/rag/rag_core_v3.py`
- `src/uav_risk/stage2/rag/hybrid_retriever.py`
- `src/uav_risk/stage2/rag/adapter.py`
- `src/uav_risk/stage2/rag/quality.py`
- `src/uav_risk/stage2/rag/runtime_diagnostics.py`
- `src/uav_risk/stage2/pipeline_v2.py`
- `src/uav_risk/stage2/reporting.py`

## Compatibility / Legacy / Optional
- Compatibility bridge: `src/uav_risk/stage2/rag/schemas.py`
- Legacy do-not-use in v2 runtime path: `src/uav_risk/stage2/pipeline.py`, `src/uav_risk/stage2/agent/ace_agent.py`
- Optional future (not used in current evidence path):
  - `src/uav_risk/stage2/rag/groq_llm.py`
  - `src/uav_risk/stage2/rag/hyde_pipeline.py`
  - `src/uav_risk/stage2/rag/prompts_v3.py`

## Non-Canonical Home Cache Cleanup Note
If stale indices exist under `/home/vscode/.uav_rag/indices`, they are non-canonical for current default runtime.
Manual cleanup (optional):
- `rm -rf /home/vscode/.uav_rag/indices`

Run this only after confirming diagnostics and smoke use canonical `src/uav_risk/stage2/knowledge/vectdb/` paths.


## Score / Rank / Confidence Visibility
- Citations are sorted by highest score first (descending by `final_score` fallback chain).
- Each citation exposes metadata fields used by downstream agent/report tooling:
  - `rank` (1-based)
  - `final_score`
  - `retrieval_score`
  - `top_score` (alias for visibility)
  - `dense_score`
  - `sparse_score`
  - `source_match_score`
  - `confidence_label` (`HIGH` / `MEDIUM` / `LOW` / `VERY LOW`)
- Citation provenance remains required:
  - `source_id`, `source_filename/source_title`, `chunk_id`, `page_start/page_end` when available.

## Query Guidance (Generic vs Refined)
Generic scenario phrasing can honestly abstain when evidence safety gates reject weak candidates.

- Generic (may abstain):
  - `uav operational guidance for risk class Medium Risk`
- Refined (expected source-aware support):
  - `special condition UAS medium risk operational limitations`
  - `SORA medium risk UAS operational safety objectives`

- Generic airspace query (now source-intent routed to Part107/AC107 family):
  - `UAS operation controlled airspace restricted area no-fly zone guidance`
- Refined vlos/weather queries:
  - `Part 107 visual line of sight small unmanned aircraft operation`
  - `small UAS preflight weather assessment wind conditions guidance`

## Legacy / Quarantine Notes
- Current default runtime does **not** use `/home/vscode/.uav_rag/indices`.
- Optional future components are kept but not active in evidence retrieval path:
  - `groq_llm.py`, `hyde_pipeline.py`, `prompts_v3.py`
- Legacy do-not-use runtime files remain quarantined by policy and inventory:
  - `src/uav_risk/stage2/pipeline.py`
  - `src/uav_risk/stage2/agent/ace_agent.py`
