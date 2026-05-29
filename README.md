# Smart Skies

Smart Skies is an evidence-grounded hybrid AI decision framework for UAV flight risk management.

It combines:
- Core structural validation / hard-veto
- Stage1 ML risk signal + SHAP risk drivers
- Stage2 RAG evidence and citation retrieval
- OperationalAgentV2 operational analysis
- DecisionEngine deterministic final decision authority
- Optional LLM synthesis for interpretation/report wording
- Public-safe System Work Trace and persisted assessment lifecycle

## Canonical Backend Flow

`User Raw Input`
-> `API`
-> `Core Validation / Hard Veto`
-> `Raw Feature Assembly`
-> `Stage1 ML`
-> `SHAP`
-> `Stage2PipelineV2`
-> `Stage2RAGAdapter`
-> `OperationalAgentV2`
-> `DecisionEngine`
-> `Optional LLMOrchestrator`
-> `Reporting`
-> `Stage2AssessmentResponse`
-> `Persistence`
-> `Frontend`

## API Run

```bash
uvicorn uav_risk.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Current Backend Capabilities

- Raw-first frontend contract (frontend sends raw profile/scenario/secondary overrides/operator notes only).
- Stage2 full assessment endpoint with typed response contract.
- Public-safe persistence for Stage2 assessments.
- History/retrieval endpoints:
  - `GET /users/{user_id}/assessments`
  - `GET /users/{user_id}/assessments/{assessment_id}`
  - Optional `profile_id` filter on list endpoint.
- RAG diagnostics metadata in API response:
  - `rag_quality_is_proven`
  - corpus coverage (`expected_source_count`, `indexed_source_count`, missing sources)
  - reranker status fields (`reranker_configured`, `reranker_available`, `reranker_used`, `reranker_reason`)
- Synthetic evidence is marked and must not be treated as grounded evidence.

## Current Validation Status

- RAG source coverage status: 9 expected / 9 indexed.
- RAG quality is proven by diagnostics harness.
- Real RAG smoke succeeded independently.
- Groq reachability succeeded independently.
- Combined real-RAG + env-LLM smoke may terminate due to runtime memory/resource pressure.

## Known Limitations

1. FAISS secret:
   - Default FAISS secret is insecure.
   - Set `UAV_FAISS_SECRET` outside git for production/release.
2. Combined RAG + LLM runtime:
   - Combined real-RAG + env-LLM smoke can terminate from resource pressure.
   - Treat as runtime efficiency/loading limitation, not confirmed logic failure.
3. Runtime efficiency:
   - Possible repeated loading of embedding/reranker/RAG/LLM resources.
   - Future improvement: in-process caching/reuse.
4. Legacy files:
   - ACE stack and older simulation scripts are not the current runtime path.
5. Local JSON persistence:
   - Suitable for demo/graduation readiness, not a production multi-process database.
   - No pagination/retention/delete lifecycle in this stage.
6. Reranker:
   - Status fields are exposed, but live production use depends on local model availability/runtime state.
7. LLM/Groq:
   - LLM is optional and does not decide.

## Manual Validation Commands

Do not run these casually in automated checks; they are manual readiness commands.

```bash
python scripts/validate_stage2_rag_index.py
python scripts/run_stage2_rag_diagnostic.py --run-quality
python scripts/run_stage2_pipeline_v2_smoke.py --use-real-rag
python scripts/check_groq_provider_reachability.py
python scripts/run_stage2_pipeline_v2_smoke.py --use-real-rag --use-env-llm
```

Note: The combined `--use-real-rag --use-env-llm` smoke may terminate due to resource pressure until runtime caching is improved.

## Security / Repo Hygiene

- Do not commit secrets.
- Configure runtime secrets (for example `UAV_FAISS_SECRET`, provider keys) outside git.

## Documentation

- [Frontend Handoff](docs/FRONTEND_HANDOFF.md)
- [API Examples](docs/API_EXAMPLES.md)
- [Stage2 RAG Readiness](docs/STAGE2_RAG_READINESS.md)
