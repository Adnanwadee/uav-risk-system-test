# Smart Skies — API Examples

## Endpoint Map

- Create profile: `POST /users/{user_id}/profiles`
- Stage2 full assessment: `POST /users/{user_id}/profiles/{profile_id}/assessments/stage2`
- List persisted assessments: `GET /users/{user_id}/assessments`
  - Optional query: `profile_id`
- Get persisted assessment by id: `GET /users/{user_id}/assessments/{assessment_id}`

## Frontend Input Contract (Raw-First)

Frontend sends only:
- Raw drone profile fields (via profile endpoints)
- Raw `scenario` fields
- Raw `secondary_overrides.values`
- `operator_notes`

Frontend must not send:
- Processed features
- One-hot columns
- Model predictions
- Transformed 198-feature vectors
- Legacy `MasterFlightPayload`

## Create Profile Example

```bash
curl -sS -X POST http://localhost:8000/users/user_1/profiles \
  -H "Content-Type: application/json" \
  -d @examples/frontend/create_profile_request.json
```

## Create Stage2 Assessment Example

```bash
curl -sS -X POST http://localhost:8000/users/user_1/profiles/profile_1/assessments/stage2 \
  -H "Content-Type: application/json" \
  -d @examples/frontend/stage2_assessment_request.json
```

## Get Assessment By ID Example

```bash
curl -sS "http://localhost:8000/users/user_1/assessments/<assessment_id>"
```

## List Assessments Example

```bash
curl -sS "http://localhost:8000/users/user_1/assessments"
curl -sS "http://localhost:8000/users/user_1/assessments?profile_id=profile_1"
```

## Response Examples

- Completed: `examples/frontend/stage2_assessment_response_example.json`
- Blocked hard-veto: `examples/frontend/blocked_stage2_response_example.json`

## Stage2 Response Interpretation

Top-level fields include:
- `status`, `user_id`, `profile_id`, `assessment_id`
- `created_at`, `persisted`, `persistence_status`
- `system_work_trace`
- `warnings`, `errors`
- `stage1`, `stage2`, `diagnostics`

`stage1`:
- ML signal (`predicted_class`, `probabilities`, feature counts)
- SHAP top features

`stage2.rag`:
- Evidence bundles, citations, support status
- Insufficient-evidence signals
- Coverage status (`corpus_coverage_status`, expected/indexed/missing counts)
- Reranker runtime status
- Retrieval origin and synthetic flags where available

`stage2.agent`:
- Recommendation, findings, action items
- Selected/skipped queries
- Sanitized `tool_trace`
- `working_memory_summary`

`stage2.decision`:
- `final_decision`, `decision_score`, `confidence_level`
- Stage weights/contributions
- Decision reasons / blocking reasons / required actions

`stage2.llm_synthesis`:
- `status` + compatibility alias `synthesis_status`
- Provider/model/runtime flags
- Narrative sections and consistency warnings

`stage2.report`:
- `sections` (structured report)
- `markdown`

## Persisted Contract Notes

Persisted assessment records include:
- `assessment_id`, `user_id`, `profile_id`, `created_at`, `status`
- `final_decision`, `decision_score`, `confidence_level`
- `stage1`, `stage2`, `report`, `system_work_trace`, `diagnostics`
- Public-safe normalized `warnings` and `errors`

History list items include:
- `assessment_id`, `user_id`, `profile_id`, `created_at`, `status`
- `final_decision`, `decision_score`, `confidence_level`, `summary`

## Authority Boundaries

- DecisionEngine is final authority for `final_decision` and `decision_score`.
- LLM synthesis is interpretation-only and never decides.
- RAG is evidence retrieval/provenance support and never decides.

## Blocked Behavior

When structural hard-veto triggers:
- `status = "blocked"`
- `stage1.ml = null`
- Decision is forced safe (`no_go`)
- No fabricated ML/RAG outputs are inserted
