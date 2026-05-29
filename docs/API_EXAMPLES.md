# Smart Skies — API Examples

## Primary Assessment Execution Endpoint
`POST /users/{user_id}/profiles/{profile_id}/assessments/stage2`

Use this endpoint for the full backend flow:
Core validation -> Stage1 ML + SHAP -> Stage2 RAG -> OperationalAgentV2 -> DecisionEngine -> optional LLM synthesis -> report output.

## Persistence Retrieval Endpoints
- `GET /users/{user_id}/assessments`
  - Optional query: `profile_id`
  - Returns persisted assessment summaries for the user (filtered when `profile_id` is provided).
- `GET /users/{user_id}/assessments/{assessment_id}`
  - Returns one persisted assessment record.

## Request Contract (Frontend)
Frontend sends only:
- Raw profile fields (created/stored through profile endpoints)
- Raw `scenario` fields
- Raw `secondary_overrides.values` scalar overrides
- `operator_notes` free text

Frontend must not send:
- Processed features
- One-hot columns
- Model predictions
- Full transformed 198-feature vectors
- Legacy `MasterFlightPayload`

### Create Stage2 Assessment
```bash
curl -sS -X POST http://localhost:8000/users/user_1/profiles/profile_1/assessments/stage2 \
  -H "Content-Type: application/json" \
  -d @examples/frontend/stage2_assessment_request.json
```

### List Persisted Assessments (User)
```bash
curl -sS "http://localhost:8000/users/user_1/assessments"
```

### List Persisted Assessments (User + Profile)
```bash
curl -sS "http://localhost:8000/users/user_1/assessments?profile_id=profile_1"
```

### Get Persisted Assessment By ID
```bash
curl -sS "http://localhost:8000/users/user_1/assessments/<assessment_id>"
```

## Response Examples
- Completed: `examples/frontend/stage2_assessment_response_example.json`
- Blocked hard-veto: `examples/frontend/blocked_stage2_response_example.json`

## Stage2 Response Top-Level Shape
```json
{
  "status": "completed|blocked|degraded|failed",
  "user_id": "...",
  "profile_id": "...",
  "assessment_id": "uuid",
  "warnings": [],
  "errors": [],
  "stage1": {},
  "stage2": {},
  "diagnostics": {}
}
```

## Persisted Assessment Record (Stored + GET)
Persisted records include public-safe fields such as:
- identity/timing: `assessment_id`, `user_id`, `profile_id`, `created_at`, `status`
- decision summary: `final_decision`, `decision_score`, `confidence_level`
- `stage1` ML/SHAP snapshot
- `stage2` RAG/Agent/Decision/LLM/report bundle
- `report`
- `system_work_trace`
- `diagnostics`
- normalized `warnings` and `errors`

## Decision, LLM, and RAG Authority
- Decision Engine is backend-authoritative for `stage2.decision`.
- LLM synthesis does not override `final_decision`, `decision_score`, evidence support status, or citations.
- RAG provides evidence/citations and provenance support, not final authority.

## Persistence Safety Guarantees
- Persisted records are public-safe by default.
- Hidden reasoning, raw prompts/completions, raw tool history, and secret-bearing keys are stripped.
- System Work Trace / Transparency Trace is stored as structured summary metadata only.

## Blocked Behavior
When structural hard-veto blocks execution:
- `status = "blocked"`
- `stage1.ml = null`
- Decision is forced safe (`no_go`) with blocking reasons
- No fabricated ML/RAG outputs are injected
