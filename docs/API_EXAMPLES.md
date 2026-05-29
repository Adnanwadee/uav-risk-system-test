# Smart Skies — API Examples

## Primary Frontend Endpoint
`POST /users/{user_id}/profiles/{profile_id}/assessments/stage2`

Use this endpoint for the full backend flow:
Core validation -> Stage1 ML + SHAP -> Stage2 RAG -> OperationalAgentV2 -> DecisionEngine -> optional LLM synthesis -> report output.

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

### Example Request
```bash
curl -sS -X POST http://localhost:8000/users/user_1/profiles/profile_1/assessments/stage2 \
  -H "Content-Type: application/json" \
  -d @examples/frontend/stage2_assessment_request.json
```

## Response Examples
- Completed: `examples/frontend/stage2_assessment_response_example.json`
- Blocked hard-veto: `examples/frontend/blocked_stage2_response_example.json`

## Response Top-Level Shape
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

## Stage2 Highlights
- `stage2.rag`: evidence/citation retrieval status and bundle summaries.
- `stage2.agent`: findings, action items, `working_memory_summary`, sanitized `tool_trace`, and sanitized `system_work_trace` (Transparency Trace).
- `stage2.decision`: deterministic final authority (`final_decision`, `decision_score`, reasons, required actions).
- `stage2.llm_synthesis`: interpretation/synthesis only.
- `stage2.report`: structured report sections plus markdown rendering.

## Decision, LLM, and RAG Authority
- Decision Engine is backend-authoritative for `stage2.decision`.
- LLM synthesis does not override `final_decision`, `decision_score`, evidence support status, or citations.
- RAG provides evidence/citations and provenance support, not final authority.

## Blocked Behavior
When structural hard-veto blocks execution:
- `status = "blocked"`
- `stage1.ml = null`
- Decision is forced safe (`no_go`) with blocking reasons
- No fabricated ML/RAG outputs are injected

## Persistence Status
`assessment_id` is generated per response for traceability.
Assessment persistence (GET/LIST history retrieval) is not implemented in this stage.

## Public-Safe Trace Policy
System Work Trace / Transparency Trace is public-safe metadata:
- Includes step summaries, statuses, and evidence IDs
- Excludes hidden chain-of-thought, raw prompts/completions, raw tool history, and secrets
