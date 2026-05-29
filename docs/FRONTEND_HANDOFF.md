# Smart Skies — Frontend Handoff

## 1. Product Identity

Smart Skies is an evidence-grounded hybrid AI decision framework for UAV flight risk management.

It is not only:
- an ML API
- a generic RAG chatbot
- an LLM decision maker

Frontend should clearly present separate subsystem outputs:
- ML risk signal
- SHAP risk drivers
- RAG evidence/citations
- Agent recommendation
- DecisionEngine final decision
- Optional LLM synthesis
- System Work Trace
- Persistence/history lifecycle

## 2. Recommended Frontend Flow

1. Create or select a profile (`user_id`, `profile_id`).
2. Submit Stage2 assessment with raw scenario + optional overrides + notes.
3. Render top decision header first.
4. Render ML, RAG, Agent, DecisionEngine, LLM, Trace cards.
5. Save/use `assessment_id` for reload/history.
6. Use history endpoints for timeline views and profile-filtered history.

## 3. Endpoint Map

- `POST /users/{user_id}/profiles`
- `GET /users/{user_id}/profiles`
- `GET /users/{user_id}/profiles/{profile_id}`
- `PUT /users/{user_id}/profiles/{profile_id}`

- `POST /users/{user_id}/profiles/{profile_id}/assessments/stage2`
- `GET /users/{user_id}/assessments`
  - Optional query: `profile_id`
- `GET /users/{user_id}/assessments/{assessment_id}`

## 4. Input Contract (Raw-First)

Send only:
- raw profile data
- raw scenario fields
- raw `secondary_overrides.values`
- `operator_notes`

Never send:
- processed model features
- one-hot columns
- transformed feature vectors
- model predictions
- legacy `MasterFlightPayload`

## 5. Stage2 Top-Level Response Contract

Top-level keys:
- `status`
- `user_id`
- `profile_id`
- `assessment_id`
- `created_at`
- `persisted`
- `persistence_status`
- `system_work_trace`
- `warnings`
- `errors`
- `stage1`
- `stage2`
- `diagnostics`

## 6. Rendering Contract (UI Cards)

### Header / Status Card
Render:
- `status`
- `assessment_id`
- `persisted` + `persistence_status`
- `stage2.decision.final_decision`
- `stage2.decision.decision_score`
- `stage2.decision.confidence_level`

### ML Card
Render:
- `stage1.ml.predicted_class`
- `stage1.ml.probabilities`
- `stage1.shap.top_features`

Display note:
- ML is a risk signal, not legal authority.

### RAG Evidence Card
Render:
- `stage2.rag.evidence_bundle_count`
- `stage2.rag.citations`
  - `source_title`
  - page/chunk fields where available
  - `support_status`
  - `retrieval_origin`
  - `synthetic`
- `stage2.rag.scenario_evidence_status`
- `stage2.rag.retrieval_usable`
- `stage2.rag.rag_quality_is_proven`
- `stage2.rag.corpus_coverage_status`
- `stage2.rag.expected_source_count`
- `stage2.rag.indexed_source_count`
- `stage2.rag.missing_sources_count`
- `stage2.rag.reranker_configured`
- `stage2.rag.reranker_available`
- `stage2.rag.reranker_used`
- `stage2.rag.reranker_reason`

Insufficient evidence behavior:
- If insufficient, highlight limitations and avoid fake certainty.

### Agent Card
Render:
- `stage2.agent.recommendation`
- `stage2.agent.findings`
- `stage2.agent.action_items`
- evidence references in findings/actions
- `stage2.agent.selected_rag_queries`
- `stage2.agent.skipped_rag_queries`
- sanitized `stage2.agent.tool_trace`
- `stage2.agent.working_memory_summary`

### DecisionEngine Card
Render:
- `stage2.decision.final_decision`
- `stage2.decision.decision_score`
- `stage2.decision.confidence_level`
- `stage2.decision.stage_contributions`
- `stage2.decision.required_actions`
- `stage2.decision.blocking_reasons`

Display note:
- DecisionEngine is final authority.

### LLM Synthesis Card
Render only when present (or status not-generated/fallback):
- `stage2.llm_synthesis.status` (or compatibility alias `synthesis_status`)
- `provider`, `model_name`, `external_provider_used`
- narrative fields (`executive_summary`, `operational_interpretation`, `decision_explanation`, `mitigation_narrative`)
- `consistency_warnings`

Display note:
- LLM does not decide.

### System Work Trace Card
Render as timeline/table:
- `stage`
- `tool_name`
- `status`
- `input_summary`
- `output_summary`
- `evidence_ids`
- `warnings`

Do not expect hidden reasoning/raw prompts/raw completions.

## 7. Report Rendering Guide

Use `stage2.report.sections` in stable order:
1. Executive Summary
2. Input Summary
3. ML Assessment
4. SHAP Risk Drivers
5. RAG Evidence
6. Agent Operational Analysis
7. LLM Synthesis
8. DecisionEngine Final Decision
9. Required Actions
10. System Work Trace
11. Diagnostics

Optional markdown rendering is in `stage2.report.markdown`.

## 8. Persistence / History Guide

### List endpoint
`GET /users/{user_id}/assessments` (optional `profile_id` filter)

List item fields:
- `assessment_id`, `user_id`, `profile_id`, `created_at`, `status`
- `final_decision`, `decision_score`, `confidence_level`
- `summary`

### Get endpoint
`GET /users/{user_id}/assessments/{assessment_id}`

Returns full persisted assessment record for view reload without rerunning pipeline:
- `stage1`, `stage2`, `report`, `system_work_trace`, `diagnostics`, warnings/errors

## 9. RAG / Agent / Decision / LLM Separation

- RAG retrieves evidence and citation provenance.
- Agent performs evidence-grounded operational analysis.
- DecisionEngine deterministically computes final decision.
- LLM is optional synthesis/wording only.

## 10. Known Limitations

1. FAISS secret must be configured outside git (`UAV_FAISS_SECRET`) for production.
2. Combined real-RAG + env-LLM smoke may terminate under resource pressure.
3. Runtime may reload heavy resources repeatedly (future caching/reuse improvement).
4. Legacy ACE/simulation files are not the current runtime path.
5. Persistence is local JSON (not production multi-process DB behavior).
6. Reranker production usage depends on local model/runtime availability.
7. LLM is optional and never decision-authoritative.

## 11. Blocked Behavior

When `status == "blocked"`:
- `stage1.ml` is `null`
- decision may be forced to `no_go`
- show blocking reasons
- do not render fabricated ML/RAG certainty
