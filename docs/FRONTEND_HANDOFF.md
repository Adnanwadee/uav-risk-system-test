# Smart Skies — Frontend Handoff

## 1. Project Overview
Smart Skies is a backend service for UAV operational risk assessment. It combines:
- Core input validation and structural checks
- ML risk model (Stage1) with SHAP explanations
- Retrieval-Augmented Generation (RAG) evidence retrieval
- Deterministic Operational Agent analyses
- Weighted Decision Engine consolidation
- LLM deterministic fallback synthesis for human-friendly summaries

The backend is intended to serve drone operators and dashboard frontends that need an evidence-driven operational decision and explanation bundle.

## 2. Recommended Frontend Flow
1. Create or select a drone profile.
2. Submit a scenario assessment for that profile.
3. Prefer the Stage2 endpoint for the full AI report (see below).
4. Render a decision card first (go / caution / no_go) with score/confidence.
5. Show ML/SHAP drivers, RAG evidence/citations, Agent findings + action items, and the LLM summary and operational report.
6. Surface warnings/limitations and a developer diagnostics panel.

## 3. Available Endpoints (summary)
- Health
  - `GET /health`

- Feature metadata
  - `GET /model/metadata`
  - `GET /features/raw-schema`
  - `GET /features/profile-fields`
  - `GET /features/scenario-fields`
  - `GET /features/secondary-overrides`

- Profile endpoints
  - `POST /users/{user_id}/profiles`
  - `GET /users/{user_id}/profiles`
  - `GET /users/{user_id}/profiles/{profile_id}`
  - `PUT /users/{user_id}/profiles/{profile_id}`
  - `DELETE /users/{user_id}/profiles/{profile_id}`

- Assessment endpoints
  - `POST /users/{user_id}/profiles/{profile_id}/assessments` (Stage1: Core/ML/SHAP)
  - `POST /users/{user_id}/profiles/{profile_id}/assessments/stage2` (Stage2: full AI pipeline + report)

## 4. Which Endpoint Should Frontend Use?
- Use the Stage2 endpoint for the final UI experience (full decision, evidence, agent, and report).
- Keep the Stage1 endpoint for fast, lightweight ML-only diagnostics or developer debug screens.

## 5. Stage2 Request Shape
- Path params: `user_id`, `profile_id`
- Body (JSON):
  - `scenario` — raw scenario fields (use the scenario raw schema from the backend; send only raw fields)
  - `secondary_overrides` — `{ "values": { <raw_feature_name>: <numeric_scalar> } }` (optional)
  - `operator_notes` — optional free-text

Important: Do not send processed one-hot/model features or prediction outputs; send raw profile/scenario/override inputs only.

## 6. Stage2 Response Shape (top-level)
Top-level keys returned in Stage2 response:
- `status` — e.g., `completed`, `blocked`, `degraded`, `failed`
- `user_id`
- `profile_id`
- `assessment_id` — generated per response for traceability (not persisted for retrieval in current stage)
- `warnings` — array
- `errors` — array
- `stage1` — Stage1/ML/SHAP snapshot
- `stage2` — RAG/Agent/Decision/LLM/Report bundle
- `diagnostics` — system-level diagnostics

### stage1
- `ml.predicted_class`
- `ml.probabilities`
- `ml.raw_feature_count`
- `ml.processed_feature_count`
- `shap.top_features` — list of SHAP driver objects (feature, value, importance)

### stage2.rag
- `retrieval_usable` (bool)
- `rag_quality_is_proven` (bool)
- `evidence_bundle_count` (int)
- `insufficient_evidence_count` (int)
- `evidence_bundle_details` (list)

### stage2.agent
- `recommendation` (e.g., `caution`)
- `findings` (list)
- `action_items` (list)
- `limitations` (list)
- `working_memory_summary` (public-safe summary only)
- `tool_trace` (sanitized public-safe tool trace)
- `system_work_trace` (Transparency Trace; structured, summarized, public-safe)

### stage2.decision
- `final_decision` (`go` / `caution` / `no_go`)
- `decision_score` (0.0–1.0)
- `confidence_level` (low/medium/high)
- `stage_weights` (breakdown by component)
- `stage_contributions` (list)
- `decision_reasons` (list)
- `blocking_reasons` (list)
- `required_actions` (list)
- `limitations` (list)

### stage2.llm_synthesis
- `status` (e.g., `fallback`)
- `executive_summary`
- `operational_interpretation`
- `decision_explanation`
- `key_risk_drivers`
- `mitigation_narrative`
- `consistency_warnings`

### stage2.report
- `structured` (object) — structured operational report
- `markdown` (string) — optional markdown rendering of the report

### diagnostics
- `path_resolution_status`
- `index_provenance_status`
- `retrieval_usable`
- `rag_quality_is_proven`
- `scenario_evidence_complete`
- `llm_mode` (e.g., `fallback`)
- `external_llm_provider_used` (bool)

## 7. Status Meanings
- `completed` — pipeline completed successfully
- `blocked` — structural hard veto prevented ML/Stage2 execution
- `degraded` — partial results (e.g., RAG unavailable) but a decision returned
- `failed` — unrecoverable error

## 8. Decision Meanings
- `go` — mission appears acceptable under provided data
- `caution` — mission requires review or mitigations
- `no_go` — mission should not proceed

## 9. Confidence / Score Meaning
- `decision_score` is a normalized operational concern signal (0.0 = low concern, 1.0 = high concern).
- `confidence_level` is an overall confidence heuristic (low/medium/high).
- These are operational guidance only and not legal authority.

## 10. Rendering Recommendations (UI panels)
- Final Decision Card (decision + score + confidence)
- Required Actions / Checklist
- ML Prediction + Probabilities
- SHAP Top Drivers (expandable list)
- RAG Evidence / Citations (click to view document preview/page)
- Agent Findings and Action Items
- LLM Operational Summary (concise paragraph)
- Limitations / Warnings
- Diagnostics (developer mode)

## 11. Evidence / Citation Rendering
- Surface the source document and a short quote or page hint when available.
- Treat RAG as evidence augmentation — do not over-interpret as legal proof.
- If insufficient evidence, highlight the limitation clearly in the UI.

## 12. LLM Synthesis Rendering
- LLM text is deterministic fallback by default unless a provider is configured.
- It is a synthesis of existing signals, not a primary source of evidence.
- Display as human-readable summary; always link back to ML/Agent/RAG sections.

## 12A. Public-Safe Trace Rendering
- Treat `working_memory_summary`, `tool_trace`, and `system_work_trace` as frontend-displayable transparency metadata.
- Do not expect hidden chain-of-thought, raw prompts/completions, or secret-bearing internal logs in these fields.
- Render summaries, statuses, evidence/citation IDs, and action/finding references.

## 13. Blocked Response Behavior
- When `status == "blocked"`:
  - `ml` is `null` (no model output)
  - `decision.final_decision` can be `no_go`
  - Show blocking reasons and do not display fabricated ML/RAG results

## 14. Known Limitations
- `assessment_id` is generated in responses but persistence/GET/LIST retrieval is not implemented yet.
- Stage2 results are not persisted in storage.
- RAG adapter may be missing if local index files are absent.
- LLM is deterministic fallback (no external provider configured by default).
- Raw feature map is trimmed from responses to ensure JSON-safe metadata.
- FAISS secret warning may appear in development — treat as config issue.

## 15. Minimal Frontend Acceptance Checklist
- Can create/select a profile
- Can submit Stage2 assessment
- Can display final decision, score, and confidence
- Can display ML/SHAP drivers
- Can display RAG evidence and citations
- Can display Agent findings and action items
- Can display LLM synthesis and the operational report
- Can correctly handle `blocked` responses


----
Generated by Smart Skies backend docs generator. Keep this in sync with backend changes.
