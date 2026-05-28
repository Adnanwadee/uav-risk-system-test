# Smart Skies — API Examples

This page provides practical examples for frontend integration.

## Create Profile (curl)
```
curl -sS -X POST http://localhost:8000/users/user_1/profiles \
  -H "Content-Type: application/json" \
  -d @examples/frontend/create_profile_request.json
```

## Stage1 (ML-only) Assessment (curl)
```
curl -sS -X POST http://localhost:8000/users/user_1/profiles/profile_1/assessments \
  -H "Content-Type: application/json" \
  -d @examples/frontend/stage1_assessment_request.json
```

## Stage2 Full Assessment (curl)
```
curl -sS -X POST http://localhost:8000/users/user_1/profiles/profile_1/assessments/stage2 \
  -H "Content-Type: application/json" \
  -d @examples/frontend/stage2_assessment_request.json
```

## Example Completed Response Interpretation
- `status: completed` — full pipeline returned a decision and report.
- Inspect `stage1.ml` for model probabilities and `stage1.shap` for drivers.
- Inspect `stage2.rag` for evidence summary and `stage2.agent` for findings.
- `stage2.decision` contains `final_decision`, `decision_score`, and `confidence_level`.
- `stage2.llm_synthesis.status` will often be `fallback` in development.

## Example Blocked Response Interpretation
- `status: blocked` indicates a structural hard-veto.
- `ml` will be `null` and `decision.final_decision` may be `no_go`.
- Show blocking reasons (do not render ML/RAG results).

## Frontend Render Mapping (example)
- Decision Card: `stage2.decision.final_decision`, `stage2.decision.decision_score`, `stage2.decision.confidence_level`
- Risk Score: `stage1.ml.probabilities` or `stage2.decision.decision_score`
- SHAP Drivers: `stage1.shap.top_features`
- Evidence List: `stage2.rag.evidence_bundle_details`
- Agent Findings: `stage2.agent.findings`
- Action Items: `stage2.agent.action_items`
- Executive Summary: `stage2.llm_synthesis.executive_summary` or `stage2.report.markdown`

---
Keep examples in sync with the API responses returned by the backend.