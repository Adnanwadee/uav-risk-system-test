# Smart Skies — UAV Risk Assessment System
https://smart-skies.com/


Smart Skies is a hybrid AI decision-support system for UAV pre-flight risk assessment. It evaluates a mission before takeoff and returns an operational decision posture such as **GO**, **CAUTION**, or **NO-GO**.

The system is designed as a decision-support framework, not as a single black-box model. It separates machine-learning risk prediction, evidence retrieval, agentic operational analysis, deterministic decision policy, and optional LLM report synthesis.

---

## 1. System Architecture

```text
Raw Frontend Input
  -> FastAPI Backend
  -> Core Validation / Hard Veto
  -> Raw Feature Assembly
  -> Stage 1 ML Risk Model
  -> SHAP Risk Drivers
  -> Stage2PipelineV2
  -> RAG Evidence Retrieval
  -> OperationalAgentV2
  -> DecisionEngine
  -> Optional LLM Synthesis
  -> Operational Report
  -> Local Persistence
  -> API / Frontend Response
```

### Authority boundaries

| Component | Role | Final authority? |
|---|---|---:|
| Core validation / hard veto | Blocks structurally unsafe or invalid missions early | Yes, for hard veto cases |
| Stage 1 ML | Produces risk class, probabilities, and SHAP risk drivers | No |
| RAG | Retrieves scenario-specific aviation evidence and citations | No |
| OperationalAgentV2 | Performs operational analysis using ML + RAG context | No |
| DecisionEngine | Computes final decision, score, confidence, and required actions | Yes |
| LLM synthesis | Generates readable interpretation/report wording when enabled | No |

---

## 2. Repository / Submission Note

This repository uses **Git LFS** for large local runtime artifacts, especially local embedding/reranker model files and some RAG artifacts.

Do **not** rely on GitHub's normal **Download ZIP** as the only full runtime package unless LFS objects are explicitly included. A normal source ZIP may contain Git LFS pointer files instead of real model files.

To get a complete local runtime copy, use:

```bash
git clone https://github.com/Adnanwadee/uav-risk-system-test.git
cd uav-risk-system-test
git lfs install
git lfs pull
```

Verify that no LFS pointer files remain:

```bash
grep -RIl "version https://git-lfs.github.com/spec/v1" . \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=node_modules \
  --exclude-dir=__pycache__
```

Expected result after a correct LFS pull: **no output**.

If this repository remains private, the evaluator must be added as a collaborator or the repository must be made public before submission review.

---

## 3. Main Project Structure

```text
.
├── artifacts/                         # Stage 1 ML artifacts and datasets
├── docs/                              # API, frontend, and RAG documentation
├── examples/frontend/                 # Example API request/response payloads
├── frontend/index.html                # Current static frontend client
├── scripts/                           # Manual validation and diagnostic scripts
├── src/uav_risk/api/                  # FastAPI app, routes, schemas, storage
├── src/uav_risk/core/                 # Raw contracts, validation, feature assembly
├── src/uav_risk/ml/                   # Stage 1 model loading/inference contract
├── src/uav_risk/stage2/               # RAG, agent, decision engine, reporting, LLM synthesis
├── tests/                             # Unit and contract tests
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

The current frontend path is:

```text
frontend/index.html
```

The old Streamlit entrypoint under `ui/app.py` is legacy and should not be used as final UI readiness evidence.

---

## 4. Requirements

Recommended runtime:

- Python 3.10 to 3.12
- Git LFS
- Local CPU runtime is supported
- Optional Groq API key for external LLM synthesis

Dependencies are defined in:

```text
pyproject.toml
```

Install locally:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

---

## 5. Environment Configuration

Create a local `.env` file from the provided template:

```bash
cp .env.example .env
```

Minimum local configuration:

```env
LLM_ENABLED=false
LLM_PROVIDER=groq
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=replace_me
UAV_FAISS_SECRET=replace_with_strong_secret
UAV_RAG_BASE_DIR=src/uav_risk/stage2/knowledge
UAV_RAG_INDEX_DIR=src/uav_risk/stage2/knowledge/vectdb
UAV_RAG_DOCS_DIR=src/uav_risk/stage2/docs
UAV_RAG_MODELS_DIR=src/uav_risk/stage2/knowledge/models
```

LLM synthesis is optional. If `LLM_ENABLED=false` or no valid provider key is available, the system keeps the deterministic ML/RAG/Agent/DecisionEngine path and disables or falls back from external LLM generation.

Do not commit `.env` or real API keys.

---

## 6. Run Backend

After installation:

```bash
uvicorn uav_risk.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Alternative without installing the package:

```bash
PYTHONPATH=src uvicorn uav_risk.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Expected shape:

```json
{
  "status": "ok",
  "service": "uav-risk-api",
  "ml_bundle_loaded": true
}
```

---

## 7. Run Frontend

The current frontend is a static HTML client:

```bash
cd frontend
python -m http.server 3000
```

Open:

```text
http://127.0.0.1:3000
```

In the frontend configuration panel, use:

```text
API Base URL: http://127.0.0.1:8000
Assessment Mode: Real
User ID: user_1
```

The backend CORS configuration supports local frontend origins on port `3000` and `3001`.

---

## 8. API Endpoint Map

### Health and metadata

```text
GET /health
GET /model/metadata
GET /features/raw-schema
GET /features/profile-fields
GET /features/scenario-fields
GET /features/secondary-overrides
```

### Profiles

```text
POST   /users/{user_id}/profiles
GET    /users/{user_id}/profiles
GET    /users/{user_id}/profiles/{profile_id}
PUT    /users/{user_id}/profiles/{profile_id}
DELETE /users/{user_id}/profiles/{profile_id}
```

### Assessments

```text
POST /users/{user_id}/profiles/{profile_id}/assessments
POST /users/{user_id}/profiles/{profile_id}/assessments/stage2
GET  /users/{user_id}/assessments
GET  /users/{user_id}/assessments?profile_id={profile_id}
GET  /users/{user_id}/assessments/{assessment_id}
```

The recommended final endpoint for the application track demo is:

```text
POST /users/{user_id}/profiles/{profile_id}/assessments/stage2
```

---

## 9. Example API Usage

Create a profile:

```bash
curl -sS -X POST http://127.0.0.1:8000/users/user_1/profiles \
  -H "Content-Type: application/json" \
  -d @examples/frontend/create_profile_request.json
```

Run a Stage 2 assessment:

```bash
curl -sS -X POST http://127.0.0.1:8000/users/user_1/profiles/profile_1/assessments/stage2 \
  -H "Content-Type: application/json" \
  -d @examples/frontend/stage2_assessment_request.json
```

List persisted assessments:

```bash
curl -sS http://127.0.0.1:8000/users/user_1/assessments
```

Read a persisted assessment:

```bash
curl -sS http://127.0.0.1:8000/users/user_1/assessments/<assessment_id>
```

---

## 10. Input Contract

The frontend sends raw input only.

Allowed frontend inputs:

- Raw drone profile fields
- Raw scenario fields
- Raw `secondary_overrides.values`
- Optional `operator_notes`

The frontend must not send:

- Processed model features
- One-hot encoded columns
- ML predictions
- SHAP values
- Transformed 198-feature vectors
- Legacy `MasterFlightPayload`

The backend owns validation, feature assembly, ML inference, SHAP computation, RAG retrieval, agent analysis, final decision policy, reporting, and persistence.

---

## 11. Stage 2 Response Summary

Top-level response fields:

```text
status
user_id
profile_id
assessment_id
created_at
persisted
persistence_status
system_work_trace
warnings
errors
stage1
stage2
diagnostics
```

Key nested sections:

```text
stage1.ml                 # predicted class, probabilities, feature counts
stage1.shap               # top SHAP risk drivers
stage2.rag                # evidence bundles, citations, coverage, reranker status
stage2.agent              # recommendation, findings, actions, tool trace
stage2.decision           # final decision, score, confidence, reasons, required actions
stage2.llm_synthesis      # optional narrative synthesis
stage2.report             # structured and markdown report
stage2.policy             # decision weights and thresholds
diagnostics               # runtime flags and backend readiness metadata
```

When structural hard-veto triggers:

```text
status = blocked
stage1.ml = null
final_decision = no_go
```

No fabricated ML/RAG certainty is inserted for blocked requests.

---

## 12. RAG Runtime Notes

The RAG layer is designed to retrieve scenario-specific aviation evidence from the indexed knowledge base.

Current indexed corpus metadata:

```text
Expected sources: 9
Indexed sources: 9
Dense chunks: 3598
Canonical index directory: src/uav_risk/stage2/knowledge/vectdb
```

Full RAG runtime requires the local embedding/reranker model files and sparse index. These are LFS-managed in this repository, so they must be pulled with Git LFS or provided in a complete artifact ZIP.

Manual RAG validation commands:

```bash
python scripts/validate_stage2_rag_index.py
python scripts/run_stage2_rag_diagnostic.py --run-quality
python scripts/run_stage2_pipeline_v2_smoke.py --use-real-rag
```

Optional external LLM validation:

```bash
python scripts/check_groq_provider_reachability.py
python scripts/run_stage2_pipeline_v2_smoke.py --use-real-rag --use-env-llm
```

The combined real-RAG + external-LLM path depends on available memory and valid provider credentials.

---

## 13. Validation Commands

Lightweight checks:

```bash
python -m compileall src
python -c "from uav_risk.api.main import app; print(app.title)"
curl http://127.0.0.1:8000/health
```

Full test suite:

```bash
pytest
```

Backend trace validation:

```bash
python scripts/run_backend_trace_validation.py
```

---

## 14. Docker Usage

Build and run backend only:

```bash
docker build -t smart-skies-backend:latest .
docker run --rm -p 8000:8000 --env-file .env smart-skies-backend:latest
```

Run with Docker Compose:

```bash
docker compose up --build
```

Then open the static frontend at:

```text
http://localhost
```

Set the frontend API base URL to:

```text
http://localhost:8000
```

Docker builds require the runtime files to be present locally. If the repository was downloaded as a source ZIP containing LFS pointers, run from a full Git LFS clone instead.

---

## 15. CI/CD Workflows

The repository contains multiple GitHub Actions workflows:

| Workflow | Purpose |
|---|---|
| `functional-ci.yml` | Functional CI: checkout with LFS, install project, import app, compile source |
| `nonfunctional-ci.yml` | Non-functional CI: Ruff critical checks and Bandit scan |
| `docker-publish.yml` | Docker image build/publish workflow |
| `faiss-signature-verify.yml` | FAISS/RAG signature-related tests |
| `main.yml` / `broken.yml` | Manual fail-demo workflows for CI/CD demonstration only |

The fail-demo workflows are intentionally manual and should not be interpreted as production readiness failures.

---

## 16. Known Limitations

1. **Git LFS dependency**  
   Full local RAG runtime needs LFS-managed model artifacts. A normal source ZIP may contain pointer files.

2. **Local JSON persistence**  
   Persistence is suitable for graduation/demo use, not production multi-process database behavior.

3. **External LLM is optional**  
   The LLM does not make the decision. It only generates optional report wording when enabled and configured.

4. **Reranker depends on local runtime artifacts**  
   Reranker status is exposed in API diagnostics and depends on model availability.

5. **Legacy UI entrypoint**  
   `ui/app.py` is legacy. Use `frontend/index.html` for the current application demo.

---

## 17. Documentation

Additional documentation:

```text
docs/API_EXAMPLES.md
docs/FRONTEND_HANDOFF.md
docs/STAGE2_RAG_READINESS.md
examples/frontend/
```

---

## 18. Final Submission Checklist

Before submitting the code package:

```bash
git lfs install
git lfs pull
grep -RIl "version https://git-lfs.github.com/spec/v1" . \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=node_modules \
  --exclude-dir=__pycache__
python -m compileall src
```

Submission should include:

```text
1. GitHub repository link
2. Full code package generated after git lfs pull
3. Final report
4. Poster
5. CI/CD demonstration video link
```
