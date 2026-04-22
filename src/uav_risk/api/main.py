#api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict

from uav_risk.stage1.loader import load_stage1_artifacts
from uav_risk.stage1.infer import run_stage1_inference


from uav_risk.stage2.api import router as stage2_router

app = FastAPI(title="UAV Risk System")

app.include_router(stage2_router)

class ScenarioPayload(BaseModel):
    scenario: Dict[str, Any]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stage1/loaded")
def stage1_loaded():
    a = load_stage1_artifacts("artifacts")
    return {
        "loaded": True,
        "policy_keys": list(a.policy.keys())[:20],
        "preprocessor_type": type(a.preprocessor).__name__,
        "reg_model_type": type(a.reg_model).__name__,
        "clf_model_type": type(a.clf_model).__name__,
    }


@app.post("/stage1/infer")
def stage1_infer(payload: ScenarioPayload):
    """
    Accepts {"scenario": {...}} and returns Stage-1 FACTS JSON.
    """
    return run_stage1_inference(payload.scenario, artifacts_dir="artifacts")
@app.get("/stage1/expected-columns")
def stage1_expected_columns():
    art = load_stage1_artifacts("artifacts")

    pre = art.preprocessor

    cols = []

    # Case 1: ColumnTransformer
    if hasattr(pre, "transformers_"):
        for name, transformer, columns in pre.transformers_:
            if columns is None:
                continue
            if isinstance(columns, list):
                cols.extend(columns)

    return {
        "expected_columns": sorted(set(map(str, cols))),
        "count": len(set(cols)),
        "preprocessor_type": type(pre).__name__,
    }

