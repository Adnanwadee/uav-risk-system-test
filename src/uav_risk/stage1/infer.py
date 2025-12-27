from __future__ import annotations

from typing import Dict, Any
import numpy as np
import xgboost as xgb

from uav_risk.stage1.loader import load_stage1_artifacts
from uav_risk.stage1.canonicalize import canonicalize_scenario


# ============================================================
# Safe XGBoost prediction bypassing sklearn wrappers
# ============================================================
def _predict_with_booster(model, X: np.ndarray):
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
    elif isinstance(model, xgb.Booster):
        booster = model
    else:
        raise RuntimeError("Unsupported XGBoost model type")

    dmat = xgb.DMatrix(X)
    return booster.predict(dmat)


def run_stage1_inference(
    scenario: Dict[str, Any],
    artifacts_dir: str = "artifacts",
) -> Dict[str, Any]:
    """
    Stage-1 UAV Risk Inference (FACTS ONLY).
    """

    # --------------------------------------------------
    # 0) Load artifacts
    # --------------------------------------------------
    art = load_stage1_artifacts(artifacts_dir)

    # --------------------------------------------------
    # 1) Canonicalize input
    # --------------------------------------------------
    df = canonicalize_scenario(scenario)

    # --------------------------------------------------
    # 2) Preprocess
    # --------------------------------------------------
    X = art.preprocessor.transform(df)

    # --------------------------------------------------
    # 3) Regression (risk score)
    # --------------------------------------------------
    risk_score = float(_predict_with_booster(art.reg_model, X)[0])

    # --------------------------------------------------
    # 4) Classification (RAW probabilities – NO calibration)
    # --------------------------------------------------
    proba = _predict_with_booster(art.clf_model, X)[0]

    class_names = art.label_encoder.inverse_transform(
        np.arange(len(proba))
    )

    proba_dict = {
        str(cls): float(p)
        for cls, p in zip(class_names, proba)
    }

    # --------------------------------------------------
    # 5) Decision logic
    # --------------------------------------------------
    policy = art.policy

    confidence = float(np.max(proba))
    predicted_class = str(class_names[int(np.argmax(proba))])

    if confidence < policy.get("min_confidence_any_decision", 0.0):
        decision = "INSUFFICIENT_CONFIDENCE"
    elif predicted_class.lower().startswith("high"):
        decision = "NO_GO"
    elif predicted_class.lower().startswith("low"):
        decision = "GO"
    else:
        decision = "CAUTION"

    # --------------------------------------------------
    # 6) FACTS JSON
    # --------------------------------------------------
    return {
        "status": "OK",
        "decision": decision,
        "predicted_class": predicted_class,
        "risk_score": risk_score,
        "confidence": confidence,
        "probabilities": proba_dict,
    }
