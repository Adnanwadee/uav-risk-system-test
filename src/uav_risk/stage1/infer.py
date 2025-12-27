from __future__ import annotations
from typing import Dict, Any
import pandas as pd
import numpy as np

from uav_risk.stage1.loader import load_stage1_artifacts

from uav_risk.stage1.canonicalize import canonicalize_scenario

def run_stage1_inference(
    scenario: Dict[str, Any],
    artifacts_dir: str = "artifacts",
) -> Dict[str, Any]:
    """
    Run Stage-1 risk inference.

    Returns a FACTS JSON (no text, no LLM).
    """

    # Load artifacts
    art = load_stage1_artifacts(artifacts_dir)

    # --------------------------------------------------
    # 1) Convert scenario dict -> DataFrame (single row)
    # --------------------------------------------------
    df = canonicalize_scenario(scenario)

    # --------------------------------------------------
    # 2) Preprocess
    # --------------------------------------------------
    X = art.preprocessor.transform(df)

    # --------------------------------------------------
    # 3) Regression (continuous risk score)
    # --------------------------------------------------
    risk_score = float(art.reg_model.predict(X)[0])

    # --------------------------------------------------
    # 4) Classification (probabilities)
    # --------------------------------------------------
    proba = art.clf_model.predict_proba(X)[0]

    class_names = art.label_encoder.inverse_transform(
        np.arange(len(proba))
    )

    proba_dict = {
        str(cls): float(p)
        for cls, p in zip(class_names, proba)
    }

    # Optional calibration
    if art.clf_calibrator is not None:
        proba_cal = art.clf_calibrator.predict_proba(proba.reshape(1, -1))[0]
        proba_dict = {
            f"{cls}_cal": float(p)
            for cls, p in zip(class_names, proba_cal)
        }

    # --------------------------------------------------
    # 5) Decision logic (policy-driven)
    # --------------------------------------------------
    policy = art.policy

    confidence = max(proba)
    predicted_class = class_names[int(np.argmax(proba))]

    if confidence < policy.get("min_confidence_any_decision", 0.0):
        decision = "INSUFFICIENT_CONFIDENCE"
    elif predicted_class.lower().startswith("high"):
        decision = "NO_GO"
    elif predicted_class.lower().startswith("low"):
        decision = "GO"
    else:
        decision = "CAUTION"

    # --------------------------------------------------
    # 6) Return FACTS JSON
    # --------------------------------------------------
    return {
        "status": "OK",
        "decision": decision,
        "predicted_class": str(predicted_class),
        "risk_score": risk_score,
        "confidence": float(confidence),
        "probabilities": proba_dict,
    }
