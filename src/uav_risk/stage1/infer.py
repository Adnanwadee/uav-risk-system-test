# src/uav_risk/stage1/infer.py
from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
import xgboost as xgb
import traceback
import logging
from functools import lru_cache

from uav_risk.stage1.loader import load_stage1_artifacts
from uav_risk.stage1.canonicalize import canonicalize_scenario
from uav_risk.stage2.schemas import MLResult

logger = logging.getLogger(__name__)

# ============================================================
# 1. Lazy-Loaded Artifact Cache (Production Essential)
# ============================================================
@lru_cache(maxsize=1)
def _get_cached_artifacts(artifacts_dir: str):
    """Loads artifacts once per runtime to prevent I/O bottleneck."""
    return load_stage1_artifacts(artifacts_dir)

# ============================================================
# 2. Safe XGBoost Prediction
# ============================================================
def _predict_with_booster(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
    elif isinstance(model, xgb.Booster):
        booster = model
    else:
        raise RuntimeError("Unsupported XGBoost model type")
    
    dmat = xgb.DMatrix(X)
    return booster.predict(dmat)

# ============================================================
# 3. Pure Statistical Oracle
# ============================================================
def run_stage1_inference(
    scenario: Dict[str, Any],
    artifacts_dir: str = "artifacts",
) -> MLResult:
    """
    Stage-1 UAV Risk Inference.
    Returns ONLY statistical facts. No decision authority.
    """
    try:
        # 0) Load cached artifacts
        art = _get_cached_artifacts(artifacts_dir)

        # 1) Canonicalize input (Ensures feature alignment)
        df = canonicalize_scenario(scenario)

        # 2) Preprocess
        X = art.preprocessor.transform(df)

        # 3) Regression (Continuous Risk Score)
        raw_risk = float(_predict_with_booster(art.reg_model, X)[0])
        
        # 4) Classification (Probabilities)
        raw_proba = _predict_with_booster(art.clf_model, X)[0]
        
        # Sanitize probabilities (handle NaN/Inf from numerical instability)
        raw_proba = np.nan_to_num(raw_proba, nan=0.0, posinf=0.0, neginf=0.0)
        proba_sum = raw_proba.sum()
        if proba_sum > 0:
            raw_proba /= proba_sum  # Normalize to sum=1
            
        class_names = art.label_encoder.inverse_transform(np.arange(len(raw_proba)))
        proba_dict = {str(cls): float(p) for cls, p in zip(class_names, raw_proba)}
        
        # 5) Finalize & Clip to safe bounds [0.0, 1.0]
        risk_score = float(np.clip(raw_risk, 0.0, 1.0))
        confidence = float(np.max(raw_proba))
        predicted_class = str(class_names[int(np.argmax(raw_proba))])

        # Return Pydantic model for LangGraph State compatibility
        return MLResult(
            predicted_class=predicted_class,
            confidence=confidence,
            risk_score=risk_score,
        )

    except Exception as e:
        logger.error(f"[ML_TOOL_ERROR] Inference failed: {e}\n{traceback.format_exc()}")
        
        # Failsafe: Return neutral state. Let InputContractEngine handle IGNORE_ML
        return MLResult(
            predicted_class="UNKNOWN",
            confidence=0.0,
            risk_score=0.5,  # Neutral baseline, not max risk
        )