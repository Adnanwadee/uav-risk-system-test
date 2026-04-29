"""
Stage 1 Inference (V13.0 - Consultant Mode)
============================================
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any

from uav_risk.stage1.loader import load_stage1_artifacts
from uav_risk.stage1.canonicalize import canonicalize_scenario
from uav_risk.stage2.schemas import MLResult

logger = logging.getLogger(__name__)
_artifacts_cache = None

def calculate_drift_score(df: pd.DataFrame, stats: Dict[str, Any]) -> float:
    z_scores = []
    for col, stat in stats.items():
        if col in df.columns:
            val = float(df[col].iloc[0])
            z = abs((val - stat['mean']) / (stat['std'] + 1e-6))
            z_scores.append(z)
    return float(np.mean(z_scores)) if z_scores else 0.0

def run_stage1_inference(scenario: Dict[str, Any], artifacts_dir: str = "artifacts") -> MLResult:
    global _artifacts_cache
    try:
        if _artifacts_cache is None:
            _artifacts_cache = load_stage1_artifacts(artifacts_dir)
            
        policy = _artifacts_cache.policy_config
        expected_cols = _artifacts_cache.preprocessor.feature_names_in_

        df, status = canonicalize_scenario(scenario, policy, expected_cols)
        if status != "OK":
            return MLResult(predicted_class="DATA_ERROR", risk_score=1.0, confidence=0.0)

        drift_score = calculate_drift_score(df, _artifacts_cache.training_stats)
        X_transformed = _artifacts_cache.preprocessor.transform(df)
        
        raw_risk = _artifacts_cache.reg_model.predict(X_transformed)[0]
        risk_score = float(np.clip(raw_risk, 0.0, 1.0))
        
        # [حماية ضد أخطاء Scikit-Learn Calibration]
        try:
            calibrated_probas = _artifacts_cache.calibrator_model.predict_proba(X_transformed)[0]
            confidence = float(np.max(calibrated_probas))
        except Exception as cal_err:
            logger.warning(f"Calibrator skipped due to scikit-learn format mismatch. Using robust fallback.")
            confidence = float(0.5 + abs(risk_score - 0.5))
        
        if drift_score > 3.0:
            confidence *= (3.0 / drift_score)

        predicted_class = "HIGH_RISK" if risk_score > 0.5 else "LOW_RISK"
        return MLResult(predicted_class=predicted_class, risk_score=risk_score, confidence=round(confidence, 4), drift_score=drift_score)

    except Exception as e:
        logger.critical(f"Stage 1 Inference Fatal Crash: {e}", exc_info=True)
        return MLResult(predicted_class="CRASH_ERROR", risk_score=1.0, confidence=0.0)