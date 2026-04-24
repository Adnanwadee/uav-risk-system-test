# src/uav_risk/stage1/infer.py
from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd
import logging
from uav_risk.stage1.loader import load_stage1_artifacts
from uav_risk.stage1.canonicalize import canonicalize_scenario
from uav_risk.stage2.schemas import MLResult

logger = logging.getLogger(__name__)
_artifacts_cache = None

def calculate_drift_score(df: pd.DataFrame, stats: Dict[str, Any]) -> float:
    """يحسب مدى انحراف المدخلات عن متوسطات التدريب (Z-Score)."""
    drifts = []
    for col, stat in stats.items():
        if col in df.columns:
            val = df[col].iloc[0]
            z_score = abs((val - stat['mean']) / (stat['std'] + 1e-6))
            drifts.append(z_score)
    return float(np.mean(drifts)) if drifts else 0.0

def run_stage1_inference(scenario: Dict[str, Any], artifacts_dir: str = "artifacts") -> MLResult:
    global _artifacts_cache
    if _artifacts_cache is None: _artifacts_cache = load_stage1_artifacts(artifacts_dir)
        
    policy = _artifacts_cache.policy_config
    df, status = canonicalize_scenario(scenario, policy, _artifacts_cache.preprocessor.feature_names_in_)
    
    if status != "OK":
        return MLResult(predicted_class=status, confidence=0.0, risk_score=1.0)

    # كشف الانحراف (Drift Detection)
    drift_score = calculate_drift_score(df, _artifacts_cache.training_stats)
    if drift_score > 3.0: # إذا كان الانحراف أكبر من 3 انحرافات معيارية
        logger.warning(f"⚠️ DATA_DRIFT_DETECTED: Input is statistically abnormal (Score: {drift_score:.2f})")

    try:
        X_transformed = _artifacts_cache.preprocessor.transform(df)
        risk_score = float(np.clip(_artifacts_cache.reg_model.predict(X_transformed)[0], 0.0, 1.0))
        
        # المعايرة الاحترافية (Isotonic Calibration)
        calibrated_probas = _artifacts_cache.calibrator_model.predict_proba(X_transformed)[0]
        max_idx = np.argmax(calibrated_probas)
        confidence = float(calibrated_probas[max_idx])
        raw_class = str(_artifacts_cache.label_encoder.inverse_transform([max_idx])[0])
        
        # تطبيق سياسة الطيران الديناميكية
        predicted_class = "CAUTION"
        if raw_class == "GO" and confidence >= policy.get("min_confidence_go", 0.8):
            predicted_class = "GO"
        elif raw_class == "NO_GO" and confidence >= policy.get("high_risk_confidence_no_go", 0.65):
            predicted_class = "NO_GO"
            
        # دمج وسم الانحراف في النتيجة إذا كان عالياً
        final_status = f"{predicted_class}_WITH_DRIFT" if drift_score > 3.0 else predicted_class

        return MLResult(predicted_class=final_status, confidence=confidence, risk_score=risk_score)

    except Exception as e:
        return MLResult(predicted_class="SYSTEM_ERROR", confidence=0.0, risk_score=1.0)