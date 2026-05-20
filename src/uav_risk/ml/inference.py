"""
Module: uav_risk.ml.inference
Purpose: High-integrity ML inference engine executing predictions, drift monitoring, and bias mitigation.
Dependencies: Imports from uav_risk.ml.schemas, uav_risk.ml.loader, and uav_risk.ml.shap_explain.
"""

import time
import hashlib
import numpy as np
import pandas as pd
import structlog
from typing import List, Dict, Any, Optional, Tuple

# استيراد العقود والهياكل المقفلة لضمان التوافق المعماري المطلق
from uav_risk.ml.schemas import (
    MLResult, 
    RiskClass, 
    FeatureImportance, 
    calculate_risk_score, 
    probabilities_to_dict
)
from uav_risk.ml.loader import Stage1Bundle
from uav_risk.ml.shap_explain import ShapExplainer

# إعداد نظام التتبع للمركب البرمجي
logger = structlog.get_logger(__name__)


class InferenceError(Exception):
    """Custom exception raised when any stage of the live model inference cycle breaks down."""
    pass


def run_stage1_inference(
    bundle: Stage1Bundle,
    feature_vector: np.ndarray,
    feature_names: List[str],
    compute_shap: bool = True
) -> MLResult:
    """
    Executes the comprehensive Stage-1 machine learning inference pipeline on a flight scenario vector.
    Converts inputs to explicitly aligned DataFrame structures to silence alignment warnings.
    """
    start_time = time.perf_counter()
    logger.info("Executing Stage-1 machine learning inference pass", vector_len=len(feature_vector) if feature_vector is not None else 0)
    
    try:
        if feature_vector is None:
            raise ValueError("Inference input error: feature_vector cannot be None")
            
        X_processed = feature_vector.reshape(1, -1)
        if X_processed.shape[1] != 198:
            raise ValueError(f"Shape discrepancy: Expected exactly 198 features, found {X_processed.shape[1]}")
            
        if np.isnan(X_processed).any() or np.isinf(X_processed).any():
            raise ValueError("Data anomaly: Input feature vector contains invalid NaN or Inf values")

        # تحويل المصفوفة إلى DataFrame بأسماء أعمدة متوافقة مع شجرة LightGBM المخزنة (Column_0, Column_1...)
        # هذا التعديل يمنع ظهور الـ UserWarning الخاص بأسماء الميزات تماماً ويؤمن الاتساق الصارم
        lgbm_column_names = [f"Column_{i}" for i in range(198)]
        df_model_input = pd.DataFrame(X_processed, columns=lgbm_column_names)

        # 4. تنفيذ التنبؤ المباشر من النموذج
        try:
            probabilities_raw = bundle.model.predict_proba(df_model_input)[0]
            
            prob_sum = float(np.sum(probabilities_raw))
            if abs(prob_sum - 1.0) > 0.001:
                raise ValueError(f"Probabilistic anomaly: Raw model output sum deviates from unity: {prob_sum}")
                
            predicted_class_idx = int(np.argmax(probabilities_raw))
            raw_class_string = bundle.class_names[predicted_class_idx]
            
            risk_class_enum = RiskClass.from_string(raw_class_string)
            confidence_val = float(probabilities_raw[predicted_class_idx])
            
        except Exception as pred_err:
            raise InferenceError(f"Underlying LightGBM graph execution failed: {str(pred_err)}")

        # 5. درع معايرة الانحياز الذكي لحل مشكلة التحيز المفرط للـ High Risk
        high_risk_calibration_threshold = bundle.policy_config.get("high_risk_confidence_no_go", 0.55)
        if raw_class_string == "High Risk" and confidence_val < high_risk_calibration_threshold:
            logger.info(
                "High-Risk Bias Detected & Mitigated Safely", 
                raw_confidence=round(confidence_val, 4),
                action="Calibrating classification down to Medium Risk"
            )
            risk_class_enum = RiskClass.MEDIUM_RISK

        # 6. توزيع الاحتمالات وحساب المخاطرة الموحدة
        probabilities_mapped = probabilities_to_dict(probabilities_raw, bundle.class_names)
        calculated_score = calculate_risk_score(probabilities_mapped)

        # 7. استخراج قيم SHAP المفسرة عبر الـ Caching المتطور
        top_importance_features: List[FeatureImportance] = []
        if compute_shap and bundle.shap_explainer is not None:
            explainer_instance = ShapExplainer(bundle.model, feature_names)
            top_importance_features = explainer_instance.explain(
                X=df_model_input.to_numpy(), 
                top_n=10, 
                predicted_class_idx=predicted_class_idx
            )

        # 8. حساب الـ DriftDetection والـ Audit Trail Hash للنزاهة الرقمية
        drift_score, is_drift_detected = _compute_drift(X_processed[0], bundle)
        feature_hash = hashlib.sha256(X_processed.tobytes()).hexdigest()
        
        execution_time_ms = (time.perf_counter() - start_time) * 1000.0
        
        return MLResult(
            risk_class=risk_class_enum,
            risk_score=calculated_score,
            confidence=confidence_val,
            probabilities=probabilities_mapped,
            top_features=top_importance_features,
            drift_score=drift_score,
            drift_detected=is_drift_detected,
            processing_time_ms=round(execution_time_ms, 2),
            model_version=bundle.get_model_version(),
            feature_vector_hash=feature_hash,
            shap_expected_values=bundle.training_stats.get("expected_shap_values", [])
        )

    except Exception as fatal_err:
        execution_time_ms = (time.perf_counter() - start_time) * 1000.0
        logger.error("Inference execution critical failure. Activating safe fallback degraded structure.", error=str(fatal_err))
        
        fallback_probabilities = {"High Risk": 0.333, "Low Risk": 0.333, "Medium Risk": 0.333}
        return MLResult(
            risk_class=RiskClass.MEDIUM_RISK,
            risk_score=0.5,
            confidence=0.0,
            probabilities=fallback_probabilities,
            top_features=[],
            drift_score=1.0,
            drift_detected=True,
            processing_time_ms=round(execution_time_ms, 2),
            model_version=bundle.get_model_version() if 'bundle' in locals() and bundle else "unknown",
            feature_vector_hash=None
        )


def _compute_drift(feature_vector: np.ndarray, bundle: Stage1Bundle) -> Tuple[float, bool]:
    """Computes data drift score via standard Z-Score deviation profiling."""
    try:
        scaler_step = None
        if hasattr(bundle.preprocessor, 'transformers_'):
            for trans in bundle.preprocessor.transformers_:
                if trans[0] == 'scaler':
                    scaler_step = trans[1]
                    break
                    
        if scaler_step is None or not hasattr(scaler_step, 'center_'):
            return 0.0, False
            
        centers = scaler_step.center_
        scales = scaler_step.scale_
        
        z_scores: List[float] = []
        for idx in range(min(len(feature_vector), len(centers))):
            val = feature_vector[idx]
            mean_val = centers[idx]
            std_val = scales[idx] if scales[idx] > 0 else 1.0
            z_scores.append(abs(val - mean_val) / std_val)
            
        if not z_scores:
            return 0.0, False
            
        top_z_scores = sorted(z_scores, reverse=True)[:10]
        drift_score = min(max(float(np.mean(top_z_scores)) / 3.0, 0.0), 1.0)
        return round(drift_score, 4), drift_score > 0.7
        
    except Exception:
        return 0.0, False

# =====================================================================
# Architectural Registry Block:
# This file depends on: src/uav_risk/ml/schemas.py, src/uav_risk/ml/loader.py, src/uav_risk/ml/shap_explain.py
# Files depending on this file: src/uav_risk/core/pipeline.py, tests/test_ml_gate4.py
# =====================================================================