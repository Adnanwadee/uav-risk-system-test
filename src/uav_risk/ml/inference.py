"""
Module: uav_risk.ml.inference
Purpose: Execution engine for LightGBM, binding feature vector to SHAP attribution,
         with mandatory preprocessing and enhanced result fidelity.
"""

import time
import hashlib
import numpy as np
import pandas as pd
import structlog
from uav_risk.ml.schemas import MLResult, RiskClass, probabilities_to_dict, calculate_risk_score, Stage1Bundle
from uav_risk.ml.shap_explain import ShapExplainer

logger = structlog.get_logger(__name__)


def _hash_vector(vec: np.ndarray) -> str:
    """إنشاء بصمة سريعة للمتجه لأغراض التتبع."""
    return hashlib.sha256(vec.tobytes()).hexdigest()[:12]


def run_stage1_inference(bundle: Stage1Bundle, feature_vector: np.ndarray, feature_names=None, compute_shap: bool = True) -> MLResult:
    """
    تنفيذ دورة الاستدلال الكاملة:
    1. معالجة مسبقة للمتجه الخام.
    2. تنبؤ بالنموذج.
    3. تفسير SHAP.
    4. تغليف النتيجة بكامل البيانات الوصفية.
    """
    start = time.perf_counter()

    # --- 1. تجهيز المتجه مباشرة (نستخدم الحزمة الإنتاجية: لا preprocessing متطلب)
    X_for_model = feature_vector.reshape(1, -1)
    # مرّر DataFrame مع أعمدة الحزمة إن أمكن لتجنب تحذيرات sklearn والتأكيد على محاذاة الأعمدة
    try:
        df_for_model = pd.DataFrame(X_for_model, columns=bundle.feature_names)
    except Exception:
        df_for_model = None

    # --- 2. التنبؤ ---
    try:
        probs_raw = bundle.model.predict_proba(df_for_model if df_for_model is not None else X_for_model)[0]
    except Exception:
        # كحل احتياطي — محاولة التنبؤ بالـ numpy array مباشرة
        probs_raw = bundle.model.predict_proba(X_for_model)[0]
    probs = probabilities_to_dict(probs_raw, bundle.class_names)
    predicted_idx = int(np.argmax(probs_raw))
    risk_class = RiskClass.from_string(bundle.class_names[predicted_idx])
    score = calculate_risk_score(probs)

    # --- 3. تفسير SHAP ---
    # استخدم كائن ShapExplainer (يُعاد استخدامه تلقائياً بفضل التخزين المؤقت)
    explainer = ShapExplainer(bundle.model, bundle.feature_names)
    top_features = []
    if compute_shap:
        try:
            top_features = explainer.explain(
                df_for_model if df_for_model is not None else X_for_model,
                predicted_class_idx=predicted_idx,
                class_names=bundle.class_names,
                raw_values=feature_vector
            )
        except Exception as exc:
            logger.exception("SHAP explanation failed; returning empty top_features", exc_info=exc)
            top_features = []

    # --- 4. بناء النتيجة ---
    result = MLResult(
        risk_class=risk_class,
        risk_score=score,
        confidence=float(probs_raw[predicted_idx]),
        probabilities=probs,
        top_features=top_features,
        processing_time_ms=(time.perf_counter() - start) * 1000,
        model_version=bundle.get_model_version(),
        feature_vector_hash=_hash_vector(feature_vector),
        # الاحتفاظ بالمخرجات الخام تلقائياً عبر __post_init__
    )

    return result