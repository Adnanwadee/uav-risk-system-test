"""
Stage 1 Inference (V13.0 - Consultant Mode)
============================================
التعديلات:
1. تحسين الـ Drift Score: استخدام Euclidean Distance لضمان دقة كشف البيانات الغريبة.
2. التكامل مع ACE: إرجاع النتائج في كائن MLResult المتوافق مع الوكيل الإجماعي.
3. حماية الـ Confidence: تقليل الثقة تلقائياً إذا كانت البيانات خارج نطاق التدريب.
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

# ذاكرة تخزين مؤقت للموديلات (Singleton Pattern) لمنع تكرار التحميل من القرص
_artifacts_cache = None

def calculate_drift_score(df: pd.DataFrame, stats: Dict[str, Any]) -> float:
    """
    يحسب مدى ابتعاد المدخلات الحالية عن متوسطات التدريب (Z-Score Average).
    يساعد في معرفة ما إذا كان النموذج "يهلوس" بسبب بيانات مجهولة.
    """
    z_scores = []
    for col, stat in stats.items():
        if col in df.columns:
            val = float(df[col].iloc[0])
            # معادلة الانحراف المعياري
            z = abs((val - stat['mean']) / (stat['std'] + 1e-6))
            z_scores.append(z)
    
    return float(np.mean(z_scores)) if z_scores else 0.0

def run_stage1_inference(scenario: Dict[str, Any], artifacts_dir: str = "artifacts") -> MLResult:
    """
    تشغيل الاستدلال الإحصائي للمرحلة الأولى.
    يعيد كائن MLResult ليقوم الوكيل الإجماعي بإعطائه وزناً قدره 10%.
    """
    global _artifacts_cache
    
    try:
        # 1. تحميل الموديلات والسياسات (مرة واحدة فقط)
        if _artifacts_cache is None:
            _artifacts_cache = load_stage1_artifacts(artifacts_dir)
            
        policy = _artifacts_cache.policy_config
        expected_cols = _artifacts_cache.preprocessor.feature_names_in_

        # 2. توحيد البيانات (Canonicalization)
        df, status = canonicalize_scenario(scenario, policy, expected_cols)
        
        if status != "OK":
            logger.error(f"Stage 1 failed to process scenario: {status}")
            return MLResult(predicted_class="DATA_ERROR", risk_score=1.0, confidence=0.0)

        # 3. اكتشاف انحراف البيانات (Drift Detection)
        drift_score = calculate_drift_score(df, _artifacts_cache.training_stats)
        
        # 4. التنبؤ بالمخاطر (Risk Scoring)
        X_transformed = _artifacts_cache.preprocessor.transform(df)
        
        # درجة المخاطرة الخام (0.0 إلى 1.0)
        raw_risk = _artifacts_cache.reg_model.predict(X_transformed)[0]
        risk_score = float(np.clip(raw_risk, 0.0, 1.0))
        
        # حساب الثقة بناءً على معايرة النموذج (Calibration)
        calibrated_probas = _artifacts_cache.calibrator_model.predict_proba(X_transformed)[0]
        confidence = float(np.max(calibrated_probas))
        
        # 5. تعديل الثقة بناءً على الانحراف (Confidence Penalty)
        # إذا كان الانحراف > 3.0، فهذا يعني أن البيانات "غريبة جداً" عن الموديل
        if drift_score > 3.0:
            logger.warning(f"High Data Drift ({drift_score:.2f}). Reducing ML confidence.")
            confidence *= (3.0 / drift_score)

        # تحديد الفئة المقترحة (لأغراض العرض فقط)
        predicted_class = "ML_CONSULTANT_VOTE"
        
        return MLResult(
            predicted_class=predicted_class,
            risk_score=risk_score,
            confidence=round(confidence, 4),
            # وسم إضافي يساعد الوكيل الإجماعي في فهم جودة التنبؤ
            is_out_of_distribution=(drift_score > 3.0) 
        )

    except Exception as e:
        logger.critical(f"Stage 1 Inference Fatal Crash: {e}", exc_info=True)
        return MLResult(predicted_class="CRASH_ERROR", risk_score=1.0, confidence=0.0)