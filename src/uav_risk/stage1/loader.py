"""
Stage 1 Artifact Loader (V2.2 - Compatibility Fix)
===================================================
تعديلات V2.2:
- إزالة الاعتماد على get_feature_names_out() لحل مشكلة التوافق بين الإصدارات.
- استخدام طرق بديلة للتحقق من صحة الـ preprocessor.
- الحفاظ على الوظائف الأساسية مع تجنب الأخطاء.

Author: Stage 1 — ACE System V2
"""

from __future__ import annotations
import os
import json
import joblib
import logging
import __main__
import numpy as np
import pandas as pd
from typing import NamedTuple, Any, Dict

# ---------------------------------------------------------------------------
# دعم دالة to_string_safe المطلوبة من قبل preprocessor القديم (Pickle Hell Fix)
# ---------------------------------------------------------------------------
def to_string_safe(x):
    """تحويل آمن إلى string لتغذية OneHotEncoder داخل الـ preprocessor."""
    try:
        return x.astype(str)
    except Exception:
        return x

__main__.to_string_safe = to_string_safe

# ---------------------------------------------------------------------------
# استيراد FeatureRegistry
# ---------------------------------------------------------------------------
from uav_risk.schema.feature_registry import FeatureRegistry

logger = logging.getLogger("Stage1Loader")

# ===========================================================================
# 1. NamedTuple: Stage1Artifacts (محدث)
# ===========================================================================
class Stage1Artifacts(NamedTuple):
    """
    الحزمة الكاملة لمكونات Stage 1.
    """
    reg_model: Any
    calibrator_model: Any
    preprocessor: Any
    label_encoder: Any
    feature_registry: FeatureRegistry
    policy_config: Dict[str, Any]
    training_stats: Dict[str, Any]

# ===========================================================================
# 2. دوال التحميل الآمن (Safe Loaders)
# ===========================================================================

def _safe_load_json(filepath: str, default: Dict[str, Any]) -> Dict[str, Any]:
    """تحميل ملف JSON بأمان مع قيمة افتراضية إذا كان الملف مفقوداً."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"✅ Loaded JSON: {filepath}")
            return data
        except Exception as e:
            logger.error(f"❌ Failed to parse JSON {filepath}: {e}")
            return default
    else:
        logger.warning(f"⚠️ JSON file not found: {filepath}. Using default: {default}")
        return default


def _safe_load_joblib(filepath: str, name: str) -> Any:
    """تحميل ملف joblib مع معالجة أخطاء واضحة."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ CRITICAL: {name} not found at: {filepath}")
    try:
        obj = joblib.load(filepath)
        logger.info(f"✅ Loaded {name} from {filepath}")
        return obj
    except Exception as e:
        logger.critical(f"❌ CRITICAL: Failed to load {name}: {e}")
        raise


def _get_preprocessor_feature_count(preprocessor: Any) -> int:
    """
    محاولة الحصول على عدد الميزات التي يخرجها الـ preprocessor
    باستخدام طرق متعددة للتوافق مع الإصدارات المختلفة.
    """
    # المحاولة 1: استخدام get_feature_names_out() إذا كان موجوداً
    if hasattr(preprocessor, 'get_feature_names_out'):
        try:
            features = list(preprocessor.get_feature_names_out())
            logger.info(f"✅ Got {len(features)} features via get_feature_names_out()")
            return len(features)
        except Exception as e:
            logger.warning(f"get_feature_names_out() failed: {e}")
    
    # المحاولة 2: استخدام transform على عينة وهمية
    try:
        # إنشاء عينة وهمية باستخدام feature_names_in_ إذا كان موجوداً
        if hasattr(preprocessor, 'feature_names_in_'):
            sample_df = pd.DataFrame([[0] * len(preprocessor.feature_names_in_)], 
                                      columns=preprocessor.feature_names_in_)
        else:
            # إذا لم يكن لدينا feature_names_in_، نستخدم قائمة فارغة (قد تفشل)
            sample_df = pd.DataFrame([[0]])
        
        sample_transformed = preprocessor.transform(sample_df)
        feature_count = sample_transformed.shape[1]
        logger.info(f"✅ Got {feature_count} features via sample transform")
        return feature_count
    except Exception as e:
        logger.warning(f"Sample transform failed: {e}")
    
    # المحاولة 3: استخدام n_features_in_ إذا كان موجوداً (قد يعطي عدد الميزات المدخلة)
    if hasattr(preprocessor, 'n_features_in_'):
        logger.warning(f"⚠️ Using n_features_in_={preprocessor.n_features_in_} (input features, not output)")
        return preprocessor.n_features_in_
    
    # إذا فشل كل شيء، نرجع None ونحذر
    logger.warning("⚠️ Could not determine preprocessor output feature count")
    return None


def _compute_training_stats_from_preprocessor(preprocessor: Any) -> Dict[str, Any]:
    """
    بناء إحصائيات احتياطية (Fallback) من الـ preprocessor إذا لم يكن
    ملف training_stats.json موجوداً.

    ملاحظة: هذه الإحصائيات احتياطية فقط (كلها أصفار).
    للاستخدام الإنتاجي، يجب بناء training_stats.json من بيانات التدريب الحقيقية.
    """
    logger.warning("⚠️ Building fallback training_stats from preprocessor output features.")
    stats = {}
    try:
        # محاولة الحصول على أسماء الميزات
        feature_names = None
        if hasattr(preprocessor, 'get_feature_names_out'):
            try:
                feature_names = list(preprocessor.get_feature_names_out())
            except Exception:
                pass
        
        if feature_names is None:
            # استخدام أسماء افتراضية
            feature_count = _get_preprocessor_feature_count(preprocessor) or 58
            feature_names = [f"feature_{i}" for i in range(feature_count)]
        
        for col in feature_names:
            stats[col] = {
                "mean": 0.0,
                "std": 1.0,
                "note": "FALLBACK: Real training_stats.json is missing. Drift detection will be unreliable."
            }
        logger.info(f"✅ Built fallback stats for {len(stats)} features.")
    except Exception as e:
        logger.error(f"❌ Could not extract feature names from preprocessor: {e}")
        stats = {"error": "Failed to build fallback stats"}
    return stats


# ===========================================================================
# 3. دالة التحميل الرئيسية
# ===========================================================================

def load_stage1_artifacts(artifacts_dir: str = "artifacts") -> Stage1Artifacts:
    """
    تحميل جميع مكونات Stage 1 بشكل آمن ومتسق.

    Args:
        artifacts_dir: المسار إلى مجلد artifacts (نسبي أو مطلق).

    Returns:
        Stage1Artifacts: حزمة المكونات الكاملة.

    Raises:
        FileNotFoundError: إذا كان أحد الملفات الأساسية مفقوداً.
    """
    logger.info(f"🚀 Loading Stage-1 Artifacts from: {artifacts_dir}")

    # --- 3.1 تحميل المكونات الأساسية (إلزامية) ---
    preprocessor = _safe_load_joblib(
        os.path.join(artifacts_dir, "uav_stage1_preprocessor_v2.pkl"),
        "Preprocessor"
    )
    reg_model = _safe_load_joblib(
        os.path.join(artifacts_dir, "xgb_reg_stage1_v2.pkl"),
        "XGBoost Regressor"
    )
    calibrator_model = _safe_load_joblib(
        os.path.join(artifacts_dir, "clf_calibrator_stage1_v2.pkl"),
        "Calibrated Classifier"
    )
    label_encoder = _safe_load_joblib(
        os.path.join(artifacts_dir, "label_encoder_stage1_v2.pkl"),
        "Label Encoder"
    )

    # --- 3.2 تحميل المكونات الاختيارية (لها قيم افتراضية) ---
    policy_config = _safe_load_json(
        os.path.join(artifacts_dir, "stage1_policy_config_v2.json"),
        default={
            "min_dq_core_present": 0.75,
            "min_confidence_any_decision": 0.55,
            "min_confidence_go": 0.80,
            "high_risk_confidence_no_go": 0.65,
            "low_risk_min_confidence": 0.60,
            "calibration": "isotonic",
            "notes": "Default thresholds. Real policy config was not found."
        }
    )

    training_stats = _safe_load_json(
        os.path.join(artifacts_dir, "training_stats.json"),
        default=None
    )
    if training_stats is None:
        training_stats = _compute_training_stats_from_preprocessor(preprocessor)

    # --- 3.3 بناء FeatureRegistry (بدون تحقق صارم لتجنب مشاكل التوافق) ---
    feature_registry = FeatureRegistry()
    expected_feature_count = feature_registry.expected_count
    
    # محاولة التحقق من عدد الميزات (بدون إيقاف التشغيل في حالة الفشل)
    preprocessor_feature_count = _get_preprocessor_feature_count(preprocessor)
    
    if preprocessor_feature_count is not None:
        if preprocessor_feature_count != expected_feature_count:
            logger.warning(
                f"⚠️ Feature count mismatch! Preprocessor outputs {preprocessor_feature_count} features, "
                f"FeatureRegistry expects {expected_feature_count} features. "
                f"Proceeding with caution - this may cause prediction errors."
            )
        else:
            logger.info(f"✅ Feature count validated: {preprocessor_feature_count} features match registry")
    else:
        logger.warning(f"⚠️ Could not verify feature count. Assuming registry is correct with {expected_feature_count} features")
    
    # ملاحظة: تخطي التحقق من أسماء الميزات بسبب مشاكل التوافق بين الإصدارات
    logger.info("ℹ️ Feature name validation skipped for compatibility. Relying on feature count only.")
    logger.info(f"✅ FeatureRegistry initialized: expecting {feature_registry.expected_count} features")
    
    # تحقق إضافي: التأكد من أن training_stats لها العدد الصحيح من الميزات
    if training_stats and "error" not in training_stats:
        stats_feature_count = len(training_stats)
        if stats_feature_count != expected_feature_count:
            logger.warning(
                f"⚠️ training_stats has {stats_feature_count} features, "
                f"but registry expects {expected_feature_count}. Drift detection may be affected."
            )
        else:
            # التحقق من أن training_stats ليست Fallback (لا تحتوي على حقل 'note')
            first_feature = list(training_stats.keys())[0] if training_stats else None
            if first_feature and 'note' in training_stats.get(first_feature, {}):
                logger.warning("⚠️ training_stats is using FALLBACK values (all zeros). Drift detection will be unreliable!")
            else:
                logger.info("✅ training_stats loaded successfully with real values")

    logger.info("✅ Stage-1 Artifacts Loaded Successfully.")

    return Stage1Artifacts(
        reg_model=reg_model,
        calibrator_model=calibrator_model,
        preprocessor=preprocessor,
        label_encoder=label_encoder,
        feature_registry=feature_registry,
        policy_config=policy_config,
        training_stats=training_stats
    )