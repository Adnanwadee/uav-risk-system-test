from __future__ import annotations

# ============================================================
# CRITICAL FIX: inject missing function into __main__ namespace
# ============================================================
import __main__

def to_string_safe(x):
    """
    MUST match EXACTLY the function used during Stage-1 preprocessing.
    This is required so legacy pickles can be loaded safely.
    """
    try:
        return x.astype(str)
    except Exception:
        return x

# Inject into __main__ so pickle can resolve the symbol
__main__.to_string_safe = to_string_safe  # ✅ KEY LINE

# src/uav_risk/stage1/loader.py
from __future__ import annotations
import os
import json
import joblib
import logging
from typing import NamedTuple, Any, Dict
import __main__
from uav_risk.stage1.utils import to_string_safe

__main__.to_string_safe = to_string_safe
logger = logging.getLogger(__name__)

class Stage1Artifacts(NamedTuple):
    reg_model: Any
    calibrator_model: Any
    preprocessor: Any
    label_encoder: Any
    policy_config: Dict[str, Any]
    training_stats: Dict[str, Any] # لإحصائيات Drift Detection

def load_stage1_artifacts(artifacts_dir: str = "artifacts") -> Stage1Artifacts:
    logger.info(f"🚀 Loading Stage-1 Aviation Artifacts from: {artifacts_dir}")
    
    def _load(name: str):
        path = os.path.join(artifacts_dir, name)
        return joblib.load(path)

    # تحميل السياسات والإحصائيات
    with open(os.path.join(artifacts_dir, "stage1_policy_config_v2.json"), 'r') as f:
        policy = json.load(f)
    
    # سنفترض وجود قيم إحصائية مرجعية داخل الـ JSON أو ملف منفصل
    # إذا لم توجد، سنستخدم قيم افتراضية آمنة
    stats = policy.get("training_stats", {
        "uav.mass_kg": {"mean": 5.0, "std": 2.0},
        "environment.weather.wind_mps": {"mean": 4.0, "std": 3.5}
    })

    return Stage1Artifacts(
        reg_model=_load("xgb_reg_stage1_v2.pkl"),
        calibrator_model=_load("clf_calibrator_stage1_v2.pkl"),
        preprocessor=_load("uav_stage1_preprocessor_v2.pkl"),
        label_encoder=_load("label_encoder_stage1_v2.pkl"),
        policy_config=policy,
        training_stats=stats
    )