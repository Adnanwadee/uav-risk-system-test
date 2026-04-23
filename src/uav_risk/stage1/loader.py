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

import os
import joblib
import logging
from typing import NamedTuple, Any

logger = logging.getLogger(__name__)

class Stage1Artifacts(NamedTuple):
    """Container for immutable Stage-1 ML models and processors."""
    clf_model: Any
    reg_model: Any
    preprocessor: Any
    label_encoder: Any

def load_stage1_artifacts(artifacts_dir: str = "artifacts") -> Stage1Artifacts:
    """
    Loads ML artifacts from disk. 
    Designed to be called via lru_cache for high-performance inference.
    """
    logger.info(f"🚀 Loading Stage-1 ML Artifacts from: {artifacts_dir}")
    
    # 1. التحقق من وجود المجلد
    if not os.path.exists(artifacts_dir):
        raise FileNotFoundError(f"Artifacts directory '{artifacts_dir}' not found. Cannot start ML Tool.")

    def _load(name: str):
        path = os.path.join(artifacts_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required artifact missing: {path}")
        return joblib.load(path)

    try:
        # 2. تحميل النماذج والمعالجات
        # تأكد أن هذه الأسماء تطابق تماماً ما هو موجود في مجلد artifacts لديك
        clf = _load("xgb_clf_stage1_v2.joblib")
        reg = _load("xgb_reg_stage1_v2.joblib")
        pre = _load("preprocessor.joblib")
        enc = _load("label_encoder.joblib")

        logger.info("✅ ML Artifacts loaded successfully.")
        return Stage1Artifacts(
            clf_model=clf,
            reg_model=reg,
            preprocessor=pre,
            label_encoder=enc
        )

    except Exception as e:
        logger.error(f"❌ Failed to load artifacts: {e}")
        raise