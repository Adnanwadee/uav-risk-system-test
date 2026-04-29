"""
Stage 1 Artifact Loader (V16.6 - String Safety Patch)
======================================================
حل مشكلة AttributeError: 'str' object has no attribute 'predictor'
عن طريق حماية دالة الترقيع من التعامل مع النصوص.
"""
from __future__ import annotations
import os
import joblib
import logging
import __main__
from typing import NamedTuple, Any, Dict

def to_string_safe(x):
    try: return x.astype(str)
    except: return x
__main__.to_string_safe = to_string_safe

logger = logging.getLogger("Stage1Loader")

class Stage1Artifacts(NamedTuple):
    reg_model: Any
    calibrator_model: Any
    preprocessor: Any
    label_encoder: Any
    policy_config: Dict[str, Any]
    training_stats: Dict[str, Any]

def load_stage1_artifacts(artifacts_dir: str = "artifacts") -> Stage1Artifacts:
    logger.info(f"🚀 Initiating Stage-1 Heavy Artifact Loading from: {artifacts_dir}")
    
    def _force_patch_model(model):
        """[رقعة عميقة]: حقن السمات المفقودة مع حماية ضد النصوص (Strings)."""
        attributes = {
            'predictor': 'cpu_predictor',
            'gpu_id': -1,
            'use_label_encoder': False,
            'n_jobs': 1,
            'tree_method': 'auto'
        }
        
        def patch_obj(obj):
            # [إصلاح حاسم]: تجاوز العملية إذا كان الكائن مجرد نص (String) أو فارغ
            if isinstance(obj, str) or obj is None:
                return
                
            for attr, val in attributes.items():
                try:
                    setattr(obj, attr, val)
                except Exception:
                    pass
        
        # 1. حقن الموديل الرئيسي
        patch_obj(model)
        
        # 2. حقن الموديل الأساسي (Base Estimator)
        if hasattr(model, 'base_estimator'):
            patch_obj(model.base_estimator)
            
        # 3. حقن الموديلات المحملة داخل CalibratedClassifierCV
        if hasattr(model, 'calibrated_classifiers_'):
            for cc in model.calibrated_classifiers_:
                if hasattr(cc, 'base_estimator'):
                    patch_obj(cc.base_estimator)
                    
        return model

    # تحميل الموديلات مع الرقعة الشاملة والآمنة
    reg_model = _force_patch_model(joblib.load(os.path.join(artifacts_dir, "xgb_reg_stage1_v2.pkl")))
    clf_model = _force_patch_model(joblib.load(os.path.join(artifacts_dir, "clf_calibrator_stage1_v2.pkl")))

    return Stage1Artifacts(
        reg_model=reg_model,
        calibrator_model=clf_model,
        preprocessor=joblib.load(os.path.join(artifacts_dir, "uav_stage1_preprocessor_v2.pkl")),
        label_encoder=joblib.load(os.path.join(artifacts_dir, "label_encoder_stage1_v2.pkl")),
        policy_config={}, 
        training_stats={}
    )