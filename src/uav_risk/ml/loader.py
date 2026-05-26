"""
Module: uav_risk.ml.loader
Purpose: High-integrity loader for Stage-1 production bundle and separate model/preprocessor artifacts,
         strictly aligned with feature_defs and schematized structures.
Dependencies: uav_risk.ml.schemas, uav_risk.ml.feature_defs, uav_risk.core, uav_risk.ml.bundle_security
"""

import os
import json
import structlog
import numpy as np
from typing import Any, Dict, Optional, Tuple, List
from pathlib import Path

from uav_risk.ml import feature_defs
from uav_risk.ml.schemas import Stage1Bundle
from uav_risk.ml.bundle_security import safe_load_bundle
from uav_risk.core.feature_engineering import generate_all_features_map, split_primary_and_secondary_overrides
from uav_risk.core.feature_router import FeatureRouter
from uav_risk.core.data_validator import DataValidator

logger = structlog.get_logger(__name__)

class ModelLoadError(Exception):
    """استثناء مخصص لانهيار دورة تجميع عناصر الحزمة."""
    pass

# أسماء الملفات المعيارية
BUNDLE_FILE = "stage1_production_bundle.pkl"
MODEL_FILE = "final_model.pkl"
PREPROCESSOR_FILE = "preprocessing_pipeline_final.pkl"
FEATURE_MAPPING_FILE = "stage1_feature_mapping.json"
MODEL_CARD_FILE = "model_card.json"
INFERENCE_CONFIG_FILE = "stage1_inference_config.json"


def load_stage1_bundle(artifacts_dir: str) -> Stage1Bundle:
    """
    تحميل الحزمة المركزية من الملفات المنفصلة.
    يتم تحميل النموذج والمعالج المسبق من ملفين منفصلين،
    والميزات والبيانات الوصفية من حزمة الـ PKL الأساسية وملفات JSON.
    """
    artifacts_path = Path(artifacts_dir)

    # --- تحميل المكوّنات الأساسية ---
    model_path = artifacts_path / MODEL_FILE
    preprocessor_path = artifacts_path / PREPROCESSOR_FILE
    bundle_path = artifacts_path / BUNDLE_FILE

    if not model_path.exists():
        raise ModelLoadError(f"Model file not found: {model_path}")
    if not bundle_path.exists():
        raise ModelLoadError(f"Bundle file not found: {bundle_path}")

    # تحميل النموذج
    model = safe_load_bundle(str(model_path))

    # تحميل المعالج المسبق إن وُجد (غير إلزامي - نستخدم الحزمة مباشرة)
    preprocessor = None
    if preprocessor_path.exists():
        try:
            preprocessor = safe_load_bundle(str(preprocessor_path))
        except Exception:
            logger.warning(f"Preprocessor present but failed to load: {preprocessor_path}; continuing without it")
    else:
        logger.info(f"Preprocessor not found, continuing without it: {preprocessor_path}")

    # تحميل الحزمة الوصفية (تحتوي على feature_names, class_names, إلخ)
    bundle_data = safe_load_bundle(str(bundle_path))
    if not isinstance(bundle_data, dict):
        raise ModelLoadError("Bundle file must contain a dictionary with feature_names and class_names.")

    feature_names = bundle_data.get("feature_names")
    class_names = bundle_data.get("class_names")

    if not feature_names or not class_names:
        raise ModelLoadError("Bundle is missing 'feature_names' or 'class_names'.")

    # --- فحص التطابق الدستوري ---
    is_aligned, alignment_err = feature_defs.validate_feature_registry_against_artifact(feature_names)
    if not is_aligned:
        logger.critical(f"ALIGNMENT BREACH: {alignment_err}")
        raise AssertionError(f"Model Feature Shift Blocked: {alignment_err}")

    feature_mapping = {name: idx for idx, name in enumerate(feature_names)}

    # --- تحميل البيانات الوصفية الإضافية (إن وُجدت) ---
    metadata = {}
    policy = {}

    # model_card.json
    model_card_path = artifacts_path / MODEL_CARD_FILE
    if model_card_path.exists():
        with open(model_card_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        logger.warning(f"Optional metadata file not found: {model_card_path}")

    # stage1_inference_config.json
    config_path = artifacts_path / INFERENCE_CONFIG_FILE
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            policy = json.load(f)
    else:
        logger.warning(f"Optional policy config file not found: {config_path}")

    # --- بناء الحزمة النهائية ---
    return Stage1Bundle(
        model=model,
        preprocessor=preprocessor,
        label_encoder=bundle_data.get("label_encoder"),   # قد يكون None
        shap_explainer=None,                              # يُنشأ لاحقاً عند الحاجة
        feature_names=feature_names,
        feature_mapping=feature_mapping,
        class_names=class_names,
        training_stats=bundle_data.get("training_stats", {}),
        policy_config=policy,
        model_metadata=metadata,
        bundle_path=str(bundle_path)
    )


def assemble_feature_vector_from_dict(
    input_mapping: Dict[str, Any],
    bundle: Stage1Bundle,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    الأنبوب المركزي: تحويل القاموس الخام إلى متجه 198 مع تطبيق
    Hard Veto → فصل الميزات → محرك DAG → موجه الميزات.
    """
    # 1. بوابة الـ Hard Veto
    validator = DataValidator()
    validation_result = validator.validate_and_store(input_mapping)
    if not validation_result.is_usable:
        logger.error("Inference blocked: Hard Veto rejected input", errors=validation_result.errors)
        raise ModelLoadError(f"Pipeline Blocked: {validation_result.errors}")

    # 2. فصل الميزات (أساسية / تجاوزات)
    primary, overrides, extras = split_primary_and_secondary_overrides(
        input_mapping, bundle.feature_names
    )

    # 3. محرك DAG – توليد الميزات الثانوية المفقودة
    feature_map = generate_all_features_map(
        primary, overrides=overrides, feature_order=bundle.feature_names
    )

    # 4. موجه الميزات – بناء متجه (198,) بالترتيب الصحيح
    router = FeatureRouter()
    vector = router.route_to_vector(feature_map)

    # 5. بيانات وصفية للجنائية
    metadata = {
        "feature_map": feature_map,
        "primary_inputs": primary,
        "secondary_overrides": overrides,
        "ignored_extras": extras,
        "validator_errors": validation_result.errors,
        "is_usable": validation_result.is_usable,
    }
    return vector, metadata