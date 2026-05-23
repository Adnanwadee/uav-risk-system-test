"""
Module: uav_risk.ml.loader
Purpose: High-integrity loader for Stage-1 production bundle, feature mappings, and model cards.
Dependencies: Strictly follows the architectural specifications outlined in "Plan K".
"""

import os
import json
import joblib
import structlog
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import math
from uav_risk.ml import feature_defs
from uav_risk.core import data_validator

# إعداد نظام التتبع والـ Logger المركزي للمنظومة
logger = structlog.get_logger(__name__)


class ModelLoadError(Exception):
    """Custom exception raised when any stage of the model bundle assembly sequence fails."""
    pass


@dataclass
class Stage1Bundle:
    """
    Complete representation of all loaded machine learning artifacts and metadata.
    Strictly aligned with the fields required by "Plan K" and runtime execution realities.
    """
    model: Any                      # LightGBM model instance
    preprocessor: Any               # sklearn ColumnTransformer instance
    feature_names: List[str]        # List of the 198 features in exact sequence
    feature_mapping: Dict[str, int] # Mapping of {feature_name: index}
    training_stats: Dict[str, Any]  # Baseline statistics for data drift detection
    policy_config: Dict[str, Any]   # Decision thresholds and operational boundaries
    model_metadata: Dict[str, Any]  # Content from model_card.json (version, metrics)
    shap_explainer: Optional[Any]   # Loaded SHAP TreeExplainer instance
    bundle_path: str                # File path from which the bundle was retrieved
    label_encoder: Optional[Any] = None
    class_names: List[str] = field(default_factory=list)
    def get_model_version(self) -> str:
        """Dynamic retrieval of the model execution iteration version from metadata."""
        if isinstance(self.model_metadata, dict):
            return self.model_metadata.get("version", self.model_metadata.get("pipeline_version", "unknown"))
        return "unknown"


def load_stage1_bundle(artifacts_dir: str) -> Stage1Bundle:
    """
    Loads the comprehensive production bundle and verified context files from the artifacts directory.
    """
    logger.info("Initializing Stage-1 high-integrity load sequence", artifacts_dir=artifacts_dir)
    
    if not os.path.exists(artifacts_dir):
        error_msg = f"Target artifacts directory not found: {artifacts_dir}"
        logger.critical("Loading failed: Missing directory", error=error_msg)
        raise ModelLoadError(error_msg)
        
    bundle_pkl_path = os.path.join(artifacts_dir, "stage1_production_bundle.pkl")
    feature_mapping_json_path = os.path.join(artifacts_dir, "stage1_feature_mapping.json")
    model_card_json_path = os.path.join(artifacts_dir, "model_card.json")
    inference_config_json_path = os.path.join(artifacts_dir, "stage1_inference_config.json")
    metadata_json_path = os.path.join(artifacts_dir, "column_metadata_final.json")
    
    required_paths = [bundle_pkl_path, feature_mapping_json_path, model_card_json_path, inference_config_json_path, metadata_json_path]
    for path in required_paths:
        if not os.path.exists(path):
            error_msg = f"Critical ML component file is missing from artifacts: {path}"
            logger.critical("Loading aborted due to unfulfilled dependency", error=error_msg)
            raise ModelLoadError(error_msg)
            
    try:
        logger.info("Loading master binary package", path=bundle_pkl_path)
        raw_bundle = joblib.load(bundle_pkl_path)
        
        with open(feature_mapping_json_path, 'r') as f:
            raw_feature_mapping = json.load(f)
            
        with open(model_card_json_path, 'r') as f:
            model_card_data = json.load(f)
            
        with open(inference_config_json_path, 'r') as f:
            policy_config_data = json.load(f)
            
        with open(metadata_json_path, 'r') as f:
            column_metadata = json.load(f)
            
        validate_bundle(raw_bundle, policy_config_data)
        
        extracted_feature_names = raw_bundle.get("feature_names", raw_feature_mapping.get("feature_names", []))

        # Enforce artifact-authoritative feature ordering and presence
        artifact_order = feature_defs.get_all_feature_names()
        if artifact_order:
            if extracted_feature_names != artifact_order:
                error_msg = (
                    "Feature ordering mismatch: extracted bundle feature_names do not match artifact-authoritative mapping."
                )
                logger.critical("Bundle feature mapping validation failed", error=error_msg)
                raise ModelLoadError(error_msg)

        # Ensure all mandatory core features exist in the bundle mapping
        core_feats = feature_defs.get_core_features()
        missing_core = [f for f in core_feats if f not in extracted_feature_names]
        if missing_core:
            error_msg = f"Bundle missing required core features: {missing_core}"
            logger.critical("Bundle validation failed - core features absent", missing=missing_core)
            raise ModelLoadError(error_msg)
        generated_feature_mapping = {name: idx for idx, name in enumerate(extracted_feature_names)}
        
        extracted_training_stats = {
            "timestamp": column_metadata.get("timestamp", "unknown"),
            "train_shape": column_metadata.get("train_shape", [0, 0]),
            "selected_strategy": column_metadata.get("selected_strategy", "unknown"),
            "expected_shap_values": policy_config_data.get("expected_shap_values", [])
        }
        
        logger.info("Stage-1 bundle successfully assembled and validated in memory",
                    total_features=len(extracted_feature_names),
                    target_classes=raw_bundle["class_names"])
                    
        return Stage1Bundle(
            model=raw_bundle["model"],
            preprocessor=raw_bundle["preprocessor"],
            feature_names=extracted_feature_names,
            feature_mapping=generated_feature_mapping,
            training_stats=extracted_training_stats,
            policy_config=policy_config_data,
            model_metadata=model_card_data,
            shap_explainer=raw_bundle["shap_explainer"],
            bundle_path=bundle_pkl_path,
            label_encoder=raw_bundle["label_encoder"],
            class_names=raw_bundle["class_names"]
        )
        
    except Exception as e:
        error_msg = f"Bundle validation failed: Runtime assembly error at {artifacts_dir}. Details: {str(e)}"
        logger.critical("Fatal loading sequence failure", error=str(e))
        raise ModelLoadError(error_msg) from e


def validate_bundle(bundle_data: dict, policy_config: dict) -> None:
    """Validates structural constraints to prevent silent failures in runtime."""
    if "model" not in bundle_data or "preprocessor" not in bundle_data:
        raise ValueError("Bundle validation failed: Core components 'model' or 'preprocessor' are missing")
        
    if "feature_names" not in bundle_data or len(bundle_data["feature_names"]) != 198:
        raise ValueError(f"Bundle validation failed: Expected exactly 198 feature entries")
        
    if not hasattr(bundle_data["model"], "predict_proba"):
        raise ValueError("Bundle validation failed: Loaded model instance does not expose 'predict_proba'")
        
    if "class_names" not in policy_config or len(policy_config["class_names"]) != 3:
        raise ValueError("Bundle validation failed: Configuration alignment must match triple categories")
        
    logger.debug("Structural verification constraints successfully cleared")


def assemble_feature_vector_from_dict(input_mapping: Dict[str, Any], bundle: Stage1Bundle) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Given a user-provided mapping of feature_name->value and a loaded Stage1Bundle,
    produce a numeric numpy feature vector ordered exactly as `bundle.feature_names`.

    Behavior:
    - If a core feature value is missing or non-numeric -> raises ModelLoadError
    - For non-core features missing in input -> fill from `feature_defs.get_safe_value()`
    - If input_mapping contains keys not in feature_names, they are collected as `free_text` (string)
    Returns (feature_vector, metadata) where metadata contains lists: 'imputed', 'provided', 'free_text'
    """
    if not isinstance(input_mapping, dict):
        raise ModelLoadError("Input mapping must be a dict of feature_name->value")

    feature_names = bundle.feature_names
    if not feature_names or len(feature_names) == 0:
        raise ModelLoadError("Bundle has no feature names to assemble vector")

    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    vec = np.zeros(len(feature_names), dtype=float)
    imputed = []
    provided = []
    free_text_parts = []

    # Run core validation and enrichment early so we can apply derived values
    try:
        vr = data_validator.validate_and_enrich(input_mapping)
    except Exception as e:
        raise ModelLoadError(f"Validation error: {e}")

    if not vr.is_usable:
        raise ModelLoadError(f"Core validation failed: {vr.errors}")

    enriched = vr.validated_features

    # treat any unknown keys as free-text intended for the agent
    for key, val in input_mapping.items():
        if key not in name_to_idx:
            # collect free-textizable entries
            if isinstance(val, str):
                free_text_parts.append(f"{key}: {val}")
            else:
                free_text_parts.append(f"{key}: {repr(val)}")

    # populate vector
    core_feats = feature_defs.get_core_features()
    for i, fname in enumerate(feature_names):
        if fname in input_mapping:
            raw = input_mapping[fname]
            try:
                num = float(raw)
                if math.isfinite(num):
                    vec[i] = num
                    provided.append(fname)
                    continue
            except Exception:
                # fall through to use enriched or safe
                pass

        # if enriched has a value (core or derived), use it
        if fname in enriched:
            try:
                num = float(enriched[fname])
                vec[i] = num
                # mark as provided if originally in input, else imputed
                if fname in input_mapping:
                    provided.append(fname)
                else:
                    imputed.append(fname)
                continue
            except Exception:
                pass

        # fallback to safe registry
        vec[i] = feature_defs.get_safe_value(fname)
        imputed.append(fname)

    metadata = {
        "imputed": imputed,
        "provided": provided,
        "free_text": "\n".join(free_text_parts),
        "validator_warnings": vr.warnings,
        "validator_errors": vr.errors
    }
    return vec, metadata

# =====================================================================
# Architectural Registry Block:
# This file depends on: None (Foundational Subsystem Loader).
# Files depending on this file: src/uav_risk/ml/inference.py, src/uav_risk/ml/shap_explain.py
# =====================================================================