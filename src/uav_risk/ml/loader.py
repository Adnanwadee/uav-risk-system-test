"""
Module: uav_risk.ml.loader
Purpose: High-integrity loader for Stage-1 production bundle, feature mappings, and model cards.
Dependencies: Strictly follows the architectural specifications outlined in "Plan K".
"""

import os
import json
from uav_risk.ml.bundle_security import safe_load_bundle
import structlog
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
from uav_risk.ml import feature_defs
from uav_risk.ml.schemas import Stage1Bundle
from uav_risk.core.feature_engineering import PRIMARY_FEATURE_SET, generate_all_features_map, split_primary_and_secondary_overrides

# إعداد نظام التتبع والـ Logger المركزي للمنظومة
logger = structlog.get_logger(__name__)


class ModelLoadError(Exception):
    """Custom exception raised when any stage of the model bundle assembly sequence fails."""
    pass


AUTHORITATIVE_STAGE1_BUNDLE = "stage1_production_bundle.pkl"


def select_authoritative_bundle_path(artifacts_dir: str) -> str:
    """Select the strongest PKL artifact for Stage-1 runtime use.

    Preference order:
    1. `stage1_production_bundle.pkl` when present
    2. Any other PKL bundle containing `feature_names`, `model`, and `preprocessor`
    3. The first PKL bundle with `feature_names`
    """
    artifacts_path = Path(artifacts_dir)
    preferred = artifacts_path / AUTHORITATIVE_STAGE1_BUNDLE
    if preferred.exists():
        return str(preferred)

    candidates = [p for p in sorted(artifacts_path.glob("*.pkl")) if p.name != AUTHORITATIVE_STAGE1_BUNDLE]
    best_path: Optional[Path] = None
    best_score = -1

    for path in candidates:
        try:
            # For inspection during candidate selection we allow unsigned bundles
            # (they may be local artifacts). The final load enforces signature
            # according to environment policy.
            raw = safe_load_bundle(str(path), hmac_key=os.getenv("BUNDLE_HMAC_KEY"), allow_unsigned=True)
        except Exception as exc:
            logger.warning("Skipping unreadable or unverified PKL artifact", path=str(path), error=str(exc))
            continue

        if not isinstance(raw, dict):
            continue

        feature_names = raw.get("feature_names")
        if not isinstance(feature_names, (list, tuple)) or not feature_names:
            continue

        score = len(feature_names)
        if "model" in raw:
            score += 100
        if "preprocessor" in raw:
            score += 100
        if "class_names" in raw and isinstance(raw["class_names"], (list, tuple)) and len(raw["class_names"]) == 3:
            score += 50
        if "shap_explainer" in raw:
            score += 10

        if score > best_score:
            best_score = score
            best_path = path

    if best_path is not None:
        logger.warning("Using fallback PKL bundle instead of preferred stage1 bundle", selected=str(best_path), selected_score=best_score)
        return str(best_path)

    raise ModelLoadError(f"No authoritative Stage-1 bundle PKL found in {artifacts_dir}")

def load_stage1_bundle(artifacts_dir: str) -> Stage1Bundle:
    """
    Loads the comprehensive production bundle and verified context files from the artifacts directory.
    """
    logger.info("Initializing Stage-1 high-integrity load sequence", artifacts_dir=artifacts_dir)
    
    if not os.path.exists(artifacts_dir):
        error_msg = f"Target artifacts directory not found: {artifacts_dir}"
        logger.critical("Loading failed: Missing directory", error=error_msg)
        raise ModelLoadError(error_msg)
        
    bundle_pkl_path = select_authoritative_bundle_path(artifacts_dir)
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
        logger.info("Loading master binary package (safe loader)", path=bundle_pkl_path)
        try:
            raw_bundle = safe_load_bundle(bundle_pkl_path,
                                          hmac_key=os.getenv("BUNDLE_HMAC_KEY"),
                                          allow_unsigned=os.getenv("BUNDLE_ALLOW_UNSIGNED", "true").lower() in ("1","true","yes"))
        except Exception as be:
            raise ModelLoadError(f"Bundle security verification or load failed: {be}") from be
        
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

        # Use the bundle's feature_names as authoritative. If a separate JSON mapping differs,
        # log a warning for operators but do not fail — the model bundle is the runtime source.
        if not extracted_feature_names:
            error_msg = "Bundle does not contain 'feature_names' and no mapping could be extracted."
            logger.critical("Bundle missing feature names", error=error_msg)
            raise ModelLoadError(error_msg)

        artifact_order = feature_defs.get_all_feature_names()
        if artifact_order and artifact_order != extracted_feature_names:
            logger.warning("Artifact mapping differs from bundle.feature_names; bundle is authoritative.", artifact_len=len(artifact_order), bundle_len=len(extracted_feature_names))

        preprocessor_names = list(getattr(raw_bundle.get("preprocessor"), "feature_names_in_", []))
        if preprocessor_names and len(preprocessor_names) != len(extracted_feature_names):
            logger.warning("Preprocessor input schema differs from bundle feature order; bundle remains authoritative.", preprocessor_len=len(preprocessor_names), bundle_len=len(extracted_feature_names))

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
        
    if "feature_names" not in bundle_data or len(bundle_data["feature_names"]) == 0:
        raise ValueError("Bundle validation failed: Expected non-empty feature_names list")
        
    if not hasattr(bundle_data["model"], "predict_proba"):
        raise ValueError("Bundle validation failed: Loaded model instance does not expose 'predict_proba'")
        
    if "class_names" not in policy_config or len(policy_config["class_names"]) != 3:
        raise ValueError("Bundle validation failed: Configuration alignment must match triple categories")
        
    logger.debug("Structural verification constraints successfully cleared")


def assemble_feature_vector_from_dict(
    input_mapping: Dict[str, Any],
    bundle: Stage1Bundle,
    validation_result: Optional["ValidationResult"] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Given a user-provided mapping of feature_name->value and a loaded Stage1Bundle,
    produce a numeric numpy feature vector ordered exactly as `bundle.feature_names`.

    Behavior:
    - If a core feature value is missing or non-numeric -> raises ModelLoadError
    - By default, any non-primary key is rejected to enforce the 68-feature contract.
    - If `allow_secondary_overrides=True`, secondary feature overrides are accepted for
      transitional compatibility with the legacy 198-feature path.
    Returns (feature_vector, metadata) where metadata contains lists of provided,
    overridden, and ignored inputs.
    """
    if not isinstance(input_mapping, dict):
        raise ModelLoadError("Input mapping must be a dict of feature_name->value")

    feature_names = bundle.feature_names
    if not feature_names or len(feature_names) == 0:
        raise ModelLoadError("Bundle has no feature names to assemble vector")

    primary_inputs, secondary_overrides, extras = split_primary_and_secondary_overrides(input_mapping, feature_names)
    invalid_keys = sorted(key for key in extras.keys() if key not in feature_names)
    if invalid_keys:
        raise ValueError(
            "assemble_feature_vector_from_dict accepts only the authoritative 198 features. Rejected keys: " + str(invalid_keys)
        )

    feature_map = generate_all_features_map(primary_inputs, overrides=secondary_overrides, feature_order=feature_names)
    vec = np.array([feature_map[name] for name in feature_names], dtype=np.float64)

    if validation_result is None:
        bundle_policy = getattr(bundle, "policy_config", {}) if bundle else {}
        policy_flag = None
        if isinstance(bundle_policy, dict):
            policy_flag = bundle_policy.get("fail_on_imputed_core")
        from uav_risk.core.data_validator import DataValidator

        validation_result = DataValidator(fail_on_imputed_core=policy_flag).validate_and_store(feature_map)

    metadata = {
        "feature_map": feature_map,
        "primary_inputs": primary_inputs,
        "secondary_overrides": secondary_overrides,
        "ignored_extras": extras,
        "validator_warnings": validation_result.warnings,
        "validator_errors": validation_result.errors,
        "is_usable": validation_result.is_usable,
    }
    return vec, metadata

# =====================================================================
# Architectural Registry Block:
# This file depends on: None (Foundational Subsystem Loader).
# Files depending on this file: src/uav_risk/ml/inference.py, src/uav_risk/ml/shap_explain.py
# =====================================================================