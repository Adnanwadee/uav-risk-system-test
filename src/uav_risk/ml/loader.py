"""
Module: uav_risk.ml.loader
Purpose: High-integrity loader for Stage-1 production bundle and separate model/preprocessor artifacts,
         strictly aligned with feature_defs and schematized structures.
Dependencies: uav_risk.ml.schemas, uav_risk.ml.feature_defs, uav_risk.core, uav_risk.ml.bundle_security
"""
# STAGE6_CLEANUP_REVIEW:
# Classification: MIXED_ACTIVE_LEGACY
# Plan lineage: PLAN3_ACTIVE raw loader plus PLAN1/PLAN2 compatibility bridge.
# Runtime status: load_stage1_bundle(), assemble_raw_feature_vector(), and transform_raw_vector are active/diagnostic paths.
# Legacy signal: assemble_feature_vector_from_dict() remains a legacy processed/mixed bridge for compatibility tests only.
# Replacement: Use assemble_raw_feature_vector() for production assessment paths.
# Action rule: Do not delete this file. Review legacy bridge function only after raw-first tests and API guards remain green.

import json
import structlog
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple, Mapping
from pathlib import Path

from uav_risk.ml import feature_defs
from uav_risk.ml.schemas import Stage1Bundle
from uav_risk.ml.bundle_security import safe_load_bundle
from uav_risk.core.feature_engineering import generate_all_features_map, generate_raw_feature_map, split_primary_and_secondary_overrides
from uav_risk.core.data_validator import DataValidator
from uav_risk.ml.raw_schema import (
    PROCESSED_FEATURE_NAMES,
    RAW_FEATURE_NAMES,
    build_raw_feature_map,
    categorical_to_processed_flags,
    get_raw_feature_names,
    reject_processed_onehot_inputs,
)

logger = structlog.get_logger(__name__)

class ModelLoadError(Exception):
    pass

BUNDLE_FILE = "stage1_production_bundle.pkl"
MODEL_FILE = "final_model.pkl"
PREPROCESSOR_FILE = "preprocessing_pipeline_final.pkl"
FEATURE_MAPPING_FILE = "stage1_feature_mapping.json"
MODEL_CARD_FILE = "model_card.json"
INFERENCE_CONFIG_FILE = "stage1_inference_config.json"


def load_stage1_bundle(artifacts_dir: str) -> Stage1Bundle:
  
    artifacts_path = Path(artifacts_dir)

    model_path = artifacts_path / MODEL_FILE
    preprocessor_path = artifacts_path / PREPROCESSOR_FILE
    bundle_path = artifacts_path / BUNDLE_FILE

    if not model_path.exists():
        raise ModelLoadError(f"Model file not found: {model_path}")
    if not bundle_path.exists():
        raise ModelLoadError(f"Bundle file not found: {bundle_path}")

    model = safe_load_bundle(str(model_path))

    preprocessor = None
    if preprocessor_path.exists():
        try:
            preprocessor = safe_load_bundle(str(preprocessor_path))
        except Exception:
            logger.warning(f"Preprocessor present but failed to load: {preprocessor_path}; continuing without it")
    else:
        logger.info(f"Preprocessor not found, continuing without it: {preprocessor_path}")

    bundle_data = safe_load_bundle(str(bundle_path))
    if not isinstance(bundle_data, dict):
        raise ModelLoadError("Bundle file must contain a dictionary with feature_names and class_names.")

    feature_names = bundle_data.get("feature_names")
    class_names = bundle_data.get("class_names")

    if not feature_names or not class_names:
        raise ModelLoadError("Bundle is missing 'feature_names' or 'class_names'.")

    is_aligned, alignment_err = feature_defs.validate_feature_registry_against_artifact(feature_names)
    if not is_aligned:
        logger.critical(f"ALIGNMENT BREACH: {alignment_err}")
        raise AssertionError(f"Model Feature Shift Blocked: {alignment_err}")

    feature_mapping = {name: idx for idx, name in enumerate(feature_names)}

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

    # Prefer the canonical packed objects when present; the standalone artifacts
    # are compatibility fallbacks for older bundles.
    model = bundle_data.get("model", model)
    preprocessor = bundle_data.get("preprocessor", preprocessor)

    if preprocessor is None or not hasattr(preprocessor, "feature_names_in_"):
        raise ModelLoadError("Stage-1 bundle is missing the fitted raw-input preprocessor.")

    try:
        transformed_names = list(preprocessor.get_feature_names_out())
    except Exception as exc:
        raise ModelLoadError(f"Unable to inspect preprocessor output names: {exc}") from exc

    if transformed_names != list(feature_names):
        raise AssertionError("Preprocessor output order does not match Stage-1 processed feature order.")

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


def _mapping_from_contract(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    return dict(value)


def assemble_raw_feature_vector(
    profile: Mapping[str, Any] | Any,
    scenario: Mapping[str, Any] | Any,
    overrides: Mapping[str, Any] | Any | None = None,
    bundle: Stage1Bundle | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Production raw assembly path: profile/scenario -> raw 197 vector.

    This function intentionally does not call the legacy processed 198 bridge,
    the fitted preprocessor, or the model. Normal inference should pass the
    returned raw vector to run_stage1_inference(), which performs preprocessing.
    """
    profile_map = _mapping_from_contract(profile)
    scenario_map = _mapping_from_contract(scenario)
    override_map = _mapping_from_contract(overrides)
    if "values" in override_map and isinstance(override_map.get("values"), Mapping):
        override_values = dict(override_map["values"])
    else:
        override_values = override_map

    try:
        reject_processed_onehot_inputs(profile_map)
        reject_processed_onehot_inputs(scenario_map)
        reject_processed_onehot_inputs(override_values)
    except ValueError as exc:
        raise ModelLoadError(str(exc)) from exc

    assembly = generate_raw_feature_map(profile_map, scenario_map, overrides=override_values)
    if assembly.hard_vetoes:
        raise ModelLoadError("Raw assembly blocked by hard veto: " + "; ".join(assembly.hard_vetoes))

    raw_feature_names = list(RAW_FEATURE_NAMES)
    if assembly.raw_feature_names != raw_feature_names:
        raise ModelLoadError("Raw assembly feature order does not match RAW_FEATURE_NAMES.")
    if set(assembly.raw_feature_map) != set(raw_feature_names):
        raise ModelLoadError("Raw assembly did not produce exactly the raw 197 feature schema.")

    raw_vector = np.array([assembly.raw_feature_map[name] for name in raw_feature_names], dtype=object)
    if raw_vector.shape != (len(raw_feature_names),):
        raise ModelLoadError(f"Raw vector shape mismatch: expected ({len(raw_feature_names)},), got {raw_vector.shape}")

    processed_names = list(PROCESSED_FEATURE_NAMES)
    preprocessor_output_names = None
    if bundle is not None:
        preprocessor_output_names = list(bundle.preprocessor.get_feature_names_out())
        if preprocessor_output_names != processed_names:
            raise ModelLoadError("Bundle preprocessor output names do not match PROCESSED_FEATURE_NAMES.")

    metadata = {
        "raw_assembly_result": assembly,
        "raw_feature_map": assembly.raw_feature_map,
        "raw_feature_names": raw_feature_names,
        "raw_feature_vector_length": len(raw_vector),
        "processed_feature_names": processed_names,
        "processed_feature_vector_length": len(processed_names),
        "preprocessor_output_length": len(preprocessor_output_names or processed_names),
        "preprocessor_output_names": preprocessor_output_names or processed_names,
        "secondary_overrides": override_values,
        "hard_vetoes": [],
    }
    return raw_vector, metadata


def assemble_feature_vector_from_dict(
    input_mapping: Dict[str, Any],
    bundle: Stage1Bundle,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Legacy compatibility only. Do not use in production raw path.

    This bridge accepts the historical processed/mixed 68-core input shape,
    generates a legacy processed physical map, and then reconstructs raw 197 for
    the preprocessor. New production callers must use assemble_raw_feature_vector().
    """
    try:
        reject_processed_onehot_inputs(input_mapping)
    except ValueError as exc:
        raise ModelLoadError(str(exc)) from exc

    raw_input_with_flags = dict(input_mapping)
    try:
        raw_input_with_flags.update(categorical_to_processed_flags(input_mapping, bundle))
    except ValueError as exc:
        raise ModelLoadError(str(exc)) from exc

    primary, overrides, extras = split_primary_and_secondary_overrides(
        raw_input_with_flags, bundle.feature_names
    )

    try:
        processed_physical_map = generate_all_features_map(
            primary, overrides=overrides, feature_order=bundle.feature_names
        )
    except Exception as exc:
        raise ModelLoadError(f"Raw feature generation failed: {exc}") from exc

    validator = DataValidator()
    validation_result = validator.validate_and_store(processed_physical_map)
    if not validation_result.is_usable:
        logger.error("Inference blocked: generated feature validation rejected input", errors=validation_result.errors)
        raise ModelLoadError(f"Pipeline Blocked: {validation_result.errors}")

    try:
        raw_feature_map = build_raw_feature_map(input_mapping, processed_physical_map, bundle)
    except Exception as exc:
        raise ModelLoadError(f"Raw preprocessor feature assembly failed: {exc}") from exc

    raw_feature_names = get_raw_feature_names(bundle)
    raw_vector = np.array([raw_feature_map[name] for name in raw_feature_names], dtype=object)

    metadata = {
        "raw_feature_map": raw_feature_map,
        "raw_feature_names": raw_feature_names,
        "processed_physical_map": processed_physical_map,
        "feature_map": processed_physical_map,
        "primary_inputs": primary,
        "secondary_overrides": overrides,
        "ignored_extras": extras,
        "validator_errors": validation_result.errors,
        "is_usable": validation_result.is_usable,
        "raw_feature_vector_length": len(raw_vector),
        "processed_feature_vector_length": len(bundle.feature_names),
    }
    return raw_vector, metadata


def transform_raw_vector(bundle: Stage1Bundle, raw_vector: np.ndarray) -> np.ndarray:
    """Diagnostic helper: transform an already assembled raw 197 vector to processed 198."""
    raw_names = get_raw_feature_names(bundle)
    if raw_vector.shape != (len(raw_names),):
        raise ModelLoadError(f"Raw vector shape mismatch: expected ({len(raw_names)},), got {raw_vector.shape}")
    raw_frame = pd.DataFrame([raw_vector.tolist()], columns=raw_names)
    processed = bundle.preprocessor.transform(raw_frame)
    processed_array = np.asarray(processed, dtype=np.float64)
    if processed_array.shape != (1, len(bundle.feature_names)):
        raise ModelLoadError(
            f"Processed vector shape mismatch: expected (1, {len(bundle.feature_names)}), got {processed_array.shape}"
        )
    return processed_array.reshape(-1)
