from __future__ import annotations

from fastapi import APIRouter, Depends

from uav_risk.api.dependencies import get_stage1_bundle
from uav_risk.ml.raw_schema import (
    DROPPED_RAW_METADATA_FEATURES,
    FORBIDDEN_USER_FEATURES,
    INTERNAL_ONLY_RAW_FEATURES,
    OPTIONAL_RAW_OVERRIDE_FEATURES,
    PROFILE_CAPABILITY_FIELDS,
    PROFILE_DERIVED_RAW_FEATURES,
    PROFILE_IDENTITY_FIELDS,
    RAW_CATEGORICAL_FEATURES,
    RAW_FEATURE_NAMES,
    SCENARIO_REQUIRED_RAW_FEATURES,
)
from uav_risk.ml.schemas import Stage1Bundle

router = APIRouter()


@router.get("/model/metadata")
def model_metadata(bundle: Stage1Bundle = Depends(get_stage1_bundle)) -> dict[str, object]:
    return {
        "raw_feature_count": len(RAW_FEATURE_NAMES),
        "processed_feature_count": len(bundle.feature_names),
        "class_names": list(bundle.class_names),
        "production_path": "raw_197 -> preprocessor -> processed_198 -> model",
    }


@router.get("/features/raw-schema")
def raw_schema(bundle: Stage1Bundle = Depends(get_stage1_bundle)) -> dict[str, object]:
    return {
        "raw_feature_names": list(RAW_FEATURE_NAMES),
        "processed_feature_names": list(bundle.feature_names),
        "raw_categorical_features": {k: list(v) for k, v in RAW_CATEGORICAL_FEATURES.items()},
        "forbidden_user_features": list(FORBIDDEN_USER_FEATURES),
        "raw_feature_count": len(RAW_FEATURE_NAMES),
        "processed_feature_count": len(bundle.feature_names),
    }


@router.get("/features/profile-fields")
def profile_fields() -> dict[str, object]:
    return {
        "profile_identity_fields": list(PROFILE_IDENTITY_FIELDS),
        "profile_derived_raw_features": list(PROFILE_DERIVED_RAW_FEATURES),
        "profile_capability_fields": list(PROFILE_CAPABILITY_FIELDS),
        "count_profile_derived_raw_features": len(PROFILE_DERIVED_RAW_FEATURES),
    }


@router.get("/features/scenario-fields")
def scenario_fields() -> dict[str, object]:
    scenario_categories = {
        k: list(v) for k, v in RAW_CATEGORICAL_FEATURES.items() if k in set(SCENARIO_REQUIRED_RAW_FEATURES)
    }
    return {
        "scenario_required_raw_features": list(SCENARIO_REQUIRED_RAW_FEATURES),
        "raw_categorical_features": scenario_categories,
        "count_scenario_required_raw_features": len(SCENARIO_REQUIRED_RAW_FEATURES),
    }


@router.get("/features/secondary-overrides")
def secondary_overrides() -> dict[str, object]:
    return {
        "optional_raw_override_features": list(OPTIONAL_RAW_OVERRIDE_FEATURES),
        "internal_only_raw_features": list(INTERNAL_ONLY_RAW_FEATURES),
        "dropped_raw_metadata_features": list(DROPPED_RAW_METADATA_FEATURES),
        "controls_actions_first_allowed_values": list(RAW_CATEGORICAL_FEATURES["controls_actions_first"]),
    }
