from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from uav_risk.ml import inference as inference_module
from uav_risk.ml.inference import predict_processed_vector, run_stage1_inference
from uav_risk.ml.loader import (
    ModelLoadError,
    assemble_feature_vector_from_dict,
    assemble_raw_feature_vector,
    load_stage1_bundle,
)
from uav_risk.ml.raw_schema import FORBIDDEN_USER_FEATURES, RAW_FEATURE_NAMES
from uav_risk.ml import loader as loader_module
from uav_risk.ml.raw_schema import PROFILE_DERIVED_RAW_FEATURES, SCENARIO_REQUIRED_RAW_FEATURES


def _value_for(name: str):
    if name == "uav_energy_source":
        return "battery"
    if name == "mission_pattern":
        return "custom"
    if name == "controls_mode":
        return "discrete"
    if name == "swarm_roles_first":
        return "single"
    if name == "spawn_xyz_first":
        return 50.0
    if name in {"mission_runway_required", "swarm_enabled", "environment_gnss_multipath", "environment_em_interference"}:
        return 0.0
    if name in {"comms_uplink_ok", "comms_downlink_ok"}:
        return 1.0
    if name.endswith("_count"):
        return 1.0
    if name == "mission_waypoints_count":
        return 2.0
    if name == "mission_time_budget_s":
        return 600.0
    if name == "mission_loiter_radius_m":
        return 30.0
    if name == "airspace_altitude_agl_min_m":
        return 10.0
    if name == "airspace_altitude_agl_max_m":
        return 50.0
    if name == "uav_rotorcraft_hover_ceiling_m":
        return 1000.0
    if name == "uav_rotorcraft_rotor_count":
        return 4.0
    if name == "uav_mass_kg":
        return 2.0
    if name == "uav_payload_mass_kg":
        return 0.5
    if name == "uav_max_tilt_deg":
        return 30.0
    if name == "uav_max_speed_mps":
        return 15.0
    if name == "uav_reserve_fraction":
        return 0.7
    if name == "comms_rssi_dbm_min":
        return -50.0
    if name == "environment_gnss_jam_dbm":
        return -125.0
    if name == "environment_weather_wind_mps":
        return 8.0
    if name == "environment_weather_gust_mps":
        return 8.0
    if name == "environment_weather_wind_dir_deg":
        return 240.0
    if name == "environment_weather_phenomena_count":
        return 1.0
    return 1.0


def _profile(**updates):
    data = {name: _value_for(name) for name in PROFILE_DERIVED_RAW_FEATURES}
    data.update(
        {
            "user_id": "user_1",
            "profile_id": "profile_1",
            "profile_name": "Raw Profile",
            "max_payload_kg": 5.0,
            "max_takeoff_mass_kg": 20.0,
            "runway_capable": True,
            "swarm_capable": True,
            "max_swarm_size": 5,
        }
    )
    data.update(updates)
    return data


def _scenario(**updates):
    data = {name: _value_for(name) for name in SCENARIO_REQUIRED_RAW_FEATURES}
    data.update(updates)
    return data


def test_production_raw_assembly_returns_raw_197():
    bundle = load_stage1_bundle("artifacts")
    raw_vector, meta = assemble_raw_feature_vector(_profile(), _scenario(), bundle=bundle)

    assert raw_vector.shape == (197,)
    assert meta["raw_feature_vector_length"] == 197


def test_production_raw_assembly_feature_names_equal_raw_schema():
    bundle = load_stage1_bundle("artifacts")
    _, meta = assemble_raw_feature_vector(_profile(), _scenario(), bundle=bundle)

    assert meta["raw_feature_names"] == list(RAW_FEATURE_NAMES)


def test_production_raw_assembly_feature_map_contains_all_raw_names():
    bundle = load_stage1_bundle("artifacts")
    _, meta = assemble_raw_feature_vector(_profile(), _scenario(), bundle=bundle)

    assert set(meta["raw_feature_map"]) == set(RAW_FEATURE_NAMES)


def test_production_raw_assembly_contains_no_forbidden_processed_onehots():
    bundle = load_stage1_bundle("artifacts")
    _, meta = assemble_raw_feature_vector(_profile(), _scenario(), bundle=bundle)

    assert set(FORBIDDEN_USER_FEATURES).isdisjoint(meta["raw_feature_map"])


def test_production_raw_assembly_runs_stage1_inference_with_probabilities_and_top_features():
    bundle = load_stage1_bundle("artifacts")
    raw_vector, _ = assemble_raw_feature_vector(_profile(), _scenario(), bundle=bundle)

    result = run_stage1_inference(bundle, raw_vector, compute_shap=True)

    assert result.risk_class.value in set(bundle.class_names)
    assert result.probabilities
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-6
    assert isinstance(result.top_features, list)


def test_bundle_preprocessor_transform_produces_processed_198():
    bundle = load_stage1_bundle("artifacts")
    _, meta = assemble_raw_feature_vector(_profile(), _scenario(), bundle=bundle)
    frame = pd.DataFrame([[meta["raw_feature_map"][name] for name in RAW_FEATURE_NAMES]], columns=list(RAW_FEATURE_NAMES))

    processed = bundle.preprocessor.transform(frame)

    assert processed.shape == (1, 198)
    assert meta["preprocessor_output_length"] == 198
    assert meta["processed_feature_vector_length"] == 198


def test_production_path_does_not_call_legacy_generate_all_features_map(monkeypatch):
    bundle = load_stage1_bundle("artifacts")

    def boom(*args, **kwargs):
        raise AssertionError("legacy bridge called")

    monkeypatch.setattr(loader_module, "generate_all_features_map", boom)

    raw_vector, _ = assemble_raw_feature_vector(_profile(), _scenario(), bundle=bundle)

    assert raw_vector.shape == (197,)


def test_production_assembly_source_is_raw_first_and_not_legacy_bridge():
    source = inspect.getsource(assemble_raw_feature_vector)

    assert "generate_raw_feature_map" in source
    assert "generate_all_features_map" not in source
    assert "build_raw_feature_map" not in source
    assert "get_core_features" not in source
    assert "preprocessor.transform" not in source
    assert "predict_proba" not in source


def test_stage1_inference_source_transforms_raw_before_model_prediction():
    source = inspect.getsource(inference_module.run_stage1_inference)

    assert "preprocessor.transform" in source
    assert "_build_ml_result" in source
    assert source.index("preprocessor.transform") < source.index("_build_ml_result")


def test_legacy_and_diagnostic_paths_are_labeled():
    assert "Legacy compatibility only" in (assemble_feature_vector_from_dict.__doc__ or "")
    assert "diagnostic" in (predict_processed_vector.__doc__ or "").lower()


@pytest.mark.parametrize(
    "target,feature",
    [("profile", "uav_energy_source_fuel"), ("scenario", "mission_pattern_custom"), ("overrides", "controls_mode_discrete")],
)
def test_processed_onehot_field_is_rejected_in_profile_scenario_or_override(target, feature):
    bundle = load_stage1_bundle("artifacts")
    profile = _profile()
    scenario = _scenario()
    overrides = {}
    if target == "profile":
        profile[feature] = 1.0
    elif target == "scenario":
        scenario[feature] = 1.0
    else:
        overrides[feature] = 1.0

    with pytest.raises(ModelLoadError, match="Processed one-hot"):
        assemble_raw_feature_vector(profile, scenario, overrides=overrides, bundle=bundle)


def test_invalid_hard_veto_scenario_does_not_return_raw_vector():
    bundle = load_stage1_bundle("artifacts")
    scenario = _scenario(uav_payload_mass_kg=10.0)

    with pytest.raises(ModelLoadError, match="PAYLOAD_EXCEEDS_PROFILE_LIMIT"):
        assemble_raw_feature_vector(_profile(max_payload_kg=1.0), scenario, bundle=bundle)


def test_legacy_assemble_feature_vector_from_dict_remains_import_compatible():
    assert callable(assemble_feature_vector_from_dict)


def test_explicit_processed_prediction_diagnostic_path_still_works():
    bundle = load_stage1_bundle("artifacts")
    splits = np.load("artifacts/processed_splits_final.npz", allow_pickle=True)
    processed_vector = splits["X_test"][0]

    result = predict_processed_vector(bundle, processed_vector, compute_shap=False)

    assert result.risk_class.value in set(bundle.class_names)
