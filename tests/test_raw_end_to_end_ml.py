from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from uav_risk.core.contracts import AssessmentCoreInput, DroneProfileRaw, ScenarioRawInput
from uav_risk.ml import loader as loader_module
from uav_risk.ml.inference import predict_processed_vector, run_stage1_inference
from uav_risk.ml.loader import ModelLoadError, assemble_raw_feature_vector, load_stage1_bundle
from uav_risk.ml.raw_schema import (
    FORBIDDEN_USER_FEATURES,
    GENERATED_RAW_FEATURES,
    INTERNAL_ONLY_RAW_FEATURES,
    PROFILE_DERIVED_RAW_FEATURES,
    RAW_FEATURE_NAMES,
    SCENARIO_REQUIRED_RAW_FEATURES,
)


def _base_value(name: str):
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
        return 3.0
    if name == "environment_weather_gust_mps":
        return 3.0
    if name == "environment_weather_wind_dir_deg":
        return 240.0
    if name == "environment_weather_phenomena_count":
        return 0.0
    return 1.0


def valid_profile(**updates) -> dict[str, object]:
    data = {name: _base_value(name) for name in PROFILE_DERIVED_RAW_FEATURES}
    data.update(
        {
            "user_id": "user_1",
            "profile_id": "profile_1",
            "profile_name": "Selected Drone",
            "max_payload_kg": 8.0,
            "max_takeoff_mass_kg": 60.0,
            "runway_capable": True,
            "swarm_capable": True,
            "max_swarm_size": 8,
        }
    )
    data.update(updates)
    return data


def low_like_scenario(**updates) -> dict[str, object]:
    data = {name: _base_value(name) for name in SCENARIO_REQUIRED_RAW_FEATURES}
    data.update(updates)
    return data


def medium_like_scenario(**updates) -> dict[str, object]:
    data = low_like_scenario(
        environment_weather_wind_mps=8.0,
        environment_weather_gust_mps=10.0,
        traffic_count=2.0,
        faults_count=1.0,
        faults_sample_severity=0.8,
        faults_sample_duration_s=30.0,
        environment_weather_phenomena_count=1.0,
        airspace_altitude_agl_max_m=100.0,
    )
    data.update(updates)
    return data


def high_like_scenario(**updates) -> dict[str, object]:
    data = low_like_scenario(
        environment_weather_wind_mps=14.0,
        environment_weather_gust_mps=18.0,
        traffic_count=4.0,
        faults_count=2.0,
        faults_sample_severity=1.0,
        faults_sample_duration_s=120.0,
        environment_weather_phenomena_count=3.0,
        airspace_altitude_agl_max_m=250.0,
        uav_payload_mass_kg=5.0,
        swarm_enabled=1.0,
        swarm_size=4.0,
        comms_rssi_dbm_min=-110.0,
        comms_uplink_ok=0.0,
        moving_obstacles_count=3.0,
        airspace_no_fly_zones_count=3.0,
    )
    data.update(updates)
    return data


def _run_case(bundle, scenario, label: str, capsys):
    raw_vector, meta = assemble_raw_feature_vector(valid_profile(), scenario, bundle=bundle)
    assert raw_vector.shape == (197,)
    assert meta["raw_feature_names"] == list(RAW_FEATURE_NAMES)
    assert set(meta["raw_feature_map"]) == set(RAW_FEATURE_NAMES)
    assert set(FORBIDDEN_USER_FEATURES).isdisjoint(meta["raw_feature_map"])

    result = run_stage1_inference(bundle, raw_vector, compute_shap=True)
    print(f"{label}: {result.risk_class.value} {result.probabilities}")
    captured = capsys.readouterr()
    assert label in captured.out
    assert result.risk_class.value in set(bundle.class_names)
    assert set(result.probabilities) == set(bundle.class_names)
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-6
    assert isinstance(result.top_features, list)
    return result


def test_raw_profile_scenario_to_ml_prediction_for_three_scenarios(capsys):
    bundle = load_stage1_bundle("artifacts")
    results = [
        _run_case(bundle, low_like_scenario(), "low-like", capsys),
        _run_case(bundle, medium_like_scenario(), "medium-like", capsys),
        _run_case(bundle, high_like_scenario(), "high-like", capsys),
    ]
    probability_vectors = [tuple(round(result.probabilities[name], 12) for name in bundle.class_names) for result in results]

    assert len(set(probability_vectors)) > 1


def test_missing_or_empty_profile_cannot_reach_inference():
    bundle = load_stage1_bundle("artifacts")

    with pytest.raises(ModelLoadError):
        assemble_raw_feature_vector({}, low_like_scenario(), bundle=bundle)


def test_missing_required_profile_field_cannot_reach_inference():
    bundle = load_stage1_bundle("artifacts")
    profile = valid_profile()
    profile.pop("uav_mass_kg")

    with pytest.raises(ModelLoadError, match="MISSING_PROFILE_FIELD"):
        assemble_raw_feature_vector(profile, low_like_scenario(), bundle=bundle)


def test_invalid_profile_category_cannot_reach_inference():
    bundle = load_stage1_bundle("artifacts")

    with pytest.raises(ModelLoadError, match="INVALID_CATEGORY"):
        assemble_raw_feature_vector(valid_profile(uav_energy_source="steam"), low_like_scenario(), bundle=bundle)


def test_assessment_identity_mismatch_cannot_pass_contract_path():
    with pytest.raises(ValidationError):
        AssessmentCoreInput(
            user_id="other_user",
            profile_id="profile_1",
            drone_profile=DroneProfileRaw(**valid_profile()),
            scenario=ScenarioRawInput(**low_like_scenario()),
        )


def test_missing_required_scenario_field_cannot_reach_inference():
    bundle = load_stage1_bundle("artifacts")
    scenario = low_like_scenario()
    scenario.pop("mission_time_budget_s")

    with pytest.raises(ModelLoadError, match="MISSING_SCENARIO_FIELD"):
        assemble_raw_feature_vector(valid_profile(), scenario, bundle=bundle)


def test_invalid_scenario_category_cannot_reach_inference():
    bundle = load_stage1_bundle("artifacts")

    with pytest.raises(ModelLoadError, match="INVALID_CATEGORY"):
        assemble_raw_feature_vector(valid_profile(), low_like_scenario(mission_pattern="zigzag"), bundle=bundle)


def test_invalid_scenario_numeric_cannot_reach_inference():
    bundle = load_stage1_bundle("artifacts")

    with pytest.raises(ModelLoadError, match="INVALID_NUMERIC"):
        assemble_raw_feature_vector(valid_profile(), low_like_scenario(uav_payload_mass_kg=-1.0), bundle=bundle)
    with pytest.raises(ModelLoadError, match="INVALID_NUMERIC"):
        assemble_raw_feature_vector(valid_profile(), low_like_scenario(mission_time_budget_s=-1.0), bundle=bundle)


def test_invalid_altitude_range_cannot_reach_inference():
    bundle = load_stage1_bundle("artifacts")

    with pytest.raises(ModelLoadError, match="ALTITUDE_RANGE_INVALID"):
        assemble_raw_feature_vector(
            valid_profile(),
            low_like_scenario(airspace_altitude_agl_min_m=100.0, airspace_altitude_agl_max_m=100.0),
            bundle=bundle,
        )


@pytest.mark.parametrize(
    ("profile_updates", "scenario_updates", "code"),
    [
        ({"max_payload_kg": 1.0}, {"uav_payload_mass_kg": 2.0}, "PAYLOAD_EXCEEDS_PROFILE_LIMIT"),
        ({"uav_rotorcraft_hover_ceiling_m": 40.0}, {"airspace_altitude_agl_max_m": 50.0}, "ALTITUDE_EXCEEDS_HOVER_CEILING"),
        ({"swarm_capable": False}, {"swarm_enabled": 1.0}, "SWARM_NOT_CAPABLE"),
        ({"max_swarm_size": 2}, {"swarm_enabled": 1.0, "swarm_size": 3.0}, "SWARM_SIZE_EXCEEDS_PROFILE_LIMIT"),
        ({"runway_capable": False}, {"mission_runway_required": 1.0}, "RUNWAY_NOT_CAPABLE"),
    ],
)
def test_structural_hard_veto_cases_cannot_reach_inference(profile_updates, scenario_updates, code):
    bundle = load_stage1_bundle("artifacts")

    with pytest.raises(ModelLoadError, match=code):
        assemble_raw_feature_vector(valid_profile(**profile_updates), low_like_scenario(**scenario_updates), bundle=bundle)


def test_no_overrides_generates_values_and_ml_can_run():
    bundle = load_stage1_bundle("artifacts")
    raw_vector, meta = assemble_raw_feature_vector(valid_profile(), low_like_scenario(), bundle=bundle)

    assert meta["raw_feature_map"]["environment_thermal_plumes_sample_radius_m"] == 50.0
    assert meta["raw_feature_map"]["sim_policy_frequency"] == 15.0
    result = run_stage1_inference(bundle, raw_vector, compute_shap=False)
    assert result.probabilities


def test_valid_secondary_override_precedes_generated_value_and_updates_missing_flag():
    bundle = load_stage1_bundle("artifacts")
    _, meta = assemble_raw_feature_vector(
        valid_profile(),
        low_like_scenario(),
        overrides={"uav_aero_wing_area_m2": 2.0},
        bundle=bundle,
    )

    assert meta["raw_feature_map"]["uav_aero_wing_area_m2"] == 2.0
    assert meta["raw_feature_map"]["uav_aero_wing_area_m2_was_missing"] == 0.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"uav_energy_source_fuel": 1.0}, "Processed one-hot"),
        ({"unknown_feature": 1.0}, "INVALID_OVERRIDE_KEY"),
        ({INTERNAL_ONLY_RAW_FEATURES[0]: 1.0}, "INTERNAL_ONLY_OVERRIDE"),
        ({"uav_aero_wing_area_m2": "2.0"}, "INVALID_NUMERIC"),
    ],
)
def test_invalid_overrides_are_rejected_before_ml(overrides, message):
    bundle = load_stage1_bundle("artifacts")

    with pytest.raises(ModelLoadError, match=message):
        assemble_raw_feature_vector(valid_profile(), low_like_scenario(), overrides=overrides, bundle=bundle)


def test_valid_controls_actions_first_override_is_accepted():
    bundle = load_stage1_bundle("artifacts")
    _, meta = assemble_raw_feature_vector(
        valid_profile(),
        low_like_scenario(),
        overrides={"controls_actions_first": "hold"},
        bundle=bundle,
    )

    assert meta["raw_feature_map"]["controls_actions_first"] == "hold"


def test_generated_features_are_not_required_from_user_scenario():
    bundle = load_stage1_bundle("artifacts")
    scenario = low_like_scenario()

    assert set(scenario) == set(SCENARIO_REQUIRED_RAW_FEATURES)
    assert set(GENERATED_RAW_FEATURES).isdisjoint(scenario)
    _, meta = assemble_raw_feature_vector(valid_profile(), scenario, bundle=bundle)
    assert set(meta["raw_feature_map"]) == set(RAW_FEATURE_NAMES)


def test_normal_production_path_does_not_call_legacy_generate_all_features_map(monkeypatch):
    bundle = load_stage1_bundle("artifacts")

    def boom(*args, **kwargs):
        raise AssertionError("legacy generate_all_features_map called")

    monkeypatch.setattr(loader_module, "generate_all_features_map", boom)
    raw_vector, _ = assemble_raw_feature_vector(valid_profile(), low_like_scenario(), bundle=bundle)
    result = run_stage1_inference(bundle, raw_vector, compute_shap=False)

    assert result.probabilities


def test_explicit_predict_processed_vector_diagnostic_path_still_works():
    bundle = load_stage1_bundle("artifacts")
    splits = np.load("artifacts/processed_splits_final.npz", allow_pickle=True)

    result = predict_processed_vector(bundle, splits["X_test"][0], compute_shap=False)

    assert result.risk_class.value in set(bundle.class_names)
