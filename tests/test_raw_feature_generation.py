from __future__ import annotations

import pandas as pd

from uav_risk.core.feature_engineering import generate_all_features_map, generate_raw_feature_map
from uav_risk.ml.loader import load_stage1_bundle
from uav_risk.ml.raw_schema import (
    FORBIDDEN_USER_FEATURES,
    PROFILE_DERIVED_RAW_FEATURES,
    RAW_FEATURE_NAMES,
    SCENARIO_REQUIRED_RAW_FEATURES,
)


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
    if name in {
        "mission_runway_required",
        "swarm_enabled",
        "environment_gnss_multipath",
        "environment_em_interference",
    }:
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


def test_valid_profile_and_scenario_return_raw_feature_assembly_result():
    result = generate_raw_feature_map(_profile(), _scenario())

    assert result.hard_vetoes == []
    assert result.raw_vector_length == 197
    assert result.raw_feature_names == list(RAW_FEATURE_NAMES)
    assert set(result.raw_feature_map) == set(RAW_FEATURE_NAMES)


def test_forbidden_processed_features_never_appear_in_raw_map():
    result = generate_raw_feature_map(_profile(), _scenario())

    assert set(FORBIDDEN_USER_FEATURES).isdisjoint(result.raw_feature_map)


def test_generated_defaults_match_calibration():
    result = generate_raw_feature_map(_profile(), _scenario())
    fmap = result.raw_feature_map

    assert fmap["sim_policy_frequency"] == 15.0
    assert fmap["sim_duration_steps"] == 900.0
    assert fmap["controls_actions_first"] == "fwd"
    assert fmap["environment_wind_profile_count"] == 3.0
    assert fmap["environment_thermal_plumes_count"] == 1.0
    assert fmap["environment_thermal_plumes_sample_radius_m"] == 50.0
    assert fmap["environment_thermal_plumes_sample_w_up_mps"] == 1.8
    assert fmap["uav_aero_wing_area_m2"] == 1.2
    assert fmap["uav_aero_aspect_ratio"] == 10.2
    assert fmap["uav_aero_cl_max"] == 1.4
    assert fmap["uav_aero_cd0"] == 0.025
    assert fmap["uav_aero_stall_speed_mps"] == 12.5
    assert fmap["airspace_runway_threshold_count"] == 3.0


def test_spawn_xyz_first_collection_returns_hard_veto_not_silent_conversion():
    result = generate_raw_feature_map(_profile(), _scenario(spawn_xyz_first=[1.0, 2.0, 3.0]))

    assert result.raw_vector_length == 0
    assert result.raw_feature_map == {}
    assert any("INVALID_NUMERIC" in veto for veto in result.hard_vetoes)


def test_secondary_override_changes_allowed_generated_field():
    result = generate_raw_feature_map(
        _profile(),
        _scenario(),
        overrides={"environment_thermal_plumes_sample_radius_m": 75.0},
    )

    assert result.hard_vetoes == []
    assert result.raw_feature_map["environment_thermal_plumes_sample_radius_m"] == 75.0


def test_secondary_override_updates_related_missing_flag_to_zero():
    result = generate_raw_feature_map(
        _profile(),
        _scenario(),
        overrides={"uav_aero_wing_area_m2": 2.0},
    )

    assert result.raw_feature_map["uav_aero_wing_area_m2"] == 2.0
    assert result.raw_feature_map["uav_aero_wing_area_m2_was_missing"] == 0.0


def test_processed_onehot_override_is_rejected():
    result = generate_raw_feature_map(_profile(), _scenario(), overrides={"uav_energy_source_fuel": 1.0})

    assert result.raw_vector_length == 0
    assert any("FORBIDDEN_PROCESSED_FEATURE" in veto for veto in result.hard_vetoes)


def test_raw_result_transforms_to_processed_198_with_bundle_preprocessor():
    bundle = load_stage1_bundle("artifacts")
    result = generate_raw_feature_map(_profile(), _scenario())
    frame = pd.DataFrame([[result.raw_feature_map[name] for name in RAW_FEATURE_NAMES]], columns=list(RAW_FEATURE_NAMES))

    processed = bundle.preprocessor.transform(frame)

    assert processed.shape == (1, 198)
    assert list(bundle.preprocessor.get_feature_names_out()) == bundle.feature_names


def test_legacy_generate_all_features_map_still_imports():
    assert callable(generate_all_features_map)
