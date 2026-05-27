from __future__ import annotations

import math

from uav_risk.core.data_validator import (
    run_structural_hard_veto,
    validate_assessment_core_input_raw,
    validate_drone_profile_raw,
    validate_scenario_raw,
    validate_secondary_overrides_raw,
)
from uav_risk.ml.raw_schema import (
    DROPPED_RAW_METADATA_FEATURES,
    FORBIDDEN_USER_FEATURES,
    GENERATED_RAW_FEATURES,
    INTERNAL_ONLY_RAW_FEATURES,
    OPTIONAL_RAW_OVERRIDE_FEATURES,
    PROFILE_DERIVED_RAW_FEATURES,
    RAW_CATEGORICAL_FEATURES,
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
        return 0.0
    if name in {
        "mission_runway_required",
        "swarm_enabled",
        "comms_uplink_ok",
        "comms_downlink_ok",
        "environment_gnss_multipath",
        "environment_em_interference",
    }:
        return 1.0 if name in {"comms_uplink_ok", "comms_downlink_ok"} else 0.0
    if name.endswith("_count"):
        return 1.0
    if name == "mission_time_budget_s":
        return 600.0
    if name == "airspace_altitude_agl_min_m":
        return 10.0
    if name == "airspace_altitude_agl_max_m":
        return 50.0
    if name == "uav_rotorcraft_hover_ceiling_m":
        return 1000.0
    if name == "comms_rssi_dbm_min":
        return -50.0
    if name == "environment_gnss_jam_dbm":
        return -125.0
    return 1.0


def _profile(**updates):
    data = {name: _value_for(name) for name in PROFILE_DERIVED_RAW_FEATURES}
    data.update(
        {
            "user_id": "user_1",
            "profile_id": "profile_1",
            "profile_name": "Test Profile",
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


def _assessment(profile=None, scenario=None, overrides=None, **updates):
    profile = _profile() if profile is None else profile
    scenario = _scenario() if scenario is None else scenario
    data = {
        "user_id": profile.get("user_id", "user_1"),
        "profile_id": profile.get("profile_id", "profile_1"),
        "drone_profile": profile,
        "scenario": scenario,
        "secondary_overrides": {"values": overrides or {}},
    }
    data.update(updates)
    return data


def _codes(result):
    return {issue.code for issue in result.issues}


def _numeric_override_key():
    return next(name for name in OPTIONAL_RAW_OVERRIDE_FEATURES if name != "controls_actions_first")


def test_valid_profile_passes():
    assert validate_drone_profile_raw(_profile()).passed


def test_valid_scenario_passes():
    assert validate_scenario_raw(_scenario()).passed


def test_valid_assessment_passes_hard_veto():
    result = run_structural_hard_veto(_assessment())

    assert result.passed
    assert result.issues == []


def test_missing_profile_field_fails():
    profile = _profile()
    profile.pop("uav_mass_kg")

    result = validate_drone_profile_raw(profile)

    assert "MISSING_PROFILE_FIELD" in _codes(result)


def test_missing_scenario_field_fails():
    scenario = _scenario()
    scenario.pop("mission_time_budget_s")

    result = validate_scenario_raw(scenario)

    assert "MISSING_SCENARIO_FIELD" in _codes(result)


def test_generated_raw_features_are_not_required_from_scenario_input():
    scenario = _scenario()

    assert set(GENERATED_RAW_FEATURES).isdisjoint(scenario)
    assert validate_scenario_raw(scenario).passed


def test_invalid_profile_category_fails():
    result = validate_drone_profile_raw(_profile(uav_energy_source="steam"))

    assert "INVALID_CATEGORY" in _codes(result)


def test_invalid_scenario_category_fails():
    result = validate_scenario_raw(_scenario(mission_pattern="zigzag"))

    assert "INVALID_CATEGORY" in _codes(result)


def test_processed_onehot_override_fails():
    result = validate_secondary_overrides_raw({FORBIDDEN_USER_FEATURES[0]: 1.0})

    assert "FORBIDDEN_PROCESSED_FEATURE" in _codes(result)


def test_unknown_override_fails():
    result = validate_secondary_overrides_raw({"unknown_feature": 1.0})

    assert "INVALID_OVERRIDE_KEY" in _codes(result)


def test_internal_only_override_fails():
    result = validate_secondary_overrides_raw({INTERNAL_ONLY_RAW_FEATURES[0]: 1.0})

    assert "INTERNAL_ONLY_OVERRIDE" in _codes(result)


def test_dropped_metadata_override_fails():
    result = validate_secondary_overrides_raw({DROPPED_RAW_METADATA_FEATURES[0]: 1.0})

    assert "DROPPED_METADATA_OVERRIDE" in _codes(result)


def test_invalid_controls_actions_first_override_fails():
    result = validate_secondary_overrides_raw({"controls_actions_first": "climb"})

    assert "INVALID_CATEGORY" in _codes(result)


def test_valid_controls_actions_first_override_passes():
    result = validate_secondary_overrides_raw({"controls_actions_first": RAW_CATEGORICAL_FEATURES["controls_actions_first"][0]})

    assert result.passed


def test_non_numeric_override_fails():
    result = validate_secondary_overrides_raw({_numeric_override_key(): "12.0"})

    assert "INVALID_NUMERIC" in _codes(result)


def test_nan_and_infinity_override_fail():
    key = _numeric_override_key()

    assert "INVALID_NUMERIC" in _codes(validate_secondary_overrides_raw({key: math.nan}))
    assert "INVALID_NUMERIC" in _codes(validate_secondary_overrides_raw({key: math.inf}))


def test_max_altitude_less_than_or_equal_to_min_altitude_fails():
    result = validate_scenario_raw(_scenario(airspace_altitude_agl_min_m=100.0, airspace_altitude_agl_max_m=100.0))

    assert "ALTITUDE_RANGE_INVALID" in _codes(result)


def test_payload_above_profile_limit_fails():
    result = run_structural_hard_veto(_assessment(scenario=_scenario(uav_payload_mass_kg=6.0)))

    assert "PAYLOAD_EXCEEDS_PROFILE_LIMIT" in _codes(result)


def test_mass_above_takeoff_mass_limit_fails():
    result = run_structural_hard_veto(_assessment(profile=_profile(uav_mass_kg=30.0, max_takeoff_mass_kg=20.0)))

    assert "MASS_EXCEEDS_PROFILE_LIMIT" in _codes(result)


def test_mission_altitude_above_hover_ceiling_fails():
    result = run_structural_hard_veto(
        _assessment(profile=_profile(uav_rotorcraft_hover_ceiling_m=40.0), scenario=_scenario(airspace_altitude_agl_max_m=50.0))
    )

    assert "ALTITUDE_EXCEEDS_HOVER_CEILING" in _codes(result)


def test_swarm_enabled_but_profile_not_swarm_capable_fails():
    result = run_structural_hard_veto(
        _assessment(profile=_profile(swarm_capable=False), scenario=_scenario(swarm_enabled=1.0))
    )

    assert "SWARM_NOT_CAPABLE" in _codes(result)


def test_swarm_size_above_profile_limit_fails():
    result = run_structural_hard_veto(
        _assessment(profile=_profile(max_swarm_size=2), scenario=_scenario(swarm_size=3.0))
    )

    assert "SWARM_SIZE_EXCEEDS_PROFILE_LIMIT" in _codes(result)


def test_runway_required_but_profile_not_runway_capable_fails():
    result = run_structural_hard_veto(
        _assessment(profile=_profile(runway_capable=False), scenario=_scenario(mission_runway_required=1.0))
    )

    assert "RUNWAY_NOT_CAPABLE" in _codes(result)


def test_no_universal_altitude_wind_or_mass_policy_threshold_is_enforced():
    profile = _profile(uav_mass_kg=30.0, max_takeoff_mass_kg=None, uav_rotorcraft_hover_ceiling_m=500.0)
    scenario = _scenario(environment_weather_wind_mps=20.0, airspace_altitude_agl_min_m=10.0, airspace_altitude_agl_max_m=150.0)

    result = run_structural_hard_veto(_assessment(profile=profile, scenario=scenario))

    assert result.passed
    assert "MASS_EXCEEDS_PROFILE_LIMIT" not in _codes(result)
    assert "ALTITUDE_EXCEEDS_HOVER_CEILING" not in _codes(result)


def test_assessment_user_and_profile_mismatch_fail():
    result = validate_assessment_core_input_raw(_assessment(user_id="other", profile_id="other_profile"))

    assert "USER_ID_MISMATCH" in _codes(result)
    assert "PROFILE_ID_MISMATCH" in _codes(result)
