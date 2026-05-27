from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from uav_risk.core.contracts import (
    AssessmentCoreInput,
    DroneProfileRaw,
    EnvironmentData,
    GPSData,
    MasterFlightPayload,
    MissionParams,
    OperatorData,
    RawFeatureAssemblyResult,
    RawSecondaryOverrides,
    ScenarioRawInput,
    UAVSpecs,
)
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
    if name.endswith("_ok") or name in {"swarm_enabled", "mission_runway_required"}:
        return 0.0
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


def _profile_payload(profile_id: str = "prof_a") -> dict[str, object]:
    payload = {name: _value_for(name) for name in PROFILE_DERIVED_RAW_FEATURES}
    payload.update(
        {
            "user_id": "user_1",
            "profile_id": profile_id,
            "profile_name": f"Profile {profile_id}",
            "max_payload_kg": 2.5,
            "max_takeoff_mass_kg": 10.0,
            "runway_capable": False,
            "swarm_capable": True,
            "max_swarm_size": 5,
        }
    )
    return payload


def _scenario_payload() -> dict[str, object]:
    return {name: _value_for(name) for name in SCENARIO_REQUIRED_RAW_FEATURES}


def test_drone_profile_raw_accepts_profile_identity_capability_and_16_raw_fields():
    profile = DroneProfileRaw(**_profile_payload())

    assert len([name for name in PROFILE_DERIVED_RAW_FEATURES if name in DroneProfileRaw.model_fields]) == 16
    assert set(PROFILE_IDENTITY_FIELDS).issubset(DroneProfileRaw.model_fields)
    assert set(PROFILE_CAPABILITY_FIELDS).issubset(DroneProfileRaw.model_fields)
    assert profile.profile_id == "prof_a"


def test_drone_profile_raw_supports_multiple_profiles_for_same_user():
    first = DroneProfileRaw(**_profile_payload("prof_a"))
    second = DroneProfileRaw(**_profile_payload("prof_b"))

    assert first.user_id == second.user_id == "user_1"
    assert first.profile_id != second.profile_id


def test_drone_profile_raw_rejects_processed_onehot_extras():
    payload = _profile_payload()
    payload["uav_energy_source_fuel"] = 1.0

    with pytest.raises(ValidationError):
        DroneProfileRaw(**payload)


def test_scenario_raw_input_has_all_45_scenario_fields():
    scenario = ScenarioRawInput(**_scenario_payload())

    assert len([name for name in SCENARIO_REQUIRED_RAW_FEATURES if name in ScenarioRawInput.model_fields]) == 45


def test_scenario_raw_input_includes_payload_mass():
    scenario = ScenarioRawInput(**_scenario_payload())

    assert "uav_payload_mass_kg" in ScenarioRawInput.model_fields
    assert scenario.uav_payload_mass_kg == _scenario_payload()["uav_payload_mass_kg"]


def test_scenario_raw_input_excludes_profile_derived_fields():
    assert set(SCENARIO_REQUIRED_RAW_FEATURES).isdisjoint(PROFILE_DERIVED_RAW_FEATURES)
    assert set(PROFILE_DERIVED_RAW_FEATURES).isdisjoint(ScenarioRawInput.model_fields)


def test_scenario_raw_input_rejects_processed_onehot_extras():
    payload = _scenario_payload()
    payload["mission_pattern_custom"] = 1.0

    with pytest.raises(ValidationError):
        ScenarioRawInput(**payload)


def test_payload_mass_is_scenario_field_not_profile_field():
    assert "uav_payload_mass_kg" in ScenarioRawInput.model_fields
    assert "uav_payload_mass_kg" not in DroneProfileRaw.model_fields


def test_raw_secondary_overrides_rejects_processed_onehot_features():
    with pytest.raises(ValidationError):
        RawSecondaryOverrides(values={FORBIDDEN_USER_FEATURES[0]: 1.0})


def test_raw_secondary_overrides_rejects_unknown_keys():
    with pytest.raises(ValidationError):
        RawSecondaryOverrides(values={"unknown_feature": 1.0})


def test_raw_secondary_overrides_rejects_internal_only_keys():
    with pytest.raises(ValidationError):
        RawSecondaryOverrides(values={INTERNAL_ONLY_RAW_FEATURES[0]: 1.0})


def test_raw_secondary_overrides_rejects_dropped_metadata_keys():
    with pytest.raises(ValidationError):
        RawSecondaryOverrides(values={DROPPED_RAW_METADATA_FEATURES[0]: 1.0})


def test_raw_secondary_overrides_accepts_controls_action_categorical_override():
    overrides = RawSecondaryOverrides(values={"controls_actions_first": "hold"})

    assert overrides.values["controls_actions_first"] == "hold"


def test_raw_secondary_overrides_rejects_invalid_controls_action():
    with pytest.raises(ValidationError):
        RawSecondaryOverrides(values={"controls_actions_first": "climb"})


def test_raw_secondary_overrides_accepts_numeric_scalar_generated_override():
    candidate = next(
        name
        for name in OPTIONAL_RAW_OVERRIDE_FEATURES
        if name != "controls_actions_first"
    )
    overrides = RawSecondaryOverrides(values={candidate: 12.5})

    assert overrides.values[candidate] == 12.5


@pytest.mark.parametrize("bad_value", ["1.2", [1.2], {"value": 1.2}, math.nan, math.inf])
def test_raw_secondary_overrides_rejects_non_numeric_values_for_numeric_overrides(bad_value):
    candidate = next(
        name
        for name in OPTIONAL_RAW_OVERRIDE_FEATURES
        if name != "controls_actions_first"
    )

    with pytest.raises(ValidationError):
        RawSecondaryOverrides(values={candidate: bad_value})


def test_assessment_core_input_validates_matching_profile_identity():
    profile = DroneProfileRaw(**_profile_payload())
    scenario = ScenarioRawInput(**_scenario_payload())
    request = AssessmentCoreInput(
        user_id=profile.user_id,
        profile_id=profile.profile_id,
        drone_profile=profile,
        scenario=scenario,
    )

    assert request.user_id == profile.user_id
    assert request.profile_id == profile.profile_id


@pytest.mark.parametrize(
    ("user_id", "profile_id"),
    [("other_user", "prof_a"), ("user_1", "other_profile")],
)
def test_assessment_core_input_rejects_mismatched_profile_identity(user_id, profile_id):
    profile = DroneProfileRaw(**_profile_payload())
    scenario = ScenarioRawInput(**_scenario_payload())

    with pytest.raises(ValidationError):
        AssessmentCoreInput(
            user_id=user_id,
            profile_id=profile_id,
            drone_profile=profile,
            scenario=scenario,
        )


def test_raw_feature_assembly_result_contract_instantiates():
    result = RawFeatureAssemblyResult(
        user_id="user_1",
        profile_id="prof_a",
        raw_feature_names=list(RAW_FEATURE_NAMES),
        raw_feature_map={},
        raw_vector_length=197,
        profile_features={},
        scenario_features={},
        generated_features={},
        secondary_overrides={},
        dropped_metadata_defaults={},
        ignored_extras={},
        hard_vetoes=[],
        warnings=[],
    )

    assert result.raw_vector_length == 197


def test_legacy_contracts_still_import_and_instantiate():
    assert UAVSpecs()
    assert MissionParams()
    assert EnvironmentData()
    assert GPSData()
    assert OperatorData()
    assert MasterFlightPayload()
    assert RAW_CATEGORICAL_FEATURES["controls_actions_first"] == ("fwd", "hold", "throttle")
