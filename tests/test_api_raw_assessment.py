from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from uav_risk.api.main import create_app
from uav_risk.ml.raw_schema import GENERATED_RAW_FEATURES, SCENARIO_REQUIRED_RAW_FEATURES
from test_api_profiles import profile_payload


def _value_for(name: str):
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
    if name == "uav_payload_mass_kg":
        return 0.5
    if name == "comms_rssi_dbm_min":
        return -50.0
    if name == "environment_gnss_jam_dbm":
        return -125.0
    if name == "environment_weather_wind_mps":
        return 6.0
    if name == "environment_weather_gust_mps":
        return 6.0
    if name == "environment_weather_wind_dir_deg":
        return 240.0
    if name == "environment_weather_phenomena_count":
        return 0.0
    return 1.0


def scenario_payload(**updates):
    data = {name: _value_for(name) for name in SCENARIO_REQUIRED_RAW_FEATURES}
    data.update(updates)
    return data


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("UAV_PROFILE_STORAGE_DIR", str(tmp_path / "profiles"))
    with TestClient(create_app()) as c:
        yield c


@pytest.fixture()
def profile(client):
    payload = profile_payload(user_id="user_1", profile_id="profile_1")
    response = client.post("/users/user_1/profiles", json=payload)
    assert response.status_code == 201
    return payload


def _assess(client, scenario=None, overrides=None, user_id="user_1", profile_id="profile_1"):
    payload = {
        "scenario": scenario or scenario_payload(),
        "secondary_overrides": {"values": overrides or {}},
        "operator_notes": "English notes here.",
    }
    return client.post(f"/users/{user_id}/profiles/{profile_id}/assessments", json=payload)


def test_valid_profile_and_scenario_without_overrides_completes(client, profile):
    response = _assess(client)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["raw_feature_count"] == 197
    assert body["processed_feature_count"] == 198
    assert set(body["ml"]["probabilities"]) == {"High Risk", "Low Risk", "Medium Risk"}


def test_generated_features_are_not_required_in_scenario_body(client, profile):
    scenario = scenario_payload()
    assert set(scenario) == set(SCENARIO_REQUIRED_RAW_FEATURES)
    assert set(GENERATED_RAW_FEATURES).isdisjoint(scenario)

    response = _assess(client, scenario=scenario)

    assert response.status_code == 200
    assert response.json()["status"] == "completed"


def test_valid_secondary_override_is_accepted_and_takes_precedence(client, profile):
    response = _assess(client, overrides={"environment_thermal_plumes_sample_radius_m": 75.0})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["raw_feature_diagnostics"]["applied_secondary_overrides"] == {
        "environment_thermal_plumes_sample_radius_m": 75.0
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"uav_energy_source_fuel": 1.0},
        {"not_a_feature": 1.0},
        {"sim_policy_frequency": 10.0},
    ],
)
def test_invalid_forbidden_unknown_or_internal_overrides_rejected_before_ml(client, profile, overrides):
    response = _assess(client, overrides=overrides)

    assert response.status_code == 422


def test_missing_selected_profile_returns_404(client):
    response = _assess(client, profile_id="missing")

    assert response.status_code == 404


def test_missing_required_scenario_field_returns_422(client, profile):
    scenario = scenario_payload()
    scenario.pop("mission_time_budget_s")

    response = _assess(client, scenario=scenario)

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("profile_updates", "scenario_updates", "code"),
    [
        ({"max_payload_kg": 0.1}, {"uav_payload_mass_kg": 2.0}, "PAYLOAD_EXCEEDS_PROFILE_LIMIT"),
        ({"uav_rotorcraft_hover_ceiling_m": 40.0}, {"airspace_altitude_agl_max_m": 50.0}, "ALTITUDE_EXCEEDS_HOVER_CEILING"),
        ({"swarm_capable": False}, {"swarm_enabled": 1.0}, "SWARM_NOT_CAPABLE"),
        ({"runway_capable": False}, {"mission_runway_required": 1.0}, "RUNWAY_NOT_CAPABLE"),
    ],
)
def test_structural_hard_veto_returns_blocked_and_no_ml(client, profile, profile_updates, scenario_updates, code):
    payload = profile_payload(user_id="user_1", profile_id="profile_1", **profile_updates)
    update_response = client.put("/users/user_1/profiles/profile_1", json=payload)
    assert update_response.status_code == 200

    response = _assess(client, scenario=scenario_payload(**scenario_updates))

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "blocked"
    assert body["ml"] is None
    assert any(issue["code"] == code for issue in body["issues"])


def test_assessment_path_does_not_call_generate_all_features_map(client, profile, monkeypatch):
    import uav_risk.core.feature_engineering as feature_engineering

    def boom(*args, **kwargs):
        raise AssertionError("legacy generate_all_features_map called")

    monkeypatch.setattr(feature_engineering, "generate_all_features_map", boom)

    response = _assess(client)

    assert response.status_code == 200
    assert response.json()["status"] == "completed"


def test_assessment_path_does_not_call_assemble_feature_vector_from_dict(client, profile, monkeypatch):
    import uav_risk.ml.loader as loader

    def boom(*args, **kwargs):
        raise AssertionError("legacy assemble_feature_vector_from_dict called")

    monkeypatch.setattr(loader, "assemble_feature_vector_from_dict", boom)

    response = _assess(client)

    assert response.status_code == 200
    assert response.json()["status"] == "completed"


def test_successful_response_probabilities_sum_to_one_and_match_model_classes(client, profile):
    response = _assess(client)

    assert response.status_code == 200
    probabilities = response.json()["ml"]["probabilities"]
    assert set(probabilities) == {"High Risk", "Low Risk", "Medium Risk"}
    assert abs(sum(probabilities.values()) - 1.0) < 1e-6
