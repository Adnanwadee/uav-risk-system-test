from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from uav_risk.api.main import create_app
from uav_risk.api.storage import LocalProfileStorage
from uav_risk.core.contracts import DroneProfileRaw
from uav_risk.ml.raw_schema import (
    PROFILE_CAPABILITY_FIELDS,
    PROFILE_DERIVED_RAW_FEATURES,
    RAW_FEATURE_NAMES,
    SCENARIO_REQUIRED_RAW_FEATURES,
)
from test_api_profiles import profile_payload
from test_api_raw_assessment import scenario_payload


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("UAV_PROFILE_STORAGE_DIR", str(tmp_path / "profiles"))
    with TestClient(create_app()) as c:
        yield c


def _create_profile(client, user_id="user_1", profile_id="profile_1", **updates):
    payload = profile_payload(user_id=user_id, profile_id=profile_id, **updates)
    response = client.post(f"/users/{user_id}/profiles", json=payload)
    assert response.status_code == 201
    return payload


def _assessment_payload(**scenario_updates):
    return {
        "scenario": scenario_payload(**scenario_updates),
        "secondary_overrides": {"values": {}},
        "operator_notes": "Closure audit notes",
    }


def test_api_app_imports_and_expected_endpoints_exist():
    app = create_app()
    routes = {(tuple(sorted(route.methods)), route.path) for route in app.routes if hasattr(route, "methods")}

    expected_paths = {
        "/health",
        "/model/metadata",
        "/features/raw-schema",
        "/features/profile-fields",
        "/features/scenario-fields",
        "/features/secondary-overrides",
        "/users/{user_id}/profiles",
        "/users/{user_id}/profiles/{profile_id}",
        "/users/{user_id}/profiles/{profile_id}/assessments",
    }
    assert expected_paths.issubset({path for _, path in routes})


def test_local_profile_storage_writes_expected_json_shape_and_isolates_users(tmp_path):
    storage = LocalProfileStorage(tmp_path / "profiles")
    profile_a = DroneProfileRaw(**profile_payload(user_id="user_a", profile_id="shared"))
    profile_b = DroneProfileRaw(**profile_payload(user_id="user_b", profile_id="shared"))

    storage.create_profile("user_a", profile_a)
    storage.create_profile("user_b", profile_b)

    path_a = tmp_path / "profiles" / "user_a" / "shared.json"
    path_b = tmp_path / "profiles" / "user_b" / "shared.json"
    assert path_a.exists()
    assert path_b.exists()
    assert path_a != path_b

    stored = json.loads(path_a.read_text(encoding="utf-8"))
    expected_keys = {"user_id", "profile_id", "profile_name", *PROFILE_DERIVED_RAW_FEATURES, *PROFILE_CAPABILITY_FIELDS}
    assert expected_keys.issubset(stored)
    assert stored["user_id"] == "user_a"
    assert stored["profile_id"] == "shared"


def test_profile_get_and_list_response_shapes_and_multiple_profiles(client):
    _create_profile(client, profile_id="profile_1")
    _create_profile(client, profile_id="profile_2")

    get_response = client.get("/users/user_1/profiles/profile_1")
    assert get_response.status_code == 200
    assert set(get_response.json()) == {"status", "user_id", "profile_id", "profile"}
    assert get_response.json()["status"] == "found"

    list_response = client.get("/users/user_1/profiles")
    assert list_response.status_code == 200
    assert list_response.json()["user_id"] == "user_1"
    assert [p["profile_id"] for p in list_response.json()["profiles"]] == ["profile_1", "profile_2"]


def test_feature_metadata_counts_and_no_legacy_processed_core_contract(client):
    raw_schema = client.get("/features/raw-schema").json()
    profile_fields = client.get("/features/profile-fields").json()
    scenario_fields = client.get("/features/scenario-fields").json()
    overrides = client.get("/features/secondary-overrides").json()

    assert raw_schema["raw_feature_count"] == 197
    assert raw_schema["processed_feature_count"] == 198
    assert profile_fields["count_profile_derived_raw_features"] == 16
    assert scenario_fields["count_scenario_required_raw_features"] == 45
    assert "uav_energy_source_fuel" in raw_schema["forbidden_user_features"]
    assert "controls_actions_first_allowed_values" in overrides
    assert "core_features" not in scenario_fields


def test_assessment_success_response_shape_and_probabilities(client):
    _create_profile(client)

    response = client.post("/users/user_1/profiles/profile_1/assessments", json=_assessment_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["user_id"] == "user_1"
    assert body["profile_id"] == "profile_1"
    assert set(body["ml"]) == {"predicted_class", "probabilities"}
    assert set(body["ml"]["probabilities"]) == {"High Risk", "Low Risk", "Medium Risk"}
    assert abs(sum(body["ml"]["probabilities"].values()) - 1.0) < 1e-6
    assert "top_features" in body["shap"]
    assert body["raw_feature_count"] == 197
    assert body["processed_feature_count"] == 198
    assert body["operator_notes"] == "Closure audit notes"


def test_hard_veto_blocked_response_shape_and_ml_not_called(client, monkeypatch):
    _create_profile(client, max_payload_kg=0.1)
    called = False

    def boom(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("ML should not run for blocked assessments")

    monkeypatch.setattr("uav_risk.api.routes.assessments.run_stage1_inference", boom)

    response = client.post(
        "/users/user_1/profiles/profile_1/assessments",
        json=_assessment_payload(uav_payload_mass_kg=2.0),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "blocked"
    assert body["ml"] is None
    assert {"code", "field", "message", "details"}.issubset(body["issues"][0])
    assert any(issue["code"] == "PAYLOAD_EXCEEDS_PROFILE_LIMIT" for issue in body["issues"])
    assert called is False


def test_selected_profile_is_mandatory(client):
    response = client.post("/users/user_1/profiles/missing/assessments", json=_assessment_payload())

    assert response.status_code == 404


def test_api_assessment_does_not_call_legacy_core_or_loader_paths(client, monkeypatch):
    _create_profile(client)

    def boom(*args, **kwargs):
        raise AssertionError("legacy path called")

    monkeypatch.setattr("uav_risk.core.feature_engineering.generate_all_features_map", boom)
    monkeypatch.setattr("uav_risk.ml.loader.assemble_feature_vector_from_dict", boom)

    response = client.post("/users/user_1/profiles/profile_1/assessments", json=_assessment_payload())

    assert response.status_code == 200
    assert response.json()["status"] == "completed"


def test_scenario_body_rejects_generated_and_processed_onehot_fields(client):
    _create_profile(client)
    payload = _assessment_payload()
    payload["scenario"]["sim_duration_steps"] = 900.0
    response = client.post("/users/user_1/profiles/profile_1/assessments", json=payload)
    assert response.status_code == 422

    payload = _assessment_payload()
    payload["scenario"]["uav_energy_source_fuel"] = 1.0
    response = client.post("/users/user_1/profiles/profile_1/assessments", json=payload)
    assert response.status_code == 422


def test_raw_feature_schema_constant_still_has_exact_length():
    assert len(RAW_FEATURE_NAMES) == 197
    assert len(SCENARIO_REQUIRED_RAW_FEATURES) == 45
