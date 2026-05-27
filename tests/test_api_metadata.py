from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from uav_risk.api.main import create_app


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


def test_raw_schema_metadata_counts(client):
    response = client.get("/features/raw-schema")

    assert response.status_code == 200
    body = response.json()
    assert body["raw_feature_count"] == 197
    assert body["processed_feature_count"] == 198
    assert len(body["raw_feature_names"]) == 197
    assert len(body["processed_feature_names"]) == 198


def test_profile_fields_metadata(client):
    response = client.get("/features/profile-fields")

    assert response.status_code == 200
    assert response.json()["count_profile_derived_raw_features"] == 16


def test_scenario_fields_metadata(client):
    response = client.get("/features/scenario-fields")

    assert response.status_code == 200
    assert response.json()["count_scenario_required_raw_features"] == 45


def test_secondary_overrides_metadata(client):
    response = client.get("/features/secondary-overrides")

    assert response.status_code == 200
    body = response.json()
    assert "environment_thermal_plumes_sample_radius_m" in body["optional_raw_override_features"]
    assert "sim_policy_frequency" in body["internal_only_raw_features"]
    assert body["controls_actions_first_allowed_values"] == ["fwd", "hold", "throttle"]


def test_model_metadata(client):
    response = client.get("/model/metadata")

    assert response.status_code == 200
    body = response.json()
    assert body["raw_feature_count"] == 197
    assert body["processed_feature_count"] == 198
    assert set(body["class_names"]) == {"High Risk", "Low Risk", "Medium Risk"}
    assert body["production_path"] == "raw_197 -> preprocessor -> processed_198 -> model"


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "uav-risk-api", "ml_bundle_loaded": True}
