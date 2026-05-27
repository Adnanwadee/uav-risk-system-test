from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from uav_risk.api.main import create_app
from uav_risk.ml.raw_schema import PROFILE_DERIVED_RAW_FEATURES


def _value_for(name: str):
    if name == "uav_energy_source":
        return "battery"
    if name == "uav_mass_kg":
        return 2.0
    if name == "uav_battery_wh":
        return 40.0
    if name == "uav_fuel_l":
        return 0.0
    if name == "uav_max_speed_mps":
        return 15.0
    if name == "uav_max_tilt_deg":
        return 30.0
    if name == "uav_reserve_fraction":
        return 0.3
    if name == "uav_rotorcraft_rotor_count":
        return 4.0
    if name == "uav_aero_prop_efficiency":
        return 0.85
    if name == "uav_rotorcraft_max_climb_mps":
        return 8.0
    if name == "uav_rotorcraft_hover_ceiling_m":
        return 1000.0
    if name.startswith("uav_sensors_"):
        return 1.0
    return 1.0


def profile_payload(user_id: str = "user_1", profile_id: str = "profile_1", **updates):
    data = {name: _value_for(name) for name in PROFILE_DERIVED_RAW_FEATURES}
    data.update({
        "user_id": user_id,
        "profile_id": profile_id,
        "profile_name": f"Profile {profile_id}",
        "max_payload_kg": 5.0,
        "max_takeoff_mass_kg": 20.0,
        "runway_capable": True,
        "swarm_capable": True,
        "max_swarm_size": 5,
    })
    data.update(updates)
    return data


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("UAV_PROFILE_STORAGE_DIR", str(tmp_path / "profiles"))
    with TestClient(create_app()) as c:
        yield c


def test_create_valid_profile_succeeds(client):
    response = client.post("/users/user_1/profiles", json=profile_payload())

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "created"
    assert body["profile"]["profile_id"] == "profile_1"


def test_create_profile_rejects_path_body_user_id_mismatch(client):
    response = client.post("/users/other_user/profiles", json=profile_payload(user_id="user_1"))

    assert response.status_code == 422


def test_create_profile_rejects_processed_onehot_extra_field(client):
    payload = profile_payload()
    payload["uav_energy_source_fuel"] = 1.0

    response = client.post("/users/user_1/profiles", json=payload)

    assert response.status_code == 422


def test_list_profiles_returns_multiple_profiles_for_same_user(client):
    client.post("/users/user_1/profiles", json=profile_payload(profile_id="profile_1"))
    client.post("/users/user_1/profiles", json=profile_payload(profile_id="profile_2"))

    response = client.get("/users/user_1/profiles")

    assert response.status_code == 200
    profiles = response.json()["profiles"]
    assert {p["profile_id"] for p in profiles} == {"profile_1", "profile_2"}


def test_get_profile_returns_selected_profile(client):
    client.post("/users/user_1/profiles", json=profile_payload(profile_id="profile_1"))

    response = client.get("/users/user_1/profiles/profile_1")

    assert response.status_code == 200
    assert response.json()["profile"]["profile_id"] == "profile_1"


def test_update_profile_succeeds_and_validates_path_body_ids(client):
    client.post("/users/user_1/profiles", json=profile_payload(profile_id="profile_1"))
    updated = profile_payload(profile_id="profile_1", profile_name="Updated")

    response = client.put("/users/user_1/profiles/profile_1", json=updated)

    assert response.status_code == 200
    assert response.json()["profile"]["profile_name"] == "Updated"

    mismatch = profile_payload(profile_id="other")
    response = client.put("/users/user_1/profiles/profile_1", json=mismatch)
    assert response.status_code == 422


def test_delete_profile_removes_selected_profile(client):
    client.post("/users/user_1/profiles", json=profile_payload(profile_id="profile_1"))

    response = client.delete("/users/user_1/profiles/profile_1")

    assert response.status_code == 200
    assert client.get("/users/user_1/profiles/profile_1").status_code == 404


def test_missing_profile_returns_404(client):
    response = client.get("/users/user_1/profiles/missing")

    assert response.status_code == 404
