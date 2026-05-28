from __future__ import annotations

import json

from fastapi.testclient import TestClient

from uav_risk.api.main import create_app
from test_api_profiles import profile_payload
from test_api_raw_assessment import scenario_payload


def _create_profile(client, user_id="user_1", profile_id="profile_1", **updates):
    payload = profile_payload(user_id=user_id, profile_id=profile_id, **updates)
    response = client.post(f"/users/{user_id}/profiles", json=payload)
    assert response.status_code == 201
    return payload


def _assessment_payload(**scenario_updates):
    return {
        "scenario": scenario_payload(**scenario_updates),
        "secondary_overrides": {"values": {}},
        "operator_notes": "Stage2 integration test",
    }


def test_stage2_endpoint_runs_and_returns_full_structure(tmp_path, monkeypatch):
    monkeypatch.setenv("UAV_PROFILE_STORAGE_DIR", str(tmp_path / "profiles"))
    # ensure runtime RAG adapter is not used in this test
    monkeypatch.setattr("uav_risk.api.routes.assessments.build_runtime_rag_adapter_if_available", lambda: None)
    with TestClient(create_app()) as client:
        _create_profile(client)
        response = client.post("/users/user_1/profiles/profile_1/assessments/stage2", json=_assessment_payload())
        assert response.status_code == 200
        body = response.json()
        assert body["status"] in {"completed", "degraded", "skipped", "failed"}
        assert body["user_id"] == "user_1"
        assert "stage1" in body and "ml" in body["stage1"]
        assert "stage2" in body and "agent" in body["stage2"]
        assert "llm_synthesis" in body["stage2"]
        assert "report" in body["stage2"]
        assert body["diagnostics"]["external_llm_provider_used"] is False


def test_stage2_blocked_hard_veto_returns_blocked(tmp_path, monkeypatch):
    monkeypatch.setenv("UAV_PROFILE_STORAGE_DIR", str(tmp_path / "profiles"))
    monkeypatch.setattr("uav_risk.api.routes.assessments.build_runtime_rag_adapter_if_available", lambda: None)
    with TestClient(create_app()) as client:
        # create a profile with a low max payload to trigger veto
        _create_profile(client, max_payload_kg=0.1)
        payload = _assessment_payload(uav_payload_mass_kg=2.0)
        response = client.post("/users/user_1/profiles/profile_1/assessments/stage2", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "blocked"
        assert body.get("decision", {}).get("final_decision") == "no_go"


def test_stage2_missing_profile_returns_404(tmp_path, monkeypatch):
    monkeypatch.setenv("UAV_PROFILE_STORAGE_DIR", str(tmp_path / "profiles"))
    with TestClient(create_app()) as client:
        payload = _assessment_payload()
        response = client.post("/users/user_1/profiles/missing/assessments/stage2", json=payload)
        assert response.status_code == 404


def test_stage2_response_is_json_serializable(tmp_path, monkeypatch):
    monkeypatch.setenv("UAV_PROFILE_STORAGE_DIR", str(tmp_path / "profiles"))
    monkeypatch.setattr("uav_risk.api.routes.assessments.build_runtime_rag_adapter_if_available", lambda: None)
    with TestClient(create_app()) as client:
        _create_profile(client)
        response = client.post("/users/user_1/profiles/profile_1/assessments/stage2", json=_assessment_payload())
        assert response.status_code == 200
        body = response.json()
        # should be serializable by json
        json.dumps(body)
