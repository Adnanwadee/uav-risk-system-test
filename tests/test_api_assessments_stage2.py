from __future__ import annotations

import json
import uuid

from fastapi.testclient import TestClient

from uav_risk.api.main import create_app
from test_api_profiles import profile_payload
from test_api_raw_assessment import scenario_payload


FORBIDDEN_KEYS = {
    "reasoning_steps",
    "chain_of_thought",
    "reasoning_chain",
    "thoughts",
    "thought",
    "scratchpad",
    "raw_prompt",
    "prompt",
    "raw_completion",
    "completion",
    "raw_llm_response",
    "tool_history",
    "internal_memory",
    "internal_reasoning",
    "private_reasoning",
    "hidden",
    "api_key",
    "secret",
    "token",
    "authorization",
}


def _set_storage_roots(monkeypatch, tmp_path):
    monkeypatch.setenv("UAV_PROFILE_STORAGE_DIR", str(tmp_path / "profiles"))
    monkeypatch.setenv("UAV_ASSESSMENT_STORAGE_DIR", str(tmp_path / "assessments"))


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


def _walk_keys(value):
    if isinstance(value, dict):
        for k, v in value.items():
            yield str(k).lower()
            yield from _walk_keys(v)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def test_stage2_endpoint_returns_stable_frontend_ready_structure(tmp_path, monkeypatch):
    _set_storage_roots(monkeypatch, tmp_path)
    monkeypatch.setattr("uav_risk.api.routes.assessments.build_runtime_rag_adapter_if_available", lambda: None)
    with TestClient(create_app()) as client:
        _create_profile(client)
        response = client.post("/users/user_1/profiles/profile_1/assessments/stage2", json=_assessment_payload())
        assert response.status_code == 200
        body = response.json()

        assert set(body.keys()) == {
            "status",
            "user_id",
            "profile_id",
            "assessment_id",
            "created_at",
            "persisted",
            "persistence_status",
            "system_work_trace",
            "warnings",
            "errors",
            "stage1",
            "stage2",
            "diagnostics",
        }
        assert body["status"] in {"completed", "degraded", "skipped", "failed"}
        assert body["user_id"] == "user_1"
        assert body["profile_id"] == "profile_1"
        assert isinstance(body["assessment_id"], str)
        assert isinstance(body["created_at"], str)
        assert body["persisted"] is True
        assert body["persistence_status"] == "saved"
        assert body["system_work_trace"]["public_safe"] is True
        uuid.UUID(body["assessment_id"])

        assert "ml" in body["stage1"] and "shap" in body["stage1"]
        assert "predicted_class" in body["stage1"]["ml"]
        assert "top_features" in body["stage1"]["shap"]

        s2 = body["stage2"]
        assert "profile_context" in s2
        assert "policy" in s2
        assert "rag" in s2
        assert "agent" in s2
        assert "decision" in s2
        assert "llm_synthesis" in s2
        assert "report" in s2

        assert "weights" in s2["policy"]
        assert "weight_rationales" in s2["policy"]
        assert s2["profile_context"]["profile_id"] == "profile_1"

        assert "evidence_bundle_details" in s2["rag"]
        assert "citations" in s2["rag"]
        assert "corpus_coverage_status" in s2["rag"]
        assert "reranker_configured" in s2["rag"]
        assert "source_ids" in s2["rag"]
        assert "source_titles" in s2["rag"]
        assert "missing_sources_count" in s2["rag"]
        assert "retrieval_origins" in s2["rag"]
        assert "synthetic_bundle_count" in s2["rag"]

        assert "findings" in s2["agent"]
        assert "action_items" in s2["agent"]
        assert "tool_trace" in s2["agent"]
        assert "system_work_trace" in s2["agent"]
        assert "working_memory_summary" in s2["agent"]
        assert "top_feature_assessments" in s2["agent"]

        assert "final_decision" in s2["decision"]
        assert "decision_score" in s2["decision"]
        assert "stage_contributions" in s2["decision"]

        assert "status" in s2["llm_synthesis"]
        assert "synthesis_status" in s2["llm_synthesis"]
        assert "provider" in s2["llm_synthesis"]
        assert "model_name" in s2["llm_synthesis"]
        assert "external_provider_used" in s2["llm_synthesis"]

        d = body["diagnostics"]
        assert "retrieval_usable" in d
        assert "rag_quality_is_proven" in d
        assert "corpus_coverage_status" in d
        assert "reranker_configured" in d
        assert "llm_mode" in d
        assert "external_llm_provider_used" in d


def test_stage2_create_persists_record_and_get_by_id(tmp_path, monkeypatch):
    _set_storage_roots(monkeypatch, tmp_path)
    monkeypatch.setattr("uav_risk.api.routes.assessments.build_runtime_rag_adapter_if_available", lambda: None)
    with TestClient(create_app()) as client:
        _create_profile(client)
        create_response = client.post("/users/user_1/profiles/profile_1/assessments/stage2", json=_assessment_payload())
        assert create_response.status_code == 200
        assessment_id = create_response.json()["assessment_id"]

        get_response = client.get(f"/users/user_1/assessments/{assessment_id}")
        assert get_response.status_code == 200
        body = get_response.json()

        assert body["assessment_id"] == assessment_id
        assert body["user_id"] == "user_1"
        assert body["profile_id"] == "profile_1"
        assert isinstance(body["created_at"], str)
        assert body["system_work_trace"]["public_safe"] is True
        assert body["stage2"]["agent"]["system_work_trace"]["public_safe"] is True

        keys = set(_walk_keys(body))
        assert keys.isdisjoint(FORBIDDEN_KEYS)


def test_stage2_list_assessments_returns_records_and_filters_profile(tmp_path, monkeypatch):
    _set_storage_roots(monkeypatch, tmp_path)
    monkeypatch.setattr("uav_risk.api.routes.assessments.build_runtime_rag_adapter_if_available", lambda: None)
    with TestClient(create_app()) as client:
        _create_profile(client, profile_id="profile_1")
        _create_profile(client, profile_id="profile_2")

        response_1 = client.post("/users/user_1/profiles/profile_1/assessments/stage2", json=_assessment_payload())
        response_2 = client.post("/users/user_1/profiles/profile_2/assessments/stage2", json=_assessment_payload())
        assert response_1.status_code == 200
        assert response_2.status_code == 200

        list_response = client.get("/users/user_1/assessments")
        assert list_response.status_code == 200
        items = list_response.json()
        assert len(items) == 2
        assert {item["profile_id"] for item in items} == {"profile_1", "profile_2"}

        filtered_response = client.get("/users/user_1/assessments", params={"profile_id": "profile_2"})
        assert filtered_response.status_code == 200
        filtered_items = filtered_response.json()
        assert len(filtered_items) == 1
        assert filtered_items[0]["profile_id"] == "profile_2"


def test_stage2_get_missing_assessment_returns_404(tmp_path, monkeypatch):
    _set_storage_roots(monkeypatch, tmp_path)
    with TestClient(create_app()) as client:
        response = client.get("/users/user_1/assessments/missing")
        assert response.status_code == 404


def test_stage2_blocked_response_is_json_safe_and_has_no_fake_outputs(tmp_path, monkeypatch):
    _set_storage_roots(monkeypatch, tmp_path)
    monkeypatch.setattr("uav_risk.api.routes.assessments.build_runtime_rag_adapter_if_available", lambda: None)
    with TestClient(create_app()) as client:
        _create_profile(client, max_payload_kg=0.1)
        payload = _assessment_payload(uav_payload_mass_kg=2.0)
        response = client.post("/users/user_1/profiles/profile_1/assessments/stage2", json=payload)
        assert response.status_code == 200
        body = response.json()

        assert body["status"] == "blocked"
        assert body["stage1"]["ml"] is None
        assert body["stage2"]["decision"]["final_decision"] == "no_go"
        assert body["stage2"]["rag"]["evidence_bundle_count"] == 0
        assert body["stage2"]["llm_synthesis"]["external_provider_used"] is False
        assert body["stage2"]["llm_synthesis"]["synthesis_status"] == "disabled"


def test_stage2_response_has_no_forbidden_private_reasoning_keys(tmp_path, monkeypatch):
    _set_storage_roots(monkeypatch, tmp_path)
    monkeypatch.setattr("uav_risk.api.routes.assessments.build_runtime_rag_adapter_if_available", lambda: None)
    with TestClient(create_app()) as client:
        _create_profile(client)
        response = client.post("/users/user_1/profiles/profile_1/assessments/stage2", json=_assessment_payload())
        assert response.status_code == 200
        body = response.json()
        keys = set(_walk_keys(body))
        assert keys.isdisjoint(FORBIDDEN_KEYS)


def test_stage2_missing_profile_returns_404(tmp_path, monkeypatch):
    _set_storage_roots(monkeypatch, tmp_path)
    with TestClient(create_app()) as client:
        payload = _assessment_payload()
        response = client.post("/users/user_1/profiles/missing/assessments/stage2", json=payload)
        assert response.status_code == 404


def test_stage2_response_is_json_serializable(tmp_path, monkeypatch):
    _set_storage_roots(monkeypatch, tmp_path)
    monkeypatch.setattr("uav_risk.api.routes.assessments.build_runtime_rag_adapter_if_available", lambda: None)
    with TestClient(create_app()) as client:
        _create_profile(client)
        response = client.post("/users/user_1/profiles/profile_1/assessments/stage2", json=_assessment_payload())
        assert response.status_code == 200
        body = response.json()
        json.dumps(body)


def test_stage2_env_groq_missing_key_keeps_safe_llm_mode(tmp_path, monkeypatch):
    _set_storage_roots(monkeypatch, tmp_path)
    monkeypatch.setattr("uav_risk.api.routes.assessments.build_runtime_rag_adapter_if_available", lambda: None)
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with TestClient(create_app()) as client:
        _create_profile(client)
        response = client.post("/users/user_1/profiles/profile_1/assessments/stage2", json=_assessment_payload())
        assert response.status_code == 200
        body = response.json()
        assert body["diagnostics"]["external_llm_provider_used"] is False
        assert body["diagnostics"]["llm_mode"] in {"fallback", "disabled", "generated", "failed"}


def test_openapi_exposes_stage2_and_history_endpoints():
    with TestClient(create_app()) as client:
        spec = client.get("/openapi.json")
        assert spec.status_code == 200
        body = spec.json()

    paths = body["paths"]
    assert "/users/{user_id}/profiles/{profile_id}/assessments/stage2" in paths
    assert "/users/{user_id}/assessments" in paths
    assert "/users/{user_id}/assessments/{assessment_id}" in paths

    components = body.get("components", {}).get("schemas", {})
    assert "Stage2AssessmentResponse" in components
    assert "AssessmentRecord" in components
    assert "AssessmentListItem" in components
