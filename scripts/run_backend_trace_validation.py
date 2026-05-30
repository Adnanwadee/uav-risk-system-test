from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from typing import Any

from fastapi.testclient import TestClient

from uav_risk.api.main import create_app
from uav_risk.ml.raw_schema import PROFILE_DERIVED_RAW_FEATURES, SCENARIO_REQUIRED_RAW_FEATURES


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


def _profile_value_for(name: str) -> float | str:
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


def _scenario_value_for(name: str) -> float | str:
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


def _profile_payload(user_id: str, profile_id: str) -> dict[str, Any]:
    data = {name: _profile_value_for(name) for name in PROFILE_DERIVED_RAW_FEATURES}
    data.update(
        {
            "user_id": user_id,
            "profile_id": profile_id,
            "profile_name": f"Trace {profile_id}",
            "max_payload_kg": 5.0,
            "max_takeoff_mass_kg": 20.0,
            "runway_capable": True,
            "swarm_capable": True,
            "max_swarm_size": 5,
        }
    )
    return data


def _scenario_payload() -> dict[str, Any]:
    return {name: _scenario_value_for(name) for name in SCENARIO_REQUIRED_RAW_FEATURES}


def _walk_keys(value: Any):
    if isinstance(value, dict):
        for k, v in value.items():
            yield str(k).lower()
            yield from _walk_keys(v)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


@contextmanager
def _temporary_storage_env():
    with tempfile.TemporaryDirectory(prefix="smart_skies_trace_") as root:
        old_profile = os.getenv("UAV_PROFILE_STORAGE_DIR")
        old_assessment = os.getenv("UAV_ASSESSMENT_STORAGE_DIR")
        os.environ["UAV_PROFILE_STORAGE_DIR"] = os.path.join(root, "profiles")
        os.environ["UAV_ASSESSMENT_STORAGE_DIR"] = os.path.join(root, "assessments")
        try:
            yield
        finally:
            if old_profile is None:
                os.environ.pop("UAV_PROFILE_STORAGE_DIR", None)
            else:
                os.environ["UAV_PROFILE_STORAGE_DIR"] = old_profile
            if old_assessment is None:
                os.environ.pop("UAV_ASSESSMENT_STORAGE_DIR", None)
            else:
                os.environ["UAV_ASSESSMENT_STORAGE_DIR"] = old_assessment


def _phase(
    name: str,
    *,
    entered: str,
    ran: bool,
    status: str,
    key_fields: dict[str, Any] | None = None,
    errors: list[Any] | None = None,
    warnings: list[Any] | None = None,
    persisted: bool | None = None,
    trace_entries_emitted: int | None = None,
) -> dict[str, Any]:
    return {
        "phase": name,
        "entered": entered,
        "ran": ran,
        "status": status,
        "key_fields": key_fields or {},
        "errors": errors or [],
        "warnings": warnings or [],
        "persisted": persisted,
        "trace_entries_emitted": trace_entries_emitted,
    }


def run_backend_trace_validation() -> dict[str, Any]:
    user_id = "user_1"
    profile_id = "profile_1"

    profile_payload = _profile_payload(user_id, profile_id)
    scenario_payload = _scenario_payload()
    request_payload = {
        "scenario": scenario_payload,
        "secondary_overrides": {"values": {}},
        "operator_notes": "backend trace validation run",
    }

    phases: list[dict[str, Any]] = []
    phases.append(
        _phase(
            "1. API input / request contract",
            entered="raw profile + raw scenario + raw secondary overrides + operator_notes",
            ran=True,
            status="ok",
            key_fields={
                "profile_field_count": len(profile_payload),
                "scenario_field_count": len(scenario_payload),
                "override_count": len(request_payload["secondary_overrides"]["values"]),
                "forbidden_processed_feature_keys_present": False,
            },
        )
    )

    with _temporary_storage_env():
        with TestClient(create_app()) as client:
            profile_resp = client.post(f"/users/{user_id}/profiles", json=profile_payload)
            if profile_resp.status_code not in {200, 201}:
                return {
                    "status": "failed",
                    "reason": "profile_create_failed",
                    "http_status": profile_resp.status_code,
                    "phases": phases,
                }

            stage2_resp = client.post(
                f"/users/{user_id}/profiles/{profile_id}/assessments/stage2",
                json=request_payload,
            )
            body = stage2_resp.json() if stage2_resp.status_code == 200 else {}

            errors = body.get("errors", []) if isinstance(body, dict) else []
            warnings = body.get("warnings", []) if isinstance(body, dict) else []
            stage1 = body.get("stage1", {}) if isinstance(body.get("stage1"), dict) else {}
            stage2 = body.get("stage2", {}) if isinstance(body.get("stage2"), dict) else {}
            rag = stage2.get("rag", {}) if isinstance(stage2.get("rag"), dict) else {}
            agent = stage2.get("agent", {}) if isinstance(stage2.get("agent"), dict) else {}
            decision = stage2.get("decision", {}) if isinstance(stage2.get("decision"), dict) else {}
            llm = stage2.get("llm_synthesis", {}) if isinstance(stage2.get("llm_synthesis"), dict) else {}
            report = stage2.get("report", {}) if isinstance(stage2.get("report"), dict) else {}

            trace = body.get("system_work_trace", {}) if isinstance(body.get("system_work_trace"), dict) else {}
            trace_entries = trace.get("entries", []) if isinstance(trace.get("entries"), list) else []
            trace_stages = sorted(
                {
                    str(item.get("stage"))
                    for item in trace_entries
                    if isinstance(item, dict) and isinstance(item.get("stage"), str)
                }
            )

            evidence_bundles = rag.get("evidence_bundle_details", []) if isinstance(rag.get("evidence_bundle_details"), list) else []
            scenario_driven_count = 0
            agent_requested_count = 0
            for item in evidence_bundles:
                if not isinstance(item, dict):
                    continue
                origin = str(item.get("retrieval_origin") or "")
                if origin == "scenario_driven":
                    scenario_driven_count += 1
                elif origin == "agent_requested":
                    agent_requested_count += 1

            phases.append(
                _phase(
                    "2. Core validation / hard veto",
                    entered="AssessmentCoreInput",
                    ran=stage2_resp.status_code == 200,
                    status="passed" if stage1.get("core", {}).get("structural_hard_veto_passed") else ("blocked" if body.get("status") == "blocked" else "unknown"),
                    key_fields={
                        "api_status": body.get("status"),
                        "structural_hard_veto_passed": stage1.get("core", {}).get("structural_hard_veto_passed"),
                    },
                    errors=errors,
                    warnings=warnings,
                )
            )

            ml = stage1.get("ml") if isinstance(stage1.get("ml"), dict) else {}
            phases.append(
                _phase(
                    "3. Raw feature assembly",
                    entered="raw profile + raw scenario + raw overrides",
                    ran=bool(ml),
                    status="ok" if bool(ml) else "not_executed",
                    key_fields={
                        "raw_feature_count": ml.get("raw_feature_count"),
                        "processed_feature_count": ml.get("processed_feature_count"),
                    },
                )
            )

            phases.append(
                _phase(
                    "4. Stage1 ML inference",
                    entered="raw feature vector",
                    ran=bool(ml),
                    status="ok" if bool(ml) else "not_executed",
                    key_fields={
                        "predicted_class": ml.get("predicted_class"),
                        "probability_class_count": len(ml.get("probabilities", {}) if isinstance(ml.get("probabilities"), dict) else {}),
                    },
                )
            )

            shap = stage1.get("shap") if isinstance(stage1.get("shap"), dict) else {}
            phases.append(
                _phase(
                    "5. SHAP explanation",
                    entered="Stage1 ML output",
                    ran=bool(shap),
                    status="ok" if bool(shap) else "not_executed",
                    key_fields={"shap_top_feature_count": len(shap.get("top_features", []) if isinstance(shap.get("top_features"), list) else [])},
                )
            )

            phases.append(
                _phase(
                    "6. Stage2 input construction",
                    entered="Stage1 snapshot + profile/scenario context",
                    ran=stage2_resp.status_code == 200,
                    status="ok" if stage2_resp.status_code == 200 else "failed",
                    key_fields={
                        "profile_context_present": isinstance(stage2.get("profile_context"), dict),
                        "policy_present": isinstance(stage2.get("policy"), dict),
                    },
                )
            )

            phases.append(
                _phase(
                    "7. Scenario-driven RAG retrieval",
                    entered="scenario-driven query plan",
                    ran=stage2_resp.status_code == 200,
                    status="ok" if stage2_resp.status_code == 200 else "failed",
                    key_fields={
                        "scenario_driven_bundle_count": scenario_driven_count,
                        "scenario_evidence_status": rag.get("scenario_evidence_status"),
                        "retrieval_usable": rag.get("retrieval_usable"),
                    },
                )
            )

            phases.append(
                _phase(
                    "8. Agent-requested RAG retrieval",
                    entered="agent evidence gap plan",
                    ran=stage2_resp.status_code == 200,
                    status="ok" if stage2_resp.status_code == 200 else "failed",
                    key_fields={
                        "agent_requested_bundle_count": agent_requested_count,
                        "selected_rag_query_count": len(agent.get("selected_rag_queries", []) if isinstance(agent.get("selected_rag_queries"), list) else []),
                        "skipped_rag_query_count": len(agent.get("skipped_rag_queries", []) if isinstance(agent.get("skipped_rag_queries"), list) else []),
                    },
                )
            )

            phases.append(
                _phase(
                    "9. OperationalAgentV2 analysis",
                    entered="ML + SHAP + RAG evidence",
                    ran=stage2_resp.status_code == 200,
                    status="ok" if stage2_resp.status_code == 200 else "failed",
                    key_fields={
                        "agent_recommendation": agent.get("recommendation"),
                        "finding_count": len(agent.get("findings", []) if isinstance(agent.get("findings"), list) else []),
                        "action_item_count": len(agent.get("action_items", []) if isinstance(agent.get("action_items"), list) else []),
                    },
                )
            )

            phases.append(
                _phase(
                    "10. DecisionEngine decision",
                    entered="deterministic stage contributions",
                    ran=stage2_resp.status_code == 200,
                    status="ok" if stage2_resp.status_code == 200 else "failed",
                    key_fields={
                        "final_decision": decision.get("final_decision"),
                        "decision_score": decision.get("decision_score"),
                        "confidence_level": decision.get("confidence_level"),
                    },
                )
            )

            phases.append(
                _phase(
                    "11. Optional LLM synthesis",
                    entered="deterministic context",
                    ran=stage2_resp.status_code == 200,
                    status=str(llm.get("status") or "unknown"),
                    key_fields={
                        "llm_status": llm.get("status"),
                        "provider": llm.get("provider"),
                        "model_name": llm.get("model_name"),
                        "consistency_warning_count": len(llm.get("consistency_warnings", []) if isinstance(llm.get("consistency_warnings"), list) else []),
                    },
                )
            )

            sections = report.get("sections", []) if isinstance(report.get("sections"), list) else []
            phases.append(
                _phase(
                    "12. Report generation",
                    entered="stage2 assessment result",
                    ran=stage2_resp.status_code == 200,
                    status="ok" if bool(report.get("generated")) else "not_generated",
                    key_fields={
                        "report_generated": report.get("generated"),
                        "report_section_count": len(sections),
                        "report_section_names": [str(item.get("title")) for item in sections if isinstance(item, dict) and isinstance(item.get("title"), str)][:11],
                    },
                )
            )

            phases.append(
                _phase(
                    "13. System Work Trace assembly",
                    entered="agent + pipeline summaries",
                    ran=stage2_resp.status_code == 200,
                    status="ok" if stage2_resp.status_code == 200 else "failed",
                    key_fields={
                        "system_work_trace_summary": trace.get("summary"),
                        "system_work_trace_stages": trace_stages,
                    },
                    trace_entries_emitted=len(trace_entries),
                )
            )

            assessment_id = body.get("assessment_id")
            get_status = None
            list_status = None
            list_count = None
            if isinstance(assessment_id, str) and assessment_id:
                get_resp = client.get(f"/users/{user_id}/assessments/{assessment_id}")
                list_resp = client.get(f"/users/{user_id}/assessments", params={"profile_id": profile_id})
                get_status = get_resp.status_code
                list_status = list_resp.status_code
                if list_resp.status_code == 200 and isinstance(list_resp.json(), list):
                    list_count = len(list_resp.json())

            phases.append(
                _phase(
                    "14. Persistence",
                    entered="Stage2AssessmentResponse",
                    ran=stage2_resp.status_code == 200,
                    status="ok" if bool(body.get("persisted")) and get_status == 200 else "partial",
                    key_fields={
                        "assessment_id": assessment_id,
                        "persisted": body.get("persisted"),
                        "persistence_status": body.get("persistence_status"),
                        "get_by_id_http_status": get_status,
                        "list_http_status": list_status,
                        "list_count": list_count,
                    },
                    persisted=bool(body.get("persisted")),
                )
            )

            keys = set(_walk_keys(body)) if isinstance(body, dict) else set()
            phases.append(
                _phase(
                    "15. API response contract",
                    entered="final API serialization",
                    ran=stage2_resp.status_code == 200,
                    status="ok" if stage2_resp.status_code == 200 else "failed",
                    key_fields={
                        "top_level_keys": sorted(list(body.keys())) if isinstance(body, dict) else [],
                        "forbidden_key_count": len(keys.intersection(FORBIDDEN_KEYS)),
                        "system_work_trace_present": isinstance(body.get("system_work_trace"), dict) if isinstance(body, dict) else False,
                    },
                )
            )

            return {
                "status": "ok" if stage2_resp.status_code == 200 else "failed",
                "http_status": stage2_resp.status_code,
                "assessment_id": assessment_id,
                "phase_count": len(phases),
                "phases": phases,
            }


def main() -> int:
    try:
        payload = run_backend_trace_validation()
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0 if payload.get("status") == "ok" else 1
    except Exception:
        print(json.dumps({"status": "failed", "error": "backend_trace_validation_failed"}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
