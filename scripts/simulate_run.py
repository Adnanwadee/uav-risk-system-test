"""Legacy/demo/manual-only harness.

This script is not canonical runtime and should not be used as final readiness evidence.
Canonical readiness commands are listed in README.md.

Simulates a full pipeline run using the integration test payload.
"""
import asyncio
import json
import os
import traceback
from pprint import pprint

from uav_risk.core.contracts import MasterFlightPayload
from uav_risk.ml.loader import load_stage1_bundle
from uav_risk.stage2.pipeline import run_ace_pipeline
from uav_risk.stage2.rag.rag_core import AsyncRAGCore
from uav_risk.stage2.llm.report_writer import ReportWriter

ARTIFACTS_DIR = 'artifacts'

# Minimal real_flight_scenario_payload adapted from tests/test_integrated_pipeline.py
payload = {
    "flight_id": "SIM-FLIGHT-0001",
    "uav": {
        "mass_kg": 5.0,
        "wingspan_m": 1.5,
        "max_speed_mps": 10.0,
        "battery_wh": 99.0,
        "battery_capacity_mah": 5000.0,
        "battery_voltage_v": 22.2,
        "rotorcraft_rotor_count": 4,
        "payload_mass_kg": 0.0,
        "max_takeoff_weight_kg": 10.0,
        "aero_wing_area_m2": 1.0
    },
    "mission": {
        "altitude_m": 20.0,
        "time_budget_s": 600.0,
        "waypoints_count": 2,
        "loiter_radius_m": 30.0
    },
    "environment": {
        "weather_wind_mps": 5.0,
        "weather_gust_mps": 3.0,
        "temperature_c": 24.0,
        "humidity_pct": 45.0,
        "weather_phenomena_count": 0,
        "gnss_jam_dbm": -125.0,
        "em_interference": 0
    },
    "gps": {
        "satellites_count": 14,
        "hdop": 0.75
    },
    "operator": {
        "experience_hours": 120.0,
        "in_restricted_zone": False,
        "airport_distance_km": 12.5
    },
    "free_text": "Routine automated VLOS commercial survey voyage.",
    "timestamp": "2026-05-20T17:35:00Z",
    # Root-level core features
    "uav_mass_kg": 5.0,
    "uav_battery_wh": 99.0,
    "uav_max_speed_mps": 10.0,
    "uav_rotorcraft_rotor_count": 4.0,
    "environment_weather_wind_mps": 5.0,
    "environment_weather_gust_mps": 3.0,
    "mission_waypoints_count": 2.0,
    "mission_time_budget_s": 600.0,
    "mission_loiter_radius_m": 30.0,
    "comms_uplink_ok": 1.0,
    "comms_downlink_ok": 1.0,
}


def main():
    if not os.path.exists(ARTIFACTS_DIR):
        print("Artifacts directory not found; aborting simulation.")
        return
    try:
        bundle = load_stage1_bundle(ARTIFACTS_DIR)
        print("Loaded Stage-1 bundle:", bundle.get_model_version())
    except Exception as e:
        print("Failed to load stage1 bundle:", e)
        traceback.print_exc()
        return

    # construct payload model
    try:
        master = MasterFlightPayload(**payload)
    except Exception as e:
        print("Failed to construct MasterFlightPayload:", e)
        traceback.print_exc()
        return

    # prepare degraded rag_core and groq_llm=None to simulate offline LLM
    rag_core = AsyncRAGCore(groq_api_key=None)
    groq_llm = None
    report_writer = ReportWriter(llm=groq_llm)

    try:
        res = asyncio.run(run_ace_pipeline(
            flight_id=payload.get('flight_id') or 'SIM-FLIGHT-0001',
            payload=master,
            full_telemetry=payload,
            stage1_bundle=bundle,
            rag_core=rag_core,
            groq_llm=groq_llm,
            feature_defs=bundle.policy_config or {},
            report_writer=report_writer,
            precomputed_feature_vector=None,
            precomputed_validation_result=None
        ))
        print("Simulation result summary:")
        pprint(res)
    except Exception as e:
        print("Simulation raised exception:", e)
        traceback.print_exc()


if __name__ == '__main__':
    main()
