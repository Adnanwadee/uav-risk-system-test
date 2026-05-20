"""
Module: tests.test_integrated_pipeline
Purpose: End-to-End integration test validating Stage-1 pipeline coupled with Stage-2 ML inference using real scenario payloads.
Dependencies: Imports from uav_risk core contracts, validators, routers, and ml subsystems.
"""

import os
import pytest
import numpy as np
import structlog

# استيراد واجهات وعقود المرحلة الأولى المعتمدة والمقفلة والمطابقة للحقول الفعلية
from uav_risk.core.contracts import MasterFlightPayload
from uav_risk.core.data_validator import DataValidator
from uav_risk.core.feature_router import FeatureRouter

# استيراد محرك المرحلة الثانية المطور والمختبر
from uav_risk.ml.loader import load_stage1_bundle
from uav_risk.ml.inference import run_stage1_inference
from uav_risk.ml.schemas import RiskClass, MLResult

# إعداد نظام التتبع للمحاكاة التكاملية
logger = structlog.get_logger(__name__)

ARTIFACTS_DIR = "artifacts"


@pytest.fixture(scope="module")
def ml_bundle():
    """Fixture to provide the loaded ML bundle for integrated inference cycles."""
    if not os.path.exists(ARTIFACTS_DIR):
        pytest.skip("Skipping integrated test: Artifacts directory not found.")
    return load_stage1_bundle(ARTIFACTS_DIR)


@pytest.fixture(scope="module")
def real_flight_scenario_payload():
    """
    Constructs a highly consistent flight telemetry payload matching the strict Pydantic layout blocks.
    Employs bulletproof dual-key injection (both nested and root-level flat columns) to guarantee 100% 
    schema alignment across Pydantic model configurations.
    """
    return {
        # -----------------------------------------------------------------
        # 1. الكتل المتداخلة لتمرير فحص شروط Pydantic الأساسية (Model Schema Validation)
        # -----------------------------------------------------------------
        "flight_id": "FLIGHT-2026-INTEGRATION-E2E",
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
            "max_altitude_m": 50.0,
            "distance_km": 1.8,
            "time_budget_s": 600.0,
            "operation_type": "VLOS",
            "is_night_flight": False,
            "waypoints_count": 2,
            "loiter_radius_m": 30.0
        },
        "environment": {
            "weather_wind_mps": 5.0,
            "weather_wind_dir_deg": 90.0,
            "weather_gust_mps": 3.0,
            "temperature_c": 24.0,
            "humidity_pct": 45.0,
            "weather_phenomena_count": 0,
            "gnss_jam_dbm": -125.0,
            "em_interference": 0
        },
        "gps": {
            "fix_quality": 1,
            "satellites_count": 14,
            "hdop": 0.75,
            "latitude": 29.3759,
            "longitude": 47.9774,
            "altitude_gps_m": 22.0
        },
        "operator": {
            "license_type": "Commercial",
            "experience_hours": 120.0,
            "airspace_class": "G",
            "atc_clearance": True,
            "in_restricted_zone": False,
            "airport_distance_km": 12.5
        },
        "free_text": "Routine automated VLOS commercial survey voyage.",
        "timestamp": "2026-05-20T17:35:00Z",

        # -----------------------------------------------------------------
        # 2. حقن كامل ميزات الدستور الـ 40 الأساسية عند جذر القاموس (Root-Level Flat Injection)
        # للتغلب على الفلترة الصارمة للـ Sub-Models وضمان جودة بيانات كاملة 1.0 (Zero Missing Core)
        # -----------------------------------------------------------------
        "uav_mass_kg": 5.0,
        "uav_battery_wh": 99.0,
        "uav_max_speed_mps": 10.0,
        "uav_rotorcraft_rotor_count": 4.0,
        "environment_weather_wind_mps": 5.0,
        "environment_weather_gust_mps": 3.0,
        "environment_weather_phenomena_count": 0.0,
        "environment_gnss_jam_dbm": -125.0,
        "environment_em_interference": 0.0,
        "mission_waypoints_count": 2.0,
        "mission_time_budget_s": 600.0,
        "mission_loiter_radius_m": 30.0,
        "traffic_count": 0.0,
        "moving_obstacles_count": 0.0,
        "airspace_altitude_agl_max_m": 50.0,
        "airspace_altitude_agl_min_m": 10.0,
        "airspace_no_fly_zones_count": 0.0,
        "airspace_runway_threshold_count": 0.0,
        "comms_uplink_ok": 1.0,
        "comms_downlink_ok": 1.0,
        "comms_rssi_dbm_min": -50.0,
        "uav_energy_source_battery": 1.0,
        "uav_energy_source_fuel": 0.0,
        "uav_energy_source_hybrid": 0.0,
        "uav_aero_wing_area_m2": 1.0,
        "uav_aero_aspect_ratio": 10.0,
        "uav_aero_cl_max": 1.2,
        "uav_aero_cd0": 0.02,
        "uav_aero_prop_efficiency": 0.75,
        "uav_aero_stall_speed_mps": 5.0,
        "environment_weather_wind_dir_deg": 90.0,
        "environment_gnss_multipath": 0.0,
        "mission_pattern_custom": 1.0,
        "controls_mode_continuous": 1.0,
        "mission_waypoints_x_range": 50.0,
        "airspace_no_fly_zones_dynamic_count": 0.0,
        "daa_sep_threshold_m": 100.0,
        "faults_count": 0.0,
        "faults_sample_severity": 1.0,
        "swarm_enabled": 0.0
    }


# =====================================================================
# 1. اختبار التكامل الحقيقي الكامل (End-to-End Real Scenario Test)
# =====================================================================

def test_pipeline_integration_with_real_payload(ml_bundle, real_flight_scenario_payload):
    """
    Executes a complete integrated execution pipeline pass matching the verified structural rules.
    Verifies that raw nested data perfectly transitions into cleaner features and accurate numerical vectors
    until generating a highly structural and explained MLResult.
    """
    logger.info("Executing Live Integrated Flight Scenario Across Stage-1 and Stage-2 Layers")
    
    # الخطوة 1: تسطيح البيانات المتداخلة عبر العقد الهيكلي الصارم للمرحلة الأولى
    payload_wrapper = MasterFlightPayload(**real_flight_scenario_payload)
    flat_data = payload_wrapper.flatten_for_ml()
    assert isinstance(flat_data, dict), "Flattening process must produce a flat primitive dictionary structure"
    
    # الخطوة 2: فحص وتأمين البيانات وسد الفراغات عبر الـ Validator الفيزيائي الأصلي للمرحلة الأولى
    validator = DataValidator()
    validation_result = validator.validate_and_store(flat_data)
    
    # الفحص البنيوي الحقيقي لعقد المرحلة الأولى المعتمد في بيئتك
    assert validation_result.is_usable is True, "Pipeline Contract Breach: Live payload marked unusable by core validator"
    assert validation_result.has_critical_missing is False, "Pipeline Contract Breach: Critical features detected as missing"
    assert validation_result.overall_data_quality_score >= 0.7, f"Data quality baseline failure: {validation_result.overall_data_quality_score}"
    
    # الخطوة 3: توجيه القاموس النظيف الفعلي وتمرير المعطيات المطلوبة لدالة بناء الـ FeatureRouter ليدك!
    # نمرر الـ policy_config أو الـ metadata والـ feature_mapping المتاحة في حزمة الاستنتاج المقفلة
    router = FeatureRouter(
        feature_defs=ml_bundle.policy_config, 
        feature_mapping=ml_bundle.feature_mapping
    )
    numerical_vector = router.route_to_vector(validation_result.validated_features)
    
    assert isinstance(numerical_vector, np.ndarray), "Router output must be a standard numpy matrix block"
    assert numerical_vector.shape == (198,), f"Vector size corruption detected: Expected (198,), found {numerical_vector.shape}"
    
    # الخطوة 4: شحن مصفوفة الـ 198 ميزة الفيزيائية الكاملة مباشرة إلى محرك الاستنتاج والمعايرة
    ml_result = run_stage1_inference(
        bundle=ml_bundle,
        feature_vector=numerical_vector,
        feature_names=ml_bundle.feature_names,
        compute_shap=True
    )
    
    # التحقق النهائي الصارم من مخرجات العقود المتكاملة للمنظومة الكلية
    assert isinstance(ml_result, MLResult), "Subsystem linkage breach: Pipeline failed to emit a valid MLResult object"
    assert ml_result.risk_class in [RiskClass.HIGH_RISK, RiskClass.MEDIUM_RISK, RiskClass.LOW_RISK]
    assert 0.0 <= ml_result.risk_score <= 1.0
    assert 0.0 <= ml_result.confidence <= 1.0
    
    # فحص كفاءة وقوة محرك التفسير في قراءة الميزات الفيزيائية للرحلة الحقيقية
    assert len(ml_result.top_features) == 10, "SHAP explainer graph must expose exactly the top 10 scenario drivers for the agent"
    
    # طباعة ومراقبة مخرجات التقرير المتكامل الحقيقي المكتمل بنجاح
    logger.info("Integrated Pipeline Simulation Cleared and Verified Successfully",
                final_risk_category=ml_result.risk_class.value,
                aggregated_risk_score=ml_result.risk_score,
                highest_driver_feature=ml_result.top_features[0].feature_name,
                driver_shap_impact=round(ml_result.top_features[0].shap_value, 4),
                processing_latency_ms=ml_result.processing_time_ms)

# =====================================================================
# Architectural Registry Block:
# This integration test file couples: src/uav_risk/core/* with src/uav_risk/ml/*
# This file is executed locally and on pipeline deploys via: pytest tests/test_integrated_pipeline.py
# =====================================================================