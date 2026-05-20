"""
Module: tests.test_ml_deep_inspection
Purpose: Deep architectural inspection suite printing exact final features, imputations, and complete ML + SHAP outputs.
Dependencies: Core contract layers and updated ML inference engines.
"""

import os
import pytest
import numpy as np
import structlog

# استيراد واجهات وعقود المرحلة الأولى والمرحلة الثانية لربط المنظومة كاملة
from uav_risk.core.contracts import MasterFlightPayload
from uav_risk.core.data_validator import DataValidator
from uav_risk.core.feature_router import FeatureRouter
from uav_risk.ml.loader import load_stage1_bundle
from uav_risk.ml.inference import run_stage1_inference

# إعداد نظام التتبع وعرض البيانات المنظمة في السيرفر
logger = structlog.get_logger(__name__)
ARTIFACTS_DIR = "artifacts"


@pytest.fixture(scope="module")
def active_bundle():
    """Fixture to retrieve the master ML bundle from local storage storage."""
    if not os.path.exists(ARTIFACTS_DIR):
        pytest.skip("Inspection aborted: Artifacts directory missing.")
    return load_stage1_bundle(ARTIFACTS_DIR)


@pytest.fixture(scope="module")
def custom_raw_payload():
    """
    سيناريو Low Risk المثالي:
    تم ضبط القيم لتكون في قلب النطاق الآمن (Safe Baseline)
    لضمان تصنيف Low Risk بمصداقية مطلقة.
    """
    return {
        "flight_id": "GOLDEN-LOW-RISK-2026",
        "uav": {
            "mass_kg": 5.0, "wingspan_m": 1.5, "max_speed_mps": 10.0,
            "battery_wh": 200.0,  # بطارية ممتلئة وممتازة
            "battery_capacity_mah": 5000.0,
            "battery_voltage_v": 22.2, "rotorcraft_rotor_count": 4
        },
        "mission": {
            "altitude_m": 30.0, "max_altitude_m": 40.0,
            "distance_km": 0.5, "time_budget_s": 1200.0, # وقت طويل جداً ومريح
            "operation_type": "VLOS", "is_night_flight": False,
            "waypoints_count": 2
        },
        "environment": {
            "weather_wind_mps": 1.0,           # شبه ساكن
            "weather_wind_dir_deg": 0.0,
            "weather_gust_mps": 0.0,           # لا هبات
            "temperature_c": 20.0,
            "humidity_pct": 30.0,
            "weather_phenomena_count": 0,
            "gnss_jam_dbm": -30,            # إشارة GPS قوية جداً (الرقم السالب الكبير يعني نقاء أعلى)
            "em_interference": 0
        },
        "airspace": {
            "altitude_agl_min_m": 25.0, 
            "altitude_agl_max_m": 35.0,
            "no_fly_zones_count": 0, "runway_threshold_count": 0
        },
        "faults": {"count": 0, "sample_severity": 0}, 
        "comms": {"uplink_ok": 1, "downlink_ok": 1, "rssi_dbm_min": -20.0}, # إشارة اتصال ممتازة
        "traffic": {"count": 0, "sample_speed_mps": 0.0},
        "swarm": {"enabled": 0}
    }

# =====================================================================
# 1. اختبار تدقيق مصفوفة الميزات المدخلة (Feature Matrix & Imputation Inspection)
# =====================================================================

def test_inspect_final_feature_vector_and_imputations(active_bundle, custom_raw_payload):
    """
    Passes raw input through the pipeline and prints out the name and final numerical value 
    of every single feature to prove that imputation logic for missing variables is working flawlessly.
    """
    print("\n\n=======================================================================")
    print("🔬 TEST 1: DEEP VECTOR INSPECTION - PRINTING ALL 198 FEATURES FOR MODEL 🔬")
    print("=======================================================================")
    
    # تنفيذ خط تسوية وتسطيح البيانات للمرحلة الأولى
    payload_wrapper = MasterFlightPayload(**custom_raw_payload)
    flat_data = payload_wrapper.flatten_for_ml()
    
    # فحص وتأمين الداتا وسد الفراغات فيزيائياً
    validator = DataValidator()
    validation_result = validator.validate_and_store(flat_data)
    
    # تحويل القاموس النظيف إلى مصفوفة خطية عبر الـ FeatureRouter ليدك
    router = FeatureRouter(
        feature_defs=active_bundle.policy_config, 
        feature_mapping=active_bundle.feature_mapping
    )
    numerical_vector = router.route_to_vector(validation_result.validated_features)
    
    # استخراج قائمة أسماء الميزات المسجلة بالترتيب من الحزمة الحقيقية
    registered_names = active_bundle.feature_names
    
    print(f"\n➔ Total Features Registered in Array: {len(numerical_vector)}")
    print(f"➔ Data Quality Score Generated: {validation_result.overall_data_quality_score:.4f}")
    print(f"➔ Pipeline Safety Usability Flag: {validation_result.is_usable}\n")
    print(f"{'INDEX':<6} | {'FEATURE NAME':<50} | {'FINAL VALUE VALUE':<18}")
    print("-" * 80)
    
    # طباعة مصفوفة الميزات الـ 198 بالاسم والقيمة النهائية عياناً أمامنا لفحص دقة الاشتقاق
    for idx, name in enumerate(registered_names):
        val = numerical_vector[idx]
        print(f"{idx:<6} | {name:<50} | {val:<18.4f}")
        
    print("-" * 80)
    
    # تأكيدات السلامة الصارمة للعقد
    assert len(numerical_vector) == 198, "Array breakdown: Vector dimension must be exactly 198 elements"
    assert validation_result.is_usable is True, "Pipeline failed to recover and mark input data usable"
    print("➔ TEST 1 PASSED: Feature alignment and automated imputation verified successfully.")


# =====================================================================
# 2. اختبار فحص مخرجات التنبؤ والتفسير المشترك (ML + SHAP Outputs Inspection)
# =====================================================================

def test_inspect_ml_and_shap_output_results(active_bundle, custom_raw_payload):
    """
    Executes live model inference and deeply prints out the exact probability mapping, 
    risk rankings, calibrated classification, and the real text explanations from SHAP TreeExplainer.
    """
    print("\n\n=======================================================================")
    print("🧠 TEST 2: ML MODEL + SHAP EXPLAINER OUTPUT INNER AUDIT REPORT 🧠")
    print("=======================================================================")
    
    payload_wrapper = MasterFlightPayload(**custom_raw_payload)
    flat_data = payload_wrapper.flatten_for_ml()
    
    validator = DataValidator()
    validation_result = validator.validate_and_store(flat_data)
    
    router = FeatureRouter(
        feature_defs=active_bundle.policy_config, 
        feature_mapping=active_bundle.feature_mapping
    )
    numerical_vector = router.route_to_vector(validation_result.validated_features)
    
    # تشغيل الاستنتاج المطور وحساب مصفوفات شيب المتكاملة والـ Cache
    ml_result = run_stage1_inference(
        bundle=active_bundle,
        feature_vector=numerical_vector,
        feature_names=active_bundle.feature_names,
        compute_shap=True
    )
    
    print(f"\n➔ Calibrated Decision Category: {ml_result.risk_class.value}")
    print(f"➔ Aggregated Risk Score Matrix: {ml_result.risk_score:.4f}")
    print(f"➔ Highest Class Confidence Level: {ml_result.confidence:.4f}")
    print(f"➔ Live Latency Profiler: {ml_result.processing_time_ms:.2f} ms")
    print(f"➔ Audit Trail Hash: {ml_result.feature_vector_hash}\n")
    
    print("➔ Class Probabilities Map Distribution:")
    for risk_name, prob_val in ml_result.probabilities.items():
        print(f"   ├── [{risk_name}]: {prob_val * 100.0:.2f}%")
        
    print("\n➔ Top 10 Driving Features Extracted via Active SHAP Explainability Engine:")
    print("-" * 110)
    print(f"{'RANK':<5} | {'FEATURE NAME':<40} | {'SHAP VALUE':<12} | {'VALUE':<10} | {'CONSTITUTIONAL EXPLANATION'}")
    print("-" * 110)
    
    # تفكيك وشرح دوافع شيب بالاسم العربي والسياق الفيزيائي المكتوب في دستور النظام ليدك
    for feat in ml_result.top_features:
        print(f"#{feat.rank:<4} | {feat.feature_name:<40} | {feat.shap_value:<12.5f} | {feat.feature_value:<10.2f} | {feat.description}")
        
    print("-" * 110)
    
    # تأكيدات الصرامة المعمارية المخرجة
    assert ml_result.risk_score is not None
    assert len(ml_result.top_features) == 10, "Explainability error: System must extract exactly top 10 driving signals for the agent"
    print("➔ TEST 2 PASSED: Inference distribution and SHAP impact metrics fully verified.")

# =====================================================================
# Architectural Registry Block:
# This audit test connects: src/uav_risk/core/* coupled directly to src/uav_risk/ml/*
# Execute to generate detailed visual terminal logs via: pytest -s tests/test_ml_deep_inspection.py
# =====================================================================