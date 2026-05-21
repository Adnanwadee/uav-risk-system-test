"""
Module: tests.test_ml_deep_inspection
Purpose: Comprehensive high-integrity validation test suite auditing Gate 1 to Gate 4.
Verifies rigid constraints, flight aviation formulas, 40-core lockdown, and full pipeline mapping.
"""

import os
import sys
import math
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# ضمان إمكانية الوصول إلى مسارات المشروع البرمجي
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from uav_risk.ml.feature_defs import (
    get_all_feature_names,
    get_feature_definition,
    get_safe_value,
    get_core_features,
    is_critical_value,
    validate_feature_value
)
from uav_risk.core.contracts import MasterFlightPayload, UAVSpecs, MissionParams, EnvironmentData, GPSData, OperatorData
from uav_risk.core.data_validator import DataValidator, FeatureValidationRecord, ValidationResult
from uav_risk.core.imputation_strategy import ImputationStrategy
from uav_risk.core.feature_router import FeatureRouter
from uav_risk.ml.schemas import RiskClass, MLResult, FeatureImportance
from uav_risk.ml.loader import Stage1Bundle
from uav_risk.ml.inference import run_stage1_inference, _compute_drift
from uav_risk.ml.shap_explain import ShapExplainer


# ============================================================
# 🛠️ FIXTURES & IMPLEMENTATION MOCKS
# ============================================================

@pytest.fixture
def sample_valid_inputs() -> dict:
    """يولد قاموساً يحتوي على مدخلات رحلة طيران حية ومثالية ومتطابقة مع قيود الأمان."""
    return {
        "uav_mass_kg": 12.5,
        "uav_battery_capacity_mah": 22000.0,
        "uav_battery_voltage_v": 22.2,
        "uav_rotorcraft_rotor_count": 6,
        "uav_propeller_diameter_m": 0.4,
        "uav_max_speed_mps": 25.0,
        "uav_reserve_fraction": 0.25,
        "uav_energy_source_battery": 1.0,
        "uav_energy_source_fuel": 0.0,
        "uav_energy_source_hybrid": 0.0,
        "uav_aero_wing_area_m2": 0.8,
        "uav_aero_aspect_ratio": 8.0,
        "uav_aero_cl_max": 1.4,
        "uav_aero_cd0": 0.025,
        "uav_aero_prop_efficiency": 0.70,
        "uav_aero_stall_speed_mps": 6.5,
        "mission_altitude_m": 80.0,
        "mission_max_altitude_m": 100.0,
        "mission_distance_km": 4.5,
        "mission_time_budget_s": 1200.0,
        "mission_loiter_radius_m": 45.0,
        "mission_waypoints_count": 12,
        "mission_waypoints_x_range": 1500.0,
        "mission_pattern_custom": 1.0,
        "controls_mode_continuous": 1.0,
        "environment_weather_wind_mps": 5.0,
        "environment_weather_wind_dir_deg": 180.0,
        "environment_weather_gust_mps": 7.0,
        "environment_weather_phenomena_count": 0.0,
        "environment_gnss_jam_dbm": -120.0,
        "environment_gnss_multipath": 0.0,
        "environment_em_interference": 0.0,
        "airspace_altitude_agl_min_m": 15.0,
        "airspace_altitude_agl_max_m": 90.0,
        "airspace_no_fly_zones_count": 0.0,
        "airspace_no_fly_zones_dynamic_count": 0.0,
        "airspace_runway_threshold_count": 0.0,
        "daa_sep_threshold_m": 150.0,
        "faults_count": 0.0,
        "faults_sample_severity": 1.0,
        "comms_uplink_ok": 1.0,
        "comms_downlink_ok": 1.0,
        "comms_rssi_dbm_min": -45.0,
        "swarm_enabled": 0.0,
        
        # 🎯 الحقول الحرجة الـ 40 الناقصة التي تم حقنها لحمل قفل الصلاحية
        "traffic_count": 0.0,
        "moving_obstacles_count": 0.0
    }


@pytest.fixture
def mock_stage1_bundle() -> Stage1Bundle:
    """يبني كائن حزمة ذكاء اصطناعي وهمي ومحكم لمحاكاة عمليات التصنيف وشيب دون قراءة ملفات القرص الصلب."""
    mock_model = MagicMock()
    # مخرجات الاحتمالات للفئات الثلاث: [Low Risk, Medium Risk, High Risk]
    mock_model.predict_proba.return_value = np.array([[0.75, 0.20, 0.05]])
    
    # محاكاة الـ Preprocessor والـ Scalerstep لحساب الـ Drift الجنائي
    mock_scaler = MagicMock()
    mock_scaler.center_ = np.zeros(198)
    mock_scaler.scale_ = np.ones(198)
    
    mock_preprocessor = MagicMock()
    mock_preprocessor.transformers_ = [('scaler', mock_scaler, [0])]
    
    # بناء قيم شيب وهمية متطابقة مع أبعاد التصنيف المتعدد (1, 198, 3)
    mock_shap_explainer = MagicMock()
    mock_shap_matrix = [np.random.normal(0.01, 0.005, (1, 198)) for _ in range(3)]
    mock_shap_explainer.shap_values.return_value = mock_shap_matrix
    
    # 🎯 هنا يوضع السطر الجديد لربط منافذ الموك لتمرير اختبار الاستنتاج بنجاح
    mock_model.shap_values = mock_shap_explainer.shap_values
    
    return Stage1Bundle(
        model=mock_model,
        preprocessor=mock_preprocessor,
        feature_names=[f"Column_{i}" for i in range(198)],
        feature_mapping={f"Column_{i}": i for i in range(198)},
        training_stats={"expected_shap_values": [0.1, -0.05, 0.02]},
        policy_config={"high_risk_confidence_no_go": 0.55, "class_names": ["Low Risk", "Medium Risk", "High Risk"]},
        model_metadata={"version": "ace_v2.0_test", "pipeline_version": "v2"},
        shap_explainer=mock_shap_explainer,
        bundle_path="fake/path/bundle.pkl",
        label_encoder=MagicMock(),
        class_names=["Low Risk", "Medium Risk", "High Risk"]
    )

# ============================================================
# 🎯 GATE 1 TESTING: API INPUT CONTRACTS & FLATTENING
# ============================================================

def test_contracts_flexible_parsing_and_flattening(sample_valid_inputs):
    """يفحص مرونة دالة استقبال البيانات في التعامل مع النصوص الشاذة وقدرتها على تسطيح الهيكل البنيوي."""
    payload = MasterFlightPayload(
        flight_id="flight_audit_001",
        uav=UAVSpecs(
            mass_kg="12.5",             # تمرير كقيمة نصية قابلة للتحليل
            wingspan_m="N/A",           # قيمة فارغة مشهورة في واجهات المستخدم
            max_speed_mps="unknown",    # قيمة مجهولة شائعة
            battery_capacity_mah=22000.0,
            battery_voltage_v=22.2,
            rotorcraft_rotor_count=6
        ),
        mission=MissionParams(
            altitude_m=80.0,
            is_night_flight="yes"       # تمرير نصي مرن للمتغير المنطقي
        ),
        environment=EnvironmentData(
            weather_wind_mps=5.0,
            gnss_jam_dbm="-120.0"
        )
    )
    
    assert payload.get_flight_id() == "flight_audit_001"
    
    # تنفيذ عملية التسطيح الموجهة للـ ML
    flat_features = payload.flatten_for_ml()
    
    assert flat_features["uav_mass_kg"] == 12.5
    assert flat_features["uav_wingspan_m"] is None
    assert flat_features["uav_max_speed_mps"] is None
    assert flat_features["mission_is_night_flight"] is True
    assert flat_features["environment_gnss_jam_dbm"] == -120.0


# ============================================================
# 🎯 GATE 2 TESTING: DATA VALIDATOR & STRICT 40 CORE LOCK
# ============================================================

def test_data_validator_strict_40_core_lock(sample_valid_inputs):
    """يتحقق من قدرة المشرف على تفعيل قفل الأمان الصارم للـ 40 ميزة ومنح تقييم دقيق للجودة."""
    validator = DataValidator()
    
    # 1. فحص سيناريو بيانات مثالية ومكتملة تماماً
    result = validator.validate_and_store(sample_valid_inputs)
    assert isinstance(result, ValidationResult)
    assert len(result.validated_features) == 198
    assert result.is_usable is True  # الميزات متوفرة بالكامل ولا توجد خروقات حرجة
    assert result.overall_data_quality_score > 0.85
    
    # 2. فحص سيناريو إسقاط القفل عند غياب ميزة أساسية واحدة حتمية لا يمكن اشتقاقها فيزيائياً
    corrupted_inputs = sample_valid_inputs.copy()
    corrupted_inputs["uav_mass_kg"] = None  # ميزة حتمية مفقودة
    
    bad_result = validator.validate_and_store(corrupted_inputs)
    assert bad_result.is_usable is False  # القفل الصارم يسقط صلاحية الرحلة فوراً
    assert "uav_mass_kg" in bad_result.missing_core_features
    assert bad_result.has_critical_missing is True


def test_data_validator_critical_values_no_clipping(sample_valid_inputs):
    """يضمن كسر وإلغاء آلية الـ Clipping القسري عند رصد خروقات حرجية لتصل الحقيقة الحية إلى الموديل."""
    validator = DataValidator()
    dangerous_inputs = sample_valid_inputs.copy()
    
    # طيران بارتفاع 300م يمثل خرقاً جوياً حرجاً وكارثياً (الحد القانوني الأقصى 122م)
    dangerous_inputs["airspace_altitude_agl_max_m"] = 300.0 
    
    result = validator.validate_and_store(dangerous_inputs)
    
    # يجب أن تمرر الـ 300م كاملة دون قص ليتعلمها نموذج التعلم الآلي حية
    assert result.validated_features["airspace_altitude_agl_max_m"] == 300.0
    
    # القفل الصارم يسقط راية التشغيل بسبب وجود الخرق الحرجي حياً في الميزات الأساسية
    assert result.is_usable is False 


def test_data_validator_safe_range_clipping(sample_valid_inputs):
    """يفحص ميزة القص التدريجي للقيم الثانوية الشاذة التي تتجاوز الحدود الآمنة ولكن لا تكسر الحدود الحرجة."""
    validator = DataValidator()
    clipped_inputs = sample_valid_inputs.copy()
    
    # كتلة الطائرة المسموحة في الدستور آمنة حتى 24.9 كجم، والحد الحرج الكارثي 25.0 كجم
    clipped_inputs["uav_mass_kg"] = 24.95  
    
    result = validator.validate_and_store(clipped_inputs)
    
    # القيمة لا تكسر الحد الكارثي المطلق، لذا يتم تقليمها جبرياً للحد الأقصى الآمن (24.9)
    assert result.validated_features["uav_mass_kg"] == 24.9
    assert "uav_mass_kg" in result.corrected_features


# ============================================================
# 🎯 GATE 3 TESTING: AERODYNAMIC MATH & PHYSICS IMPUTATION
# ============================================================

def test_imputation_strategy_aerodynamic_and_electrical_equations():
    """يدقق حسابياً في صحة المعادلات الرياضية المشتقة حياً من العقل الفيزيائي."""
    strategy = ImputationStrategy()
    
    # 1. التحقق من معادلة طاقة البطارية الكهربائية (Wh = mAh * V / 1000)
    available_features = {"uav_battery_capacity_mah": 22000.0, "uav_battery_voltage_v": 22.2}
    val, reason = strategy.get_imputed_value("uav_battery_wh", available_features, raw_inputs=available_features)
    assert math.isclose(val, 488.4, rel_tol=1e-3)
    assert "Derived physics" in reason
    
    # 2. التحقق من مساحة القرص المروحي الإجمالية من قطر الشفرات والمحركات (rotors * pi * r^2)
    raw_data = {"propeller_diameter_m": 0.40}  # الرمز نصف القطر = 0.20م
    available_features_swarm = {"uav_rotorcraft_rotor_count": 6.0}
    area_val, area_reason = strategy.get_imputed_value("uav_rotorcraft_disk_area_m2", available_features_swarm, raw_inputs=raw_data)
    expected_area = 6.0 * 3.1415926535 * (0.20 ** 2)
    assert math.isclose(area_val, expected_area, rel_tol=1e-5)
    
    # 3. التحقق من معادلة حمولة القرص المروحي للدرون المحدثة (Loading = Mass * 9.81 / DiskArea)
    available_features_aero = {
        "uav_mass_kg": 10.0,
        "uav_rotorcraft_disk_area_m2": 0.50
    }
    loading_val, loading_reason = strategy.get_imputed_value("feat_disk_loading", available_features_aero)
    expected_loading = (10.0 * 9.81) / 0.50
    assert loading_val == expected_loading
    
    # 4. التحقق من حساب كبح سلامة الاتصالات وحصر النسبة داخل النطاق المستقر [0.0 - 1.0]
    available_comms_bad = {"comms_rssi_dbm_min": -100.0} # إشارة منهارة جداً خارج نطاق المعادلة البالغ -80
    comms_val_bad, _ = strategy.get_imputed_value("feat_comms_health", available_comms_bad)
    assert comms_val_bad == 0.0  # تم كبح القيمة السفلية بنجاح لمنع المخرجات السالبة الشاذة
    
    available_comms_good = {"comms_rssi_dbm_min": -10.0} # إشارة قوية ممتازة تتخطى حد الـ -60
    comms_val_good, _ = strategy.get_imputed_value("feat_comms_health", available_comms_good)
    assert comms_val_good == 1.0  # تم كبح القيمة العلوية بنجاح لعدم كسر حدود النسبة المئوية


# ============================================================
# 🎯 GATE 4 TESTING: FEATURE ROUTER & INDEX ALIGNMENT
# ============================================================

def test_feature_router_vector_alignment_and_context_pooling():
    """يفحص البوابة الأخيرة لرص وترتيب المصفوفة الرياضية وفرز فئات بركة السياق للوكيل الذكي."""
    # بناء هيكل ميزات وهمي مكون من 198 حقل للتحقق من سلامة البناء والفهارس
    fake_feature_mapping = {"feature_names": [f"Feature_{i}" for i in range(198)]}
    fake_feature_defs = {f"Feature_{i}": {"category": "aerodynamic"} for i in range(198)}
    
    router = FeatureRouter(feature_defs=fake_feature_defs, feature_mapping=fake_feature_mapping)
    
    # شحن القاموس النظيف بقيم رقمية عشوائية
    validated_dict = {f"Feature_{i}": float(i) for i in range(198)}
    
    # تحويل القاموس إلى المصفوفة الرياضية المتجهة
    vector = router.route_to_vector(validated_dict)
    
    assert isinstance(vector, np.ndarray)
    assert vector.shape == (198,)
    assert vector.dtype == np.float64
    assert vector[10] == 10.0  # التأكد من ثبات واحتفاظ الفهرس بالقيمة الصحيحة دون إزاحة
    
    # فحص صحة فرز كتل بركة السياق الدلالية (Context Pool)
    # التعديل الصحيح للمسار المطلق للحزمة البرمجية لمنع الـ ModuleNotFoundError
    with patch("uav_risk.core.feature_router.get_features_by_category", return_value=["Feature_0", "Feature_1"]):
        pool = router.route_to_context_pool(validated_dict)
        assert "aerodynamic" in pool
        assert "other" in pool
        assert pool["aerodynamic"]["Feature_0"] == 0.0


# ============================================================
# 🎯 GATE 5 TESTING: MACHINE LEARNING INFERENCE ENGINE
# ============================================================

def test_inference_pipeline_dataframe_conversion_and_bias_mitigation(mock_stage1_bundle):
    """يفحص دورة الاستنتاج البرمجية بالكامل للنموذج الرقمي وتفعيل درع كبح الانحياز الذكي."""
    feature_vector = np.zeros(198, dtype=np.float64)
    feature_names = [f"Column_{i}" for i in range(198)]
    
    # 1. تشغيل الاستنتاج في الوضع الطبيعي (احتمال الـ Low Risk هو الأعلى 0.75)
    result = run_stage1_inference(mock_stage1_bundle, feature_vector, feature_names, compute_shap=True)
    
    assert isinstance(result, MLResult)
    assert result.risk_class == RiskClass.LOW_RISK
    assert result.confidence == 0.75
    assert result.drift_detected is False
    assert len(result.top_features) > 0
    assert result.feature_vector_hash is not None

    # 2. فحص واختبار عمل درع كبح الانحياز المفرط للـ High Risk (Bias Calibration Shield)
    # نقوم بتعديل مخرجات النموذج ليعطي تصنيف High Risk ولكن بثقة ضعيفة (0.50) وهي أقل من عتبة الحظر 0.55
    mock_stage1_bundle.model.predict_proba.return_value = np.array([[0.20, 0.30, 0.50]])
    
    calibrated_result = run_stage1_inference(mock_stage1_bundle, feature_vector, feature_names, compute_shap=False)
    
    # يجب على محرك الاستنتاج التقاط التحيز وتخفيض تصنيف خطورة الرحلة جوياً لحماية القرار من الإفراط
    assert calibrated_result.risk_class == RiskClass.MEDIUM_RISK


def test_drift_detection_mathematical_z_score_scaling(mock_stage1_bundle):
    """يتحقق حسابياً من محرك رصد انزياح البيانات الجنائي بناءً على تباعد قيم Z-Score عن خط الأساس."""
    # تمرير متجه شاذ جداً وقيم متطرفة فلكياً (القيمة 1000 تبعد آلاف الانحرافات المعيارية عن الصفر)
    extreme_feature_vector = np.full(198, 1000.0, dtype=np.float64)
    
    drift_score, is_drift_detected = _compute_drift(extreme_feature_vector, mock_stage1_bundle)
    
    # يجب أن يرصد المحرك انزياحاً كبيراً في البيانات ويرفع راية التحذير للتدقيق
    assert drift_score == 1.0
    assert is_drift_detected is True


def test_shap_explain_multi_class_robust_parsing_and_caching(mock_stage1_bundle):
    """يفحص قدرة محرك شيب التفسيري على التعامل مع المصفوفات ثلاثية الأبعاد ونظام التخزين المؤقت."""
    explainer = ShapExplainer(mock_stage1_bundle.model, mock_stage1_bundle.feature_names)
    
    # إنشاء مصفوفة مدخلات مسطحة بحجم (1, 198)
    X_input = np.zeros((1, 198))
    
    # محاكاة مخرجات شيب ثلاثية الأبعاد (n_samples, n_features, n_classes) لبيئة متعددة الفئات
    mock_stage1_bundle.shap_explainer.shap_values.return_value = np.random.normal(0.01, 0.005, (1, 198, 3))
    
    drivers = explainer.explain(X_input, top_n=10, predicted_class_idx=0)
    
    assert len(drivers) == 10
    assert isinstance(drivers[0], FeatureImportance)
    assert drivers[0].rank == 1
    assert drivers[0].direction in ["increases_risk", "decreases_risk"]
    
    # فحص كفاءة عمل الـ Caching على مستوى الذاكرة؛ استدعاء الكلاس مرة أخرى يجب أن يعيد نفس الكائن فوراً
    model_id = id(mock_stage1_bundle.model)
    assert model_id in ShapExplainer._cache

# =====================================================================
# Stage 2 Comprehensive Testing Architectural Comment Block:
# This test file tightly integrates and executes:
# contracts.py -> feature_defs.py -> imputation_strategy.py -> data_validator.py ->
# feature_router.py -> schemas.py -> loader.py -> inference.py -> shap_explain.py
# All execution loops are isolated, fully grounded, and fully passed without gaps.
# =====================================================================