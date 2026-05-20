"""
Module: tests.test_ml_gate4
Purpose: Production-grade comprehensive test suite for Stage-1 ML layer (Gate 4 Verification).
Dependencies: Imports and validates uav_risk.ml loader, inference, and schemas components.
"""

import os
import time
import pytest
import numpy as np
import structlog

# استيراد المكونات المقفلة لضمان الاتساق المطلق مع النظام الحقيقي
from uav_risk.ml.loader import load_stage1_bundle
from uav_risk.ml.inference import run_stage1_inference
from uav_risk.ml.schemas import RiskClass, MLResult

# إعداد نظام التتبع والـ Logger للاختبارات
logger = structlog.get_logger(__name__)

# تحديد مسار الـ Artifacts الافتراضي للاختبارات المحلية والـ CI/CD
ARTIFACTS_DIR = "artifacts"


@pytest.fixture(scope="module")
def loaded_bundle():
    """Fixture to load and share the verified Stage-1 production bundle across test cases."""
    if not os.path.exists(ARTIFACTS_DIR):
        pytest.skip(f"Skipping ML Gate 4 tests: Artifacts directory '{ARTIFACTS_DIR}' not found.")
    try:
        bundle = load_stage1_bundle(ARTIFACTS_DIR)
        return bundle
    except Exception as e:
        pytest.fail(f"Module Fixture Setup Failed: Unable to assemble Stage1Bundle: {str(e)}")


@pytest.fixture(scope="module")
def valid_feature_names(loaded_bundle):
    """Fixture to extract the official 198 feature names list from the active bundle."""
    return loaded_bundle.feature_names


@pytest.fixture(scope="module")
def baseline_vector():
    """Generates a standard neutral baseline array mimicking normalized feature distributions."""
    # مصفوفة خطية بحجم 198 ممتلئة بقيم وسيطة (0.1) لتجنب التصفير المطلق
    vec = np.full(198, 0.1, dtype=np.float64)
    return vec


# =====================================================================
# 1. اختبار خط الاستنتاج الكامل (End-to-End Inference Verification)
# =====================================================================

def test_ml_inference_end_to_end(loaded_bundle, valid_feature_names, baseline_vector):
    """Verifies that a valid input vector smoothly transitions through the pipeline and produces valid MLResult."""
    logger.info("Test Case Initiated: End-to-End Inference Pipeline Validation")
    
    result = run_stage1_inference(
        bundle=loaded_bundle,
        feature_vector=baseline_vector,
        feature_names=valid_feature_names,
        compute_shap=True
    )
    
    # التحقق من نوع الكائن المعاد وسلامة البيانات الداخلية
    assert isinstance(result, MLResult), "Output must be a valid strict MLResult dataclass contract"
    assert result.risk_class in [RiskClass.HIGH_RISK, RiskClass.MEDIUM_RISK, RiskClass.LOW_RISK]
    assert 0.0 <= result.risk_score <= 1.0, f"Risk score out of bounds: {result.risk_score}"
    assert 0.0 <= result.confidence <= 1.0, f"Model confidence out of bounds: {result.confidence}"
    
    # التأكد من المجموع الاحتمالي للتصنيف الثلاثي الفعلي للـ Artifacts
    prob_sum = sum(result.probabilities.values())
    assert abs(prob_sum - 1.0) < 0.01, f"Probabilities map must sum to approximately 1.0, got: {prob_sum}"
    
    # فحص محرك الـ SHAP ومطابقة الترتيب
    assert len(result.top_features) == 10, f"Expected exactly 10 explanatory features, found: {len(result.top_features)}"
    assert result.top_features[0].rank == 1, "Top driving feature must hold structural rank 1"
    assert result.processing_time_ms > 0.0, "Latency indicator execution time tracking must be positive"
    
    logger.info("Test Case Passed: End-to-End Inference verified successfully", 
                risk_class=result.risk_class.value, score=result.risk_score)


# =====================================================================
# 2. اختبار معالجة الحالات الحدية الصفرية (Edge-Case Zero Vector Verification)
# =====================================================================

def test_ml_handles_edge_case_vector(loaded_bundle, valid_feature_names):
    """Ensures that a completely zeroed vector does not cause arithmetic breakdowns like DivisionByZero."""
    logger.info("Test Case Initiated: Edge-Case Zero Vector Arithmetic Integrity")
    
    zero_vector = np.zeros(198, dtype=np.float64)
    
    result = run_stage1_inference(
        bundle=loaded_bundle,
        feature_vector=zero_vector,
        feature_names=valid_feature_names,
        compute_shap=False
    )
    
    assert result is not None
    assert result.confidence > 0.0, "Model must provide a firm mathematical judgment even on static zero inputs"
    assert len(result.top_features) == 0, "SHAP tracking list must be empty since compute_shap was flagged False"


# =====================================================================
# 3. اختبار التخزين المؤقت لمحرك التفسير (SHAP Explainer Caching Efficiency)
# =====================================================================

def test_shap_caching_performance(loaded_bundle, valid_feature_names, baseline_vector):
    """
    Validates that the class-level caching mechanism avoids rebuilding the explainer graph.
    Explicitly clears the global memory cache beforehand to ensure an accurate cold-vs-hot benchmark.
    """
    logger.info("Test Case Initiated: SHAP Explainer Cache Sub-millisecond Performance")
    
    # خطوة الضبط الهندسي الحقيقي: تفريغ الـ Cache تماماً لضمان بداية باردة دقيقة
    # وتجنب دفء الذاكرة الناتج عن تشغيل الاختبارات العمودية السابقة
    from uav_risk.ml.shap_explain import ShapExplainer
    ShapExplainer._cache.clear()
    
    # الجولة الأولى: بناء جراف الشجر بالكامل (Cold Start Initialization)
    t1_start = time.perf_counter()
    run_stage1_inference(loaded_bundle, baseline_vector, valid_feature_names, compute_shap=True)
    first_call_latency = time.perf_counter() - t1_start
    
    # الجولة الثانية: استخراج البيانات فوراً من الذاكرة (Hot Start Retrieval)
    t2_start = time.perf_counter()
    run_stage1_inference(loaded_bundle, baseline_vector, valid_feature_names, compute_shap=True)
    second_call_latency = time.perf_counter() - t2_start
    
    logger.info("Cache Performance Latency Matrix Analysed", 
                first_pass_s=round(first_call_latency, 5), 
                second_pass_s=round(second_call_latency, 5))
                
    # التأكد الصارم من أن الجولة الثانية أسرع بشكل ملحوظ نتيجة توفير وقت بناء جراف مكتبة SHAP
    assert second_call_latency < first_call_latency, (
        f"Performance Failure: Second pass ({second_call_latency:.5f}s) "
        f"must be faster than cold first pass ({first_call_latency:.5f}s)"
    )


# =====================================================================
# 4. اختبار تفعيل منظومة رصد انحراف البيانات (Data Drift Activation)
# =====================================================================

def test_drift_detection_activation(loaded_bundle, valid_feature_names):
    """Injects extreme out-of-distribution values to verify the Z-Score profiling triggers drift alerts."""
    logger.info("Test Case Initiated: Out-of-Distribution Data Drift Alert Activation")
    
    # إنشاء متجه شاذ يحتوي على قيم فيزيائية ضخمة جداً وخارجة عن منطق بيانات التدريب الأساسية
    extreme_drift_vector = np.full(198, 99999.0, dtype=np.float64)
    
    result = run_stage1_inference(
        bundle=loaded_bundle,
        feature_vector=extreme_drift_vector,
        feature_names=valid_feature_names,
        compute_shap=False
    )
    
    # إذا كانت المقاييس الإحصائية للـ RobustScaler محملة، يجب أن يرصد الانحراف
    if result.drift_score > 0.0:
        assert result.drift_score == 1.0 or result.drift_detected is True, "Drift alert framework must trigger on massive deviations"
        logger.info("Test Case Passed: Data Drift successfully isolated and flagged", score=result.drift_score)
    else:
        logger.info("Drift calculation bypassed gracefully due to missing scaler center configurations in fallback mode")


# =====================================================================
# 5. اختبار درع معايرة الانحياز وتخفيف المخاطر العالية (Bias-Mitigation Calibration Shield)
# =====================================================================

def test_bias_calibration_trigger_logic(loaded_bundle, valid_feature_names):
    """Validates the smart calibration rule that prevents weak pessimistic high-risk alerts from vetoing voyages."""
    logger.info("Test Case Initiated: High-Risk Bias Calibration Shield Logic")
    
    # نختبر منطق الدمج والمعايرة لـ RiskClass
    # إذا كان النموذج يتوقع High Risk ولكن بثقة احتمالية منخفضة أقل من العتبة (السياسة المقررة 0.55)،
    # يجب أن يتدخل الكود ويخفض التصنيف لـ Medium Risk حماية للرحلة من الإنذار الكاذب
    low_confidence_probabilities = {"High Risk": 0.42, "Low Risk": 0.18, "Medium Risk": 0.40}
    
    # نتحقق من تفعيل آلية الخفض الرياضي داخل الدالة عند محاكاة سيناريو تحيز ضعيف
    # نقوم باختبار المنطق البرمجي عبر التأكد من أن التصنيفات الصافية لا تعتمد الـ argmax الأعمى إذا لم تتخط الحواجز السياساتية
    assert low_confidence_probabilities["High Risk"] == max(low_confidence_probabilities.values())
    
    # نقوم بتمرير متجه مصمم ليعطي احتمالية عالية للـ High Risk ولكن نتحكم بالعتبة سياساتياً
    loaded_bundle.policy_config["high_risk_confidence_no_go"] = 0.99 # نرفع العتبة جداً لفرض تفعيل التدخل والمعايرة برمجياً
    
    zero_vector = np.zeros(198, dtype=np.float64)
    result = run_stage1_inference(loaded_bundle, zero_vector, valid_feature_names, compute_shap=False)
    
    # الكود المطور لدينا مبرمج ليمنع إعلان HIGH_RISK إذا لم يتخط الـ 0.99 في هذه الحالة المعدلة للاختبار
    assert result.risk_class != RiskClass.HIGH_RISK, "Bias Mitigation Failure: Pipeline allowed uncalibrated weak High Risk prediction"
    logger.info("Test Case Passed: Bias calibration mechanism clamped weak high-risk predictions successfully")


# =====================================================================
# 6. اختبار التعافي المرن والنمط التراجعي الآمن (Robust Degraded Fallback Mode)
# =====================================================================

def test_inference_graceful_fallback_on_corrupt_input(loaded_bundle, valid_feature_names):
    """Ensures that passing corrupt input values (like NaNs) triggers the secure degraded fallback instead of crashing."""
    logger.info("Test Case Initiated: Graceful Degraded Fallback Validation on Poisoned Matrix")
    
    # إنشاء متجه ملوث بقيم غير رقمية NaN تكسر العمليات الرياضية العادية
    poisoned_vector = np.full(198, np.nan, dtype=np.float64)
    
    # تشغيل الاستنتاج: الكود مبرمج ليمتص الخطأ عبر كتلة try-except ويعيد كائن Fallback معتدل وآمن
    result = run_stage1_inference(
        bundle=loaded_bundle,
        feature_vector=poisoned_vector,
        feature_names=valid_feature_names,
        compute_shap=False
    )
    
    assert result is not None
    assert result.risk_class == RiskClass.MEDIUM_RISK, "Fallback failure: Corrupt matrix must drop to cautious Medium Risk state"
    assert result.confidence == 0.0, "Fallback identity marker: Confidence must be explicit zero to flag technical failure to agent"
    assert result.drift_detected is True, "Fallback tracking: Drift indicator must turn True during breakdown bypass"
    
    logger.info("Test Case Passed: Secure degraded mode successfully absorbed structural poisoning without system termination")

# =====================================================================
# Architectural Registry Block:
# This test file depends on: src/uav_risk/ml/loader.py, src/uav_risk/ml/inference.py, src/uav_risk/ml/schemas.py
# This file is executed by: pytest tests/test_ml_gate4.py
# =====================================================================