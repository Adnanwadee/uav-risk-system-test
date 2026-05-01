"""
Test: Full Pipeline (End-to-End - V1.0 Comprehensive)
==================================================
يختبر التكامل الكامل:
1. تحميل artifacts
2. Canonicalization
3. ML Inference (باستخدام bundle مباشرة)
4. Drift Detection
5. طباعة جميع النتائج بالتفصيل
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import json
import numpy as np
from datetime import datetime
from src.uav_risk.stage1.loader import load_stage1_artifacts
from src.uav_risk.stage1.canonicalize import canonicalize_scenario, CORE_FIELDS
from src.uav_risk.stage1.infer import load_bundle, infer_from_bundle, predict_risk

print("=" * 70)
print("🧪 TEST 5: Full Pipeline (End-to-End)")
print("=" * 70)

# ============================================================================
# SECTION 0: Load artifacts and bundle
# ============================================================================
print("\n📌 SECTION 0: Loading Artifacts and Bundle")
print("-" * 50)

artifacts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'artifacts'))
start_time = time.time()
artifacts = load_stage1_artifacts(artifacts_dir=artifacts_path)
load_time = (time.time() - start_time) * 1000
print(f"[TIMING] Artifacts loaded in {load_time:.2f} ms")
print(f"[OUTPUT] training_stats features: {len(artifacts.training_stats)}")

# Load bundle for ML inference
bundle = load_bundle(os.path.join(artifacts_path, "stage1_bundle_v2.pkl"))

# ============================================================================
# SECTION 1: Create multiple test scenarios
# ============================================================================
print("\n📌 SECTION 1: Creating Test Scenarios")
print("-" * 50)

def create_scenario(name, custom_values):
    """Create a test scenario with custom values"""
    base_data = {
        "uav.mass_kg": 2.5,
        "uav.max_speed_mps": 15.0,
        "uav.battery_model.hover_power_W": 200.0,
        "environment.weather.wind_mps": 5.0,
        "environment.weather.gust_mps": 7.0,
        "environment.gnss_jam_dbm": -70.0,
        "daa.sep_threshold_m": 25.0,
        "daa.ttc_threshold_s": 10.0,
        "airspace.altitude_agl_max_m": 120.0,
        "uav.type": "quadrotor",
        "mission.type": "inspection",
        "mission.pattern": "grid",
        "environment.weather.visibility": "good",
        "mission.waypoints": [[0, 0, 0], [100, 0, 10], [200, 50, 20]],
        "uav.sensors": {"lidar": "1", "camera_rgb": "1", "gnss": "1", "imu": "1"},
        "moving_obstacles": [],
        "airspace.geofence": None,
        "comms": {"uplink_ok": True, "downlink_ok": True},
        "environment": {},
        "mission": {},
        "uav": {}
    }
    base_data.update(custom_values)
    return name, base_data

scenarios = [
    create_scenario("NOMINAL_FLIGHT", {}),
    create_scenario("HIGH_WIND", {"environment.weather.wind_mps": 15.0}),
    create_scenario("LOW_BATTERY", {"telemetry.battery_level_pct": 8.0}),
    create_scenario("HEAVY_UAV", {"uav.mass_kg": 45.0}),
    create_scenario("EMERGENCY_MISSION", {"mission.type": "emergency_medical"}),
]

print(f"[OUTPUT] Created {len(scenarios)} test scenarios")

# ============================================================================
# SECTION 2: Run pipeline for each scenario
# ============================================================================
print("\n📌 SECTION 2: Running Pipeline for All Scenarios")
print("=" * 70)

results = []

for scenario_name, flat_data in scenarios:
    print(f"\n{'='*50}")
    print(f"📋 SCENARIO: {scenario_name}")
    print(f"{'='*50}")
    
    # STEP 1: Canonicalization
    print(f"\n[STEP 1] Canonicalization...")
    start_time = time.time()
    canon_result = canonicalize_scenario(
        flat_data, 
        artifacts.feature_registry, 
        artifacts.preprocessor
    )
    canon_time = (time.time() - start_time) * 1000
    
    print(f"[TIMING] Canonicalization: {canon_time:.2f} ms")
    print(f"[OUTPUT] Status: {canon_result.status}")
    print(f"[OUTPUT] Feature vector shape: {canon_result.feature_vector.shape if canon_result.feature_vector is not None else 'None'}")
    print(f"[OUTPUT] Missing core: {canon_result.missing_core_fields}")
    print(f"[OUTPUT] Missing optional: {len(canon_result.missing_optional_fields)}")
    print(f"[OUTPUT] Warnings: {len(canon_result.warnings)}")
    
    if canon_result.status != "OK":
        print(f"[SKIP] Cannot proceed - canonicalization failed")
        results.append({
            "scenario": scenario_name,
            "status": "FAILED",
            "reason": f"Canonicalization: {canon_result.status}"
        })
        continue
    
    # STEP 2: ML Inference (using new bundle method)
    print(f"\n[STEP 2] ML Inference (using bundle)...")
    start_time = time.time()
    ml_result = infer_from_bundle(
        canon_result.feature_vector,
        bundle,
        drift_method="max",
        drift_z_threshold=3.0
    )
    infer_time = (time.time() - start_time) * 1000
    
    print(f"[TIMING] ML Inference: {infer_time:.2f} ms")
    print(f"[OUTPUT] Status: {ml_result.status}")
    print(f"[OUTPUT] risk_score: {ml_result.risk_score}")
    print(f"[OUTPUT] risk_category: {ml_result.risk_category}")
    print(f"[OUTPUT] confidence: {ml_result.confidence}")
    print(f"[OUTPUT] drift_detected: {ml_result.drift_detected}")
    print(f"[OUTPUT] drift_score: {ml_result.drift_score}")
    print(f"[OUTPUT] features_exceeding_threshold: {ml_result.features_exceeding_threshold}")
    
    if ml_result.top_offending_features:
        print(f"[OUTPUT] Top offenders: {ml_result.top_offending_features[:3]}")
    
    # STEP 3: Store result
    results.append({
        "scenario": scenario_name,
        "status": ml_result.status,
        "risk_score": ml_result.risk_score,
        "risk_category": ml_result.risk_category,
        "confidence": ml_result.confidence,
        "drift_detected": ml_result.drift_detected,
        "canon_time_ms": canon_time,
        "infer_time_ms": infer_time,
        "missing_optional": len(canon_result.missing_optional_fields)
    })

# ============================================================================
# SECTION 3: Summary Report
# ============================================================================
print("\n" + "=" * 70)
print("📊 PIPELINE EXECUTION SUMMARY")
print("=" * 70)
print(f"\n{'Scenario':<20} {'Status':<12} {'Risk Score':<12} {'Category':<15} {'Drift':<8}")
print("-" * 70)

for r in results:
    print(f"{r['scenario']:<20} {r['status']:<12} {r.get('risk_score', 'N/A'):<12} {r.get('risk_category', 'N/A'):<15} {r.get('drift_detected', 'N/A'):<8}")

print("\n" + "=" * 70)
print("✅ Full Pipeline Tests Complete!")
print("=" * 70)

# Performance summary
avg_canon = np.mean([r.get('canon_time_ms', 0) for r in results if 'canon_time_ms' in r])
avg_infer = np.mean([r.get('infer_time_ms', 0) for r in results if 'infer_time_ms' in r])
print(f"\n📈 PERFORMANCE:")
print(f"   - Avg Canonicalization: {avg_canon:.2f} ms")
print(f"   - Avg ML Inference: {avg_infer:.2f} ms")
print(f"   - Total scenarios: {len(scenarios)}")
successful = len([r for r in results if r['status'] != 'FAILED'])
print(f"   - Successful: {successful}/{len(scenarios)}")

print("\n" + "=" * 70)
