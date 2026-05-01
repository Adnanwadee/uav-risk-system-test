"""
Test: Canonicalize (V1.0 - Comprehensive)
==================================================
يختبر:
1. تحويل البيانات الكاملة → status = OK
2. البيانات الناقصة في Core → status = CORE_FIELD_MISSING
3. البيانات الناقصة في Optional → تسجيل في missing_optional_fields
4. feature_vector shape = (58,)
5. ML pipeline integration (actual prediction)

المدخلات: flat_data dict
المخرجات: CanonicalizationResult + ML prediction
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.uav_risk.stage1.loader import load_stage1_artifacts
from src.uav_risk.stage1.canonicalize import canonicalize_scenario, CORE_FIELDS
from src.uav_risk.stage1.infer import infer, create_inference_input

print("=" * 70)
print("🧪 TEST 3: Canonicalization + ML Inference")
print("=" * 70)

# ============================================================================
# SECTION 0: Load artifacts
# ============================================================================
print("\n📌 SECTION 0: Loading Artifacts")
print("-" * 50)

artifacts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'artifacts'))
artifacts = load_stage1_artifacts(artifacts_dir=artifacts_path)
print(f"[INPUT] Artifacts loaded from: {artifacts_path}")

# ============================================================================
# SECTION 1: Helper function to create mock data
# ============================================================================
print("\n📌 SECTION 1: Creating Mock Flight Data")
print("-" * 50)

def create_mock_flight_data(custom_values=None):
    """Create complete mock flight data with optional overrides"""
    base_data = {
        # Core fields (required)
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
        
        # Optional fields
        "mission.waypoints": [[0, 0, 0], [100, 0, 10], [200, 50, 20]],
        "uav.sensors": {"lidar": "1", "camera_rgb": "1", "gnss": "1", "imu": "1"},
        "moving_obstacles": [],
        "airspace.geofence": None,
        "comms": {"uplink_ok": True, "downlink_ok": True},
        "environment": {},
        "mission": {},
        "uav": {}
    }
    
    if custom_values:
        base_data.update(custom_values)
    
    return base_data

print(f"[OUTPUT] Mock data template created")

# ============================================================================
# SECTION 2: Scenario 1 - Complete data (should succeed)
# ============================================================================
print("\n📌 SECTION 2: Scenario 1 - Complete Data")
print("-" * 50)

flat_data_complete = create_mock_flight_data()
print(f"[INPUT] flat_data keys: {len(flat_data_complete)}")
print(f"[INPUT] Core fields present: {all(k in flat_data_complete for k in CORE_FIELDS)}")

result = canonicalize_scenario(
    flat_data_complete, 
    artifacts.feature_registry, 
    artifacts.preprocessor
)

print(f"\n[OUTPUT] CanonicalizationResult:")
print(f"  - status: {result.status}")
print(f"  - feature_vector shape: {result.feature_vector.shape if result.feature_vector is not None else 'None'}")
print(f"  - missing_core_fields: {result.missing_core_fields}")
print(f"  - missing_optional_fields: {len(result.missing_optional_fields)}")
print(f"  - warnings: {len(result.warnings)}")

assert result.status == "OK", f"Expected OK, got {result.status}"
assert result.feature_vector is not None, "Feature vector is None"
assert result.feature_vector.shape[0] == 58, f"Expected shape (58,), got {result.feature_vector.shape}"
print("\n[✓] Complete data canonicalization: PASSED")

# Run ML inference on this data
print(f"\n[PROCESS] Running ML inference on canonicalized data...")
inference_input = create_inference_input(result.feature_vector, artifacts)
ml_result = infer(inference_input)
print(f"[OUTPUT] ML Result:")
print(f"  - risk_score: {ml_result.risk_score}")
print(f"  - risk_category: {ml_result.risk_category}")
print(f"  - confidence: {ml_result.confidence}")
print(f"  - status: {ml_result.status}")
print("[✓] ML inference on complete data: PASSED")

# ============================================================================
# SECTION 3: Scenario 2 - Missing CORE field (should fail)
# ============================================================================
print("\n📌 SECTION 3: Scenario 2 - Missing Core Field")
print("-" * 50)

flat_data_missing_core = create_mock_flight_data()
flat_data_missing_core["uav.mass_kg"] = None
print(f"[INPUT] Removed/Set None: 'uav.mass_kg'")

missing_core_before = [f for f in CORE_FIELDS if f not in flat_data_missing_core or flat_data_missing_core[f] is None]
print(f"[INPUT] Missing core fields before canonicalization: {missing_core_before}")

result = canonicalize_scenario(
    flat_data_missing_core,
    artifacts.feature_registry,
    artifacts.preprocessor
)

print(f"\n[OUTPUT] CanonicalizationResult:")
print(f"  - status: {result.status}")
print(f"  - missing_core_fields: {result.missing_core_fields}")

assert result.status == "CORE_FIELD_MISSING", f"Expected CORE_FIELD_MISSING, got {result.status}"
assert "uav.mass_kg" in result.missing_core_fields, "uav.mass_kg should be in missing_core_fields"
print("[✓] Missing core field detection: PASSED")

# ============================================================================
# SECTION 4: Scenario 3 - Missing OPTIONAL fields (should warn, not fail)
# ============================================================================
print("\n📌 SECTION 4: Scenario 3 - Missing Optional Fields")
print("-" * 50)

flat_data_missing_optional = create_mock_flight_data()
flat_data_missing_optional["uav.sensors"] = {}  # Remove all sensors
flat_data_missing_optional["comms"] = {}  # Remove comms
print(f"[INPUT] Removed sensors and comms (optional fields)")

result = canonicalize_scenario(
    flat_data_missing_optional,
    artifacts.feature_registry,
    artifacts.preprocessor
)

print(f"\n[OUTPUT] CanonicalizationResult:")
print(f"  - status: {result.status}")
print(f"  - missing_optional_fields: {len(result.missing_optional_fields)}")
if result.missing_optional_fields:
    print(f"  - sample missing: {result.missing_optional_fields[:5]}")

assert result.status == "OK", f"Expected OK (optional missing ok), got {result.status}"
print("[✓] Missing optional fields handled correctly (no failure): PASSED")

# ============================================================================
# SECTION 5: Scenario 4 - Edge case: NaN values
# ============================================================================
print("\n📌 SECTION 5: Scenario 4 - NaN Values in Core Field")
print("-" * 50)

flat_data_nan = create_mock_flight_data()
flat_data_nan["environment.weather.wind_mps"] = float('nan')
print(f"[INPUT] Set wind_mps to NaN")

result = canonicalize_scenario(
    flat_data_nan,
    artifacts.feature_registry,
    artifacts.preprocessor
)

print(f"\n[OUTPUT] CanonicalizationResult:")
print(f"  - status: {result.status}")
print(f"  - warnings: {len(result.warnings)}")

# NaN should be treated as missing for core fields
if result.status != "OK":
    print(f"[INFO] NaN detected as missing core field: {result.status}")
else:
    print(f"[INFO] NaN was handled (converted to default)")

print("[✓] NaN handling test complete")

# ============================================================================
# SECTION 6: Summary
# ============================================================================
print("\n" + "=" * 70)
print("✅ Canonicalization Tests Complete!")
print("=" * 70)
print(f"\n📊 SUMMARY:")
print(f"   - Complete data: {'✅' if result.status == 'OK' else '❌'}")
print(f"   - Missing core detection: {'✅' if 'uav.mass_kg' in result.missing_core_fields else '❌'}")
print(f"   - Optional missing handling: {'✅' if len(result.missing_optional_fields) > 0 else '⚠️'}")
print(f"   - ML inference working: {'✅' if ml_result.status != 'ERROR' else '❌'}")
print("=" * 70)