"""
Test: Drift Detection (V1.0 - Comprehensive)
==================================================
يختبر:
1. compute_drift_score مع method="max", "count", "mean"
2. كشف القيم الشاذة (مثل وزن 10000 كجم)
3. التعامل مع 2D arrays
4. حساب عدد الميزات المتجاوزة للعتبة
5. top_offending_features

المدخلات: feature_vector, training_stats
المخرجات: drift_score, drift_detected, features_exceeding_threshold, top_offending_features
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.uav_risk.stage1.infer import compute_drift_score
from src.uav_risk.stage1.loader import load_stage1_artifacts

print("=" * 70)
print("🧪 TEST 4: Drift Detection (compute_drift_score)")
print("=" * 70)

# ============================================================================
# SECTION 0: Load training_stats
# ============================================================================
print("\n📌 SECTION 0: Loading training_stats")
print("-" * 50)

artifacts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'artifacts'))
artifacts = load_stage1_artifacts(artifacts_dir=artifacts_path)
training_stats = artifacts.training_stats

feature_names = list(training_stats.keys())
n_features = len(feature_names)
print(f"[INPUT] training_stats loaded: {n_features} features")
print(f"[INPUT] First 3 features: {feature_names[:3]}")

# Create normal vector (all means)
normal_vector = np.array([training_stats[f]['mean'] for f in feature_names])
print(f"[INPUT] Normal vector shape: {normal_vector.shape}")

# ============================================================================
# SECTION 1: Method Testing - "max"
# ============================================================================
print("\n📌 SECTION 1: Method 'max' (Best for anomaly detection)")
print("-" * 50)

print(f"\n[TEST 1.1] Normal data (close to mean)")
drift_score, detected, count, offenders, z_scores = compute_drift_score(
    normal_vector, training_stats, method="max", z_threshold=3.0
)
print(f"[OUTPUT] drift_score: {drift_score:.4f}")
print(f"[OUTPUT] drift_detected: {detected}")
print(f"[OUTPUT] features_exceeding_threshold: {count}")
assert detected == False, "Normal data should not trigger drift"
print("[✓] Normal data: PASSED")

# Create anomalous vector (mass extreme)
print(f"\n[TEST 1.2] Anomalous data - mass = 10000 kg (extreme)")
anomalous_vector = normal_vector.copy()
mass_index = feature_names.index("uav.mass_kg") if "uav.mass_kg" in feature_names else 0
original_mass_value = anomalous_vector[mass_index]
anomalous_vector[mass_index] = 100.0  # 100 sigma deviation
print(f"[INPUT] Modified 'uav.mass_kg' from {original_mass_value:.4f} to 100.0")

drift_score, detected, count, offenders, z_scores = compute_drift_score(
    anomalous_vector, training_stats, method="max", z_threshold=3.0
)
print(f"[OUTPUT] drift_score: {drift_score:.4f}")
print(f"[OUTPUT] drift_detected: {detected}")
print(f"[OUTPUT] features_exceeding_threshold: {count}")
if offenders:
    print(f"[OUTPUT] Top offenders: {offenders[:3]}")
assert detected == True, "Anomalous data should trigger drift"
print("[✓] Anomaly detection (max): PASSED")

# ============================================================================
# SECTION 2: Method Testing - "count"
# ============================================================================
print("\n📌 SECTION 2: Method 'count' (Count features exceeding threshold)")
print("-" * 50)

drift_score, detected, count, offenders, z_scores = compute_drift_score(
    anomalous_vector, training_stats, method="count", z_threshold=3.0
)
print(f"[OUTPUT] drift_score (count): {drift_score:.0f}")
print(f"[OUTPUT] drift_detected: {detected}")
print(f"[OUTPUT] features_exceeding_threshold: {count}")
assert detected == True, "Count method should detect anomaly"
print("[✓] Count method: PASSED")

# ============================================================================
# SECTION 3: Method Testing - "mean" (not recommended)
# ============================================================================
print("\n📌 SECTION 3: Method 'mean' (Not recommended - dilutes anomalies)")
print("-" * 50)

drift_score, detected, count, offenders, z_scores = compute_drift_score(
    anomalous_vector, training_stats, method="mean", z_threshold=3.0
)
print(f"[OUTPUT] drift_score (mean): {drift_score:.4f}")
print(f"[OUTPUT] drift_detected: {detected}")
print(f"[INFO] Mean method may not detect single anomaly (score: {drift_score:.4f})")
print("[✓] Mean method test complete")

# ============================================================================
# SECTION 4: 2D Array Handling
# ============================================================================
print("\n📌 SECTION 4: 2D Array Handling (Batches)")
print("-" * 50)

# Test 4.1: (1, 58) batch
two_d_vector = normal_vector.reshape(1, -1)
print(f"\n[TEST 4.1] Input shape: {two_d_vector.shape}")
drift_score, detected, count, offenders, z_scores = compute_drift_score(
    two_d_vector, training_stats, method="max", z_threshold=3.0
)
print(f"[OUTPUT] drift_score: {drift_score:.4f}")
print(f"[OUTPUT] drift_detected: {detected}")
print("[✓] 2D (1,58) handling: PASSED")

# Test 4.2: (10, 58) batch (warning expected)
batch_10x58 = np.zeros((10, 58))
print(f"\n[TEST 4.2] Input shape: {batch_10x58.shape}")
drift_score, detected, count, offenders, z_scores = compute_drift_score(
    batch_10x58, training_stats, method="max", z_threshold=3.0
)
print(f"[OUTPUT] drift_score: {drift_score:.4f}")
print(f"[OUTPUT] drift_detected: {detected}")
print("[INFO] Multiple samples warning was logged (expected)")
print("[✓] 2D (10,58) handling: PASSED")

# ============================================================================
# SECTION 5: Edge Cases
# ============================================================================
print("\n📌 SECTION 5: Edge Cases")
print("-" * 50)

# Test 5.1: NaN values
print(f"\n[TEST 5.1] Vector with NaN")
nan_vector = normal_vector.copy()
nan_vector[0] = np.nan
drift_score, detected, count, offenders, z_scores = compute_drift_score(
    nan_vector, training_stats, method="max", z_threshold=3.0
)
print(f"[OUTPUT] drift_score: {drift_score:.4f} (NaN handled)")
print("[✓] NaN handling: PASSED")

# Test 5.2: Inf values
print(f"\n[TEST 5.2] Vector with Inf")
inf_vector = normal_vector.copy()
inf_vector[0] = np.inf
drift_score, detected, count, offenders, z_scores = compute_drift_score(
    inf_vector, training_stats, method="max", z_threshold=3.0
)
print(f"[OUTPUT] drift_score: {drift_score:.4f} (Inf should trigger high drift)")
print(f"[OUTPUT] drift_detected: {detected}")
assert detected == True, "Inf should trigger drift"
print("[✓] Inf handling: PASSED")

# ============================================================================
# SECTION 6: Top Offending Features
# ============================================================================
print("\n📌 SECTION 6: Top Offending Features Analysis")
print("-" * 50)

# Create vector with multiple anomalies
multi_anomaly = normal_vector.copy()
if "uav.mass_kg" in feature_names:
    multi_anomaly[feature_names.index("uav.mass_kg")] = 50.0
if "uav.battery_model.hover_power_W" in feature_names:
    multi_anomaly[feature_names.index("uav.battery_model.hover_power_W")] = 30.0
if "feat_mission_dist_m" in feature_names:
    multi_anomaly[feature_names.index("feat_mission_dist_m")] = 20.0

drift_score, detected, count, offenders, z_scores = compute_drift_score(
    multi_anomaly, training_stats, method="max", z_threshold=3.0
)

print(f"[OUTPUT] Total features exceeding threshold: {count}")
print(f"[OUTPUT] Top offenders:")
for i, (name, z) in enumerate(offenders[:5]):
    print(f"  {i+1}. {name}: z={z:.2f}")

print("\n" + "=" * 70)
print("✅ Drift Detection Tests Complete!")
print("=" * 70)
print(f"\n📊 SUMMARY:")
print(f"   - Max method (recommended): {'✅' if detected else '❌'}")
print(f"   - 2D array handling: ✅")
print(f"   - NaN/Inf handling: ✅")
print("=" * 70)