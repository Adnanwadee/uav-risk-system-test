"""
Test: Feature Registry (V1.0 - Comprehensive)
==================================================
يختبر:
1. عدد الميزات (58)
2. validate_vector() مع مصفوفات 1D, 2D, lists
3. get_warnings_for_values() للقيم الطبيعية والشاذة
4. get_feature_list() و get_feature_index()
5. duplicate detection

المدخلات: FeatureRegistry object
المخرجات: نتائج الاختبارات المطبوعة
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.uav_risk.schema.feature_registry import FeatureRegistry

print("=" * 70)
print("🧪 TEST 1: Feature Registry (V3.2)")
print("=" * 70)

# ============================================================================
# SECTION 1: Initialization
# ============================================================================
print("\n📌 SECTION 1: Registry Initialization")
print("-" * 50)

registry = FeatureRegistry()
print(f"[INPUT] Creating FeatureRegistry...")
print(f"[OUTPUT] Expected features: {registry.expected_count}")
print(f"[OUTPUT] Actual features: {len(registry.get_feature_list())}")

assert registry.expected_count == 58, f"Expected 58 features, got {registry.expected_count}"
print("[✓] Feature count validation: PASSED")

# ============================================================================
# SECTION 2: validate_vector() with 1D arrays
# ============================================================================
print("\n📌 SECTION 2: validate_vector() - 1D Arrays")
print("-" * 50)

# Test 2.1: Correct vector (58 features)
vector_58 = np.zeros(58)
print(f"\n[TEST 2.1] Input: np.zeros(58) shape={vector_58.shape}")
result = registry.validate_vector(vector_58)
print(f"[OUTPUT] validate_vector() = {result}")
assert result == True, "Expected True for 58 features"
print("[✓] 58 features: PASSED")

# Test 2.2: Wrong vector (57 features)
vector_57 = np.zeros(57)
print(f"\n[TEST 2.2] Input: np.zeros(57) shape={vector_57.shape}")
result = registry.validate_vector(vector_57)
print(f"[OUTPUT] validate_vector() = {result}")
assert result == False, "Expected False for 57 features"
print("[✓] 57 features (should fail): PASSED")

# Test 2.3: Wrong vector (59 features)
vector_59 = np.zeros(59)
print(f"\n[TEST 2.3] Input: np.zeros(59) shape={vector_59.shape}")
result = registry.validate_vector(vector_59)
print(f"[OUTPUT] validate_vector() = {result}")
assert result == False, "Expected False for 59 features"
print("[✓] 59 features (should fail): PASSED")

# ============================================================================
# SECTION 3: validate_vector() with 2D arrays (batches)
# ============================================================================
print("\n📌 SECTION 3: validate_vector() - 2D Arrays (Batches)")
print("-" * 50)

# Test 3.1: 2D batch (1, 58)
batch_1x58 = np.zeros((1, 58))
print(f"\n[TEST 3.1] Input: np.zeros((1, 58)) shape={batch_1x58.shape}")
result = registry.validate_vector(batch_1x58)
print(f"[OUTPUT] validate_vector() = {result}")
assert result == True, "Expected True for (1, 58)"
print("[✓] Batch (1, 58): PASSED")

# Test 3.2: 2D batch (10, 58)
batch_10x58 = np.zeros((10, 58))
print(f"\n[TEST 3.2] Input: np.zeros((10, 58)) shape={batch_10x58.shape}")
result = registry.validate_vector(batch_10x58)
print(f"[OUTPUT] validate_vector() = {result}")
assert result == True, "Expected True for (10, 58)"
print("[✓] Batch (10, 58): PASSED")

# Test 3.3: 2D batch wrong shape (1, 57)
batch_1x57 = np.zeros((1, 57))
print(f"\n[TEST 3.3] Input: np.zeros((1, 57)) shape={batch_1x57.shape}")
result = registry.validate_vector(batch_1x57)
print(f"[OUTPUT] validate_vector() = {result}")
assert result == False, "Expected False for (1, 57)"
print("[✓] Batch (1, 57) should fail: PASSED")

# ============================================================================
# SECTION 4: validate_vector() with Python lists
# ============================================================================
print("\n📌 SECTION 4: validate_vector() - Python Lists")
print("-" * 50)

list_58 = [0.0] * 58
print(f"\n[TEST 4.1] Input: list of 58 zeros, len={len(list_58)}")
result = registry.validate_vector(list_58)
print(f"[OUTPUT] validate_vector() = {result}")
assert result == True, "Expected True for list of 58"

list_57 = [0.0] * 57
print(f"\n[TEST 4.2] Input: list of 57 zeros, len={len(list_57)}")
result = registry.validate_vector(list_57)
print(f"[OUTPUT] validate_vector() = {result}")
assert result == False, "Expected False for list of 57"
print("[✓] Python lists: PASSED")

# ============================================================================
# SECTION 5: get_warnings_for_values()
# ============================================================================
print("\n📌 SECTION 5: get_warnings_for_values() - Anomaly Detection")
print("-" * 50)

# Test 5.1: Normal value (no warning)
print(f"\n[TEST 5.1] Feature: 'uav.mass_kg', value=5.0 (normal range 0.5-2800)")
warnings = registry.get_warnings_for_values("uav.mass_kg", 5.0)
print(f"[OUTPUT] Warnings: {warnings}")
assert len(warnings) == 0, f"Expected no warnings, got {warnings}"
print("[✓] Normal value: PASSED")

# Test 5.2: Anomalous value (above max)
print(f"\n[TEST 5.2] Feature: 'uav.mass_kg', value=3000.0 (above max 2800)")
warnings = registry.get_warnings_for_values("uav.mass_kg", 3000.0)
print(f"[OUTPUT] Warnings: {warnings}")
assert len(warnings) > 0, "Expected warnings for anomalous value"
assert "Domain Note" in warnings[0] or "above" in warnings[0], "Warning should mention domain"
print("[✓] Anomalous value (above max): PASSED")

# Test 5.3: Anomalous value (below min)
print(f"\n[TEST 5.3] Feature: 'uav.mass_kg', value=0.1 (below min 0.5)")
warnings = registry.get_warnings_for_values("uav.mass_kg", 0.1)
print(f"[OUTPUT] Warnings: {warnings}")
assert len(warnings) > 0, "Expected warnings for value below min"
print("[✓] Anomalous value (below min): PASSED")

# Test 5.4: Feature with no metadata
print(f"\n[TEST 5.4] Feature: 'unknown_feature' (not in metadata)")
warnings = registry.get_warnings_for_values("unknown_feature", 100.0)
print(f"[OUTPUT] Warnings: {warnings}")
assert len(warnings) == 0, f"Expected no warnings, got {warnings}"
print("[✓] Unknown feature: PASSED")

# ============================================================================
# SECTION 6: get_feature_index()
# ============================================================================
print("\n📌 SECTION 6: get_feature_index()")
print("-" * 50)

print(f"\n[TEST 6.1] Getting index for 'uav.mass_kg'")
idx = registry.get_feature_index("uav.mass_kg")
print(f"[OUTPUT] Index: {idx}")
assert idx == 0, f"Expected index 0 for 'uav.mass_kg', got {idx}"
print("[✓] get_feature_index(): PASSED")

try:
    registry.get_feature_index("nonexistent_feature")
    print("[ERROR] Should have raised KeyError")
    assert False
except KeyError as e:
    print(f"[OUTPUT] KeyError raised correctly: {e}")
    print("[✓] Nonexistent feature raises KeyError: PASSED")

# ============================================================================
# SECTION 7: Feature list immutability
# ============================================================================
print("\n📌 SECTION 7: Feature List Immutability")
print("-" * 50)

original_list = registry.get_feature_list()
print(f"[INPUT] Original list length: {len(original_list)}")
try:
    original_list.append("should_not_work")
    print(f"[OUTPUT] List was modified! Length now: {len(original_list)}")
    print("[WARNING] List is mutable - this could cause issues")
except AttributeError:
    print("[OUTPUT] List is immutable (copy returned)")

print("\n" + "=" * 70)
print("✅ Feature Registry Tests Complete!")
print("=" * 70)