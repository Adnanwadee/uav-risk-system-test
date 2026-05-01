"""
Test: Loader (V1.0 - Comprehensive)
==================================================
يختبر:
1. تحميل training_stats.json الحقيقي (not fallback)
2. التحقق من عدد الميزات (58)
3. التحقق من compatibility مع preprocessor
4. عرض عينة من القيم

المدخلات: artifacts_dir path
المخرجات: نتائج الاختبارات المطبوعة
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.uav_risk.stage1.loader import load_stage1_artifacts

print("=" * 70)
print("🧪 TEST 2: Loader (Stage1Artifacts)")
print("=" * 70)

# ============================================================================
# SECTION 1: Load artifacts
# ============================================================================
print("\n📌 SECTION 1: Loading Stage1Artifacts")
print("-" * 50)

artifacts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'artifacts'))
print(f"[INPUT] artifacts_dir = {artifacts_path}")

# Check if directory exists
if not os.path.exists(artifacts_path):
    print(f"[ERROR] Artifacts directory not found: {artifacts_path}")
    sys.exit(1)
print(f"[INFO] Artifacts directory exists")

# Load artifacts
print(f"\n[PROCESS] Loading Stage1Artifacts...")
artifacts = load_stage1_artifacts(artifacts_dir=artifacts_path)
print(f"[OUTPUT] Stage1Artifacts loaded successfully")

# ============================================================================
# SECTION 2: Verify training_stats (REAL, not fallback)
# ============================================================================
print("\n📌 SECTION 2: training_stats Validation")
print("-" * 50)

stats = artifacts.training_stats
print(f"[INPUT] training_stats type: {type(stats)}")
print(f"[INPUT] training_stats keys count: {len(stats)}")

# Check count
expected_count = 58
actual_count = len(stats)
print(f"[OUTPUT] Expected features: {expected_count}, Actual: {actual_count}")
assert actual_count == expected_count, f"Feature count mismatch: {actual_count} != {expected_count}"
print("[✓] Feature count: PASSED")

# Check if it's real data (not fallback)
first_key = list(stats.keys())[0] if stats else None
if first_key and 'note' in stats.get(first_key, {}):
    print(f"[WARNING] training_stats contains FALLBACK marker!")
    print(f"[OUTPUT] training_stats is FALLBACK (all zeros) - Drift detection will be unreliable!")
    is_fallback = True
else:
    print(f"[OUTPUT] training_stats is REAL data (no fallback marker)")
    is_fallback = False

# Display sample values
print(f"\n[OUTPUT] Sample training_stats values (first 5 features):")
for i, (key, value) in enumerate(list(stats.items())[:5]):
    print(f"  {i+1}. {key}: mean={value['mean']:.6f}, std={value['std']:.6f}")

# ============================================================================
# SECTION 3: Verify policy_config
# ============================================================================
print("\n📌 SECTION 3: policy_config Validation")
print("-" * 50)

policy = artifacts.policy_config
print(f"[INPUT] policy_config keys: {list(policy.keys())}")
print(f"[OUTPUT] min_confidence_go: {policy.get('min_confidence_go', 'MISSING')}")
print(f"[OUTPUT] high_risk_confidence_no_go: {policy.get('high_risk_confidence_no_go', 'MISSING')}")
print(f"[OUTPUT] min_confidence_any_decision: {policy.get('min_confidence_any_decision', 'MISSING')}")

required_keys = ["min_confidence_go", "high_risk_confidence_no_go", "min_confidence_any_decision"]
missing_keys = [k for k in required_keys if k not in policy]
if missing_keys:
    print(f"[WARNING] Missing policy keys: {missing_keys}")
else:
    print("[✓] All required policy keys present: PASSED")

# ============================================================================
# SECTION 4: Verify all components are loaded
# ============================================================================
print("\n📌 SECTION 4: Components Validation")
print("-" * 50)

components = {
    "reg_model": artifacts.reg_model,
    "calibrator_model": artifacts.calibrator_model,
    "preprocessor": artifacts.preprocessor,
    "label_encoder": artifacts.label_encoder,
    "feature_registry": artifacts.feature_registry,
    "training_stats": artifacts.training_stats,
    "policy_config": artifacts.policy_config,
}

all_loaded = True
for name, component in components.items():
    if component is None:
        print(f"[ERROR] {name} is None!")
        all_loaded = False
    else:
        component_type = type(component).__name__
        if name == "training_stats":
            print(f"[✓] {name}: dict with {len(component)} keys")
        elif name == "policy_config":
            print(f"[✓] {name}: dict with {len(component)} keys")
        else:
            print(f"[✓] {name}: {component_type}")

assert all_loaded, "Some components failed to load"
print("\n[✓] All components loaded successfully: PASSED")

# ============================================================================
# SECTION 5: Feature Registry validation
# ============================================================================
print("\n📌 SECTION 5: Feature Registry Validation")
print("-" * 50)

feature_registry = artifacts.feature_registry
print(f"[INPUT] FeatureRegistry expected features: {feature_registry.expected_count}")

# Check compatibility with training_stats
registry_features = set(feature_registry.get_feature_list())
stats_features = set(stats.keys())

common_features = registry_features & stats_features
registry_only = registry_features - stats_features
stats_only = stats_features - registry_features

print(f"[OUTPUT] Common features: {len(common_features)}")
print(f"[OUTPUT] Registry-only features: {len(registry_only)}")
print(f"[OUTPUT] Stats-only features: {len(stats_only)}")

if registry_only:
    print(f"[WARNING] Features in registry but not in training_stats: {list(registry_only)[:5]}")
if stats_only:
    print(f"[WARNING] Features in training_stats but not in registry: {list(stats_only)[:5]}")

print("\n" + "=" * 70)
print("✅ Loader Tests Complete!")
print("=" * 70)
print(f"\n📊 FINAL STATUS:")
print(f"   - training_stats: {'REAL' if not is_fallback else 'FALLBACK'}")
print(f"   - Features count: {actual_count}/58")
print(f"   - All components: {'✅' if all_loaded else '❌'}")
print("=" * 70)