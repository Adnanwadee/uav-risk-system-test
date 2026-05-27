from __future__ import annotations

from uav_risk.ml.loader import load_stage1_bundle
from uav_risk.ml.raw_schema import (
    DROPPED_RAW_METADATA_FEATURES,
    FORBIDDEN_USER_FEATURES,
    GENERATED_RAW_FEATURES,
    INTERNAL_ONLY_RAW_FEATURES,
    OPTIONAL_RAW_OVERRIDE_FEATURES,
    PROCESSED_FEATURE_NAMES,
    PROFILE_CAPABILITY_FIELDS,
    PROFILE_DERIVED_RAW_FEATURES,
    PROFILE_IDENTITY_FIELDS,
    RAW_CATEGORICAL_FEATURES,
    RAW_FEATURE_NAMES,
    SCENARIO_REQUIRED_RAW_FEATURES,
)


def _partition_sets() -> tuple[set[str], set[str], set[str], set[str]]:
    return (
        set(PROFILE_DERIVED_RAW_FEATURES),
        set(SCENARIO_REQUIRED_RAW_FEATURES),
        set(GENERATED_RAW_FEATURES),
        set(DROPPED_RAW_METADATA_FEATURES),
    )


def test_raw_and_processed_schema_are_pinned_to_bundle():
    bundle = load_stage1_bundle("artifacts")

    assert len(RAW_FEATURE_NAMES) == 197
    assert len(PROCESSED_FEATURE_NAMES) == 198
    assert list(RAW_FEATURE_NAMES) == list(bundle.preprocessor.feature_names_in_)
    assert list(PROCESSED_FEATURE_NAMES) == list(bundle.feature_names)


def test_raw_schema_partition_counts_and_coverage():
    profile, scenario, generated, dropped = _partition_sets()

    assert len(PROFILE_DERIVED_RAW_FEATURES) == 16
    assert len(SCENARIO_REQUIRED_RAW_FEATURES) == 45
    assert len(GENERATED_RAW_FEATURES) == 129
    assert len(DROPPED_RAW_METADATA_FEATURES) == 7
    assert profile | scenario | generated | dropped == set(RAW_FEATURE_NAMES)


def test_raw_schema_partition_has_no_overlap():
    partitions = _partition_sets()

    for index, left in enumerate(partitions):
        for right in partitions[index + 1 :]:
            assert left.isdisjoint(right)


def test_forbidden_user_features_are_not_profile_or_scenario_contracts():
    user_contract = set(PROFILE_DERIVED_RAW_FEATURES) | set(SCENARIO_REQUIRED_RAW_FEATURES)

    assert set(FORBIDDEN_USER_FEATURES).isdisjoint(user_contract)


def test_profile_metadata_fields_are_not_raw_ml_features_unless_profile_derived():
    raw_profile = set(PROFILE_DERIVED_RAW_FEATURES)

    assert set(PROFILE_IDENTITY_FIELDS).isdisjoint(RAW_FEATURE_NAMES)
    for field in PROFILE_CAPABILITY_FIELDS:
        assert field not in RAW_FEATURE_NAMES or field in raw_profile


def test_optional_overrides_are_generated_and_exclude_internal_only_fields():
    optional = set(OPTIONAL_RAW_OVERRIDE_FEATURES)

    assert optional.issubset(GENERATED_RAW_FEATURES)
    assert optional.isdisjoint(INTERNAL_ONLY_RAW_FEATURES)
    assert set(INTERNAL_ONLY_RAW_FEATURES).issubset(GENERATED_RAW_FEATURES)


def test_controls_actions_first_is_optional_categorical_override():
    assert RAW_CATEGORICAL_FEATURES["controls_actions_first"] == ("fwd", "hold", "throttle")
    assert "controls_actions_first" in GENERATED_RAW_FEATURES
    assert "controls_actions_first" in OPTIONAL_RAW_OVERRIDE_FEATURES


def test_payload_mass_is_scenario_required_not_profile_derived():
    assert "uav_payload_mass_kg" in SCENARIO_REQUIRED_RAW_FEATURES
    assert "uav_payload_mass_kg" not in PROFILE_DERIVED_RAW_FEATURES
