from __future__ import annotations

# STAGE6_CLEANUP_REVIEW:
# Classification: LEGACY_PLAN1_PLAN2_TESTS
# Runtime status: These tests target removed legacy feature-engineering APIs and are not
# part of the canonical raw-first Smart Skies backend validation path.
# Legacy signal: imports _normalize_primary_inputs or generate_secondary_features.
# Replacement coverage: raw-first tests in test_raw_feature_generation.py,
# test_raw_loader_integration.py, test_api_raw_assessment.py, and test_core_raw_contracts.py.
# Action rule: Temporarily skipped during Stage 6 cleanup; later delete or rewrite explicitly.

import pytest

pytest.skip(
    "STAGE6_CLEANUP_REVIEW: legacy PLAN1/PLAN2 feature-engineering tests "
    "target removed APIs (_normalize_primary_inputs/generate_secondary_features).",
    allow_module_level=True,
)

import json
import importlib
import shutil
import tempfile
import sys
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from types import ModuleType

import numpy as np
import pytest
from fastapi.testclient import TestClient

from uav_risk.core.contracts import MasterFlightPayload
from uav_risk.core.feature_engineering import (
    PRIMARY_FEATURES,
    _normalize_primary_inputs,
    generate_all_features,
    generate_all_features_map,
    generate_secondary_features,
    load_authoritative_feature_order,
)
from uav_risk.ml import feature_defs
from uav_risk.ml.inference import run_stage1_inference
from uav_risk.ml.loader import assemble_feature_vector_from_dict, load_stage1_bundle
from uav_risk.ml.schemas import RiskClass


@dataclass
class CompatStage1Bundle:
    model: Any
    preprocessor: Any
    label_encoder: Any
    shap_explainer: Any
    feature_names: list[str]
    feature_mapping: dict[str, int]
    training_stats: dict[str, Any]
    policy_config: dict[str, Any]
    model_metadata: dict[str, Any]
    bundle_path: str
    class_names: list[str]

    def get_model_version(self) -> str:
        return self.model_metadata.get("version", self.model_metadata.get("pipeline_version", "unknown"))


def _print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def _print_mapping(title: str, mapping: dict[str, float], keys: list[str] | None = None) -> None:
    print(f"{title} | count={len(mapping)}")
    items = mapping.items() if keys is None else ((key, mapping[key]) for key in keys if key in mapping)
    for key, value in items:
        print(f"  {key}: {value}")


def _assert_vector(vector: np.ndarray, expected_len: int = 198) -> None:
    assert isinstance(vector, np.ndarray)
    assert vector.ndim == 1
    assert vector.shape == (expected_len,)
    assert vector.dtype == np.float64
    assert np.isfinite(vector).all()


def _assert_feature_order(feature_map: dict[str, float]) -> None:
    order = load_authoritative_feature_order()
    assert list(feature_map.keys()) == order


def _base_primary() -> dict[str, float]:
    primary = {
        "uav_energy_source_fuel": 0.0,
        "uav_energy_source_hybrid": 0.0,
        "mission_pattern_custom": 1.0,
        "mission_pattern_grid": 0.0,
        "mission_pattern_orbit": 0.0,
        "mission_pattern_spiral": 0.0,
        "controls_mode_discrete": 1.0,
        "swarm_enabled": 0.0,
        "swarm_size": 1.0,
        "swarm_inter_uav_sep_min_m": 0.0,
        "swarm_roles_first_relay": 0.0,
        "swarm_roles_first_scout": 0.0,
        "swarm_roles_first_single": 0.0,
        "swarm_roles_first_solo": 0.0,
        "uav_mass_kg": 2.0,
        "uav_battery_wh": 40.0,
        "uav_fuel_l": 0.0,
        "uav_payload_mass_kg": 0.0,
        "uav_max_speed_mps": 15.0,
        "uav_max_tilt_deg": 20.0,
        "uav_reserve_fraction": 0.25,
        "uav_rotorcraft_rotor_count": 4.0,
        "uav_rotorcraft_max_climb_mps": 3.0,
        "uav_rotorcraft_hover_ceiling_m": 1000.0,
        "uav_aero_prop_efficiency": 0.75,
        "uav_sensors_gnss": 1.0,
        "uav_sensors_lidar": 0.0,
        "uav_sensors_radar": 0.0,
        "uav_sensors_camera_rgb": 0.0,
        "uav_sensors_camera_thermal": 0.0,
        "environment_weather_wind_mps": 0.0,
        "environment_weather_wind_dir_deg": 0.0,
        "environment_weather_gust_mps": 0.0,
        "environment_weather_phenomena_count": 0.0,
        "environment_gnss_jam_dbm": -125.0,
        "environment_gnss_multipath": 0.0,
        "environment_em_interference": 0.0,
        "airspace_altitude_agl_min_m": 10.0,
        "airspace_altitude_agl_max_m": 50.0,
        "airspace_no_fly_zones_count": 0.0,
        "airspace_no_fly_zones_sample_radius_m": 0.0,
        "airspace_no_fly_zones_sample_floor_m": 0.0,
        "airspace_no_fly_zones_sample_ceiling_m": 0.0,
        "airspace_no_fly_zones_dynamic_count": 0.0,
        "mission_runway_required": 0.0,
        "airspace_runway_length_m": 0.0,
        "spawn_xyz_first": 0.0,
        "spawn_yaw_deg": 0.0,
        "landing_preferred_sites_count": 0.0,
        "landing_preferred_sites_z_mean": 0.0,
        "landing_emergency_sites_count": 0.0,
        "mission_waypoints_count": 2.0,
        "mission_waypoints_z_mean": 30.0,
        "mission_time_budget_s": 600.0,
        "mission_loiter_radius_m": 30.0,
        "traffic_count": 0.0,
        "traffic_sample_speed_mps": 0.0,
        "moving_obstacles_count": 0.0,
        "moving_obstacles_sample_radius_m": 0.0,
        "daa_sep_threshold_m": 100.0,
        "daa_ttc_threshold_s": 0.0,
        "comms_uplink_ok": 1.0,
        "comms_downlink_ok": 1.0,
        "comms_rssi_dbm_min": -50.0,
        "comms_loss_windows_count": 0.0,
        "faults_count": 0.0,
        "faults_sample_severity": 1.0,
        "faults_sample_duration_s": 0.0,
    }
    assert set(primary) == set(PRIMARY_FEATURES)
    return primary


def _simple_primary() -> dict[str, float]:
    primary = _base_primary()
    primary.update(
        {
            "landing_preferred_sites_count": 1.0,
            "landing_emergency_sites_count": 1.0,
            "mission_waypoints_count": 2.0,
            "spawn_xyz_first": 0.0,
        }
    )
    return primary


def _complex_primary() -> dict[str, float]:
    primary = _base_primary()
    primary.update(
        {
            "uav_energy_source_fuel": 1.0,
            "uav_energy_source_hybrid": 1.0,
            "mission_pattern_custom": 0.0,
            "mission_pattern_grid": 1.0,
            "controls_mode_discrete": 0.0,
            "swarm_enabled": 1.0,
            "swarm_size": 3.0,
            "swarm_inter_uav_sep_min_m": 20.0,
            "swarm_roles_first_relay": 1.0,
            "swarm_roles_first_scout": 1.0,
            "uav_mass_kg": 14.0,
            "uav_battery_wh": 250.0,
            "uav_fuel_l": 4.5,
            "uav_payload_mass_kg": 3.0,
            "uav_max_speed_mps": 35.0,
            "uav_max_tilt_deg": 30.0,
            "uav_reserve_fraction": 0.35,
            "uav_rotorcraft_rotor_count": 8.0,
            "uav_rotorcraft_max_climb_mps": 8.0,
            "uav_rotorcraft_hover_ceiling_m": 2000.0,
            "uav_sensors_lidar": 1.0,
            "uav_sensors_radar": 1.0,
            "uav_sensors_camera_rgb": 1.0,
            "uav_sensors_camera_thermal": 1.0,
            "environment_weather_wind_mps": 8.0,
            "environment_weather_wind_dir_deg": 270.0,
            "environment_weather_gust_mps": 12.0,
            "environment_weather_phenomena_count": 1.0,
            "environment_gnss_jam_dbm": -120.0,
            "environment_gnss_multipath": 1.0,
            "environment_em_interference": 1.0,
            "airspace_altitude_agl_min_m": 30.0,
            "airspace_altitude_agl_max_m": 120.0,
            "airspace_no_fly_zones_count": 2.0,
            "airspace_no_fly_zones_sample_radius_m": 500.0,
            "airspace_no_fly_zones_sample_floor_m": 10.0,
            "airspace_no_fly_zones_sample_ceiling_m": 100.0,
            "airspace_no_fly_zones_dynamic_count": 1.0,
            "mission_runway_required": 1.0,
            "airspace_runway_length_m": 1200.0,
            "spawn_xyz_first": 100.0,
            "spawn_yaw_deg": 90.0,
            "landing_preferred_sites_count": 3.0,
            "landing_preferred_sites_z_mean": 40.0,
            "landing_emergency_sites_count": 2.0,
            "mission_waypoints_count": 10.0,
            "mission_waypoints_z_mean": 80.0,
            "mission_time_budget_s": 2400.0,
            "mission_loiter_radius_m": 100.0,
            "traffic_count": 4.0,
            "traffic_sample_speed_mps": 25.0,
            "moving_obstacles_count": 2.0,
            "moving_obstacles_sample_radius_m": 20.0,
            "daa_sep_threshold_m": 150.0,
            "daa_ttc_threshold_s": 30.0,
            "comms_rssi_dbm_min": -60.0,
            "comms_loss_windows_count": 2.0,
            "faults_count": 1.0,
            "faults_sample_severity": 2.0,
            "faults_sample_duration_s": 15.0,
        }
    )
    return primary


def _zero_counts_primary() -> dict[str, float]:
    primary = _base_primary()
    primary.update(
        {
            "landing_preferred_sites_count": 0.0,
            "landing_emergency_sites_count": 0.0,
            "mission_waypoints_count": 0.0,
            "mission_waypoints_z_mean": 0.0,
            "traffic_count": 0.0,
            "moving_obstacles_count": 0.0,
            "comms_loss_windows_count": 0.0,
            "faults_count": 0.0,
            "airspace_no_fly_zones_count": 0.0,
            "airspace_no_fly_zones_dynamic_count": 0.0,
            "environment_weather_wind_mps": 0.0,
            "environment_weather_gust_mps": 0.0,
            "spawn_xyz_first": 0.0,
        }
    )
    return primary


def _load_compat_stage1_bundle() -> CompatStage1Bundle:
    with tempfile.TemporaryDirectory() as tempdir:
        temp_path = Path(tempdir)
        shutil.copy("artifacts/stage1_production_bundle.pkl", temp_path / "stage1_production_bundle.pkl")
        shutil.copy("artifacts/stage1_feature_mapping.json", temp_path / "stage1_feature_mapping.json")
        (temp_path / "model_card.json").write_text(json.dumps({"version": "qa-test"}), encoding="utf-8")
        (temp_path / "stage1_inference_config.json").write_text(
            json.dumps(
                {
                    "class_names": ["High Risk", "Low Risk", "Medium Risk"],
                    "high_risk_confidence_no_go": 0.55,
                    "expected_shap_values": [],
                }
            ),
            encoding="utf-8",
        )
        (temp_path / "column_metadata_final.json").write_text(
            json.dumps({"timestamp": "qa-test", "train_shape": [0, 0], "selected_strategy": "qa-test"}),
            encoding="utf-8",
        )

        original_stage1_bundle = load_stage1_bundle.__globals__["Stage1Bundle"]
        load_stage1_bundle.__globals__["Stage1Bundle"] = CompatStage1Bundle
        try:
            bundle = load_stage1_bundle(str(temp_path))
        finally:
            load_stage1_bundle.__globals__["Stage1Bundle"] = original_stage1_bundle

        return bundle


@lru_cache(maxsize=1)
def _cached_compat_stage1_bundle() -> CompatStage1Bundle:
    return _load_compat_stage1_bundle()


def _run_and_print_inference(bundle: CompatStage1Bundle, primary: dict[str, float], title: str) -> Any:
    feature_vec, meta = assemble_feature_vector_from_dict(primary, bundle)
    print(f"{title} feature vector shape: {feature_vec.shape}")
    print(f"{title} metadata usable: {meta['is_usable']}")
    print(f"{title} metadata warnings: {meta['validator_warnings']}")
    result = run_stage1_inference(
        bundle=bundle,
        feature_vector=feature_vec,
        feature_names=bundle.feature_names,
        compute_shap=True,
    )
    print(f"{title} risk_class: {result.risk_class.value}")
    print(f"{title} risk_score: {result.risk_score}")
    print(f"{title} probabilities: {result.probabilities}")
    print(f"{title} top 5 SHAP features:")
    for item in result.top_features[:5]:
        print(f"  - {item.feature_name}: shap={item.shap_value} value={item.feature_value} desc={item.description}")
    return result


def test_01_simple_quadcopter_pipeline():
    _print_header("TEST 1: Simple Quadcopter Scenario")
    primary = _simple_primary()
    print("Input primary feature count:", len(primary))
    print("Input primary keys:", sorted(primary.keys()))

    secondary = generate_secondary_features(primary)
    feature_map = generate_all_features_map(primary)
    vector = generate_all_features(primary)

    print("Generated secondary feature count:", len(secondary))
    _print_mapping("All generated secondary features", secondary)
    print("Generated feature map count:", len(feature_map))
    print("Vector shape:", vector.shape)
    _assert_vector(vector)
    assert len(feature_map) == 198
    assert len(secondary) == 130
    _assert_feature_order(feature_map)

    top_keys = [
        "feat_disk_loading",
        "uav_rotorcraft_disk_area_m2",
        "feat_sensor_redundancy",
        "feat_comms_health",
        "feat_wind_gust_ratio",
        "feat_weather_severity",
        "comms_loss_windows_x_range",
        "mission_waypoints_z_range",
        "landing_preferred_sites_x_mean",
        "uav_battery_model_hover_power_w",
    ]
    _print_mapping("Top 10 selected outputs", feature_map, top_keys)

    print("RESULT: PASS - simple quadcopter pipeline produced 198 ordered features")


def test_02_complex_hybrid_swarm_pipeline():
    _print_header("TEST 2: Complex Hybrid Swarm Scenario")
    primary = _complex_primary()
    feature_map = generate_all_features_map(primary)
    secondary = generate_secondary_features(primary)

    print("Generated secondary feature count:", len(secondary))
    _print_mapping("Secondary outputs", secondary)
    print("uav_aero_wing_area_m2:", feature_map["uav_aero_wing_area_m2"])
    print("swarm_roles_count:", feature_map["swarm_roles_count"])
    print("feat_comms_health:", feature_map["feat_comms_health"])

    assert feature_map["uav_aero_wing_area_m2"] > 0.0
    assert feature_map["swarm_roles_count"] == 3.0
    assert 0.0 <= feature_map["feat_comms_health"] <= 1.0
    print("RESULT: PASS - complex hybrid swarm pipeline satisfied all checks")


def test_03_zero_counts_scenario():
    _print_header("TEST 3: Zero Counts Scenario")
    primary = _zero_counts_primary()
    feature_map = generate_all_features_map(primary)

    spatial_mean_keys = [
        key
        for key in feature_map
        if key.endswith("_mean")
        and (
            "landing_preferred_sites" in key
            or "landing_emergency_sites" in key
            or "mission_waypoints" in key
            or "comms_loss_windows" in key
        )
    ]
    spatial_std_keys = [
        key
        for key in feature_map
        if key.endswith("_std")
        and (
            "landing_preferred_sites" in key
            or "landing_emergency_sites" in key
            or "mission_waypoints" in key
            or "comms_loss_windows" in key
        )
    ]

    _print_mapping("Spatial means", feature_map, spatial_mean_keys)
    _print_mapping("Spatial stds", feature_map, spatial_std_keys)
    print("faults_sample_t_s:", feature_map["faults_sample_t_s"])

    assert all(feature_map[key] == 0.0 for key in spatial_mean_keys)
    assert all(feature_map[key] == 0.0 for key in spatial_std_keys)
    assert feature_map["faults_sample_t_s"] == 0.0
    print("RESULT: PASS - zero-count scenario collapsed spatial statistics to zero")


def test_04_missing_primary_features_hard_veto():
    _print_header("TEST 4: Missing Primary Features Hard Veto")
    primary = _base_primary()
    primary.pop("uav_mass_kg")
    primary.pop("mission_time_budget_s")

    try:
        _normalize_primary_inputs(primary)
        pytest.fail("Expected ValueError for missing primary features")
    except ValueError as exc:
        print("Caught ValueError:", exc)
        assert "Missing primary features" in str(exc)

    try:
        _normalize_primary_inputs({})
        pytest.fail("Expected ValueError for empty primary payload")
    except ValueError as exc:
        print("Caught ValueError for empty payload:", exc)
        assert "Missing primary features" in str(exc)

    print("RESULT: PASS - missing primaries are rejected before generation")


def test_05_accept_secondary_keys_in_loader():
    _print_header("TEST 5: Accept Secondary Keys in Loader")

    @dataclass
    class FakeBundle:
        feature_names: list[str]
        policy_config: dict[str, float]

    bundle = FakeBundle(load_authoritative_feature_order(), {})
    payload = _base_primary()
    payload["uav_aero_wing_area_m2"] = 1.0

    vector, metadata = assemble_feature_vector_from_dict(payload, bundle)
    print("Assembled vector shape:", vector.shape)
    print("Metadata keys:", sorted(metadata.keys()))
    print("uav_aero_wing_area_m2 position:", bundle.feature_names.index("uav_aero_wing_area_m2"))
    assert vector.shape == (198,)
    assert vector[bundle.feature_names.index("uav_aero_wing_area_m2")] == 1.0
    assert metadata["secondary_overrides"]["uav_aero_wing_area_m2"] == 1.0

    print("RESULT: PASS - loader accepted secondary input without altering it")


def test_06_overrides_are_used_as_provided():
    _print_header("TEST 6: Secondary Overrides")
    primary = _base_primary()

    overridden_secondary = generate_secondary_features(primary, overrides={"feat_disk_loading": 99999.0})
    print("Overridden feat_disk_loading:", overridden_secondary["feat_disk_loading"])
    assert overridden_secondary["feat_disk_loading"] == 99999.0

    feature_map = generate_all_features_map(primary, overrides={"feat_sensor_redundancy": 0.6})
    generated_default = generate_secondary_features(primary)["feat_sensor_redundancy"]
    print("User override feat_sensor_redundancy:", feature_map["feat_sensor_redundancy"])
    print("Generated default feat_sensor_redundancy:", generated_default)
    assert feature_map["feat_sensor_redundancy"] == 0.6

    print("RESULT: PASS - overrides were preserved exactly as provided")


def test_07_spawn_xyz_first_list_triplet():
    _print_header("TEST 7: spawn_xyz_first as Triplet")
    primary = _base_primary()
    primary.update(
        {
            "spawn_xyz_first": [100.0, 200.0, 50.0],
            "landing_preferred_sites_count": 1.0,
            "landing_emergency_sites_count": 1.0,
            "mission_waypoints_count": 1.0,
            "comms_loss_windows_count": 1.0,
        }
    )

    print("Normalized spawn value:", _normalize_primary_inputs(primary)["spawn_xyz_first"])
    try:
        feature_map = generate_all_features_map(primary)
        print("Unexpected success. Feature map count:", len(feature_map))
        _print_mapping(
            "Spatial outputs driven by spawn_xyz_first list",
            feature_map,
            [
                "landing_preferred_sites_x_mean",
                "landing_preferred_sites_y_mean",
                "landing_preferred_sites_z_mean",
                "landing_emergency_sites_x_mean",
                "landing_emergency_sites_y_mean",
                "landing_emergency_sites_z_mean",
                "mission_waypoints_x_mean",
                "mission_waypoints_y_mean",
                "mission_waypoints_z_mean",
                "comms_loss_windows_x_mean",
                "comms_loss_windows_y_mean",
            ],
        )
        assert feature_map["landing_preferred_sites_x_mean"] == 100.0
        assert feature_map["landing_preferred_sites_y_mean"] == 200.0
        assert feature_map["landing_preferred_sites_z_mean"] == 50.0
        assert feature_map["mission_waypoints_x_mean"] == 100.0
        assert feature_map["mission_waypoints_y_mean"] == 200.0
        assert feature_map["mission_waypoints_z_mean"] == 50.0
    except Exception as exc:
        print("Triplet path failed as a QA defect:", type(exc).__name__, exc)
        pytest.fail(f"spawn_xyz_first triplet is not accepted cleanly: {exc}")


def test_08_spawn_xyz_first_scalar_deprecation_warning():
    _print_header("TEST 8: spawn_xyz_first Scalar Compatibility")
    primary = _base_primary()
    primary.update(
        {
            "spawn_xyz_first": 100.0,
            "landing_preferred_sites_count": 1.0,
            "mission_waypoints_count": 1.0,
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        feature_map = generate_all_features_map(primary)

    print("Captured warnings:")
    for item in caught:
        print("  -", item.category.__name__, str(item.message))

    assert any(item.category is DeprecationWarning for item in caught)
    assert feature_map["landing_preferred_sites_x_mean"] == 100.0
    print("RESULT: PASS - scalar spawn_xyz_first still works with DeprecationWarning")


def test_09_real_model_inference_lightgbm():
    _print_header("TEST 9: Real LightGBM Inference")
    bundle = _cached_compat_stage1_bundle()
    print("Loaded bundle feature count:", len(bundle.feature_names))
    print("Loaded bundle class names:", bundle.class_names)
    print("Bundle order matches authoritative order:", bundle.feature_names == load_authoritative_feature_order())

    scenarios = [
        ("simple", _simple_primary()),
        ("complex", _complex_primary()),
        ("zero", _zero_counts_primary()),
    ]

    for scenario_name, primary in scenarios:
        print(f"\n--- Inference scenario: {scenario_name} ---")
        result = _run_and_print_inference(bundle, primary, scenario_name)
        assert result.risk_class in {RiskClass.HIGH_RISK, RiskClass.MEDIUM_RISK, RiskClass.LOW_RISK}
        assert abs(sum(result.probabilities.values()) - 1.0) < 1e-6
        assert len(result.top_features) > 0

    print("RESULT: PASS - real bundle produced inference outputs for all three scenarios")


def test_10_full_trace_from_payload_to_model_acceptance():
    _print_header("TEST 10: Full Trace From Payload to Model Acceptance")
    payload_kwargs = _simple_primary()
    payload_kwargs["uav_aero_wing_area_m2"] = 1.0
    payload = MasterFlightPayload(**payload_kwargs)

    flat_primary = payload.flatten_for_ml(primary_only=True)
    flat_all = payload.flatten_for_ml(primary_only=False)

    print("flatten_for_ml(primary_only=True) count:", len(flat_primary))
    print("flatten_for_ml(primary_only=False) count:", len(flat_all))
    _print_mapping("Primary-only flattened payload", flat_primary)
    _print_mapping("Full flattened payload", flat_all, ["uav_aero_wing_area_m2"])

    assert len(flat_primary) == 68
    assert set(flat_primary.keys()) == set(PRIMARY_FEATURES)
    assert "uav_aero_wing_area_m2" in flat_all

    compat_bundle = _cached_compat_stage1_bundle()

    input_map = payload.flatten_for_ml(primary_only=True)
    input_map["uav_aero_wing_area_m2"] = 1.0

    feature_vec, feature_meta = assemble_feature_vector_from_dict(input_map, compat_bundle)
    print("Assembled feature vector shape:", feature_vec.shape)
    print("Assembly metadata keys:", sorted(feature_meta.keys()))
    assert feature_vec.shape == (198,)
    assert feature_meta["feature_map"]["uav_aero_wing_area_m2"] == 1.0

    model_result = run_stage1_inference(
        bundle=compat_bundle,
        feature_vector=feature_vec,
        feature_names=compat_bundle.feature_names,
        compute_shap=True,
    )
    print("Model risk class:", model_result.risk_class.value)
    print("Model risk score:", model_result.risk_score)
    print("Model probabilities:", model_result.probabilities)
    assert model_result.risk_class in {RiskClass.HIGH_RISK, RiskClass.MEDIUM_RISK, RiskClass.LOW_RISK}

    print("RESULT: PASS - payload flattened into the authoritative 198-feature vector and the model accepted it")

def test_14_qa_impossible_wind_boundary_should_preserve_value():
    _print_header("TEST 14: QA Impossible Wind Boundary")
    primary = _base_primary()
    primary["environment_weather_wind_mps"] = 500.0
    feature_map = generate_all_features_map(primary)

    print("environment_weather_wind_mps input:", primary["environment_weather_wind_mps"])
    print("environment_weather_wind_mps output:", feature_map["environment_weather_wind_mps"])
    print("feat_wind_speed_ratio:", feature_map["feat_wind_speed_ratio"])
    print("feat_weather_severity:", feature_map["feat_weather_severity"])
    print("Why this test exists: physically impossible wind must be preserved, not clipped.")

    assert feature_map["environment_weather_wind_mps"] == 500.0
    assert feature_map["feat_wind_speed_ratio"] == 500.0 / primary["uav_max_speed_mps"]
    print("RESULT: PASS - impossible wind was preserved without clipping")
