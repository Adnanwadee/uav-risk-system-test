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
import shutil
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from types import ModuleType

import numpy as np

from uav_risk.core.feature_engineering import (
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


TRACE_DIR = Path("artifacts") / "scenario_traces"


def _print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def _print_mapping(title: str, mapping: dict[str, float], keys: list[str] | None = None) -> None:
    print(f"{title} | count={len(mapping)}")
    items = mapping.items() if keys is None else ((key, mapping[key]) for key in keys if key in mapping)
    for key, value in items:
        print(f"  {key}: {value}")


def _base_primary() -> dict[str, float]:
    return {
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


def _trace_path(scenario_name: str) -> Path:
    return TRACE_DIR / f"{scenario_name}.json"


def _print_trace_summary(scenario_name: str, trace: dict[str, Any]) -> None:
    print(f"Trace saved to: {_trace_path(scenario_name)}")
    print("Primary input sample:")
    _print_mapping(
        "Primary",
        trace["primary_input"],
        [
            "uav_mass_kg",
            "mission_waypoints_count",
            "spawn_xyz_first",
            "environment_weather_wind_mps",
            "landing_preferred_sites_count",
        ],
    )
    print("Normalized primary sample:")
    _print_mapping(
        "Normalized primary",
        trace["normalized_primary"],
        [
            "uav_mass_kg",
            "mission_waypoints_count",
            "spawn_xyz_first",
            "environment_weather_wind_mps",
        ],
    )
    print("Feature map sample:")
    _print_mapping(
        "Feature map",
        trace["feature_map"],
        [
            "uav_battery_model_hover_power_w",
            "uav_battery_model_k_drag",
            "feat_disk_loading",
            "feat_sensor_redundancy",
            "mission_waypoints_z_mean",
            "landing_preferred_sites_z_mean",
            "feat_wind_speed_ratio",
        ],
    )
    print("Model result:")
    print(f"  risk_class: {trace['model_result']['risk_class']}")
    print(f"  risk_score: {trace['model_result']['risk_score']}")
    print(f"  probabilities: {trace['model_result']['probabilities']}")
    print(f"  feature_vector_shape: {trace['feature_vector_shape']}")


def _save_trace(scenario_name: str, trace: dict[str, Any]) -> None:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    with _trace_path(scenario_name).open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(trace), handle, ensure_ascii=False, indent=2)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _run_trace_scenario(scenario_name: str, primary: dict[str, float], preview_keys: list[str]) -> dict[str, Any]:
    bundle = _cached_compat_stage1_bundle()
    normalized_primary = _normalize_primary_inputs(primary)
    secondary = generate_secondary_features(primary)
    feature_map = generate_all_features_map(primary)
    feature_vector = generate_all_features(primary)
    assembled_vector, metadata = assemble_feature_vector_from_dict(primary, bundle)
    result = run_stage1_inference(
        bundle=bundle,
        feature_vector=feature_vector,
        feature_names=bundle.feature_names,
        compute_shap=True,
    )

    trace = {
        "scenario_name": scenario_name,
        "bundle_feature_count": len(bundle.feature_names),
        "bundle_order_matches_authoritative": bundle.feature_names == load_authoritative_feature_order(),
        "primary_input": primary,
        "normalized_primary": normalized_primary,
        "secondary_features": secondary,
        "feature_map": feature_map,
        "feature_vector_shape": list(feature_vector.shape),
        "assembled_vector_shape": list(assembled_vector.shape),
        "feature_vector": feature_vector.tolist(),
        "assembled_vector": assembled_vector.tolist(),
        "validator_metadata": metadata,
        "model_result": {
            "risk_class": result.risk_class.value,
            "risk_score": result.risk_score,
            "probabilities": result.probabilities,
            "top_features": [
                {
                    "feature_name": item.feature_name,
                    "shap_value": item.shap_value,
                    "feature_value": item.feature_value,
                    "description": item.description,
                }
                for item in result.top_features[:5]
            ],
        },
        "preview_keys": preview_keys,
        "feature_preview": {key: feature_map[key] for key in preview_keys if key in feature_map},
    }

    _save_trace(scenario_name, trace)
    _print_header(f"SCENARIO TRACE: {scenario_name}")
    _print_trace_summary(scenario_name, trace)
    print("Top 5 SHAP features:")
    for item in trace["model_result"]["top_features"]:
        print(
            f"  - {item['feature_name']}: shap={item['shap_value']} value={item['feature_value']} desc={item['description']}"
        )
    return trace


@lru_cache(maxsize=1)
def _cached_compat_stage1_bundle() -> CompatStage1Bundle:
    with tempfile.TemporaryDirectory() as tempdir:
        temp_path = Path(tempdir)
        shutil.copy("artifacts/stage1_production_bundle.pkl", temp_path / "stage1_production_bundle.pkl")
        shutil.copy("artifacts/stage1_feature_mapping.json", temp_path / "stage1_feature_mapping.json")
        (temp_path / "model_card.json").write_text(json.dumps({"version": "qa-trace"}), encoding="utf-8")
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
            json.dumps({"timestamp": "qa-trace", "train_shape": [0, 0], "selected_strategy": "qa-trace"}),
            encoding="utf-8",
        )

        original_stage1_bundle = load_stage1_bundle.__globals__["Stage1Bundle"]
        load_stage1_bundle.__globals__["Stage1Bundle"] = CompatStage1Bundle
        try:
            bundle = load_stage1_bundle(str(temp_path))
        finally:
            load_stage1_bundle.__globals__["Stage1Bundle"] = original_stage1_bundle

        return bundle


def test_01_nominal_simple_quadcopter_trace() -> None:
    trace = _run_trace_scenario(
        "scenario_01_nominal_simple_quadcopter",
        _simple_primary(),
        [
            "uav_battery_model_hover_power_w",
            "feat_disk_loading",
            "feat_sensor_redundancy",
            "mission_waypoints_z_range",
            "landing_preferred_sites_x_mean",
        ],
    )
    assert trace["model_result"]["risk_class"] in {RiskClass.HIGH_RISK.value, RiskClass.MEDIUM_RISK.value, RiskClass.LOW_RISK.value}
    assert trace["feature_vector_shape"] == [198]


def test_02_nominal_complex_hybrid_swarm_trace() -> None:
    trace = _run_trace_scenario(
        "scenario_02_nominal_complex_hybrid_swarm",
        _complex_primary(),
        [
            "uav_aero_wing_area_m2",
            "uav_battery_model_hover_power_w",
            "feat_comms_health",
            "feat_weather_severity",
            "swarm_roles_count",
        ],
    )
    assert trace["feature_map"]["uav_aero_wing_area_m2"] > 0.0
    assert trace["feature_map"]["swarm_roles_count"] == 3.0


def test_03_zero_counts_trace() -> None:
    trace = _run_trace_scenario(
        "scenario_03_zero_counts",
        _zero_counts_primary(),
        [
            "landing_preferred_sites_x_mean",
            "landing_emergency_sites_x_mean",
            "mission_waypoints_x_mean",
            "comms_loss_windows_x_mean",
            "faults_sample_t_s",
        ],
    )
    zero_keys = [
        key
        for key in trace["feature_map"]
        if (key.endswith("_mean") or key.endswith("_std"))
        and (
            "landing_preferred_sites" in key
            or "landing_emergency_sites" in key
            or "mission_waypoints" in key
            or "comms_loss_windows" in key
        )
    ]
    assert all(trace["feature_map"][key] == 0.0 for key in zero_keys)


def test_04_triplet_spawn_xyz_trace() -> None:
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
    trace = _run_trace_scenario(
        "scenario_04_triplet_spawn_xyz",
        primary,
        [
            "landing_preferred_sites_x_mean",
            "landing_preferred_sites_y_mean",
            "landing_preferred_sites_z_mean",
            "landing_emergency_sites_z_mean",
            "mission_waypoints_z_mean",
        ],
    )
    assert trace["feature_map"]["landing_preferred_sites_z_mean"] == 50.0
    assert trace["feature_map"]["mission_waypoints_z_mean"] == 50.0


def test_05_negative_mass_preservation_trace() -> None:
    primary = _base_primary()
    primary["uav_mass_kg"] = -1.0
    trace = _run_trace_scenario(
        "scenario_05_negative_mass_preservation",
        primary,
        [
            "uav_mass_kg",
            "uav_battery_model_hover_power_w",
            "feat_disk_loading",
            "feat_weather_severity",
        ],
    )
    assert trace["normalized_primary"]["uav_mass_kg"] == -1.0
    assert trace["feature_map"]["uav_mass_kg"] == -1.0


def test_06_impossible_wind_preservation_trace() -> None:
    primary = _base_primary()
    primary["environment_weather_wind_mps"] = 500.0
    trace = _run_trace_scenario(
        "scenario_06_impossible_wind_preservation",
        primary,
        [
            "environment_weather_wind_mps",
            "feat_wind_speed_ratio",
            "feat_weather_severity",
            "uav_battery_model_hover_power_w",
        ],
    )
    assert trace["normalized_primary"]["environment_weather_wind_mps"] == 500.0
    assert trace["feature_map"]["environment_weather_wind_mps"] == 500.0