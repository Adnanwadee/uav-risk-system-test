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

from dataclasses import dataclass

import numpy as np

from uav_risk.core.feature_engineering import (
    PRIMARY_FEATURES,
    generate_all_features,
    generate_all_features_map,
    generate_secondary_features,
    load_authoritative_feature_order,
)
from uav_risk.ml import feature_defs
from uav_risk.ml.loader import assemble_feature_vector_from_dict


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


def _complex_primary() -> dict[str, float]:
    base = _base_primary()
    base.update(
        {
            "uav_energy_source_fuel": 1.0,
            "uav_energy_source_hybrid": 1.0,
            "mission_pattern_custom": 0.0,
            "mission_pattern_grid": 1.0,
            "controls_mode_discrete": 0.0,
            "swarm_enabled": 1.0,
            "swarm_size": 5.0,
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
    return base


def _zero_counts_primary() -> dict[str, float]:
    base = _base_primary()
    base.update(
        {
            "mission_waypoints_count": 0.0,
            "landing_preferred_sites_count": 0.0,
            "landing_emergency_sites_count": 0.0,
            "traffic_count": 0.0,
            "moving_obstacles_count": 0.0,
            "comms_loss_windows_count": 0.0,
            "faults_count": 0.0,
        }
    )
    return base


def _assert_vector_sane(vector: np.ndarray) -> None:
    assert vector.shape == (198,)
    assert vector.dtype == np.float64
    assert np.isfinite(vector).all()


def test_generate_all_features_minimal_uav(capsys):
    vector = generate_all_features(_base_primary())
    _assert_vector_sane(vector)
    print("minimal", vector[:20].round(4).tolist())
    out = capsys.readouterr().out
    assert "minimal" in out


def test_generate_all_features_complex_mission(capsys):
    vector = generate_all_features(_complex_primary())
    _assert_vector_sane(vector)
    print("complex", vector[:20].round(4).tolist())
    out = capsys.readouterr().out
    assert "complex" in out


def test_generate_all_features_zero_counts_and_loader_assembly(capsys):
    @dataclass
    class FakeBundle:
        feature_names: list[str]
        policy_config: dict[str, float]

    primary = _zero_counts_primary()
    feature_order = load_authoritative_feature_order()
    bundle = FakeBundle(feature_names=feature_order, policy_config={})
    vector, metadata = assemble_feature_vector_from_dict(primary, bundle)
    _assert_vector_sane(vector)
    assert metadata["feature_map"]["traffic_count"] == 0.0
    assert metadata["feature_map"]["comms_loss_windows_x_range"] == 0.0
    print("zero_counts", vector[:20].round(4).tolist())
    out = capsys.readouterr().out
    assert "zero_counts" in out


def test_generate_secondary_features_override_is_preserved():
    secondary = generate_secondary_features(_base_primary(), overrides={"feat_disk_loading": 99999.0})
    assert secondary["feat_disk_loading"] == 99999.0


def test_generate_all_features_map_uses_order_and_bounds():
    feature_map = generate_all_features_map(_complex_primary())
    assert len(feature_map) == 198
    assert list(feature_map.keys()) == load_authoritative_feature_order()
    assert feature_map["sim_policy_frequency"] == 10.0
