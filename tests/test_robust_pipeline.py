from __future__ import annotations

import warnings
import numpy as np
import pytest

from uav_risk.core.feature_engineering import (
    generate_all_features,
    generate_all_features_map,
    generate_secondary_features,
    load_authoritative_feature_order,
)
from uav_risk.ml.loader import assemble_feature_vector_from_dict
from uav_risk.ml.inference import run_stage1_inference
from uav_risk.ml.schemas import RiskClass


def _base_primary() -> dict[str, float]:
    # Reuse the minimal primary set used across existing tests
    feature_order = load_authoritative_feature_order()
    # Build a simple valid primary dict by taking the first 68 primary keys
    # For compatibility with the SSoT this test provides expected canonical primaries
    base = {
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
    return base


def _complex_primary():
    base = _base_primary()
    base.update({
        "uav_energy_source_fuel": 1.0,
        "uav_energy_source_hybrid": 1.0,
        "mission_pattern_custom": 0.0,
        "mission_pattern_grid": 1.0,
        "controls_mode_discrete": 0.0,
        "swarm_enabled": 1.0,
        "swarm_size": 5.0,
        "uav_mass_kg": 14.0,
        "uav_battery_wh": 250.0,
        "uav_fuel_l": 4.5,
        "uav_payload_mass_kg": 3.0,
        "uav_max_speed_mps": 35.0,
        "uav_sensors_lidar": 1.0,
        "uav_sensors_radar": 1.0,
        "uav_sensors_camera_rgb": 1.0,
        "uav_sensors_camera_thermal": 1.0,
        "environment_weather_wind_mps": 8.0,
        "environment_weather_gust_mps": 12.0,
        "environment_gnss_jam_dbm": -120.0,
        "environment_gnss_multipath": 1.0,
        "environment_em_interference": 1.0,
        "airspace_altitude_agl_max_m": 120.0,
        "airspace_no_fly_zones_count": 2.0,
        "airspace_no_fly_zones_sample_radius_m": 500.0,
        "mission_runway_required": 1.0,
        "airspace_runway_length_m": 1200.0,
        "spawn_xyz_first": 100.0,
        "spawn_yaw_deg": 90.0,
        "landing_preferred_sites_count": 3.0,
        "landing_emergency_sites_count": 2.0,
        "mission_waypoints_count": 10.0,
        "mission_time_budget_s": 2400.0,
        "mission_loiter_radius_m": 100.0,
        "traffic_count": 4.0,
        "moving_obstacles_count": 2.0,
        "daa_sep_threshold_m": 150.0,
        "daa_ttc_threshold_s": 30.0,
        "faults_count": 1.0,
        "faults_sample_severity": 2.0,
        "faults_sample_duration_s": 15.0,
    })
    return base


def _zero_counts_primary():
    base = _base_primary()
    base.update({
        "mission_waypoints_count": 0.0,
        "landing_preferred_sites_count": 0.0,
        "landing_emergency_sites_count": 0.0,
        "traffic_count": 0.0,
        "moving_obstacles_count": 0.0,
        "comms_loss_windows_count": 0.0,
        "faults_count": 0.0,
    })
    return base


def test_missing_primary_features_raise_value_error():
    prim = _base_primary()
    # remove a required primary
    prim.pop("uav_mass_kg")
    with pytest.raises(ValueError):
        generate_all_features(prim)


def test_generation_three_scenarios_and_stage8_zero():
    v1 = generate_all_features(_base_primary())
    v2 = generate_all_features(_complex_primary())
    v3 = generate_all_features(_zero_counts_primary())

    assert isinstance(v1, np.ndarray) and v1.ndim == 1
    assert isinstance(v2, np.ndarray) and v2.ndim == 1
    assert isinstance(v3, np.ndarray) and v3.ndim == 1
    # spot-check stage8 zero entries are present in generated map
    fmap = generate_all_features_map(_base_primary())
    assert fmap.get("comms_loss_windows_x_range") == 0.0
    assert fmap.get("comms_loss_windows_y_range") == 0.0


def test_secondary_generation_spot_checks_and_override_preservation():
    primary = _base_primary()
    secondary = generate_secondary_features(primary, overrides={"feat_disk_loading": 99999.0})
    # spot-check derived feature exists
    assert "sim_policy_frequency" in secondary
    # override is preserved exactly as provided
    assert secondary["feat_disk_loading"] == 99999.0


def test_spawn_xyz_first_tuple_and_scalar_warning():
    p = _base_primary()
    # tuple/list usage is accepted and preserved in the normalized trace
    p["spawn_xyz_first"] = [1.0, 2.0, 3.0]
    vector = generate_all_features(p)
    assert isinstance(vector, np.ndarray)
    assert vector.shape == (198,)

    # scalar usage should emit DeprecationWarning and be accepted
    p["spawn_xyz_first"] = 7.0
    with pytest.deprecated_call():
        scalar_vector = generate_all_features(p)
    assert isinstance(scalar_vector, np.ndarray)
    assert scalar_vector.shape == (198,)


def test_assemble_accepts_non_primary_keys():
    primary = _base_primary()

    class FakeBundle:
        def __init__(self, feature_names):
            self.feature_names = feature_names
            self.policy_config = {}

    bundle = FakeBundle(load_authoritative_feature_order())
    primary_with_secondary = dict(primary)
    primary_with_secondary["feat_disk_loading"] = 1.23
    vector, metadata = assemble_feature_vector_from_dict(primary_with_secondary, bundle)
    assert vector.shape == (198,)
    assert metadata["feature_map"]["feat_disk_loading"] == 1.23


def test_stage8_features_explicitly_zero_in_map():
    fmap = generate_all_features_map(_base_primary())
    stage8_keys = [
        "comms_loss_windows_x_range",
        "comms_loss_windows_y_range",
    ]
    for k in stage8_keys:
        assert fmap.get(k) == 0.0


def test_real_model_inference_with_fake_bundle():
    primary = _base_primary()

    class FakeModel:
        def predict_proba(self, X):
            # return a deterministic distribution biased to High Risk
            return np.array([[0.05, 0.10, 0.85]])

    class FakeBundle:
        def __init__(self, feature_names):
            self.model = FakeModel()
            self.preprocessor = None
            self.label_encoder = None
            self.shap_explainer = None
            self.feature_names = feature_names
            self.class_names = ["Low Risk", "Medium Risk", "High Risk"]
            self.training_stats = {}
            self.policy_config = {}
            self.model_metadata = {"version": "test"}

    bundle = FakeBundle(load_authoritative_feature_order())
    vec, meta = assemble_feature_vector_from_dict(primary, bundle)
    result = run_stage1_inference(bundle=bundle, feature_vector=vec, feature_names=bundle.feature_names, compute_shap=False)
    assert isinstance(result, object)
    assert result.risk_class == RiskClass.HIGH_RISK
