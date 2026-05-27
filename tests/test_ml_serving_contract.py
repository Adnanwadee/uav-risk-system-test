from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from uav_risk.ml.inference import predict_processed_vector, run_stage1_inference
from uav_risk.ml.loader import ModelLoadError, assemble_feature_vector_from_dict, load_stage1_bundle
from uav_risk.ml.raw_schema import PROCESSED_ONEHOT_FEATURES, get_raw_schema


class SpyPreprocessor:
    def __init__(self, wrapped):
        self._wrapped = wrapped
        self.calls = 0
        self.feature_names_in_ = wrapped.feature_names_in_
        self.transformers_ = wrapped.transformers_

    def get_feature_names_out(self):
        return self._wrapped.get_feature_names_out()

    def transform(self, frame):
        self.calls += 1
        return self._wrapped.transform(frame)


def _raw_primary() -> dict[str, object]:
    return {
        "uav_energy_source": "battery",
        "mission_pattern": "custom",
        "controls_mode": "discrete",
        "swarm_roles_first": "single",
        "swarm_enabled": 0.0,
        "swarm_size": 1.0,
        "swarm_inter_uav_sep_min_m": 0.0,
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


def test_raw_and_processed_schema_counts_and_order():
    bundle = load_stage1_bundle("artifacts")
    schema = get_raw_schema(bundle)

    assert len(schema.raw_feature_names) == 197
    assert len(schema.processed_feature_names) == 198
    assert len(bundle.preprocessor.get_feature_names_out()) == 198
    assert list(bundle.preprocessor.get_feature_names_out()) == bundle.feature_names
    assert schema.processed_onehot_feature_names == list(PROCESSED_ONEHOT_FEATURES)


def test_assembly_returns_raw_vector_and_rejects_processed_onehot_inputs():
    bundle = load_stage1_bundle("artifacts")
    raw_vector, meta = assemble_feature_vector_from_dict(_raw_primary(), bundle)

    assert raw_vector.shape == (197,)
    assert meta["raw_feature_vector_length"] == 197
    assert meta["processed_feature_vector_length"] == 198
    assert meta["raw_feature_names"] == list(bundle.preprocessor.feature_names_in_)

    bad = _raw_primary()
    bad["uav_energy_source_fuel"] = 1.0
    with pytest.raises(ModelLoadError, match="Processed one-hot"):
        assemble_feature_vector_from_dict(bad, bundle)


def test_normal_inference_uses_preprocessor():
    bundle = load_stage1_bundle("artifacts")
    spy = SpyPreprocessor(bundle.preprocessor)
    bundle.preprocessor = spy
    raw_vector, _ = assemble_feature_vector_from_dict(_raw_primary(), bundle)

    result = run_stage1_inference(bundle, raw_vector, compute_shap=False)

    assert result.probabilities
    assert spy.calls >= 1


def test_processed_prediction_path_is_explicit_and_predicts_saved_split():
    bundle = load_stage1_bundle("artifacts")
    splits = np.load("artifacts/processed_splits_final.npz", allow_pickle=True)
    processed_vector = splits["X_test"][0]

    result = predict_processed_vector(bundle, processed_vector, compute_shap=False)

    assert result.risk_class.value in set(bundle.class_names)
    with pytest.raises(ValueError, match="Processed vector shape mismatch"):
        predict_processed_vector(bundle, processed_vector[:-1], compute_shap=False)


def test_model_predicts_all_classes_on_saved_processed_split():
    bundle = load_stage1_bundle("artifacts")
    splits = np.load("artifacts/processed_splits_final.npz", allow_pickle=True)
    preds = [
        predict_processed_vector(bundle, vector, compute_shap=False).risk_class.value
        for vector in splits["X_test"][:500]
    ]

    assert set(preds) == set(bundle.class_names)
