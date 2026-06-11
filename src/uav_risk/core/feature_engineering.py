from __future__ import annotations

# STAGE6_CLEANUP_REVIEW:
# Classification: MIXED_ACTIVE_LEGACY
# Plan lineage: PLAN3_ACTIVE raw assembly plus PLAN1/PLAN2 legacy processed-feature compatibility.
# Runtime status: generate_raw_feature_map() is the canonical production path.
# Legacy signal: generate_all_features_map(), generate_all_features(), and split_primary_and_secondary_overrides()
# remain compatibility helpers for old processed/mixed feature paths.
# Removed legacy API: _normalize_primary_inputs and generate_secondary_features are not restored.
# Action rule: Do not delete this file. Review legacy helpers function-by-function after raw-first tests remain green.
import json
import logging
import math
import os
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import warnings

from uav_risk.ml import feature_defs
from uav_risk.core.contracts import RawFeatureAssemblyResult
from uav_risk.core.data_validator import run_structural_hard_veto
from uav_risk.ml.raw_schema import (
    DROPPED_RAW_DEFAULTS,
    DROPPED_RAW_METADATA_FEATURES,
    GENERATED_RAW_FEATURES,
    PROFILE_DERIVED_RAW_FEATURES,
    RAW_FEATURE_NAMES,
    SCENARIO_REQUIRED_RAW_FEATURES,
)

logger = logging.getLogger(__name__)

# Legacy processed 68-feature contract. Production raw assembly uses raw_schema.py
# and generate_raw_feature_map() instead.
PRIMARY_FEATURES: list[str] = feature_defs.get_core_features()
PRIMARY_FEATURE_SET = set(PRIMARY_FEATURES)


def _coerce_float(feature_name: str, value: Any) -> float:
    if value is None:
        raise ValueError(f"Missing required feature entry: {feature_name}")
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"", "none", "null", "n/a", "na", "unknown"}:
            raise ValueError(f"Missing required feature entry: {feature_name}")
        if cleaned in {"true", "yes", "on"}:
            return 1.0
        if cleaned in {"false", "no", "off"}:
            return 0.0
        return float(cleaned)
    return float(value)


def generate_all_features_map(
    primary_dict: Mapping[str, Any],
    overrides: Optional[Mapping[str, Any]] = None,
    feature_order: Optional[Sequence[str]] = None,
) -> OrderedDict[str, float]:
    """Legacy compatibility only. Do not use in production raw path.

    This builds the historical processed/mixed 198-feature map used by older
    compatibility tests. Production assembly must use generate_raw_feature_map(),
    which emits the raw 197-column preprocessor contract.
    """
    logger.info("Initiating 8-Stage Deterministic DAG Feature Engineering Pipeline.")
    
    order = list(feature_order) if feature_order is not None else feature_defs.get_all_feature_names()
    overrides_map = dict(overrides) if overrides is not None else {}

    normalized_primary: Dict[str, Any] = {}
    missing_primaries = [name for name in PRIMARY_FEATURES if name not in primary_dict or primary_dict[name] is None]
    if missing_primaries:
        logger.error(f"Pipeline Interrupted: Missing mandatory core features: {missing_primaries}")
        raise ValueError(f"Missing mandatory primary features: {missing_primaries}")

    for name in PRIMARY_FEATURES:
        if name == "spawn_xyz_first":
            spawn_val = primary_dict[name]
            if isinstance(spawn_val, (list, tuple)):
                if len(spawn_val) != 3:
                    raise ValueError("spawn_xyz_first must contain exactly 3 spatial values: [x, y, z]")
                normalized_primary[name] = [float(coord) for coord in spawn_val]
                continue
            normalized_primary[name] = _coerce_float(name, spawn_val)
            continue
        normalized_primary[name] = _coerce_float(name, primary_dict[name])

    v: Dict[str, Any] = dict(normalized_primary)

    def inject(key: str, computed_value: float) -> None:
        if key in overrides_map:
            v[key] = _coerce_float(key, overrides_map[key])
        else:
            v[key] = float(computed_value)

   
    inject("sim_policy_frequency", 10.0)
    inject("autofix_uav_physics_count", 0.0)
    inject("environment_thermal_plumes_count", 0.0)
    inject("environment_wind_profile_count", 1.0)

    zero_flags = [
        "uav_rotorcraft_rotor_count_was_missing", "autofix_uav_physics_count_was_missing",
        "autofix_uav_physics_first_was_missing", "uav_aero_wing_area_m2_was_missing",
        "uav_aero_aspect_ratio_was_missing", "uav_aero_cl_max_was_missing",
        "uav_aero_cd0_was_missing", "uav_aero_prop_efficiency_was_missing",
        "uav_aero_stall_speed_mps_was_missing", "airspace_runway_threshold_count_was_missing",
        "airspace_runway_threshold_first_was_missing", "airspace_runway_heading_deg_was_missing",
        "airspace_runway_length_m_was_missing", "mission_transition_profile_vtol_to_ff_t_s_was_missing",
        "mission_transition_profile_ff_to_vtol_t_s_was_missing", "swarm_size_was_missing",
        "swarm_roles_count_was_missing", "swarm_roles_first_was_missing",
        "swarm_inter_uav_sep_min_m_was_missing", "uav_rotorcraft_max_climb_mps_was_missing",
        "uav_rotorcraft_hover_ceiling_m_was_missing", "mission_loiter_radius_m_was_missing"
    ]
    for flag in zero_flags:
        inject(flag, 0.0)

    inject("airspace__no__fly__zones__sample__center_count", float(v["airspace_no_fly_zones_count"]))
    inject("airspace__no__fly__zones__dynamic__sampl_count", float(v["airspace_no_fly_zones_dynamic_count"]))
    inject("landing__preferred__sites__sample_count", float(v["landing_preferred_sites_count"]))
    inject("landing__emergency__sites__sample_count", float(v["landing_emergency_sites_count"]))
    inject("mission__waypoints__sample_count", float(v["mission_waypoints_count"]))
    inject("traffic__sample__spawn_count", float(v["traffic_count"]))
    inject("moving__obstacles__sample__center_count", float(v["moving_obstacles_count"]))
    inject("moving__obstacles__sample__vel_count", float(v["moving_obstacles_count"]))
    inject("comms__loss__windows__sample_count", float(v["comms_loss_windows_count"]))
    inject("environment__thermal__plumes__sample__ce_count", float(v["environment_thermal_plumes_count"]))

   
    discrete_mode = float(v["controls_mode_discrete"])
    wp_count = float(v["mission_waypoints_count"])
    swarm_on = float(v["swarm_enabled"])
    s_size = float(v["swarm_size"])
    
    inject("controls_actions_first_hold", 1.0 if discrete_mode == 1.0 else 0.0)
    inject("controls_actions_first_throttle", 1.0 if discrete_mode == 0.0 else 0.0)
    inject("controls_actions_count", wp_count + 1.0)
    inject("sim_duration_steps", float(v["mission_time_budget_s"]) * 10.0)
    
    s_roles = 1.0 if (swarm_on == 0.0 or s_size == 1.0) else (2.0 if s_size == 2.0 else 3.0)
    inject("swarm_roles_count", s_roles)
    inject("airspace_runway_threshold_count", float(v["mission_runway_required"]))
    inject("airspace__geofence__sample__points_count", float(v["airspace_no_fly_zones_count"]) * 4.0)

    
    mass = float(v["uav_mass_kg"])
    payload = float(v["uav_payload_mass_kg"])
    fuel = float(v["uav_energy_source_fuel"])
    hybrid = float(v["uav_energy_source_hybrid"])
    rotors = max(float(v["uav_rotorcraft_rotor_count"]), 1.0)
    tilt = float(v["uav_max_tilt_deg"])

    w_area = 0.0 if (fuel == 0.0 and hybrid == 0.0) else (mass / 35.0)
    inject("uav_aero_wing_area_m2", w_area)
    inject("uav_aero_aspect_ratio", 0.0 if w_area == 0.0 else 10.2)
    inject("uav_aero_cl_max", 0.0 if w_area == 0.0 else 1.4)
    inject("uav_aero_cd0", 0.0 if w_area == 0.0 else 0.025)
    
    cl_max_val = float(v["uav_aero_cl_max"])
    st_speed = 0.0 if (w_area == 0.0 or cl_max_val <= 0.0) else math.sqrt((2.0 * mass * 9.81) / (1.225 * w_area * cl_max_val))
    inject("uav_aero_stall_speed_mps", st_speed)
    
    d_area = mass / (40.0 / math.sqrt(rotors)) if mass > 0.0 else 0.0
    inject("uav_rotorcraft_disk_area_m2", d_area)
    
    h_power = 0.0 if d_area <= 0.0 else (pow((mass * 9.81), 1.5) / math.sqrt(2 * 1.225 * d_area * 0.75))
    inject("uav_battery_model_hover_power_w", h_power)
    
    k_drag_val = 0.08 if payload == 0.0 else (0.12 + (payload / max(mass, 1e-6)) * 0.15)
    inject("uav_battery_model_k_drag", float(np.clip(k_drag_val, 0.05, 0.5)))
    inject("uav_battery_model_k_manoeuvre", 1.0 + (tilt / 45.0) * 0.3)
    
    p_drag = 0.0 if payload == 0.0 else (0.15 if payload <= 0.5 else 0.25)
    inject("uav_payload_drag_coeff", p_drag)
    inject("mission_transition_profile_vtol_to_ff_t_s", 0.0 if w_area == 0.0 else 10.0)
    inject("mission_transition_profile_ff_to_vtol_t_s", 0.0 if w_area == 0.0 else 10.0)

  
    wind = float(v["environment_weather_wind_mps"])
    gust = float(v["environment_weather_gust_mps"])
    max_sp = float(v["uav_max_speed_mps"])
    t_budget = float(v["mission_time_budget_s"])
    f_count = float(v["faults_count"])
    f_sev = float(v["faults_sample_severity"])
    f_dur = float(v["faults_sample_duration_s"])
    
    inject("feat_altitude_range", float(v["airspace_altitude_agl_max_m"]) - float(v["airspace_altitude_agl_min_m"]))
    inject("feat_wind_gust_ratio", (gust / wind) if wind != 0.0 else 1.0)
    inject("feat_wind_speed_ratio", wind / max(max_sp, 1e-6))
    
    s_red = (float(v["uav_sensors_gnss"]) + float(v["uav_sensors_lidar"]) + float(v["uav_sensors_radar"]) + 
             float(v["uav_sensors_camera_rgb"]) + float(v["uav_sensors_camera_thermal"])) / 5.0
    inject("feat_sensor_redundancy", s_red)
    inject("feat_reserve_utilization", float(v["uav_reserve_fraction"]) * 100.0)
    inject("feat_traffic_density", float(v["traffic_count"]) / max(t_budget, 1e-6))
    inject("feat_fault_risk", f_count * f_sev * (f_dur / max(t_budget, 1e-6)))
    
    disk_area_val = float(v["uav_rotorcraft_disk_area_m2"])
    inject("feat_disk_loading", (mass / disk_area_val) if disk_area_val > 0.0 else 0.0)
    
    comm_h = ((float(v["comms_uplink_ok"]) + float(v["comms_downlink_ok"])) / 2.0) * (1.0 + (float(v["comms_rssi_dbm_min"]) + 100.0) / 100.0)
    inject("feat_comms_health", float(np.clip(comm_h, 0.0, 1.0)))
    inject("feat_weather_severity", (wind / 10.0 + gust / 15.0 + float(v["environment_weather_phenomena_count"])) / 3.0)

   
    spawn_raw = v["spawn_xyz_first"]
    if isinstance(spawn_raw, (list, tuple)):
        sx, sy, sz = float(spawn_raw[0]), float(spawn_raw[1]), float(spawn_raw[2])
    else:
        sx = sy = sz = float(spawn_raw)

    pref_c = int(float(v["landing_preferred_sites_count"]))
    em_c = int(float(v["landing_emergency_sites_count"]))
    wp_c = int(float(v["mission_waypoints_count"]))
    obs_c = int(float(v["moving_obstacles_count"]))
    comms_c = int(float(v["comms_loss_windows_count"]))
    runway_req = int(float(v["mission_runway_required"]))
    
    inject("landing_preferred_sites_x_mean", sx if pref_c >= 1 else 0.0)
    inject("landing_preferred_sites_x_std", 57.74 if pref_c > 1 else 0.0)
    inject("landing_preferred_sites_y_mean", sy if pref_c >= 1 else 0.0)
    inject("landing_preferred_sites_y_std", 57.74 if pref_c > 1 else 0.0)
    inject("landing_preferred_sites_z_mean", float(v["landing_preferred_sites_z_mean"]))
    inject("landing_preferred_sites_z_std", 2.89 if pref_c > 1 else 0.0)

    inject("landing_emergency_sites_x_mean", sx if em_c >= 1 else 0.0)
    inject("landing_emergency_sites_x_std", 115.47 if em_c > 1 else 0.0)
    inject("landing_emergency_sites_y_mean", sy if em_c >= 1 else 0.0)
    inject("landing_emergency_sites_y_std", 115.47 if em_c > 1 else 0.0)
    inject("landing_emergency_sites_z_mean", float(v["landing_preferred_sites_z_mean"]) if em_c >= 1 else 0.0)
    inject("landing_emergency_sites_z_std", 5.77 if em_c > 1 else 0.0)

    inject("mission_waypoints_x_mean", sx if wp_c >= 1 else 0.0)
    # Use loiter radius-derived std as a reasonable default instead of an extreme sentinel (100.0)
    wp_std_calc = 0.0 if wp_c <= 2 else (float(v["mission_loiter_radius_m"]) / 1.414)
    inject("mission_waypoints_x_std", wp_std_calc)
    inject("mission_waypoints_y_mean", sy if wp_c >= 1 else 0.0)
    inject("mission_waypoints_y_std", wp_std_calc)
    
    min_agl = float(v["airspace_altitude_agl_min_m"])
    max_agl = float(v["airspace_altitude_agl_max_m"])
    h_ceiling = float(v["uav_rotorcraft_hover_ceiling_m"])
    w_z_min = min_agl + 5.0
    w_z_max = min(max_agl - 10.0, h_ceiling - 20.0)
    w_z_max = max(w_z_max, w_z_min)
    
    inject("mission_waypoints_z_mean", float(v["mission_waypoints_z_mean"]))
    inject("mission_waypoints_z_min", w_z_min)
    inject("mission_waypoints_z_max", w_z_max)
    inject("mission_waypoints_z_std", (w_z_max - w_z_min) / 4.0 if wp_c > 0 else 0.0)

    inject("moving_obstacles_sample_center_x_mean", sx if obs_c > 0 else 0.0)
    inject("moving_obstacles_sample_center_x_std", 115.47 if obs_c > 1 else 0.0)
    inject("moving_obstacles_sample_center_y_mean", sy if obs_c > 0 else 0.0)
    inject("moving_obstacles_sample_center_y_std", 115.47 if obs_c > 1 else 0.0)
    inject("moving_obstacles_sample_center_z_mean", float(v["mission_waypoints_z_mean"]) if obs_c > 0 else 0.0)
    inject("moving_obstacles_sample_center_z_std", 10.0 if obs_c > 1 else 0.0)

    inject("comms_loss_windows_x_mean", sx if comms_c > 0 else 0.0)
    inject("comms_loss_windows_x_std", 150.0 if comms_c > 0 else 0.0)
    inject("comms_loss_windows_y_mean", sy if comms_c > 0 else 0.0)
    inject("comms_loss_windows_y_std", 150.0 if comms_c > 0 else 0.0)

    inject("airspace_no_fly_zones_dynamic_sample_radius_m", float(v["airspace_no_fly_zones_sample_radius_m"]))
    inject("airspace_no_fly_zones_dynamic_sample_floor_m", min_agl)
    inject("airspace_no_fly_zones_dynamic_sample_ceiling_m", max_agl)
    inject("faults_sample_t_s", 0.0 if f_count == 0.0 else (t_budget / 2.0))
    inject("environment_wind_profile_sample_alt_m", (min_agl + max_agl) / 2.0)
    inject("environment_wind_profile_sample_wind_mps", wind)
    inject("environment_wind_profile_sample_dir_deg", float(v["environment_weather_wind_dir_deg"]))
    inject("environment_thermal_plumes_sample_radius_m", 0.0)
    inject("environment_thermal_plumes_sample_w_up_mps", 0.0)
    inject("airspace_runway_heading_deg", float(v["spawn_yaw_deg"]) if runway_req == 1 else 0.0)
    inject("traffic_sample_heading_deg", 0.0)
    inject("airspace_runway_threshold_first", sx if runway_req == 1 else 0.0)

    
    def inject_bounds(prefix: str) -> None:
        mean_x, std_x = float(v[f"{prefix}_x_mean"]), float(v[f"{prefix}_x_std"])
        mean_y, std_y = float(v[f"{prefix}_y_mean"]), float(v[f"{prefix}_y_std"])
        inject(f"{prefix}_x_min", mean_x - (2.0 * std_x))
        inject(f"{prefix}_x_max", mean_x + (2.0 * std_x))
        inject(f"{prefix}_y_min", mean_y - (2.0 * std_y))
        inject(f"{prefix}_y_max", mean_y + (2.0 * std_y))
        if prefix != "mission_waypoints":
            mean_z, std_z = float(v[f"{prefix}_z_mean"]), float(v[f"{prefix}_z_std"])
            inject(f"{prefix}_z_min", mean_z - (2.0 * std_z))
            inject(f"{prefix}_z_max", mean_z + (2.0 * std_z))

    inject_bounds("landing_preferred_sites")
    inject_bounds("landing_emergency_sites")
    inject_bounds("mission_waypoints")
    
    mx, sx_obs = float(v["moving_obstacles_sample_center_x_mean"]), float(v["moving_obstacles_sample_center_x_std"])
    my, sy_obs = float(v["moving_obstacles_sample_center_y_mean"]), float(v["moving_obstacles_sample_center_y_std"])
    mz, sz_obs = float(v["moving_obstacles_sample_center_z_mean"]), float(v["moving_obstacles_sample_center_z_std"])
    inject("moving_obstacles_sample_center_x_min", mx - (2.0 * sx_obs))
    inject("moving_obstacles_sample_center_x_max", mx + (2.0 * sx_obs))
    inject("moving_obstacles_sample_center_y_min", my - (2.0 * sy_obs))
    inject("moving_obstacles_sample_center_y_max", my + (2.0 * sy_obs))
    inject("moving_obstacles_sample_center_z_min", mz - (2.0 * sz_obs))
    inject("moving_obstacles_sample_center_z_max", mz + (2.0 * sz_obs))

    cx, sc_x = float(v["comms_loss_windows_x_mean"]), float(v["comms_loss_windows_x_std"])
    cy, sc_y = float(v["comms_loss_windows_y_mean"]), float(v["comms_loss_windows_y_std"])
    inject("comms_loss_windows_x_min", cx - (2.0 * sc_x))
    inject("comms_loss_windows_x_max", cx + (2.0 * sc_x))
    inject("comms_loss_windows_y_min", cy - (2.0 * sc_y))
    inject("comms_loss_windows_y_max", cy + (2.0 * sc_y))

   
    ranges = [
        "landing_preferred_sites_x", "landing_preferred_sites_y", "landing_preferred_sites_z",
        "landing_emergency_sites_x", "landing_emergency_sites_y", "landing_emergency_sites_z",
        "mission_waypoints_x", "mission_waypoints_y", "mission_waypoints_z",
        "moving_obstacles_sample_center_x", "moving_obstacles_sample_center_y", "moving_obstacles_sample_center_z"
    ]
    for r_key in ranges:
        inject(f"{r_key}_range", float(v[f"{r_key}_max"]) - float(v[f"{r_key}_min"]))

    inject("comms_loss_windows_x_range", 0.0)
    inject("comms_loss_windows_y_range", 0.0)
    
    vel_features = [
        "moving_obstacles_sample_vel_x_mean", "moving_obstacles_sample_vel_x_std",
        "moving_obstacles_sample_vel_x_min", "moving_obstacles_sample_vel_x_max", "moving_obstacles_sample_vel_x_range",
        "moving_obstacles_sample_vel_y_mean", "moving_obstacles_sample_vel_y_std",
        "moving_obstacles_sample_vel_y_min", "moving_obstacles_sample_vel_y_max", "moving_obstacles_sample_vel_y_range",
        "moving_obstacles_sample_vel_z_mean"
    ]
    for v_feat in vel_features:
        inject(v_feat, 0.0)

    ordered_map: OrderedDict[str, float] = OrderedDict()
    for name in order:
        if name not in v:
            logger.critical(f"CRITICAL DESIGN FLAW: Feature '{name}' missing from DAG execution stages!")
            raise KeyError(f"Feature '{name}' was not generated by any DAG stage. Pipeline corrupted.")
        
        val_entry = v[name]
        if name == "spawn_xyz_first" and isinstance(val_entry, (list, tuple)):
            ordered_map[name] = float(val_entry[0])
        else:
            ordered_map[name] = float(val_entry)

    if len(ordered_map) != len(order):
        raise RuntimeError(f"Feature vector size mismatch: expected {len(order)}, got {len(ordered_map)}")

    return ordered_map


def _as_mapping(value: Any) -> Dict[str, Any]:
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    return dict(value or {})


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(float(value), low, high))


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return float(value) == 1.0


def _scalar_float(name: str, value: Any) -> float:
    if isinstance(value, (list, tuple, dict)):
        raise ValueError(f"{name} must be a scalar raw value, not a collection.")
    return float(value)


def _set_bounds(v: Dict[str, Any], prefix: str, include_z: bool = True) -> None:
    for axis in ("x", "y"):
        mean = float(v[f"{prefix}_{axis}_mean"])
        std = float(v[f"{prefix}_{axis}_std"])
        v[f"{prefix}_{axis}_min"] = mean - 2.0 * std
        v[f"{prefix}_{axis}_max"] = mean + 2.0 * std
        v[f"{prefix}_{axis}_range"] = v[f"{prefix}_{axis}_max"] - v[f"{prefix}_{axis}_min"]
    if include_z:
        mean = float(v[f"{prefix}_z_mean"])
        std = float(v[f"{prefix}_z_std"])
        v[f"{prefix}_z_min"] = mean - 2.0 * std
        v[f"{prefix}_z_max"] = mean + 2.0 * std
        v[f"{prefix}_z_range"] = v[f"{prefix}_z_max"] - v[f"{prefix}_z_min"]


def _issue_strings(result: Any) -> list[str]:
    return [f"{issue.code}: {issue.message}" for issue in getattr(result, "issues", [])]


def generate_raw_feature_map(
    profile: Mapping[str, Any],
    scenario: Mapping[str, Any],
    overrides: Optional[Mapping[str, Any]] = None,
    raw_feature_order: Optional[Sequence[str]] = None,
) -> RawFeatureAssemblyResult:
    """Production raw assembly: build the 197-column preprocessor input map in artifact order."""
    profile_map = _as_mapping(profile)
    scenario_map = _as_mapping(scenario)
    override_values = _as_mapping(overrides)
    order = list(raw_feature_order) if raw_feature_order is not None else list(RAW_FEATURE_NAMES)

    assessment = {
        "user_id": profile_map.get("user_id"),
        "profile_id": profile_map.get("profile_id"),
        "drone_profile": profile_map,
        "scenario": scenario_map,
        "secondary_overrides": {"values": override_values},
    }
    validation = run_structural_hard_veto(assessment)
    if not validation.passed:
        return RawFeatureAssemblyResult(
            user_id=str(profile_map.get("user_id", "")),
            profile_id=str(profile_map.get("profile_id", "")),
            raw_feature_names=order,
            raw_feature_map={},
            raw_vector_length=0,
            profile_features={name: profile_map.get(name) for name in PROFILE_DERIVED_RAW_FEATURES if name in profile_map},
            scenario_features={name: scenario_map.get(name) for name in SCENARIO_REQUIRED_RAW_FEATURES if name in scenario_map},
            generated_features={},
            secondary_overrides=override_values,
            dropped_metadata_defaults={},
            ignored_extras={},
            hard_vetoes=_issue_strings(validation),
            warnings=[],
        )

    v: Dict[str, Any] = {}
    for name in PROFILE_DERIVED_RAW_FEATURES:
        v[name] = profile_map[name]
    for name in SCENARIO_REQUIRED_RAW_FEATURES:
        v[name] = scenario_map[name]

    dropped_defaults = dict(DROPPED_RAW_DEFAULTS)
    dropped_defaults["uav_type"] = profile_map.get("uav_type", "unknown")
    for name in DROPPED_RAW_METADATA_FEATURES:
        v[name] = dropped_defaults[name]

    mass = float(v["uav_mass_kg"])
    payload = float(v["uav_payload_mass_kg"])
    rotors = int(round(float(v["uav_rotorcraft_rotor_count"])))
    tilt = int(round(float(v["uav_max_tilt_deg"])))
    mission_time = float(v["mission_time_budget_s"])
    wind = float(v["environment_weather_wind_mps"])
    gust = float(v["environment_weather_gust_mps"])
    max_speed = float(v["uav_max_speed_mps"])
    min_alt = float(v["airspace_altitude_agl_min_m"])
    max_alt = float(v["airspace_altitude_agl_max_m"])
    hover_ceiling = float(v["uav_rotorcraft_hover_ceiling_m"])
    spawn_anchor = _scalar_float("spawn_xyz_first", v["spawn_xyz_first"])

    v["sim_policy_frequency"] = 15.0
    if v["sim_policy_frequency"] == 10.0:
        duration = 600.0
    elif v["sim_policy_frequency"] == 20.0 and mission_time >= 850.0:
        duration = 1200.0
    elif v["sim_policy_frequency"] == 20.0:
        duration = 800.0
    else:
        duration = 900.0
    v["sim_duration_steps"] = _clip(duration, 600.0, 1200.0)

    disk_map = {1: 3.5, 2: 3.5, 4: 1.2, 6: 0.5, 8: 2.54}
    v["uav_rotorcraft_disk_area_m2"] = _clip(disk_map.get(rotors, 1.13), 0.12, 5.7)
    if mass <= 3.2:
        hover_power = 111.8
    elif mass <= 5.8:
        hover_power = 335.4
    elif mass <= 12.5:
        hover_power = 658.8
    elif mass <= 25.0:
        hover_power = 1336.4
    else:
        hover_power = 4200.0
    v["uav_battery_model_hover_power_w"] = _clip(hover_power, 20.0, 4460.4)
    payload_mass_ratio = payload / mass if mass > 0.0 else 0.0
    if payload_mass_ratio <= 0.096:
        k_drag = 0.15
    elif payload_mass_ratio <= 0.12:
        k_drag = 0.20
    elif payload_mass_ratio <= 0.141:
        k_drag = 0.15
    elif payload_mass_ratio <= 0.16:
        k_drag = 0.08
    else:
        k_drag = 0.15
    v["uav_battery_model_k_drag"] = _clip(k_drag, 0.008, 0.64)
    v["uav_battery_model_k_manoeuvre"] = _clip({15: 0.25, 30: 0.15, 35: 0.25, 90: 1.2}.get(tilt, 0.25), 0.005, 47.55)
    if payload <= 0.5:
        payload_drag = 0.8
    elif payload <= 0.8:
        payload_drag = 0.1
    elif payload <= 1.2:
        payload_drag = 0.3
    elif payload <= 5.0:
        payload_drag = 0.35
    else:
        payload_drag = 0.8
    v["uav_payload_drag_coeff"] = _clip(payload_drag, 0.01, 1.2)
    v["uav_aero_wing_area_m2"] = _clip(1.2, 0.3, 3.6)
    v["uav_aero_aspect_ratio"] = _clip(10.2, 8.5, 12.0)
    v["uav_aero_cl_max"] = _clip(1.4, 1.4, 1.8)
    v["uav_aero_cd0"] = _clip(0.025, 0.015, 0.05)
    v["uav_aero_stall_speed_mps"] = _clip(12.5, 4.0, 21.5)

    pref_c = float(v["landing_preferred_sites_count"])
    v["landing_preferred_sites_x_mean"] = spawn_anchor if pref_c >= 1.0 else 0.0
    v["landing_preferred_sites_y_mean"] = spawn_anchor if pref_c >= 1.0 else 0.0
    v["landing_preferred_sites_x_std"] = 57.74 if pref_c > 1.0 else 0.0
    v["landing_preferred_sites_y_std"] = 57.74 if pref_c > 1.0 else 0.0
    v["landing_preferred_sites_z_std"] = 2.89 if pref_c > 1.0 else 0.0
    _set_bounds(v, "landing_preferred_sites")

    em_c = float(v["landing_emergency_sites_count"])
    v["landing_emergency_sites_x_mean"] = spawn_anchor if em_c >= 1.0 else 0.0
    v["landing_emergency_sites_y_mean"] = spawn_anchor if em_c >= 1.0 else 0.0
    v["landing_emergency_sites_z_mean"] = float(v["landing_preferred_sites_z_mean"]) if em_c >= 1.0 else 0.0
    v["landing_emergency_sites_x_std"] = 115.47 if em_c > 1.0 else 0.0
    v["landing_emergency_sites_y_std"] = 115.47 if em_c > 1.0 else 0.0
    v["landing_emergency_sites_z_std"] = 5.77 if em_c > 1.0 else 0.0
    _set_bounds(v, "landing_emergency_sites")

    wp_c = float(v["mission_waypoints_count"])
    wp_std = 0.0 if wp_c <= 2.0 else float(v["mission_loiter_radius_m"]) / 1.414
    v["mission_waypoints_x_mean"] = spawn_anchor if wp_c >= 1.0 else 0.0
    v["mission_waypoints_y_mean"] = spawn_anchor if wp_c >= 1.0 else 0.0
    v["mission_waypoints_x_std"] = wp_std
    v["mission_waypoints_y_std"] = wp_std
    _set_bounds(v, "mission_waypoints", include_z=False)
    v["mission_waypoints_z_min"] = min_alt + 5.0
    v["mission_waypoints_z_max"] = max(min(max_alt - 10.0, hover_ceiling - 20.0), v["mission_waypoints_z_min"])
    v["mission_waypoints_z_std"] = (v["mission_waypoints_z_max"] - v["mission_waypoints_z_min"]) / 4.0 if wp_c > 0.0 else 0.0
    v["mission_waypoints_z_range"] = v["mission_waypoints_z_max"] - v["mission_waypoints_z_min"]

    v["traffic_sample_heading_deg"] = 0.0
    v["controls_actions_count"] = float(v["mission_waypoints_count"]) + 1.0
    v["controls_actions_first"] = "fwd"
    v["faults_sample_t_s"] = 0.0 if float(v["faults_count"]) == 0.0 else mission_time / 2.0

    comms_c = float(v["comms_loss_windows_count"])
    for axis in ("x", "y"):
        if comms_c > 0.0:
            v[f"comms_loss_windows_{axis}_mean"] = spawn_anchor
            v[f"comms_loss_windows_{axis}_std"] = 150.0
            v[f"comms_loss_windows_{axis}_min"] = spawn_anchor - 300.0
            v[f"comms_loss_windows_{axis}_max"] = spawn_anchor + 300.0
            v[f"comms_loss_windows_{axis}_range"] = 0.0
        else:
            v[f"comms_loss_windows_{axis}_mean"] = 0.0
            v[f"comms_loss_windows_{axis}_std"] = 0.0
            v[f"comms_loss_windows_{axis}_min"] = 0.0
            v[f"comms_loss_windows_{axis}_max"] = 0.0
            v[f"comms_loss_windows_{axis}_range"] = 0.0

    v["airspace_no_fly_zones_dynamic_sample_radius_m"] = float(v["airspace_no_fly_zones_sample_radius_m"])
    v["airspace_no_fly_zones_dynamic_sample_floor_m"] = min_alt
    v["airspace_no_fly_zones_dynamic_sample_ceiling_m"] = max_alt
    v["autofix_uav_physics_count"] = 0.0
    runway_required = _truthy(v["mission_runway_required"])
    v["airspace_runway_threshold_count"] = 3.0
    v["airspace_runway_threshold_first"] = spawn_anchor if runway_required else 0.0
    v["airspace_runway_heading_deg"] = float(v["spawn_yaw_deg"]) if runway_required else 0.0
    v["environment_wind_profile_count"] = 3.0
    v["environment_wind_profile_sample_alt_m"] = 0.0
    v["environment_wind_profile_sample_wind_mps"] = wind
    v["environment_wind_profile_sample_dir_deg"] = float(v["environment_weather_wind_dir_deg"])
    v["environment_thermal_plumes_count"] = 1.0
    v["environment_thermal_plumes_sample_radius_m"] = 50.0
    v["environment_thermal_plumes_sample_w_up_mps"] = 1.8
    v["mission_transition_profile_vtol_to_ff_t_s"] = 10.0
    v["mission_transition_profile_ff_to_vtol_t_s"] = 10.0
    swarm_size = float(v["swarm_size"])
    v["swarm_roles_count"] = 1.0 if (not _truthy(v["swarm_enabled"]) or swarm_size <= 1.0) else min(swarm_size, 3.0)

    v["airspace__geofence__sample__points_count"] = float(v["airspace_no_fly_zones_count"]) * 4.0
    v["airspace__no__fly__zones__sample__center_count"] = float(v["airspace_no_fly_zones_count"])
    v["landing__preferred__sites__sample_count"] = float(v["landing_preferred_sites_count"])
    v["landing__emergency__sites__sample_count"] = float(v["landing_emergency_sites_count"])
    v["mission__waypoints__sample_count"] = float(v["mission_waypoints_count"])
    v["traffic__sample__spawn_count"] = float(v["traffic_count"])
    v["moving__obstacles__sample__center_count"] = float(v["moving_obstacles_count"])
    v["moving__obstacles__sample__vel_count"] = float(v["moving_obstacles_count"])
    v["comms__loss__windows__sample_count"] = float(v["comms_loss_windows_count"])
    v["airspace__no__fly__zones__dynamic__sampl_count"] = float(v["airspace_no_fly_zones_dynamic_count"])
    v["environment__thermal__plumes__sample__ce_count"] = float(v["environment_thermal_plumes_count"])

    v["feat_disk_loading"] = mass / float(v["uav_rotorcraft_disk_area_m2"]) if float(v["uav_rotorcraft_disk_area_m2"]) > 0.0 else 0.0
    v["feat_altitude_range"] = max_alt - min_alt
    v["feat_reserve_utilization"] = float(v["uav_reserve_fraction"])
    v["feat_wind_gust_ratio"] = gust / wind if wind > 0.0 else 1.0
    v["feat_wind_speed_ratio"] = wind / max_speed if max_speed > 0.0 else 0.0
    v["feat_sensor_redundancy"] = (
        float(v["uav_sensors_gnss"]) + float(v["uav_sensors_lidar"]) + float(v["uav_sensors_radar"])
        + float(v["uav_sensors_camera_rgb"]) + float(v["uav_sensors_camera_thermal"]) + 1.0
    )
    v["feat_comms_health"] = 1.0 if _truthy(v["comms_uplink_ok"]) and _truthy(v["comms_downlink_ok"]) and float(v["comms_rssi_dbm_min"]) > -100.0 else 0.0
    v["feat_traffic_density"] = _clip(0.0 if float(v["traffic_count"]) == 0.0 else 0.009009 * float(v["traffic_count"]), 0.0, 0.222222)
    v["feat_fault_risk"] = _clip(float(v["faults_count"]) * float(v["faults_sample_severity"]), 0.0, 2.0)
    if wind <= 6.2:
        weather = 6.0
    elif wind <= 8.0:
        weather = 8.0
    elif wind <= 8.5:
        weather = 8.45
    else:
        weather = 11.5
    v["feat_weather_severity"] = _clip(weather + 0.1 * float(v["environment_weather_phenomena_count"]), 0.0, 18.0)

    for key, value in override_values.items():
        v[key] = value

    missing_flags = {
        "uav_rotorcraft_rotor_count_was_missing": 0.0,
        "uav_aero_prop_efficiency_was_missing": 0.0,
        "airspace_runway_length_m_was_missing": 0.0,
        "swarm_size_was_missing": 0.0,
        "swarm_roles_first_was_missing": 0.0,
        "swarm_inter_uav_sep_min_m_was_missing": 0.0,
        "uav_rotorcraft_max_climb_mps_was_missing": 0.0,
        "uav_rotorcraft_hover_ceiling_m_was_missing": 0.0,
        "mission_loiter_radius_m_was_missing": 0.0,
        "autofix_uav_physics_count_was_missing": 1.0,
        "autofix_uav_physics_first_was_missing": 1.0,
        "uav_aero_wing_area_m2_was_missing": 1.0,
        "uav_aero_aspect_ratio_was_missing": 1.0,
        "uav_aero_cl_max_was_missing": 1.0,
        "uav_aero_cd0_was_missing": 1.0,
        "uav_aero_stall_speed_mps_was_missing": 1.0,
        "airspace_runway_threshold_count_was_missing": 1.0,
        "airspace_runway_threshold_first_was_missing": 1.0,
        "airspace_runway_heading_deg_was_missing": 1.0,
        "mission_transition_profile_vtol_to_ff_t_s_was_missing": 1.0,
        "mission_transition_profile_ff_to_vtol_t_s_was_missing": 1.0,
        "swarm_roles_count_was_missing": 1.0,
    }
    override_missing_map = {
        "uav_aero_wing_area_m2": "uav_aero_wing_area_m2_was_missing",
        "uav_aero_aspect_ratio": "uav_aero_aspect_ratio_was_missing",
        "uav_aero_cl_max": "uav_aero_cl_max_was_missing",
        "uav_aero_cd0": "uav_aero_cd0_was_missing",
        "uav_aero_stall_speed_mps": "uav_aero_stall_speed_mps_was_missing",
        "airspace_runway_threshold_count": "airspace_runway_threshold_count_was_missing",
        "airspace_runway_threshold_first": "airspace_runway_threshold_first_was_missing",
        "airspace_runway_heading_deg": "airspace_runway_heading_deg_was_missing",
        "mission_transition_profile_vtol_to_ff_t_s": "mission_transition_profile_vtol_to_ff_t_s_was_missing",
        "mission_transition_profile_ff_to_vtol_t_s": "mission_transition_profile_ff_to_vtol_t_s_was_missing",
        "swarm_roles_count": "swarm_roles_count_was_missing",
        "autofix_uav_physics_count": "autofix_uav_physics_count_was_missing",
    }
    for base, flag in override_missing_map.items():
        if base in override_values:
            missing_flags[flag] = 0.0
    v.update(missing_flags)

    ordered = OrderedDict((name, v[name]) for name in order)
    generated = {name: ordered[name] for name in GENERATED_RAW_FEATURES if name in ordered}
    return RawFeatureAssemblyResult(
        user_id=str(profile_map.get("user_id", "")),
        profile_id=str(profile_map.get("profile_id", "")),
        raw_feature_names=order,
        raw_feature_map=dict(ordered),
        raw_vector_length=len(ordered),
        profile_features={name: profile_map[name] for name in PROFILE_DERIVED_RAW_FEATURES},
        scenario_features={name: scenario_map[name] for name in SCENARIO_REQUIRED_RAW_FEATURES},
        generated_features=generated,
        secondary_overrides=override_values,
        dropped_metadata_defaults={name: dropped_defaults[name] for name in DROPPED_RAW_METADATA_FEATURES},
        ignored_extras={},
        hard_vetoes=[],
        warnings=[],
    )


def generate_all_features(
    primary_dict: Mapping[str, Any],
    overrides: Optional[Mapping[str, Any]] = None,
    feature_order: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Legacy compatibility only. Do not use in production raw path.

    Converts the legacy generate_all_features_map() output to a processed/mixed
    numeric vector. Production inference must receive raw 197 and transform via
    the fitted preprocessor.
    """
    feature_map = generate_all_features_map(primary_dict, overrides=overrides, feature_order=feature_order)
    return np.array(list(feature_map.values()), dtype=np.float64)


def split_primary_and_secondary_overrides(
    input_mapping: Mapping[str, Any],
    feature_order: Optional[Sequence[str]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Legacy compatibility helper for the processed 68-feature bridge."""
    order = set(feature_order or feature_defs.get_all_feature_names())
    primary: Dict[str, Any] = {}
    overrides: Dict[str, Any] = {}
    extras: Dict[str, Any] = {}
    for key, value in input_mapping.items():
        if key in PRIMARY_FEATURE_SET:
            primary[key] = value
        elif key in order:
            overrides[key] = value
        else:
            extras[key] = value
    return primary, overrides, extras


# =====================================================================
# Architectural Registry Block:
# This file serves as the Single Point of Truth for 8-Stage DAG propagation.
# This file depends on: src/uav_risk/ml/feature_defs.py
# Files depending on this file: src/uav_risk/ml/feature_generation.py, src/uav_risk/stage2/pipeline.py
# =====================================================================
