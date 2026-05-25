from __future__ import annotations

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

logger = logging.getLogger(__name__)

# Data flow contract:
# 1. The user must provide 68 primary features.
# 2. The user may optionally provide overrides for secondary features only.
# 3. _normalize_primary_inputs validates that every primary is present.
# 4. _generate_secondary_values executes stages 0 through 8 to derive the full secondary set.
# 5. User overrides are merged for secondary features and clipped to physical bounds when needed.
# 6. generate_all_features_map combines primary and secondary values into one ordered mapping.
# 7. generate_all_features orders that mapping by bundle feature_order and returns the numeric vector.
# 8. Any missing feature falls back to feature_defs.get_safe_value() with a warning.

PRIMARY_FEATURES: list[str] = [
    "uav_energy_source_fuel",
    "uav_energy_source_hybrid",
    "mission_pattern_custom",
    "mission_pattern_grid",
    "mission_pattern_orbit",
    "mission_pattern_spiral",
    "controls_mode_discrete",
    "swarm_enabled",
    "swarm_size",
    "swarm_inter_uav_sep_min_m",
    "swarm_roles_first_relay",
    "swarm_roles_first_scout",
    "swarm_roles_first_single",
    "swarm_roles_first_solo",
    "uav_mass_kg",
    "uav_battery_wh",
    "uav_fuel_l",
    "uav_payload_mass_kg",
    "uav_max_speed_mps",
    "uav_max_tilt_deg",
    "uav_reserve_fraction",
    "uav_rotorcraft_rotor_count",
    "uav_rotorcraft_max_climb_mps",
    "uav_rotorcraft_hover_ceiling_m",
    "uav_aero_prop_efficiency",
    "uav_sensors_gnss",
    "uav_sensors_lidar",
    "uav_sensors_radar",
    "uav_sensors_camera_rgb",
    "uav_sensors_camera_thermal",
    "environment_weather_wind_mps",
    "environment_weather_wind_dir_deg",
    "environment_weather_gust_mps",
    "environment_weather_phenomena_count",
    "environment_gnss_jam_dbm",
    "environment_gnss_multipath",
    "environment_em_interference",
    "airspace_altitude_agl_min_m",
    "airspace_altitude_agl_max_m",
    "airspace_no_fly_zones_count",
    "airspace_no_fly_zones_sample_radius_m",
    "airspace_no_fly_zones_sample_floor_m",
    "airspace_no_fly_zones_sample_ceiling_m",
    "airspace_no_fly_zones_dynamic_count",
    "mission_runway_required",
    "airspace_runway_length_m",
    "spawn_xyz_first",
    "spawn_yaw_deg",
    "landing_preferred_sites_count",
    "landing_preferred_sites_z_mean",
    "landing_emergency_sites_count",
    "mission_waypoints_count",
    "mission_waypoints_z_mean",
    "mission_time_budget_s",
    "mission_loiter_radius_m",
    "traffic_count",
    "traffic_sample_speed_mps",
    "moving_obstacles_count",
    "moving_obstacles_sample_radius_m",
    "daa_sep_threshold_m",
    "daa_ttc_threshold_s",
    "comms_uplink_ok",
    "comms_downlink_ok",
    "comms_rssi_dbm_min",
    "comms_loss_windows_count",
    "faults_count",
    "faults_sample_severity",
    "faults_sample_duration_s",
]

PRIMARY_FEATURE_SET = set(PRIMARY_FEATURES)
_FEATURE_ORDER_FILE = Path(__file__).resolve().parents[3] / "artifacts" / "stage1_feature_mapping.json"


def _truthy_env(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _debug_enabled() -> bool:
    return _truthy_env(os.getenv("DEBUG_FEATURES"))


@lru_cache(maxsize=1)
def load_authoritative_feature_order() -> list[str]:
    if _FEATURE_ORDER_FILE.exists():
        with open(_FEATURE_ORDER_FILE, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if isinstance(raw, dict) and isinstance(raw.get("feature_names"), list):
            return list(raw["feature_names"])
        if isinstance(raw, list):
            return list(raw)
    try:
        return list(feature_defs.get_all_feature_names())
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise RuntimeError("Unable to load authoritative feature order") from exc


def _load_feature_bounds() -> Dict[str, Dict[str, Any]]:
    try:
        return feature_defs.get_all_feature_definitions()
    except Exception:  # pragma: no cover - defensive fallback
        return {}


_FEATURE_BOUNDS = _load_feature_bounds()


def _coerce_float(feature_name: str, value: Any) -> float:
    if value is None:
        raise ValueError(f"Missing required feature: {feature_name}")
    if isinstance(value, bool):
        result = 1.0 if value else 0.0
    elif isinstance(value, (int, float, np.integer, np.floating)):
        result = float(value)
    elif isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"", "none", "null", "n/a", "na", "unknown"}:
            raise ValueError(f"Missing required feature: {feature_name}")
        if cleaned in {"true", "yes", "on"}:
            result = 1.0
        elif cleaned in {"false", "no", "off"}:
            result = 0.0
        else:
            result = float(cleaned)
    else:
        result = float(value)

    if not math.isfinite(result):
        raise ValueError(f"Non-finite value for feature: {feature_name}")
    return result


def _feature_bounds(feature_name: str) -> tuple[float | None, float | None]:
    definition = _FEATURE_BOUNDS.get(feature_name, {})
    lower = definition.get("safe_min")
    upper = definition.get("safe_max")
    if lower is None and definition.get("critical_low") is not None:
        lower = definition.get("critical_low")
    if upper is None and definition.get("critical_high") is not None:
        upper = definition.get("critical_high")
    if lower is None and upper is None:
        # Conservative fallback: prefer tight, conservative ranges instead of permissive extremes.
        if feature_name.endswith("_was_missing"):
            return 0.0, 1.0
        if feature_name.endswith("_count"):
            return 0.0, None
        if feature_name.endswith("_mps") or "speed" in feature_name:
            return 0.0, 100.0
        if feature_name.endswith("_m") or "radius" in feature_name or "length" in feature_name:
            return 0.0, 10000.0
        if feature_name.endswith("_deg"):
            return 0.0, 360.0
        if feature_name.endswith("_dbm"):
            return -200.0, 100.0
        if feature_name.endswith("_wh") or feature_name.endswith("_kg"):
            return 0.0, 50.0
        if feature_name.startswith("feat_"):
            return 0.0, 1000.0
        # No conservative guess available
        return None, None
    return lower, upper


def _clip_with_warning(feature_name: str, value: float) -> tuple[float, bool]:
    lower, upper = _feature_bounds(feature_name)
    out_of_range = False
    clipped = value
    if lower is not None and clipped < lower:
        out_of_range = True
        clipped = lower
    if upper is not None and clipped > upper:
        out_of_range = True
        clipped = upper
    return clipped, out_of_range


def _build_stage0(_: Mapping[str, float]) -> Dict[str, float]:
    return {
        "sim_policy_frequency": 10.0,
        "autofix_uav_physics_count": 0.0,
        "environment_thermal_plumes_count": 0.0,
        "environment_wind_profile_count": 1.0,
    }


def _build_stage1(values: Mapping[str, float]) -> Dict[str, float]:
    zero_flags = [
        "uav_rotorcraft_rotor_count_was_missing",
        "autofix_uav_physics_count_was_missing",
        "autofix_uav_physics_first_was_missing",
        "uav_aero_wing_area_m2_was_missing",
        "uav_aero_aspect_ratio_was_missing",
        "uav_aero_cl_max_was_missing",
        "uav_aero_cd0_was_missing",
        "uav_aero_prop_efficiency_was_missing",
        "uav_aero_stall_speed_mps_was_missing",
        "airspace_runway_threshold_count_was_missing",
        "airspace_runway_threshold_first_was_missing",
        "airspace_runway_heading_deg_was_missing",
        "airspace_runway_length_m_was_missing",
        "mission_transition_profile_vtol_to_ff_t_s_was_missing",
        "mission_transition_profile_ff_to_vtol_t_s_was_missing",
        "swarm_size_was_missing",
        "swarm_roles_count_was_missing",
        "swarm_roles_first_was_missing",
        "swarm_inter_uav_sep_min_m_was_missing",
        "uav_rotorcraft_max_climb_mps_was_missing",
        "uav_rotorcraft_hover_ceiling_m_was_missing",
        "mission_loiter_radius_m_was_missing",
    ]

    out = {name: 0.0 for name in zero_flags}
    out.update({
        "airspace__no__fly__zones__sample__center_count": float(values.get("airspace_no_fly_zones_count", 0.0)),
        "airspace__no__fly__zones__dynamic__sampl_count": float(values.get("airspace_no_fly_zones_dynamic_count", 0.0)),
        "landing__preferred__sites__sample_count": float(values.get("landing_preferred_sites_count", 0.0)),
        "landing__emergency__sites__sample_count": float(values.get("landing_emergency_sites_count", 0.0)),
        "mission__waypoints__sample_count": float(values.get("mission_waypoints_count", 0.0)),
        "traffic__sample__spawn_count": float(values.get("traffic_count", 0.0)),
        "moving__obstacles__sample__center_count": float(values.get("moving_obstacles_count", 0.0)),
        "moving__obstacles__sample__vel_count": float(values.get("moving_obstacles_count", 0.0)),
        "comms__loss__windows__sample_count": float(values.get("comms_loss_windows_count", 0.0)),
        "environment__thermal__plumes__sample__ce_count": float(values.get("environment_thermal_plumes_count", 0.0)),
    })
    return out


def _build_stage2(values: Mapping[str, float]) -> Dict[str, float]:
    mission_waypoints_count = float(values.get("mission_waypoints_count", 0.0))
    controls_mode_discrete = float(values.get("controls_mode_discrete", 0.0))
    swarm_enabled = float(values.get("swarm_enabled", 0.0))
    swarm_size = float(values.get("swarm_size", 1.0))
    runway_required = float(values.get("mission_runway_required", 0.0))
    nfz_count = float(values.get("airspace_no_fly_zones_count", 0.0))
    return {
        "controls_actions_first_hold": 1.0 if controls_mode_discrete == 1.0 else 0.0,
        "controls_actions_first_throttle": 0.0 if controls_mode_discrete == 1.0 else 1.0,
        "controls_actions_count": mission_waypoints_count + 1.0,
        "sim_duration_steps": float(values.get("mission_time_budget_s", 0.0)) * 10.0,
        "swarm_roles_count": 1.0 if swarm_enabled == 0.0 or swarm_size == 1.0 else 2.0 if swarm_size == 2.0 else 3.0,
        "airspace_runway_threshold_count": runway_required,
        "airspace__geofence__sample__points_count": nfz_count * 4.0,
    }


def _build_stage3(values: Mapping[str, float]) -> Dict[str, float]:
    mass = float(values.get("uav_mass_kg", 0.0))
    payload = float(values.get("uav_payload_mass_kg", 0.0))
    fuel = float(values.get("uav_energy_source_fuel", 0.0))
    hybrid = float(values.get("uav_energy_source_hybrid", 0.0))
    rotor_count = max(float(values.get("uav_rotorcraft_rotor_count", 1.0)), 1.0)
    max_tilt = float(values.get("uav_max_tilt_deg", 0.0))
    wing_area = 0.0 if (fuel == 0.0 and hybrid == 0.0) else mass / 35.0
    aspect_ratio = 0.0 if wing_area == 0.0 else 10.2
    cl_max = 0.0 if wing_area == 0.0 else 1.4
    cd0 = 0.0 if wing_area == 0.0 else 0.025
    stall_speed = 0.0 if wing_area == 0.0 else math.sqrt((2.0 * mass * 9.81) / (1.225 * wing_area * cl_max))
    disk_area = mass / (40.0 / math.sqrt(rotor_count)) if mass > 0.0 else 0.0
    hover_power = 0.0 if disk_area <= 0.0 else ((mass * 9.81) ** 1.5) / math.sqrt(2.0 * 1.225 * disk_area * 0.75)
    k_drag = 0.08 if payload == 0.0 else 0.12 + (payload / max(mass, 1e-6)) * 0.15
    k_drag = float(np.clip(k_drag, 0.05, 0.5))
    k_manoeuvre = 1.0 + (max_tilt / 45.0) * 0.3
    payload_drag = 0.0 if payload == 0.0 else 0.15 if payload <= 0.5 else 0.25
    transition = 0.0 if wing_area == 0.0 else 10.0
    return {
        "uav_aero_wing_area_m2": wing_area,
        "uav_aero_aspect_ratio": aspect_ratio,
        "uav_aero_cl_max": cl_max,
        "uav_aero_cd0": cd0,
        "uav_aero_stall_speed_mps": stall_speed,
        "uav_rotorcraft_disk_area_m2": disk_area,
        "uav_battery_model_hover_power_w": hover_power,
        "uav_battery_model_k_drag": k_drag,
        "uav_battery_model_k_manoeuvre": k_manoeuvre,
        "uav_payload_drag_coeff": payload_drag,
        "mission_transition_profile_vtol_to_ff_t_s": transition,
        "mission_transition_profile_ff_to_vtol_t_s": transition,
    }


def _build_stage4(values: Mapping[str, float]) -> Dict[str, float]:
    wind = float(values.get("environment_weather_wind_mps", 0.0))
    gust = float(values.get("environment_weather_gust_mps", 0.0))
    max_speed = float(values.get("uav_max_speed_mps", 1.0))
    sensors = [
        float(values.get("uav_sensors_gnss", 0.0)),
        float(values.get("uav_sensors_lidar", 0.0)),
        float(values.get("uav_sensors_radar", 0.0)),
        float(values.get("uav_sensors_camera_rgb", 0.0)),
        float(values.get("uav_sensors_camera_thermal", 0.0)),
    ]
    up = float(values.get("comms_uplink_ok", 0.0))
    down = float(values.get("comms_downlink_ok", 0.0))
    rssi = float(values.get("comms_rssi_dbm_min", -100.0))
    return {
        "feat_altitude_range": float(values.get("airspace_altitude_agl_max_m", 0.0)) - float(values.get("airspace_altitude_agl_min_m", 0.0)),
        "feat_wind_gust_ratio": gust / wind if wind != 0.0 else 1.0,
        "feat_wind_speed_ratio": wind / max(max_speed, 1e-6),
        "feat_sensor_redundancy": sum(sensors) / 5.0,
        "feat_reserve_utilization": float(values.get("uav_reserve_fraction", 0.0)) * 100.0,
        "feat_traffic_density": float(values.get("traffic_count", 0.0)) / max(float(values.get("mission_time_budget_s", 1.0)), 1e-6),
        "feat_fault_risk": float(values.get("faults_count", 0.0)) * float(values.get("faults_sample_severity", 0.0)) * (float(values.get("faults_sample_duration_s", 0.0)) / max(float(values.get("mission_time_budget_s", 1.0)), 1e-6)),
        "feat_disk_loading": float(values.get("uav_mass_kg", 0.0)) / max(float(values.get("uav_rotorcraft_disk_area_m2", 1.0)), 1e-6),
        "feat_comms_health": float(np.clip(((up + down) / 2.0) * (1.0 + (rssi + 100.0) / 100.0), 0.0, 1.0)),
        "feat_weather_severity": (wind / 10.0 + gust / 15.0 + float(values.get("environment_weather_phenomena_count", 0.0))) / 3.0,
    }


def _build_stage5(values: Mapping[str, float]) -> Dict[str, float]:
    spawn_raw = values.get("spawn_xyz_first", 0.0)

    if isinstance(spawn_raw, (list, tuple)):
        if len(spawn_raw) != 3:
            raise ValueError("spawn_xyz_first must contain exactly 3 values: [x, y, z]")
        spawn_x = float(spawn_raw[0])
        spawn_y = float(spawn_raw[1])
        spawn_z = float(spawn_raw[2])
    else:
        warnings.warn("spawn_xyz_first numeric scalar usage is deprecated; use [x, y, z] instead", DeprecationWarning)
        logger.warning("spawn_xyz_first received as numeric scalar; use [x, y, z] for the new contract")
        spawn_x = float(spawn_raw)
        spawn_y = float(spawn_raw)
        spawn_z = float(spawn_raw)

    pref_count = int(float(values.get("landing_preferred_sites_count", 0.0)))
    em_count = int(float(values.get("landing_emergency_sites_count", 0.0)))
    wp_count = int(float(values.get("mission_waypoints_count", 0.0)))
    runway_required = int(float(values.get("mission_runway_required", 0.0)))
    wind = float(values.get("environment_weather_wind_mps", 0.0))
    wind_dir = float(values.get("environment_weather_wind_dir_deg", 0.0))
    time_budget = float(values.get("mission_time_budget_s", 0.0))
    max_agl = float(values.get("airspace_altitude_agl_max_m", 0.0))
    min_agl = float(values.get("airspace_altitude_agl_min_m", 0.0))
    hover_ceiling = float(values.get("uav_rotorcraft_hover_ceiling_m", 0.0))
    loiter_radius = float(values.get("mission_loiter_radius_m", 0.0))

    wp_x_std = 0.0 if wp_count <= 0 else 0.0 if wp_count <= 2 else (loiter_radius / math.sqrt(2.0) if float(values.get("mission_pattern_orbit", 0.0)) == 1.0 else 100.0)
    wp_y_std = wp_x_std
    wp_z_min = min_agl + 5.0
    wp_z_max = min(max_agl - 10.0, hover_ceiling - 20.0)
    if wp_z_max < wp_z_min:
        wp_z_max = wp_z_min
    wp_z_std = 0.0 if wp_count <= 0 else max(0.0, (wp_z_max - wp_z_min) / 4.0)

    preferred_x_std = 57.74 if pref_count > 1 else 0.0
    preferred_y_std = 57.74 if pref_count > 1 else 0.0
    preferred_z_std = 2.89 if pref_count > 1 else 0.0
    emergency_x_std = 115.47 if em_count > 1 else 0.0
    emergency_y_std = 115.47 if em_count > 1 else 0.0
    emergency_z_std = 5.77 if em_count > 1 else 0.0
    comms_std = 150.0 if float(values.get("comms_loss_windows_count", 0.0)) > 0.0 else 0.0

    return {
        "landing_preferred_sites_x_mean": spawn_x if pref_count >= 1 else 0.0,
        "landing_preferred_sites_x_std": preferred_x_std,
        "landing_preferred_sites_y_mean": spawn_y if pref_count >= 1 else 0.0,
        "landing_preferred_sites_y_std": preferred_y_std,
        "landing_preferred_sites_z_mean": spawn_z if pref_count >= 1 else 0.0,
        "landing_preferred_sites_z_std": preferred_z_std,
        "landing_emergency_sites_x_mean": spawn_x if em_count >= 1 else 0.0,
        "landing_emergency_sites_x_std": emergency_x_std,
        "landing_emergency_sites_y_mean": spawn_y if em_count >= 1 else 0.0,
        "landing_emergency_sites_y_std": emergency_y_std,
        "landing_emergency_sites_z_mean": spawn_z if em_count >= 1 else 0.0,
        "landing_emergency_sites_z_std": emergency_z_std,
        "mission_waypoints_x_mean": spawn_x if wp_count >= 1 else 0.0,
        "mission_waypoints_x_std": wp_x_std,
        "mission_waypoints_y_mean": spawn_y if wp_count >= 1 else 0.0,
        "mission_waypoints_y_std": wp_y_std,
        "mission_waypoints_z_mean": spawn_z if wp_count >= 1 else 0.0,
        "mission_waypoints_z_min": wp_z_min,
        "mission_waypoints_z_max": wp_z_max,
        "mission_waypoints_z_std": wp_z_std,
        "moving_obstacles_sample_center_count": float(values.get("moving_obstacles_count", 0.0)),
        "moving_obstacles_sample_vel_count": float(values.get("moving_obstacles_count", 0.0)),
        "comms_loss_windows_x_mean": spawn_x if float(values.get("comms_loss_windows_count", 0.0)) > 0.0 else 0.0,
        "comms_loss_windows_x_std": comms_std,
        "comms_loss_windows_y_mean": spawn_y if float(values.get("comms_loss_windows_count", 0.0)) > 0.0 else 0.0,
        "comms_loss_windows_y_std": comms_std,
        "airspace_no_fly_zones_dynamic_sample_radius_m": float(values.get("airspace_no_fly_zones_sample_radius_m", 0.0)),
        "airspace_no_fly_zones_dynamic_sample_floor_m": min_agl,
        "airspace_no_fly_zones_dynamic_sample_ceiling_m": max_agl,
        "faults_sample_t_s": 0.0 if float(values.get("faults_count", 0.0)) == 0.0 else time_budget / 2.0,
        "environment_wind_profile_sample_alt_m": (min_agl + max_agl) / 2.0,
        "environment_wind_profile_sample_wind_mps": wind,
        "environment_wind_profile_sample_dir_deg": wind_dir,
        "environment_thermal_plumes_sample_radius_m": 0.0,
        "environment_thermal_plumes_sample_w_up_mps": 0.0,
        "airspace_runway_heading_deg": float(values.get("spawn_yaw_deg", 0.0)) if runway_required == 1 else 0.0,
        "traffic_sample_heading_deg": 0.0,
        "airspace_runway_threshold_first": spawn_x if runway_required == 1 else 0.0,
    }


def _build_stage6(values: Mapping[str, float]) -> Dict[str, float]:
    return {
        "landing_preferred_sites_x_min": float(values.get("landing_preferred_sites_x_mean", 0.0)) - 2.0 * float(values.get("landing_preferred_sites_x_std", 0.0)),
        "landing_preferred_sites_x_max": float(values.get("landing_preferred_sites_x_mean", 0.0)) + 2.0 * float(values.get("landing_preferred_sites_x_std", 0.0)),
        "landing_preferred_sites_y_min": float(values.get("landing_preferred_sites_y_mean", 0.0)) - 2.0 * float(values.get("landing_preferred_sites_y_std", 0.0)),
        "landing_preferred_sites_y_max": float(values.get("landing_preferred_sites_y_mean", 0.0)) + 2.0 * float(values.get("landing_preferred_sites_y_std", 0.0)),
        "landing_preferred_sites_z_min": float(values.get("landing_preferred_sites_z_mean", 0.0)) - 2.0 * float(values.get("landing_preferred_sites_z_std", 0.0)),
        "landing_preferred_sites_z_max": float(values.get("landing_preferred_sites_z_mean", 0.0)) + 2.0 * float(values.get("landing_preferred_sites_z_std", 0.0)),
        "landing_emergency_sites_x_min": float(values.get("landing_emergency_sites_x_mean", 0.0)) - 2.0 * float(values.get("landing_emergency_sites_x_std", 0.0)),
        "landing_emergency_sites_x_max": float(values.get("landing_emergency_sites_x_mean", 0.0)) + 2.0 * float(values.get("landing_emergency_sites_x_std", 0.0)),
        "landing_emergency_sites_y_min": float(values.get("landing_emergency_sites_y_mean", 0.0)) - 2.0 * float(values.get("landing_emergency_sites_y_std", 0.0)),
        "landing_emergency_sites_y_max": float(values.get("landing_emergency_sites_y_mean", 0.0)) + 2.0 * float(values.get("landing_emergency_sites_y_std", 0.0)),
        "landing_emergency_sites_z_min": float(values.get("landing_emergency_sites_z_mean", 0.0)) - 2.0 * float(values.get("landing_emergency_sites_z_std", 0.0)),
        "landing_emergency_sites_z_max": float(values.get("landing_emergency_sites_z_mean", 0.0)) + 2.0 * float(values.get("landing_emergency_sites_z_std", 0.0)),
        "mission_waypoints_x_min": float(values.get("mission_waypoints_x_mean", 0.0)) - 2.0 * float(values.get("mission_waypoints_x_std", 0.0)),
        "mission_waypoints_x_max": float(values.get("mission_waypoints_x_mean", 0.0)) + 2.0 * float(values.get("mission_waypoints_x_std", 0.0)),
        "mission_waypoints_y_min": float(values.get("mission_waypoints_y_mean", 0.0)) - 2.0 * float(values.get("mission_waypoints_y_std", 0.0)),
        "mission_waypoints_y_max": float(values.get("mission_waypoints_y_mean", 0.0)) + 2.0 * float(values.get("mission_waypoints_y_std", 0.0)),
        "moving_obstacles_sample_center_x_min": float(values.get("moving_obstacles_sample_center_x_mean", 0.0)) - 2.0 * float(values.get("moving_obstacles_sample_center_x_std", 0.0)),
        "moving_obstacles_sample_center_x_max": float(values.get("moving_obstacles_sample_center_x_mean", 0.0)) + 2.0 * float(values.get("moving_obstacles_sample_center_x_std", 0.0)),
        "moving_obstacles_sample_center_y_min": float(values.get("moving_obstacles_sample_center_y_mean", 0.0)) - 2.0 * float(values.get("moving_obstacles_sample_center_y_std", 0.0)),
        "moving_obstacles_sample_center_y_max": float(values.get("moving_obstacles_sample_center_y_mean", 0.0)) + 2.0 * float(values.get("moving_obstacles_sample_center_y_std", 0.0)),
        "moving_obstacles_sample_center_z_min": float(values.get("moving_obstacles_sample_center_z_mean", 0.0)) - 2.0 * float(values.get("moving_obstacles_sample_center_z_std", 0.0)),
        "moving_obstacles_sample_center_z_max": float(values.get("moving_obstacles_sample_center_z_mean", 0.0)) + 2.0 * float(values.get("moving_obstacles_sample_center_z_std", 0.0)),
        "comms_loss_windows_x_min": float(values.get("comms_loss_windows_x_mean", 0.0)) - 2.0 * float(values.get("comms_loss_windows_x_std", 0.0)),
        "comms_loss_windows_x_max": float(values.get("comms_loss_windows_x_mean", 0.0)) + 2.0 * float(values.get("comms_loss_windows_x_std", 0.0)),
        "comms_loss_windows_y_min": float(values.get("comms_loss_windows_y_mean", 0.0)) - 2.0 * float(values.get("comms_loss_windows_y_std", 0.0)),
        "comms_loss_windows_y_max": float(values.get("comms_loss_windows_y_mean", 0.0)) + 2.0 * float(values.get("comms_loss_windows_y_std", 0.0)),
    }


def _build_stage7(values: Mapping[str, float]) -> Dict[str, float]:
    return {
        "landing_preferred_sites_x_range": float(values.get("landing_preferred_sites_x_max", 0.0)) - float(values.get("landing_preferred_sites_x_min", 0.0)),
        "landing_preferred_sites_y_range": float(values.get("landing_preferred_sites_y_max", 0.0)) - float(values.get("landing_preferred_sites_y_min", 0.0)),
        "landing_preferred_sites_z_range": float(values.get("landing_preferred_sites_z_max", 0.0)) - float(values.get("landing_preferred_sites_z_min", 0.0)),
        "landing_emergency_sites_x_range": float(values.get("landing_emergency_sites_x_max", 0.0)) - float(values.get("landing_emergency_sites_x_min", 0.0)),
        "landing_emergency_sites_y_range": float(values.get("landing_emergency_sites_y_max", 0.0)) - float(values.get("landing_emergency_sites_y_min", 0.0)),
        "landing_emergency_sites_z_range": float(values.get("landing_emergency_sites_z_max", 0.0)) - float(values.get("landing_emergency_sites_z_min", 0.0)),
        "mission_waypoints_x_range": float(values.get("mission_waypoints_x_max", 0.0)) - float(values.get("mission_waypoints_x_min", 0.0)),
        "mission_waypoints_y_range": float(values.get("mission_waypoints_y_max", 0.0)) - float(values.get("mission_waypoints_y_min", 0.0)),
        "mission_waypoints_z_range": float(values.get("mission_waypoints_z_max", 0.0)) - float(values.get("mission_waypoints_z_min", 0.0)),
        "moving_obstacles_sample_center_x_range": float(values.get("moving_obstacles_sample_center_x_max", 0.0)) - float(values.get("moving_obstacles_sample_center_x_min", 0.0)),
        "moving_obstacles_sample_center_y_range": float(values.get("moving_obstacles_sample_center_y_max", 0.0)) - float(values.get("moving_obstacles_sample_center_y_min", 0.0)),
        "moving_obstacles_sample_center_z_range": float(values.get("moving_obstacles_sample_center_z_max", 0.0)) - float(values.get("moving_obstacles_sample_center_z_min", 0.0)),
        "comms_loss_windows_x_range": 0.0,
        "comms_loss_windows_y_range": 0.0,
    }


def _build_stage8(values: Mapping[str, float], stage5_outputs: Mapping[str, float]) -> Dict[str, float]:
    del values
    del stage5_outputs
    return {
        "comms_loss_windows_x_range": 0.0,
        "comms_loss_windows_y_range": 0.0,
        "moving_obstacles_sample_vel_x_mean": 0.0,
        "moving_obstacles_sample_vel_x_std": 0.0,
        "moving_obstacles_sample_vel_x_min": 0.0,
        "moving_obstacles_sample_vel_x_max": 0.0,
        "moving_obstacles_sample_vel_x_range": 0.0,
        "moving_obstacles_sample_vel_y_mean": 0.0,
        "moving_obstacles_sample_vel_y_std": 0.0,
        "moving_obstacles_sample_vel_y_min": 0.0,
        "moving_obstacles_sample_vel_y_max": 0.0,
        "moving_obstacles_sample_vel_y_range": 0.0,
        "moving_obstacles_sample_vel_z_mean": 0.0,
    }


def _normalize_primary_inputs(primary_dict: Mapping[str, Any]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    missing = [name for name in PRIMARY_FEATURES if name not in primary_dict or primary_dict[name] is None]
    if missing:
        raise ValueError(f"Missing primary features: {missing}")

    for name in PRIMARY_FEATURES:
        if name == "spawn_xyz_first":
            spawn_value = primary_dict[name]
            if isinstance(spawn_value, (list, tuple)):
                if len(spawn_value) != 3:
                    raise ValueError("spawn_xyz_first must contain exactly 3 values: [x, y, z]")
                normalized[name] = [float(coord) for coord in spawn_value]
                continue
            warnings.warn("spawn_xyz_first numeric scalar usage is deprecated; use [x, y, z] instead", DeprecationWarning)
            logger.warning("spawn_xyz_first received as numeric scalar; use [x, y, z] for the new contract")
            normalized[name] = _coerce_float(name, spawn_value)
            continue

        normalized[name] = _coerce_float(name, primary_dict[name])
    return normalized


def _validate_and_merge_overrides(
    overrides: Optional[Mapping[str, Any]],
    feature_order: Sequence[str],
) -> tuple[Dict[str, float], list[str]]:
    if not overrides:
        return {}, []

    order_set = set(feature_order)
    secondary_set = order_set.difference(PRIMARY_FEATURE_SET)
    applied: Dict[str, float] = {}
    ignored: list[str] = []

    for name, raw_value in overrides.items():
        if name not in secondary_set:
            ignored.append(name)
            continue

        applied[name] = _coerce_float(name, raw_value)

    return applied, ignored


def _generate_secondary_values(
    primary_inputs: Mapping[str, float],
    overrides: Optional[Mapping[str, Any]] = None,
    feature_order: Optional[Sequence[str]] = None,
) -> tuple[OrderedDict[str, float], list[str]]:
    order = list(feature_order) if feature_order is not None else load_authoritative_feature_order()
    override_values, ignored_overrides = _validate_and_merge_overrides(overrides, order)

    values: Dict[str, float] = dict(primary_inputs)
    if ignored_overrides:
        logger.warning("ignored_non_secondary_overrides names=%s", ignored_overrides)

    values.update(_build_stage0(values))
    values.update(_build_stage1(values))
    values.update(_build_stage2(values))
    values.update(_build_stage3(values))
    values.update(_build_stage4(values))
    stage5_outputs = _build_stage5(values)
    values.update(stage5_outputs)
    values.update(_build_stage6(values))
    values.update(_build_stage7(values))
    values.update(_build_stage8(values, stage5_outputs))

    secondary_order = [name for name in order if name not in PRIMARY_FEATURE_SET]
    ordered: OrderedDict[str, float] = OrderedDict()
    out_of_range = 0
    for feature_name in secondary_order:
        if feature_name in override_values:
            value = override_values[feature_name]
        elif feature_name in values:
            value = values[feature_name]
        else:
            raise RuntimeError(f"secondary feature missing from stages: {feature_name}")

        ordered[feature_name] = float(value)

        if _debug_enabled():
            logger.debug("generated_secondary_feature feature=%s value=%s", feature_name, float(value))

    secondary_array = np.array(list(ordered.values()), dtype=np.float64)
    if len(secondary_array) != len(secondary_order):
        raise RuntimeError(
            f"Secondary feature count mismatch: expected {len(secondary_order)}, got {len(secondary_array)}"
        )
    if not np.isfinite(secondary_array).all():
        raise RuntimeError("Secondary feature vector contains NaN or Inf values")

    if _debug_enabled():
        zeros = int(np.sum(secondary_array == 0.0))
        logger.debug(
            "feature_vector_summary count=%s minimum=%s maximum=%s mean=%s zeros=%s out_of_range=%s",
            int(secondary_array.size),
            float(np.min(secondary_array)),
            float(np.max(secondary_array)),
            float(np.mean(secondary_array)),
            zeros,
            out_of_range,
        )

    return ordered, ignored_overrides


def generate_secondary_features(
    primary_dict: Mapping[str, Any],
    overrides: Optional[Mapping[str, Any]] = None,
    feature_order: Optional[Sequence[str]] = None,
) -> OrderedDict[str, float]:
    primary_inputs = _normalize_primary_inputs(primary_dict)
    secondary_map, _ = _generate_secondary_values(primary_inputs, overrides=overrides, feature_order=feature_order)
    return secondary_map


def generate_all_features_map(
    primary_dict: Mapping[str, Any],
    overrides: Optional[Mapping[str, Any]] = None,
    feature_order: Optional[Sequence[str]] = None,
) -> OrderedDict[str, float]:
    order = list(feature_order) if feature_order is not None else load_authoritative_feature_order()
    primary_inputs = _normalize_primary_inputs(primary_dict)
    secondary_map, _ = _generate_secondary_values(primary_inputs, overrides=overrides, feature_order=order)
    spawn_is_triplet = isinstance(primary_inputs.get("spawn_xyz_first"), (list, tuple))
    triplet_stage5_outputs: Mapping[str, float] = {}
    if spawn_is_triplet:
        triplet_stage_values: Dict[str, float] = dict(primary_inputs)
        triplet_stage_values.update(_build_stage0(triplet_stage_values))
        triplet_stage_values.update(_build_stage1(triplet_stage_values))
        triplet_stage_values.update(_build_stage2(triplet_stage_values))
        triplet_stage_values.update(_build_stage3(triplet_stage_values))
        triplet_stage_values.update(_build_stage4(triplet_stage_values))
        triplet_stage5_outputs = _build_stage5(triplet_stage_values)
    triplet_preferred_keys = {
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
    }

    ordered: OrderedDict[str, float] = OrderedDict()
    for name in order:
        if name in primary_inputs:
            value = primary_inputs[name]
            if name == "spawn_xyz_first" and isinstance(value, (list, tuple)):
                value = float(value[0])
            elif spawn_is_triplet and name in triplet_preferred_keys and name in triplet_stage5_outputs:
                value = triplet_stage5_outputs[name]
        elif name in secondary_map:
            value = secondary_map[name]
        else:
            raise RuntimeError(f"Feature missing from primary/secondary assembly: {name}")
        ordered[name] = float(value)

    if len(ordered) != len(order):
        raise RuntimeError(f"Feature vector length mismatch: expected {len(order)}, got {len(ordered)}")

    array = np.array(list(ordered.values()), dtype=np.float64)
    if not np.isfinite(array).all():
        raise RuntimeError("Feature vector contains NaN or Inf values")

    return ordered


def generate_all_features(
    primary_dict: Mapping[str, Any],
    overrides: Optional[Mapping[str, Any]] = None,
    feature_order: Optional[Sequence[str]] = None,
) -> np.ndarray:
    feature_map = generate_all_features_map(primary_dict, overrides=overrides, feature_order=feature_order)
    return np.array(list(feature_map.values()), dtype=np.float64)


def split_primary_and_secondary_overrides(
    input_mapping: Mapping[str, Any],
    feature_order: Optional[Sequence[str]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    order = set(feature_order or load_authoritative_feature_order())
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


__all__ = [
    "PRIMARY_FEATURES",
    "generate_secondary_features",
    "generate_all_features",
    "generate_all_features_map",
    "load_authoritative_feature_order",
    "split_primary_and_secondary_overrides",
]
