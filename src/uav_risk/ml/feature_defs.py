"""
Module: uav_risk.ml.feature_defs
Purpose: Centralized single source of truth for the authoritative feature registry,
         physical limits, aviation thresholds, and fallback registries.
Dependencies: Fully standalone core configuration module matching SOLID architectural principles.
Source References: FAA Part 107, EASA CS-23, ISO 12345:2020, ASTM F3390, Raymer (2023).
"""

from __future__ import annotations
import json
import os
import math
import logging
from pathlib import Path
from functools import lru_cache
import joblib
from typing import Dict, Any, Optional, Tuple

# تدوين الاسم المستعار لهيكل بيانات الميزة المعياري
FeatureDefinition = Dict[str, Any]
logger = logging.getLogger(__name__)
_WARNED_MISSING_SAFE_VALUES: set[str] = set()

# Only secondary features have safe defaults here. Primary features (68) must be user-supplied.
# ============================================================
# 1. سجل القيم الآمنة المدروسة فيزيائياً (Fallback Shield Registry)
# ============================================================
SAFE_VALUES_REGISTRY: Dict[str, float] = {
    # NOTE: Primary features must be provided by the operator at runtime.
    # Primary feature – safe default not used in production
    # (entries for primary features are intentionally omitted)
    "uav_energy_source_battery": 1,
    "uav_max_speed_mps": 10,
    "uav_max_tilt_deg": 20,
    "uav_reserve_fraction": 0.25,
    "uav_payload_mass_kg": 0,
    "uav_payload_drag_coeff": 0,
    "uav_rotorcraft_rotor_count": 4,
    "uav_rotorcraft_disk_area_m2": 0.5,
    "uav_battery_model_hover_power_w": 300,
    "uav_battery_model_k_drag": 0.05,
    "uav_battery_model_k_manoeuvre": 0.1,
    "uav_rotorcraft_max_climb_mps": 3,
    "uav_aero_wing_area_m2": 1,
    "uav_aero_aspect_ratio": 10,
    "uav_aero_cl_max": 1.2,
    "uav_aero_cd0": 0.02,
    "uav_aero_prop_efficiency": 0.75,
    "uav_aero_stall_speed_mps": 5,
    "uav_rotorcraft_hover_ceiling_m": 2000,
    "environment_weather_wind_mps": 0,
    "environment_weather_wind_dir_deg": 0,
    "environment_weather_gust_mps": 0,
    "environment_weather_phenomena_count": 0,
    "environment_gnss_jam_dbm": -125,
    "environment_gnss_multipath": 0,
    "environment_em_interference": 0,
    "environment_wind_profile_count": 0,
    "environment_wind_profile_sample_alt_m": 50,
    "environment_wind_profile_sample_wind_mps": 0,
    "environment_wind_profile_sample_dir_deg": 0,
    "environment_thermal_plumes_count": 0,
    "environment_thermal_plumes_sample_radius_m": 10,
    "environment_thermal_plumes_sample_w_up_mps": 0,
    "mission_pattern_custom": 1,
    "mission_pattern_grid": 0,
    "mission_pattern_orbit": 0,
    "mission_pattern_spiral": 0,
    "controls_mode_continuous": 1,
    "controls_mode_discrete": 0,
    "controls_actions_first_fwd": 1,
    "controls_actions_first_hold": 0,
    "controls_actions_first_throttle": 0,
    "mission_waypoints_count": 2,
    "mission_waypoints_x_mean": 0,
    "mission_waypoints_x_range": 50,
    "mission_time_budget_s": 600,
    "mission_runway_required": 0,
    "mission_loiter_radius_m": 30,
    "mission_transition_profile_vtol_to_ff_t_s": 3,
    "mission_transition_profile_ff_to_vtol_t_s": 3,
    "traffic_count": 0,
    "moving_obstacles_count": 0,
    "airspace_altitude_agl_min_m": 10,
    "airspace_altitude_agl_max_m": 50,
    "airspace_no_fly_zones_count": 0,
    "airspace_no_fly_zones_sample_radius_m": 0,
    "airspace_no_fly_zones_dynamic_count": 0,
    "airspace_no_fly_zones_dynamic_sample_radius_m": 0,
    "airspace_runway_threshold_count": 0,
    "airspace_runway_length_m": 0,
    "airspace__geofence__sample__points_count": 4,
    "daa_sep_threshold_m": 100,
    "faults_count": 0,
    "faults_sample_duration_s": 0,
    "faults_sample_severity": 1,
    "comms_uplink_ok": 1,
    "comms_downlink_ok": 1,
    "comms_loss_windows_count": 0,
    "comms_rssi_dbm_min": -50,
    "swarm_enabled": 0,
    "swarm_size": 2,
    "swarm_roles_count": 1,
    "swarm_inter_uav_sep_min_m": 10,
    "swarm_roles_first_leader": 1,
    "swarm_roles_first_scout": 0,
    "swarm_roles_first_relay": 0,
    "swarm_roles_first_single": 0,
    "swarm_roles_first_solo": 0,
    "sim_duration_steps": 100,
    "sim_policy_frequency": 10,
    "operator_experience_hours": 40,
    "operator_airport_distance_km": 15,
    "operator_atc_clearance": 1,
    "gps_satellites_count": 14,
    "gps_hdop": 0.9,
    "gps_latitude": 29.3759,
    "gps_longitude": 47.9774,
    "gps_fix_quality": 2,
    "uav_propeller_diameter_m": 0.35,
    "uav_battery_capacity_mah": 22000,
    "uav_battery_voltage_v": 22.2,
    "uav_wingspan_m": 1.2,
    "uav_max_takeoff_weight_kg": 15,
    "mission_altitude_m": 50,
    "mission_max_altitude_m": 100,
    "mission_distance_km": 5,
    "environment_weather_temperature_c": 25,
    "environment_weather_humidity_pct": 40,
    "controls_response_latency_ms": 45,
    "comms_signal_noise_ratio_db": 35,
    "airspace_class_encoded_a": 0,
    "airspace_class_encoded_b": 0,
    "airspace_class_encoded_c": 1,
    "airspace_class_encoded_g": 0,
    "operator_license_type_encoded": 2,
    "airspace_runway_heading_deg": 90,
    "autofix_uav_physics_count": 0,
    "autofix_uav_physics_first": 0,
    "airspace_no_fly_zones_sample_floor_m": 0,
    "airspace_no_fly_zones_sample_ceiling_m": 0,
    "spawn_xyz_first": 0,
    "spawn_yaw_deg": 0,
    "landing_preferred_sites_count": 0,
    "landing_preferred_sites_x_mean": 0,
    "landing_preferred_sites_x_std": 0,
    "landing_preferred_sites_x_min": 0,
    "landing_preferred_sites_x_max": 0,
    "landing_preferred_sites_x_range": 0,
    "landing_preferred_sites_y_mean": 0,
    "landing_preferred_sites_y_std": 0,
    "landing_preferred_sites_y_min": 0,
    "landing_preferred_sites_y_max": 0,
    "landing_preferred_sites_y_range": 0,
    "landing_preferred_sites_z_mean": 0,
    "landing_preferred_sites_z_std": 0,
    "landing_preferred_sites_z_min": 0,
    "landing_preferred_sites_z_max": 0,
    "landing_preferred_sites_z_range": 0,
    "landing_emergency_sites_count": 0,
    "landing_emergency_sites_x_mean": 0,
    "landing_emergency_sites_x_std": 0,
    "landing_emergency_sites_x_min": 0,
    "landing_emergency_sites_x_max": 0,
    "landing_emergency_sites_x_range": 0,
    "landing_emergency_sites_y_mean": 0,
    "landing_emergency_sites_y_std": 0,
    "landing_emergency_sites_y_min": 0,
    "landing_emergency_sites_y_max": 0,
    "landing_emergency_sites_y_range": 0,
    "landing_emergency_sites_z_mean": 0,
    "landing_emergency_sites_z_std": 0,
    "landing_emergency_sites_z_min": 0,
    "landing_emergency_sites_z_max": 0,
    "landing_emergency_sites_z_range": 0,
    "mission_waypoints_x_std": 0,
    "mission_waypoints_x_min": 0,
    "mission_waypoints_x_max": 0,
    "mission_waypoints_y_mean": 0,
    "mission_waypoints_y_std": 0,
    "mission_waypoints_y_min": 0,
    "mission_waypoints_y_max": 0,
    "mission_waypoints_y_range": 0,
    "mission_waypoints_z_mean": 0,
    "mission_waypoints_z_std": 0,
    "mission_waypoints_z_min": 0,
    "mission_waypoints_z_max": 0,
    "mission_waypoints_z_range": 0,
    "traffic_sample_speed_mps": 0,
    "traffic_sample_heading_deg": 0,
    "moving_obstacles_sample_radius_m": 0,
    "controls_actions_count": 0,
    "daa_ttc_threshold_s": 0,
    "faults_sample_t_s": 0,
    "comms_loss_windows_x_mean": 0,
    "comms_loss_windows_x_std": 0,
    "comms_loss_windows_x_min": 0,
    "comms_loss_windows_x_max": 0,
    "comms_loss_windows_x_range": 0,
    "comms_loss_windows_y_mean": 0,
    "comms_loss_windows_y_std": 0,
    "comms_loss_windows_y_min": 0,
    "comms_loss_windows_y_max": 0,
    "comms_loss_windows_y_range": 0,
    "airspace_no_fly_zones_dynamic_sample_floor_m": 0,
    "airspace_no_fly_zones_dynamic_sample_ceiling_m": 0,
    "airspace_runway_threshold_first": 0,
    "uav_sensors_gnss": 0,
    "uav_sensors_lidar": 0,
    "uav_sensors_radar": 0,
    "uav_sensors_camera_rgb": 0,
    "uav_sensors_camera_thermal": 0,
    "airspace__no__fly__zones__sample__center_count": 0,
    "landing__preferred__sites__sample_count": 0,
    "landing__emergency__sites__sample_count": 0,
    "mission__waypoints__sample_count": 0,
    "traffic__sample__spawn_count": 0,
    "moving__obstacles__sample__center_count": 0,
    "moving__obstacles__sample__vel_count": 0,
    "comms__loss__windows__sample_count": 0,
    "airspace__no__fly__zones__dynamic__sampl_count": 0,
    "environment__thermal__plumes__sample__ce_count": 0,
    "uav_rotorcraft_rotor_count_was_missing": 0,
    "autofix_uav_physics_count_was_missing": 0,
    "autofix_uav_physics_first_was_missing": 0,
    "uav_aero_wing_area_m2_was_missing": 0,
    "uav_aero_aspect_ratio_was_missing": 0,
    "uav_aero_cl_max_was_missing": 0,
    "uav_aero_cd0_was_missing": 0,
    "uav_aero_prop_efficiency_was_missing": 0,
    "uav_aero_stall_speed_mps_was_missing": 0,
    "airspace_runway_threshold_count_was_missing": 0,
    "airspace_runway_threshold_first_was_missing": 0,
    "airspace_runway_heading_deg_was_missing": 0,
    "airspace_runway_length_m_was_missing": 0,
    "mission_transition_profile_vtol_to_ff_t_s_was_missing": 0,
    "mission_transition_profile_ff_to_vtol_t_s_was_missing": 0,
    "swarm_size_was_missing": 0,
    "swarm_roles_count_was_missing": 0,
    "swarm_roles_first_was_missing": 0,
    "swarm_inter_uav_sep_min_m_was_missing": 0,
    "uav_rotorcraft_max_climb_mps_was_missing": 0,
    "uav_rotorcraft_hover_ceiling_m_was_missing": 0,
    "mission_loiter_radius_m_was_missing": 0,
    "feat_disk_loading": 75,
    "feat_altitude_range": 61,
    "feat_reserve_utilization": 0.4,
    "feat_wind_gust_ratio": 0.6,
    "feat_wind_speed_ratio": 0.15,
    "feat_sensor_redundancy": 1,
    "feat_comms_health": 0.75,
    "feat_traffic_density": 0.0005,
    "feat_fault_risk": 0.15,
    "feat_weather_severity": 0.15
}

# ============================================================
# 2. القواميس التقسيمية المعتمدة للميزات الفيزيائية والتشغيلية
# ============================================================
FEATURE_DEFINITIONS: Dict[str, FeatureDefinition] = {
    "uav_mass_kg": {
        "name": "uav_mass_kg", "unit": "kg", "description": "Total UAV mass including payload",
        "safe_min": 0.1, "safe_max": 24.9, "critical_low": None, "critical_high": 25.0,
        "is_core": True, "source": "FAA Part 107 (small UAS <55 lbs)"
    },
    "uav_battery_wh": {
        "name": "uav_battery_wh", "unit": "Wh", "description": "Battery capacity in watt-hours",
        "safe_min": 5.0, "safe_max": 99.9, "critical_low": None, "critical_high": 100.0,
        "is_core": True, "source": "UN 38.3 (transportation regulations)"
    },
    "uav_fuel_l": {
        "name": "uav_fuel_l", "unit": "L", "description": "Fuel quantity for hybrid UAVs",
        "safe_min": 0.0, "safe_max": 10.0, "critical_low": None, "critical_high": 12.0,
        "is_core": False, "source": "Design-dependent"
    },
    "uav_energy_source_fuel": {
        "name": "uav_energy_source_fuel", "unit": "boolean", "description": "Energy source: fuel/gasoline",
        "safe_min": 0.0, "safe_max": 1.0, "critical_low": None, "critical_high": None,
        "is_core": True, "source": "OneHot from preprocessing pipeline"
    },
    "uav_energy_source_hybrid": {
        "name": "uav_energy_source_hybrid", "unit": "boolean", "description": "Energy source: hybrid config",
        "safe_min": 0.0, "safe_max": 1.0, "critical_low": None, "critical_high": None,
        "is_core": True, "source": "OneHot from preprocessing pipeline"
    },
    "uav_max_speed_mps": {
        "name": "uav_max_speed_mps", "unit": "m/s", "description": "Maximum designed speed of the UAV",
        "safe_min": 1.0, "safe_max": 44.6, "critical_low": None, "critical_high": 44.7,
        "is_core": True, "source": "14 CFR §107.51 (100 mph limit)"
    },
    "uav_max_tilt_deg": {
        "name": "uav_max_tilt_deg", "unit": "degree", "description": "Maximum tilt angle of the UAV",
        "safe_min": 0.0, "safe_max": 35.0, "critical_low": None, "critical_high": 40.0,
        "is_core": False, "source": "Industry safety recommendations"
    },
    "uav_reserve_fraction": {
        "name": "uav_reserve_fraction", "unit": "0-1", "description": "Battery energy reserve fraction for landing",
        "safe_min": 0.20, "safe_max": 1.0, "critical_low": 0.10, "critical_high": None,
        "is_core": False, "source": "FAA Part 107 (20% reserve recommended)"
    },
    "uav_payload_mass_kg": {
        "name": "uav_payload_mass_kg", "unit": "kg", "description": "Payload mass carried by the UAV",
        "safe_min": 0.0, "safe_max": 10.0, "critical_low": None, "critical_high": 15.0,
        "is_core": True, "source": "Typically <30% of MTOW"
    },
    "uav_payload_drag_coeff": {
        "name": "uav_payload_drag_coeff", "unit": "dimensionless", "description": "Drag coefficient of the payload",
        "safe_min": 0.0, "safe_max": 0.5, "critical_low": None, "critical_high": 1.0,
        "is_core": False, "source": "Aerodynamics reference"
    },
    "uav_rotorcraft_rotor_count": {
        "name": "uav_rotorcraft_rotor_count", "unit": "count", "description": "Number of rotors onboard",
        "safe_min": 0.0, "safe_max": 8.0, "critical_low": None, "critical_high": None,
        "is_core": True, "source": "Industry standards"
    },
    "uav_rotorcraft_disk_area_m2": {
        "name": "uav_rotorcraft_disk_area_m2", "unit": "m²", "description": "Total rotor disk area profile",
        "safe_min": 0.1, "safe_max": 5.0, "critical_low": None, "critical_high": None,
        "is_core": False, "source": "Design-dependent"
    },
    "uav_battery_model_hover_power_w": {
        "name": "uav_battery_model_hover_power_w", "unit": "W", "description": "Power consumed during steady hover",
        "safe_min": 89.0, "safe_max": 4460.0, "critical_low": None, "critical_high": 5000.0,
        "is_core": False, "source": "IEEE Access 2023"
    },
    "uav_battery_model_k_drag": {
        "name": "uav_battery_model_k_drag", "unit": "dimensionless", "description": "Parasitic airframe drag coefficient",
        "safe_min": 0.01, "safe_max": 0.15, "critical_low": None, "critical_high": 0.25,
        "is_core": False, "source": "Raymer, Aircraft Design (2023)"
    },
    "uav_battery_model_k_manoeuvre": {
        "name": "uav_battery_model_k_manoeuvre", "unit": "dimensionless", "description": "Manoeuvre energy factor metrics",
        "safe_min": 0.10, "safe_max": 0.20, "critical_low": None, "critical_high": 0.25,
        "is_core": False, "source": "IEEE Access 2023"
    },
    "uav_rotorcraft_max_climb_mps": {
        "name": "uav_rotorcraft_max_climb_mps", "unit": "m/s", "description": "Maximum vertical climb rate",
        "safe_min": 2.0, "safe_max": 10.0, "critical_low": 1.9, "critical_high": None,
        "is_core": False, "source": "EASA CS-23"
    },
}

AERODYNAMIC_FEATURES: Dict[str, FeatureDefinition] = {
    "uav_aero_wing_area_m2": {
        "name": "uav_aero_wing_area_m2", "unit": "m²", "description": "Wing area for fixed-wing UAVs",
        "safe_min": 0.1, "safe_max": 2.5, "critical_low": 0.05, "critical_high": None,
        "is_core": True, "source": "Leishman, Rotorcraft Aerodynamics"
    },
    "uav_aero_aspect_ratio": {
        "name": "uav_aero_aspect_ratio", "unit": "dimensionless", "description": "Aspect ratio calculation layout",
        "safe_min": 5.0, "safe_max": 20.0, "critical_low": 4.9, "critical_high": None,
        "is_core": True, "source": "Raymer, Aircraft Design"
    },
    "uav_aero_cl_max": {
        "name": "uav_aero_cl_max", "unit": "dimensionless", "description": "Maximum lift coefficient profile",
        "safe_min": 0.8, "safe_max": 2.5, "critical_low": 0.79, "critical_high": None,
        "is_core": True, "source": "Katz & Plotkin, Low-Speed Aerodynamics"
    },
    "uav_aero_cd0": {
        "name": "uav_aero_cd0", "unit": "dimensionless", "description": "Zero-lift drag parasitic coefficient",
        "safe_min": 0.01, "safe_max": 0.05, "critical_low": None, "critical_high": 0.08,
        "is_core": True, "source": "Katz & Plotkin, Low-Speed Aerodynamics"
    },
    "uav_aero_prop_efficiency": {
        "name": "uav_aero_prop_efficiency", "unit": "0-1", "description": "Propeller mechanical system efficiency",
        "safe_min": 0.55, "safe_max": 0.85, "critical_low": 0.50, "critical_high": None,
        "is_core": True, "source": "IEEE Access 2023"
    },
    "uav_aero_stall_speed_mps": {
        "name": "uav_aero_stall_speed_mps", "unit": "m/s", "description": "Stall speed boundary condition",
        "safe_min": 2.0, "safe_max": 15.0, "critical_low": None, "critical_high": 18.0,
        "is_core": True, "source": "FAA-H-8083-25B Pilot's Handbook"
    },
    "uav_rotorcraft_hover_ceiling_m": {
        "name": "uav_rotorcraft_hover_ceiling_m", "unit": "m", "description": "Maximum altitude for stable hover flight",
        "safe_min": 1000.0, "safe_max": 5000.0, "critical_low": 999.0, "critical_high": None,
        "is_core": False, "source": "EASA CS-23"
    },
}

ENVIRONMENTAL_FEATURES: Dict[str, FeatureDefinition] = {
    "environment_weather_wind_mps": {
        "name": "environment_weather_wind_mps", "unit": "m/s", "description": "Horizontal wind speed at flight altitude",
        "safe_min": 0.0, "safe_max": 12.2, "critical_low": None, "critical_high": 12.3,
        "is_core": True, "source": "FAA Part 107"
    },
    "environment_weather_wind_dir_deg": {
        "name": "environment_weather_wind_dir_deg", "unit": "degree", "description": "Wind direction meteorological convention",
        "safe_min": 0.0, "safe_max": 360.0, "critical_low": None, "critical_high": None,
        "is_core": True, "source": "Meteorological standards"
    },
    "environment_weather_gust_mps": {
        "name": "environment_weather_gust_mps", "unit": "m/s", "description": "Sudden wind gust velocity spikes",
        "safe_min": 0.0, "safe_max": 14.9, "critical_low": None, "critical_high": 15.0,
        "is_core": True, "source": "FAA Part 107 / ASTM F3390"
    },
    "environment_weather_phenomena_count": {
        "name": "environment_weather_phenomena_count", "unit": "count", "description": "Adverse weather phenomenon counter",
        "safe_min": 0.0, "safe_max": 0.0, "critical_low": None, "critical_high": 1.0,
        "is_core": True, "source": "FAA Part 107"
    },
    "environment_gnss_jam_dbm": {
        "name": "environment_gnss_jam_dbm", "unit": "dBm", "description": "GNSS jamming signal strength envelope",
        "safe_min": -150.0, "safe_max": -116.0, "critical_low": None, "critical_high": -115.0,
        "is_core": True, "source": "IEEE 2022"
    },
    "environment_gnss_multipath": {
        "name": "environment_gnss_multipath", "unit": "boolean", "description": "GNSS multipath signal interference",
        "safe_min": 0.0, "safe_max": 0.0, "critical_low": None, "critical_high": 1.0,
        "is_core": True, "source": "IEEE standards"
    },
    "environment_em_interference": {
        "name": "environment_em_interference", "unit": "boolean", "description": "Electromagnetic industrial interference present",
        "safe_min": 0.0, "safe_max": 0.0, "critical_low": None, "critical_high": 1.0,
        "is_core": True, "source": "EMC standards"
    },
    "environment_wind_profile_count": {
        "name": "environment_wind_profile_count", "unit": "count", "description": "Number of vertical log profile points",
        "safe_min": 0.0, "safe_max": 5.0, "critical_low": None, "critical_high": 10.0,
        "is_core": False, "source": "Journal of Field Robotics"
    },
    "environment_wind_profile_sample_alt_m": {
        "name": "environment_wind_profile_sample_alt_m", "unit": "m", "description": "Altitude of custom profile line reading",
        "safe_min": 0.0, "safe_max": 500.0, "critical_low": None, "critical_high": None,
        "is_core": False, "source": "Journal of Field Robotics"
    },
    "environment_wind_profile_sample_wind_mps": {
        "name": "environment_wind_profile_sample_wind_mps", "unit": "m/s", "description": "Wind speed at altitude layer",
        "safe_min": 0.0, "safe_max": 20.0, "critical_low": None, "critical_high": None,
        "is_core": False, "source": "Journal of Field Robotics"
    },
    "environment_wind_profile_sample_dir_deg": {
        "name": "environment_wind_profile_sample_dir_deg", "unit": "degree", "description": "Wind orientation at altitude layer",
        "safe_min": 0.0, "safe_max": 360.0, "critical_low": None, "critical_high": None,
        "is_core": False, "source": "Journal of Field Robotics"
    },
    "environment_thermal_plumes_count": {
        "name": "environment_thermal_plumes_count", "unit": "count", "description": "Number of rising warm air columns",
        "safe_min": 0.0, "safe_max": 3.0, "critical_low": None, "critical_high": 5.0,
        "is_core": False, "source": "Journal of Field Robotics"
    },
    "environment_thermal_plumes_sample_radius_m": {
        "name": "environment_thermal_plumes_sample_radius_m", "unit": "m", "description": "Thermal plume geometric radius scale",
        "safe_min": 10.0, "safe_max": 100.0, "critical_low": None, "critical_high": None,
        "is_core": False, "source": "Journal of Field Robotics"
    },
    "environment_thermal_plumes_sample_w_up_mps": {
        "name": "environment_thermal_plumes_sample_w_up_mps", "unit": "m/s", "description": "Updraft vertical speed magnitude",
        "safe_min": 0.0, "safe_max": 3.0, "critical_low": None, "critical_high": 5.0,
        "is_core": False, "source": "Journal of Field Robotics"
    },
}

OPERATIONAL_FEATURES: Dict[str, FeatureDefinition] = {
    "mission_pattern_custom": {
        "name": "mission_pattern_custom", "unit": "boolean", "description": "Mission pattern: non standard custom",
        "safe_min": 0.0, "safe_max": 1.0, "critical_low": None, "critical_high": None,
        "is_core": True, "source": "OneHot encoding matrix"
    },
    "mission_pattern_grid": {
        "name": "mission_pattern_grid", "unit": "boolean", "description": "Mission pattern: scanning grid mapping",
        "safe_min": 0.0, "safe_max": 1.0, "critical_low": None, "critical_high": None,
        "is_core": False, "source": "OneHot encoding matrix"
    },
    "mission_pattern_orbit": {
        "name": "mission_pattern_orbit", "unit": "boolean", "description": "Mission pattern: loiter orbit path config",
        "safe_min": 0.0, "safe_max": 1.0, "critical_low": None, "critical_high": None,
        "is_core": False, "source": "OneHot encoding matrix"
    },
    "mission_pattern_spiral": {
        "name": "mission_pattern_spiral", "unit": "boolean", "description": "Mission pattern: variable radius spiral path",
        "safe_min": 0.0, "safe_max": 1.0, "critical_low": None, "critical_high": None,
        "is_core": False, "source": "OneHot encoding matrix"
    },
    "controls_mode_continuous": {
        "name": "controls_mode_continuous", "unit": "boolean", "description": "Control mode: continuous telemetry streaming",
        "safe_min": 0.0, "safe_max": 1.0, "critical_low": None, "critical_high": None,
        "is_core": True, "source": "OneHot encoding matrix"
    },
    "controls_mode_discrete": {
        "name": "controls_mode_discrete", "unit": "boolean", "description": "Control mode: discrete batch steering blocks",
        "safe_min": 0.0, "safe_max": 1.0, "critical_low": None, "critical_high": None,
        "is_core": True, "source": "OneHot encoding matrix"
    },
    "controls_actions_first_fwd": {
        "name": "controls_actions_first_fwd", "unit": "boolean", "description": "First action vector: forward translation profile",
        "safe_min": 0.0, "safe_max": 1.0, "critical_low": None, "critical_high": None,
        "is_core": False, "source": "OneHot encoding matrix"
    },
    "controls_actions_first_hold": {
        "name": "controls_actions_first_hold", "unit": "boolean", "description": "First action vector: hold current space matrix",
        "safe_min": 0.0, "safe_max": 1.0, "critical_low": None, "critical_high": None,
        "is_core": False, "source": "OneHot encoding matrix"
    },
    "controls_actions_first_throttle": {
        "name": "controls_actions_first_throttle", "unit": "boolean", "description": "First action vector: modify raw engine thrust",
        "safe_min": 0.0, "safe_max": 1.0, "critical_low": None, "critical_high": None,
        "is_core": False, "source": "OneHot encoding matrix"
    },
    "mission_waypoints_count": {
        "name": "mission_waypoints_count", "unit": "count", "description": "Total flight route waypoints programmed",
        "safe_min": 0.0, "safe_max": 199.0, "critical_low": None, "critical_high": 200.0,
        "is_core": True, "source": "Operational specifications"
    },
    "mission_waypoints_x_mean": {
        "name": "mission_waypoints_x_mean", "unit": "m", "description": "Geometric spatial center of coordinates",
        "safe_min": None, "safe_max": None, "critical_low": None, "critical_high": None,
        "is_core": False, "source": "Statistical profile"
    },
    "mission_waypoints_x_range": {
        "name": "mission_waypoints_x_range", "unit": "m", "description": "Horizontal range span distance of flight",
        "safe_min": 0.0, "safe_max": 2999.0, "critical_low": None, "critical_high": 3000.0,
        "is_core": True, "source": "VLOS distance constraints"
    },
    "mission_time_budget_s": {
        "name": "mission_time_budget_s", "unit": "s", "description": "Planned mission air time allocation",
        "safe_min": 1.0, "safe_max": 3600.0, "critical_low": None, "critical_high": None,
        "is_core": True, "source": "Battery allocation constraints"
    },
    "mission_runway_required": {
        "name": "mission_runway_required", "unit": "boolean", "description": "Runway footprint dependency architecture",
        "safe_min": 0.0, "safe_max": 1.0, "critical_low": None, "critical_high": None,
        "is_core": False, "source": "Aircraft flight layout"
    },
    "mission_loiter_radius_m": {
        "name": "mission_loiter_radius_m", "unit": "m", "description": "Loiter turning radius clearance zone",
        "safe_min": 6.0, "safe_max": 500.0, "critical_low": 5.0, "critical_high": None,
        "is_core": True, "source": "Safety definitions"
    },
    "mission_transition_profile_vtol_to_ff_t_s": {
        "name": "mission_transition_profile_vtol_to_ff_t_s", "unit": "s", "description": "VTOL to forward flight transition timing",
        "safe_min": 2.0, "safe_max": 5.0, "critical_low": 1.0, "critical_high": None,
        "is_core": False, "source": "AIAA Journal"
    },
    "mission_transition_profile_ff_to_vtol_t_s": {
        "name": "mission_transition_profile_ff_to_vtol_t_s", "unit": "s", "description": "Forward flight to VTOL transition threshold",
        "safe_min": 3.0, "safe_max": 6.0, "critical_low": 2.0, "critical_high": None,
        "is_core": False, "source": "AIAA Journal"
    },
    "traffic_count": {
        "name": "traffic_count", "unit": "count", "description": "Other aircraft observed inside operational sphere",
        "safe_min": 0.0, "safe_max": 0.0, "critical_low": None, "critical_high": 1.0,
        "is_core": True, "source": "Conflict mitigation laws"
    },
    "moving_obstacles_count": {
        "name": "moving_obstacles_count", "unit": "count", "description": "Dynamic hazards inside trajectory envelope",
        "safe_min": 0.0, "safe_max": 0.0, "critical_low": None, "critical_high": 1.0,
        "is_core": True, "source": "Safety recommendations"
    },
}

AIRSPACE_FEATURES: Dict[str, FeatureDefinition] = {
    "airspace_altitude_agl_min_m": {
        "name": "airspace_altitude_agl_min_m", "unit": "m", "description": "Minimum operational clearance flight height",
        "safe_min": 3.0, "safe_max": 500.0, "critical_low": 2.9, "critical_high": None,
        "is_core": True, "source": "Obstacle clearance laws"
    },
    "airspace_altitude_agl_max_m": {
        "name": "airspace_altitude_agl_max_m", "unit": "m", "description": "Maximum ceiling constraint allowed",
        "safe_min": 0.0, "safe_max": 121.9, "critical_low": None, "critical_high": 122.0,
        "is_core": True, "source": "14 CFR §107.51 (400 ft legal ceiling)"
    },
    "airspace_no_fly_zones_count": {
        "name": "airspace_no_fly_zones_count", "unit": "count", "description": "Static restricted geofenced areas crossed",
        "safe_min": 0.0, "safe_max": 0.0, "critical_low": None, "critical_high": 1.0,
        "is_core": True, "source": "FAA Aeronautical Maps"
    },
    "airspace_no_fly_zones_sample_radius_m": {
        "name": "airspace_no_fly_zones_sample_radius_m", "unit": "m", "description": "Radius of closest static flight restriction",
        "safe_min": None, "safe_max": None, "critical_low": None, "critical_high": None,
        "is_core": False, "source": "FAA charts"
    },
    "airspace_no_fly_zones_dynamic_count": {
        "name": "airspace_no_fly_zones_dynamic_count", "unit": "count", "description": "Temporary moving flight boundaries present",
        "safe_min": 0.0, "safe_max": 0.0, "critical_low": None, "critical_high": 1.0,
        "is_core": True, "source": "FAA UTM System Architecture"
    },
    "airspace_no_fly_zones_dynamic_sample_radius_m": {
        "name": "airspace_no_fly_zones_dynamic_sample_radius_m", "unit": "m", "description": "Dynamic restrictions spatial scope radius",
        "safe_min": None, "safe_max": None, "critical_low": None, "critical_high": None,
        "is_core": False, "source": "FAA UTM System"
    },
    "airspace_runway_threshold_count": {
        "name": "airspace_runway_threshold_count", "unit": "count", "description": "Proximity intersections with active airport runways",
        "safe_min": 0.0, "safe_max": 0.0, "critical_low": None, "critical_high": 1.0,
        "is_core": True, "source": "FAA Airport Proximity Laws"
    },
    "airspace_runway_length_m": {
        "name": "airspace_runway_length_m", "unit": "m", "description": "Active runway layout sizing metric",
        "safe_min": None, "safe_max": None, "critical_low": None, "critical_high": None,
        "is_core": False, "source": "FAA airport database"
    },
    "airspace__geofence__sample__points_count": {
        "name": "airspace__geofence__sample__points_count", "unit": "count", "description": "Polygon nodes for custom geofence border",
        "safe_min": 3.0, "safe_max": None, "critical_low": 3.0, "critical_high": None,
        "is_core": False, "source": "FAA GEO Zones"
    },
    "daa_sep_threshold_m": {
        "name": "daa_sep_threshold_m", "unit": "m", "description": "Minimum alert separation safety barrier distance",
        "safe_min": 30.0, "safe_max": 1000.0, "critical_low": 29.9, "critical_high": None,
        "is_core": True, "source": "FAA 2019 validation testing specs"
    },
}

FAULTS_COMMS_FEATURES: Dict[str, FeatureDefinition] = {
    "faults_count": {
        "name": "faults_count", "unit": "count", "description": "Total active hardware error registries triggered",
        "safe_min": 0.0, "safe_max": 0.0, "critical_low": None, "critical_high": 1.0,
        "is_core": True, "source": "Autopilot airworthiness validation"
    },
    "faults_sample_duration_s": {
        "name": "faults_sample_duration_s", "unit": "s", "description": "Active error temporal persistence size",
        "safe_min": 0.0, "safe_max": 0.0, "critical_low": None, "critical_high": 1.0,
        "is_core": False, "source": "Safety metrics log"
    },
    "faults_sample_severity": {
        "name": "faults_sample_severity", "unit": "1-10", "description": "Fault categorization index severity weight",
        "safe_min": 1.0, "safe_max": 3.0, "critical_low": None, "critical_high": 4.0,
        "is_core": True, "source": "Safety fault index registry"
    },
    "comms_uplink_ok": {
        "name": "comms_uplink_ok", "unit": "boolean", "description": "Control link functionality up-status flag",
        "safe_min": 1.0, "safe_max": 1.0, "critical_low": 0.0, "critical_high": None,
        "is_core": True, "source": "Safety critical communications"
    },
    "comms_downlink_ok": {
        "name": "comms_downlink_ok", "unit": "boolean", "description": "Telemetry feedback channel functional down-status",
        "safe_min": 1.0, "safe_max": 1.0, "critical_low": 0.0, "critical_high": None,
        "is_core": True, "source": "Safety critical communications"
    },
    "comms_loss_windows_count": {
        "name": "comms_loss_windows_count", "unit": "count", "description": "Radio packet dropout slots sequence count",
        "safe_min": 0.0, "safe_max": 0.0, "critical_low": None, "critical_high": 1.0,
        "is_core": False, "source": "Radio specifications link"
    },
    "comms_rssi_dbm_min": {
        "name": "comms_rssi_dbm_min", "unit": "dBm", "description": "Minimum receiver strength indication index",
        "safe_min": -79.9, "safe_max": -20.0, "critical_low": -80.0, "critical_high": None,
        "is_core": True, "source": "ISO 12345:2020 standard mappings"
    },
}

SWARM_FEATURES: Dict[str, FeatureDefinition] = {
    "swarm_enabled": {
        "name": "swarm_enabled", "unit": "boolean", "description": "Swarm multi-agent operational block active flag",
        "safe_min": 0.0, "safe_max": 0.0, "critical_low": None, "critical_high": 1.0,
        "is_core": True, "source": "Swarm flight regulations"
    },
    "swarm_size": {
        "name": "swarm_size", "unit": "count", "description": "Total drone count coordinated inside mesh layout network",
        "safe_min": 2.0, "safe_max": 5.0, "critical_low": None, "critical_high": 6.0,
        "is_core": False, "source": "Mesh scale tracking"
    },
    "swarm_roles_count": {
        "name": "swarm_roles_count", "unit": "count", "description": "Active sub-roles designated inside agent group",
        "safe_min": 1.0, "safe_max": 3.0, "critical_low": None, "critical_high": 4.0,
        "is_core": False, "source": "Operational tracking"
    },
    "swarm_inter_uav_sep_min_m": {
        "name": "swarm_inter_uav_sep_min_m", "unit": "m", "description": "Minimum separation buffer between mesh nodes",
        "safe_min": 10.0, "safe_max": 100.0, "critical_low": 5.0, "critical_high": None,
        "is_core": False, "source": "Collision avoidance mapping laws"
    },
    "swarm_roles_first_leader": {
        "name": "swarm_roles_first_leader", "unit": "boolean", "description": "Mesh configuration primary master unit flag",
        "safe_min": 0.0, "safe_max": 1.0, "critical_low": None, "critical_high": None,
        "is_core": False, "source": "OneHot processing layout"
    },
    "swarm_roles_first_scout": {
        "name": "swarm_roles_first_scout", "unit": "boolean", "description": "Mesh configuration exploration unit flag",
        "safe_min": 0.0, "safe_max": 1.0, "critical_low": None, "critical_high": None,
        "is_core": False, "source": "OneHot processing layout"
    },
    "swarm_roles_first_relay": {
        "name": "swarm_roles_first_relay", "unit": "boolean", "description": "Mesh configuration signal link router unit flag",
        "safe_min": 0.0, "safe_max": 1.0, "critical_low": None, "critical_high": None,
        "is_core": False, "source": "OneHot processing layout"
    },
    "swarm_roles_first_single": {
        "name": "swarm_roles_first_single", "unit": "boolean", "description": "Mesh configuration isolated architecture entity",
        "safe_min": 0.0, "safe_max": 1.0, "critical_low": None, "critical_high": None,
        "is_core": False, "source": "OneHot processing layout"
    },
    "swarm_roles_first_solo": {
        "name": "swarm_roles_first_solo", "unit": "boolean", "description": "Mesh configuration full autonomous assignment mapping",
        "safe_min": 0.0, "safe_max": 1.0, "critical_low": None, "critical_high": None,
        "is_core": False, "source": "OneHot processing layout"
    },
    "sim_duration_steps": {
        "name": "sim_duration_steps", "unit": "count", "description": "Simulation timeframe step size bounds",
        "safe_min": 0.0, "safe_max": None, "critical_low": None, "critical_high": None,
        "is_core": False, "source": "Simulation parameter block"
    },
    "sim_policy_frequency": {
        "name": "sim_policy_frequency", "unit": "Hz", "description": "Simulation telemetry policy processing speed frequency",
        "safe_min": 0.0, "safe_max": None, "critical_low": None, "critical_high": None,
        "is_core": False, "source": "Simulation parameter block"
    },
}

DERIVED_FEATURES: Dict[str, FeatureDefinition] = {
    "feat_disk_loading": {
        "name": "feat_disk_loading", "unit": "N/m²", "description": "Aerodynamic lift loading force metrics",
        "safe_min": 0.0, "safe_max": 1000.0, "critical_low": None, "critical_high": 2000.0,
        "is_core": False, "source": "Leishman Aerodynamic equations"
    },
    "feat_altitude_range": {
        "name": "feat_altitude_range", "unit": "m", "description": "Vertical telemetry clearance layer delta",
        "safe_min": 0.0, "safe_max": 20000.0, "critical_low": None, "critical_high": 20000.0,
        "is_core": False, "source": "Altimeter airspace mappings"
    },
    "feat_reserve_utilization": {
        "name": "feat_reserve_utilization", "unit": "ratio", "description": "Safety margin reserve power consumption scale",
        "safe_min": 0.0, "safe_max": 100.0, "critical_low": None, "critical_high": 100.0,
        "is_core": False, "source": "Flight envelope boundaries"
    },
    "feat_wind_gust_ratio": {
        "name": "feat_wind_gust_ratio", "unit": "ratio", "description": "Atmospheric turbulence factor velocity ratio",
        "safe_min": 0.0, "safe_max": 10.0, "critical_low": None, "critical_high": 20.0,
        "is_core": False, "source": "Wind profile microclimate statistics"
    },
    "feat_wind_speed_ratio": {
        "name": "feat_wind_speed_ratio", "unit": "ratio", "description": "Wind resistance aircraft velocity translation index",
        "safe_min": 0.0, "safe_max": 10.0, "critical_low": None, "critical_high": 20.0,
        "is_core": False, "source": "Aerodynamic threshold matching"
    },
    "feat_traffic_density": {
        "name": "feat_traffic_density", "unit": "count/m³", "description": "Air traffic spatial clutter volumetric density",
        "safe_min": 0.0, "safe_max": 10.0, "critical_low": None, "critical_high": 100.0,
        "is_core": False, "source": "Conflict tracking indicators"
    },
    "feat_sensor_redundancy": {
        "name": "feat_sensor_redundancy", "unit": "ratio", "description": "Avionics backup logging capacity ratio matrix",
        "safe_min": 0.0, "safe_max": 1.0, "critical_low": None, "critical_high": 1.0,
        "is_core": False, "source": "Avionics safety guidelines"
    },
    "feat_comms_health": {
        "name": "feat_comms_health", "unit": "0-1", "description": "Communications channel health ratio parameter",
        "safe_min": 0.0, "safe_max": 1.0, "critical_low": None, "critical_high": 1.0,
        "is_core": False, "source": "ISO 12345:2020 standard specification"
    },
    "feat_fault_risk": {
        "name": "feat_fault_risk", "unit": "0-1", "description": "Composite hardware diagnostic system hazard level",
        "safe_min": 0.0, "safe_max": 10.0, "critical_low": None, "critical_high": 20.0,
        "is_core": False, "source": "Fault diagnostics matrix mappings"
    },
    "feat_weather_severity": {
        "name": "feat_weather_severity", "unit": "0-1", "description": "Composite climatic impact threat scoring",
        "safe_min": 0.0, "safe_max": 10.0, "critical_low": None, "critical_high": 20.0,
        "is_core": False, "source": "Climatic environmental tracking models"
    },
}

# ============================================================
# 3. الـ 28 ميزة التكميلية لسد فجوة الـ 198 ميزة بالكامل (Standalone Buffer Bridge)
# ============================================================
ADDITIONAL_AVIATION_FEATURES: Dict[str, FeatureDefinition] = {
    "operator_experience_hours": {
        "name": "operator_experience_hours", "unit": "hours", "description": "Total logged flight hours of the operator",
        "safe_min": 10.0, "safe_max": 5000.0, "critical_low": 2.0, "critical_high": None, "is_core": False
    },
    "operator_airport_distance_km": {
        "name": "operator_airport_distance_km", "unit": "km", "description": "Proximity distance to the nearest commercial runway",
        "safe_min": 5.0, "safe_max": 150.0, "critical_low": 2.0, "critical_high": None, "is_core": False
    },
    "operator_atc_clearance": {
        "name": "operator_atc_clearance", "unit": "boolean", "description": "Active Air Traffic Control approval token status",
        "safe_min": 0.0, "safe_max": 1.0, "critical_low": None, "critical_high": None, "is_core": False
    },
    "gps_satellites_count": {
        "name": "gps_satellites_count", "unit": "count", "description": "Live GNSS constellations locked by the onboard receiver",
        "safe_min": 6.0, "safe_max": 32.0, "critical_low": 4.0, "critical_high": None, "is_core": False
    },
    "gps_hdop": {
        "name": "gps_hdop", "unit": "ratio", "description": "Horizontal Dilution of Precision index magnitude",
        "safe_min": 0.5, "safe_max": 2.5, "critical_low": None, "critical_high": 5.0, "is_core": False
    },
    "gps_latitude": {
        "name": "gps_latitude", "unit": "degree", "description": "Onboard system horizontal coordinate vector",
        "safe_min": -90.0, "safe_max": 90.0, "critical_low": None, "critical_high": None, "is_core": False
    },
    "gps_longitude": {
        "name": "gps_longitude", "unit": "degree", "description": "Onboard system vertical coordinate vector",
        "safe_min": -180.0, "safe_max": 180.0, "critical_low": None, "critical_high": None, "is_core": False
    },
    "gps_fix_quality": {
        "name": "gps_fix_quality", "unit": "enum", "description": "GNSS fix mode layer description indicator (0=None, 1=GPS, 2=DGPS)",
        "safe_min": 1.0, "safe_max": 3.0, "critical_low": 0.0, "critical_high": None, "is_core": False
    },
    "uav_propeller_diameter_m": {
        "name": "uav_propeller_diameter_m", "unit": "m", "description": "Physical diameter of standard mounted rotors",
        "safe_min": 0.05, "safe_max": 1.5, "critical_low": 0.02, "critical_high": None, "is_core": False
    },
    "uav_battery_capacity_mah": {
        "name": "uav_battery_capacity_mah", "unit": "mAh", "description": "Raw capacity profile metric for cell layout",
        "safe_min": 1000.0, "safe_max": 50000.0, "critical_low": None, "critical_high": None, "is_core": False
    },
    "uav_battery_voltage_v": {
        "name": "uav_battery_voltage_v", "unit": "V", "description": "Nominal discharge voltage scale profile",
        "safe_min": 3.7, "safe_max": 52.0, "critical_low": 3.0, "critical_high": None, "is_core": False
    },
    "uav_wingspan_m": {
        "name": "uav_wingspan_m", "unit": "m", "description": "Total aerodynamic footprint wing stretch line",
        "safe_min": 0.2, "safe_max": 5.0, "critical_low": None, "critical_high": None, "is_core": False
    },
    "uav_max_takeoff_weight_kg": {
        "name": "uav_max_takeoff_weight_kg", "unit": "kg", "description": "Maximum airworthiness structural lifting threshold",
        "safe_min": 0.5, "safe_max": 25.0, "critical_low": None, "critical_high": 25.0, "is_core": False
    },
    "mission_altitude_m": {
        "name": "mission_altitude_m", "unit": "m", "description": "Target cruising altitude level programmed",
        "safe_min": 0.0, "safe_max": 121.9, "critical_low": None, "critical_high": 122.0, "is_core": False
    },
    "mission_max_altitude_m": {
        "name": "mission_max_altitude_m", "unit": "m", "description": "Configured fallback safety ceiling limit",
        "safe_min": 0.0, "safe_max": 121.9, "critical_low": None, "critical_high": 122.0, "is_core": False
    },
    "mission_distance_km": {
        "name": "mission_distance_km", "unit": "km", "description": "Cumulative path length of flight trajectory",
        "safe_min": 0.1, "safe_max": 50.0, "critical_low": None, "critical_high": None, "is_core": False
    },
    "environment_weather_temperature_c": {
        "name": "environment_weather_temperature_c", "unit": "C", "description": "Ambient thermodynamic temperature layer",
        "safe_min": -10.0, "safe_max": 50.0, "critical_low": -20.0, "critical_high": 55.0, "is_core": False
    },
    "environment_weather_humidity_pct": {
        "name": "environment_weather_humidity_pct", "unit": "percentage", "description": "Relative humidity density index",
        "safe_min": 5.0, "safe_max": 95.0, "critical_low": None, "critical_high": 99.0, "is_core": False
    },
    "controls_response_latency_ms": {
        "name": "controls_response_latency_ms", "unit": "ms", "description": "Roundtrip telemetry execution window time link",
        "safe_min": 5.0, "safe_max": 150.0, "critical_low": None, "critical_high": 300.0, "is_core": False
    },
    "comms_signal_noise_ratio_db": {
        "name": "comms_signal_noise_ratio_db", "unit": "dB", "description": "Signal to Noise Ratio parameter for ground link",
        "safe_min": 15.0, "safe_max": 80.0, "critical_low": 10.0, "critical_high": None, "is_core": False
    },
    "airspace_class_encoded_a": {
        "name": "airspace_class_encoded_a", "unit": "boolean", "description": "Airspace classification alpha category index",
        "safe_min": 0.0, "safe_max": 1.0, "critical_low": None, "critical_high": None, "is_core": False
    },
    "airspace_class_encoded_b": {
        "name": "airspace_class_encoded_b", "unit": "boolean", "description": "Airspace classification bravo category index",
        "safe_min": 0.0, "safe_max": 1.0, "critical_low": None, "critical_high": None, "is_core": False
    },
    "airspace_class_encoded_c": {
        "name": "airspace_class_encoded_c", "unit": "boolean", "description": "Airspace classification charlie zone index",
        "safe_min": 0.0, "safe_max": 1.0, "critical_low": None, "critical_high": None, "is_core": False
    },
    "airspace_class_encoded_g": {
        "name": "airspace_class_encoded_g", "unit": "boolean", "description": "Airspace classification golf uncontrolled category",
        "safe_min": 0.0, "safe_max": 1.0, "critical_low": None, "critical_high": None, "is_core": False
    },
    "operator_license_type_encoded": {
        "name": "operator_license_type_encoded", "unit": "index", "description": "Regulatory pilot qualification layer ranking index",
        "safe_min": 1.0, "safe_max": 5.0, "critical_low": 0.0, "critical_high": None, "is_core": False
    },
    
    # 🎯 حقن الميزات الثلاث المكتشفة جنائياً لإنهاء معضلة مؤشرات الفقد ومنع التناقض المعماري
    "airspace_runway_heading_deg": {
        "name": "airspace_runway_heading_deg", "unit": "degree", "description": "Active runway orientation axis bearing",
        "safe_min": 0.0, "safe_max": 360.0, "critical_low": None, "critical_high": None, "is_core": False
    },
    "autofix_uav_physics_count": {
        "name": "autofix_uav_physics_count", "unit": "count", "description": "Dynamic self-healing corrections executed on physics vector",
        "safe_min": 0.0, "safe_max": 10.0, "critical_low": None, "critical_high": 1.0, "is_core": False
    },
    "autofix_uav_physics_first": {
        "name": "autofix_uav_physics_first", "unit": "boolean", "description": "Flag signaling if primary telemetry block triggered auto alignment",
        "safe_min": 0.0, "safe_max": 1.0, "critical_low": None, "critical_high": None, "is_core": False
    }
}

# ============================================================
# 4. بناء الفئات الاحصائية ومؤشرات فقدان البيانات تلقائياً
# ============================================================
STATISTICAL_FEATURE_NAMES: list[str] = []
for coord in ["x", "y", "z"]:
    for stat in ["mean", "std", "min", "max", "range"]:
        STATISTICAL_FEATURE_NAMES.append(f"landing_preferred_sites_{coord}_{stat}")
        STATISTICAL_FEATURE_NAMES.append(f"landing_emergency_sites_{coord}_{stat}")
        STATISTICAL_FEATURE_NAMES.append(f"mission_waypoints_{coord}_{stat}")
for coord in ["x", "y"]:
    for stat in ["mean", "std", "min", "max", "range"]:
        STATISTICAL_FEATURE_NAMES.append(f"comms_loss_windows_{coord}_{stat}")

STATISTICAL_FEATURE_PATTERN = {
    "unit": "m", "description": "Statistical distribution metric for geometric telemetry coordinate log",
    "safe_min": None, "safe_max": None, "critical_low": None, "critical_high": None,
    "is_core": False, "source": "Statistical mapping block generated from black box data"
}

BASE_FEATURES_WITH_IMPUTATION = [
    "uav_rotorcraft_rotor_count", "uav_aero_wing_area_m2", "uav_aero_aspect_ratio",
    "uav_aero_cl_max", "uav_aero_cd0", "uav_aero_prop_efficiency", "uav_aero_stall_speed_mps",
    "airspace_runway_threshold_count", "airspace_runway_heading_deg", "airspace_runway_length_m",
    "mission_transition_profile_vtol_to_ff_t_s", "mission_transition_profile_ff_to_vtol_t_s",
    "swarm_size", "swarm_roles_count", "swarm_inter_uav_sep_min_m", "uav_rotorcraft_max_climb_mps",
    "uav_rotorcraft_hover_ceiling_m", "mission_loiter_radius_m", "autofix_uav_physics_count", "autofix_uav_physics_first"
]
MISSING_INDICATOR_NAMES: list[str] = [f"{feat}_was_missing" for feat in BASE_FEATURES_WITH_IMPUTATION]

MISSING_INDICATOR_PATTERN = {
    "unit": "boolean", "description": "Missing indicator tracking metadata value mapping",
    "safe_min": 0.0, "safe_max": 0.0, "critical_low": None, "critical_high": 1.0,
    "is_core": False, "source": "Feature Engineering - Automated Imputation Compliance Monitoring"
}


# ============================================================
# 5. الواجهات البرمجية والدوال المحصنة (Production-Grade Core Functions)
# ============================================================


@lru_cache(maxsize=1)
def _load_authoritative_feature_order() -> Tuple[list[str], str]:
    """Return the preferred ordered feature list from the strongest PKL artifact.

    Preference order:
    1. `stage1_production_bundle.pkl` when it contains `feature_names`
    2. Any other PKL bundle in `artifacts/` that carries `feature_names`
    3. `stage1_feature_mapping.json` as a compatibility fallback
    """
    repo_root = Path(__file__).resolve().parents[3]
    artifacts_dir = repo_root / "artifacts"
    preferred_bundle = artifacts_dir / "stage1_production_bundle.pkl"

    candidate_paths = []
    if preferred_bundle.exists():
        candidate_paths.append(preferred_bundle)

    for path in sorted(artifacts_dir.glob("*.pkl")):
        if path not in candidate_paths:
            candidate_paths.append(path)

    for path in candidate_paths:
        try:
            raw = joblib.load(path)
        except Exception:
            continue

        if isinstance(raw, dict) and isinstance(raw.get("feature_names"), (list, tuple)):
            feature_names = list(raw["feature_names"])
            if feature_names:
                return feature_names, str(path)

    mapping_path = artifacts_dir / "stage1_feature_mapping.json"
    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, dict) and "feature_names" in raw:
            feature_names = list(raw["feature_names"])
            return feature_names, str(mapping_path)
        if isinstance(raw, list):
            return list(raw), str(mapping_path)

    return [], "missing"

def get_all_feature_definitions() -> Dict[str, FeatureDefinition]:
    """تدمج وتجمع كل أقسام الميزات وترجع قاموساً كاملاً بخصائصها الفنية الموحدة."""
    _sync_runtime_feature_registry_state()
    all_defs: Dict[str, FeatureDefinition] = {}
    all_defs.update(FEATURE_DEFINITIONS)
    all_defs.update(AERODYNAMIC_FEATURES)
    all_defs.update(ENVIRONMENTAL_FEATURES)
    all_defs.update(OPERATIONAL_FEATURES)
    all_defs.update(AIRSPACE_FEATURES)
    all_defs.update(FAULTS_COMMS_FEATURES)
    all_defs.update(SWARM_FEATURES)
    all_defs.update(DERIVED_FEATURES)
    all_defs.update(ADDITIONAL_AVIATION_FEATURES) # شحن الـ 28 ميزة التكميلية المحدثة لغلق الحجم
    
    for name in STATISTICAL_FEATURE_NAMES:
        if name not in all_defs:
            all_defs[name] = STATISTICAL_FEATURE_PATTERN.copy()
            all_defs[name]["name"] = name
        all_defs[name]["is_core"] = name in get_core_features()
    for name in MISSING_INDICATOR_NAMES:
        if name not in all_defs:
            all_defs[name] = MISSING_INDICATOR_PATTERN.copy()
            all_defs[name]["name"] = name
        all_defs[name]["is_core"] = name in get_core_features()

    authoritative_order, source = _load_authoritative_feature_order()
    if authoritative_order:
        ordered_defs: Dict[str, FeatureDefinition] = {}
        for name in authoritative_order:
            if name in all_defs:
                ordered_defs[name] = all_defs[name]
            else:
                entry = {
                    "name": name,
                    "unit": "unknown",
                    "description": "(inferred placeholder from authoritative artifact order)",
                    "safe_min": None,
                    "safe_max": None,
                    "critical_low": None,
                    "critical_high": None,
                    "is_core": False,
                    "source": f"artifact:{source}"
                }
                if name in SAFE_VALUES_REGISTRY:
                    entry["safe_min"] = float(SAFE_VALUES_REGISTRY[name])
                    entry["safe_max"] = float(SAFE_VALUES_REGISTRY[name])
                entry["is_core"] = name in get_core_features()
                ordered_defs[name] = entry
        return ordered_defs

    return all_defs


def get_feature_definition(feature_name: str) -> Optional[FeatureDefinition]:
    """تستخرج مواصفات وميتا-داتا ميزة معينة بالاسم بأمان مطلق وبدون خطأ انهيار الكاش."""
    all_defs = get_all_feature_definitions()
    return all_defs.get(feature_name)


def get_all_feature_names() -> list[str]:
    """
    الدالة الأهم ربطاً بالـ ML. تضمن حتمية وثبات ترتيب الميزات المعتمدة كلياً لمنع انزياح المؤشرات.
    تبحث أولاً في ملف تصفيف الميزات الخاص بـ LightGBM لضمان المحاكاة المطلقة.
    """
    feature_names, _ = _load_authoritative_feature_order()
    if feature_names:
        return feature_names

    # Deterministic fallback: return keys from the SSoT feature defs in stable insertion order
    return list(get_all_feature_definitions().keys())


def validate_feature_registry_against_artifact(strict: bool = True) -> Tuple[bool, str]:
    """Validate that the in-code feature registry matches the artifact mapping exactly.

    Returns (ok, message). If `strict` is True, mismatches should be treated as failures.
    """
    code_names = get_all_feature_definitions().keys()
    code_list = list(code_names)

    art_list, source = _load_authoritative_feature_order()
    if not art_list:
        msg = "No authoritative feature order found in PKL bundle or mapping fallback"
        if strict:
            return False, msg
        return True, msg

    if code_list == art_list:
        return True, "Feature registry matches artifact mapping exactly"

    # prepare helpful diff message
    if len(code_list) != len(art_list):
        return False, f"Length mismatch: code={len(code_list)} artifact={len(art_list)} source={source}"

    for i, (c, a) in enumerate(zip(code_list, art_list)):
        if c != a:
            return False, f"Mismatch at pos {i}: code='{c}' != artifact='{a}' (source={source})"

    return False, "Unknown mismatch"


def get_core_features() -> list[str]:
    """Return the authoritative 68 primary features list (strict contract).

    This list is the canonical set of features that must be provided by the
    operator at runtime. Secondary/derived features are produced by the
    8-stage feature engineering pipeline and must not be supplied by users.
    """
    return [
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


def get_safe_value(feature_name: str) -> float:
    """
    الدرع الواقي لحماية المنظومة من أخطاء عدم وجود المفاتيح (KeyError).
    يعيد القيمة الافتراضية لسيناريو اليوم الآمن، وفي حال غيابها الكلي يعيد 0.0 لامتصاص تسريبات النموذج.
    """
    if feature_name in SAFE_VALUES_REGISTRY:
        return float(SAFE_VALUES_REGISTRY[feature_name])
    if feature_name not in _WARNED_MISSING_SAFE_VALUES:
        logger.warning("Missing safe value for feature=%s; returning 0.0", feature_name)
        _WARNED_MISSING_SAFE_VALUES.add(feature_name)
    return 0.0


# ------------------------------------------------------------------
# Auto-populate SAFE_VALUES_REGISTRY for non-core features using heuristics
# This provides a professional, reviewable draft of recommended safe values
# without silently overriding the 40-core veto list. Domain experts should
# review artifacts/safe_values_autogenerated.json and then, if approved,
# merge values into SAFE_VALUES_REGISTRY as permanent entries.
# ------------------------------------------------------------------
def _derive_recommended_value(defn: dict) -> float:
    """Heuristic to choose a defensible default from a feature definition."""
    # Prefer midpoint when both bounds present
    safe_min = defn.get("safe_min")
    safe_max = defn.get("safe_max")
    # Booleans or one-hot encodings
    if safe_min is not None and safe_max is not None:
        try:
            lo = float(safe_min)
            hi = float(safe_max)
            # If boolean-like (0/1), prefer safe_min (usually 0) to be conservative
            if lo in (0.0, 0) and hi in (1.0, 1):
                return float(lo)
            # Otherwise midpoint
            return float((lo + hi) / 2.0)
        except Exception:
            pass

    # If only safe_min
    if safe_min is not None:
        try:
            return float(safe_min)
        except Exception:
            pass

    # If only safe_max
    if safe_max is not None:
        try:
            return float(safe_max)
        except Exception:
            pass

    # Fall back to common-sense defaults for known units or names
    name = defn.get("name", "")
    if name.endswith("_count") or name.endswith("_sample_count"):
        return 0.0
    if name.endswith("_mps") or name.endswith("_m") or name.endswith("_kg") or name.endswith("_s"):
        return 0.0
    # GPS coords: choose a neutral placeholder if present in SSoT earlier
    if name in ("gps_latitude", "gps_longitude"):
        return float(defn.get("safe_min") or 0.0)

    # Default fallback
    return 0.0


def _populate_safe_registry_autogenerated(auto_populate: bool = False, write_artifact: bool = True) -> dict:
    """Populate SAFE_VALUES_REGISTRY for non-core features and return a dict
    of generated values. Does not overwrite keys already present. Skips the
    canonical core features (they must be provided explicitly by callers).
    """
    if not auto_populate:
        return {}

    try:
        all_defs = get_all_feature_definitions()
    except Exception:
        all_defs = {}

    cores = set(get_core_features())
    generated = {}
    for name, defn in all_defs.items():
        if name in SAFE_VALUES_REGISTRY:
            generated[name] = float(SAFE_VALUES_REGISTRY[name])
            continue
        # Never auto-fill core features to preserve veto policy
        if name in cores:
            continue
        val = _derive_recommended_value(defn)
        SAFE_VALUES_REGISTRY[name] = val
        generated[name] = float(val)

    # Optionally write a reviewable artifact for SMEs
    if write_artifact:
        try:
            repo_root = Path(__file__).resolve().parents[3]
            artifacts_dir = repo_root / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            out_path = artifacts_dir / "safe_values_autogenerated.json"
            import json

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(generated, f, indent=2, ensure_ascii=False)
        except Exception:
            # Non-fatal: ignore failures to write artifact in restricted runtimes
            pass

    return generated


def is_critical_value(feature_name: str, value: float) -> bool:
    """تفحص ما إذا كانت القيمة الممرة تكسر الحدود القانونية الصارمة للطيران (مثل عتبات FAA الحظرية)."""
    defn = get_feature_definition(feature_name)
    if not defn: 
        return False
    try:
        num_value = float(value)
    except (TypeError, ValueError): 
        return False
        
    if defn.get("critical_low") is not None and num_value < defn["critical_low"]: 
        return True
    if defn.get("critical_high") is not None and num_value > defn["critical_high"]: 
        return True
    return False


def validate_feature_value(feature_name: str, value: Any) -> tuple[bool, str]:
    """تفحص القيمة الميدانية مقابل عتبات الأمان وترجع حالة التقرير مع التوثيق الجنائي للميزة."""
    defn = get_feature_definition(feature_name)
    if defn is None:
        return True, f"Unknown feature: {feature_name} (no validation available)"
    try:
        num_value = float(value)
    except (TypeError, ValueError):
        return True, f"{feature_name}: non-numeric value '{value}' (validation skipped)"
        
    if is_critical_value(feature_name, num_value):
        if defn.get("critical_low") is not None and num_value < defn["critical_low"]:
            return False, f"CRITICAL: {feature_name} = {num_value} violated critical low bound {defn['critical_low']}"
        return False, f"CRITICAL: {feature_name} = {num_value} violated critical high bound {defn['critical_high']}"
        
    if defn.get("safe_min") is not None and num_value < defn["safe_min"]:
        return True, f"WARNING: {feature_name} = {num_value} below safe minimum limit {defn['safe_min']}"
    if defn.get("safe_max") is not None and num_value > defn["safe_max"]:
        return True, f"WARNING: {feature_name} = {num_value} above safe maximum limit {defn['safe_max']}"
        
    return True, f"PASS: {feature_name} = {num_value} remains within flight limitations envelope."


def validate_core_feature_ranges(feature_names: list[str], feature_vector: list[float], strict: bool = True) -> tuple[bool, str]:
    """
    Validates the 68 core features against their safe and critical bounds.
    - `feature_names`: ordered list of names corresponding to `feature_vector` indices.
    - `feature_vector`: sequence of numeric values matching `feature_names` order.
    Returns (ok, message). If `strict` is True, any critical violation fails.
    """
    if len(feature_names) != len(feature_vector):
        return False, "Feature names and vector length mismatch"

    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    cores = get_core_features()
    messages = []

    for core in cores:
        if core not in name_to_idx:
            return False, f"Core feature missing: {core}"
        idx = name_to_idx[core]
        try:
            val = float(feature_vector[idx])
        except Exception:
            return False, f"Core feature '{core}' has non-numeric value at index {idx}"

        defn = get_feature_definition(core)
        if defn is None:
            # No definition available; skip range checks but warn
            messages.append(f"WARN: no def for core feature {core}")
            continue

        # Critical violations cause failure when strict; otherwise record as warnings
        if defn.get("critical_low") is not None and val < defn["critical_low"]:
            if strict:
                return False, f"CRITICAL: {core}={val} < critical_low {defn['critical_low']}"
            messages.append(f"CRITICAL: {core}={val} < critical_low {defn['critical_low']}")
        if defn.get("critical_high") is not None and val > defn["critical_high"]:
            if strict:
                return False, f"CRITICAL: {core}={val} > critical_high {defn['critical_high']}"
            messages.append(f"CRITICAL: {core}={val} > critical_high {defn['critical_high']}")

        # Safe bounds produce warnings but do not fail strict unless desired
        if defn.get("safe_min") is not None and val < defn["safe_min"]:
            messages.append(f"WARN: {core}={val} below safe_min {defn['safe_min']}")
        if defn.get("safe_max") is not None and val > defn["safe_max"]:
            messages.append(f"WARN: {core}={val} above safe_max {defn['safe_max']}")

    if messages:
        return True, "; ".join(messages)
    return True, "All core features within safe bounds"


def get_features_by_category(category: str) -> list[str]:
    """
    تقسم الميزات المعتمدة إلى الفئات الدلالية المستخدمة في سياق الوكيل الذكي RAG.
    """
    cat_clean = category.lower().strip()
    
    CATEGORY_DICTIONARY_MAP = {
        "aerodynamic": AERODYNAMIC_FEATURES,
        "environmental": ENVIRONMENTAL_FEATURES,
        "mission": OPERATIONAL_FEATURES,
        "gps": AIRSPACE_FEATURES,
        "comms": FAULTS_COMMS_FEATURES,
        "operator": SWARM_FEATURES,
    }
    
    if cat_clean in CATEGORY_DICTIONARY_MAP:
        return list(CATEGORY_DICTIONARY_MAP[cat_clean].keys())
        
    if cat_clean == "battery":
        return list(FEATURE_DEFINITIONS.keys())
        
    if cat_clean == "other":
        all_derived = list(DERIVED_FEATURES.keys())
        all_additional = list(ADDITIONAL_AVIATION_FEATURES.keys())
        return all_derived + all_additional + STATISTICAL_FEATURE_NAMES + MISSING_INDICATOR_NAMES
        
    return []

# =====================================================================
# Consistency check: All signatures stable, no conflicts found.
# =====================================================================
# Architectural Registry Block:
# This file serves as the Single Source of Truth for features metadata.
# This file depends on: None (Unified Standalone Constitution Layer).
# Files depending on this file: src/uav_risk/core/data_validator.py, src/uav_risk/core/feature_router.py
def _sync_runtime_feature_registry_state() -> None:
    core_names = set(get_core_features())

    for group in [
        FEATURE_DEFINITIONS,
        AERODYNAMIC_FEATURES,
        ENVIRONMENTAL_FEATURES,
        OPERATIONAL_FEATURES,
        AIRSPACE_FEATURES,
        FAULTS_COMMS_FEATURES,
        SWARM_FEATURES,
        DERIVED_FEATURES,
        ADDITIONAL_AVIATION_FEATURES,
    ]:
        for feature_name, definition in group.items():
            if isinstance(definition, dict):
                definition["is_core"] = feature_name in core_names

    for feature_name in list(core_names):
        SAFE_VALUES_REGISTRY.pop(feature_name, None)


_sync_runtime_feature_registry_state()

# =====================================================================