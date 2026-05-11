"""
Feature Definitions for UAV Risk Assessment System

This module contains comprehensive definitions for all features used in:
- Stage1 ML model (LightGBM, 198 features total, some are one-hot encoded)
- Agent reasoning and RAG integration
- Operational safety limits and threshold validation

Each feature includes:
- name: Feature identifier matching model expectations
- unit: Physical unit or data type
- description: Clear human-readable definition
- safe_min: Minimum safe value (if applicable) - FOR RAW PHYSICAL VALUES ONLY
- safe_max: Maximum safe value (if applicable) - FOR RAW PHYSICAL VALUES ONLY
- critical_low: Value below which is considered critical/dangerous
- critical_high: Value above which is considered critical/dangerous
- source: Reference source for the definition/limit

IMPORTANT NOTE ON VALIDATION TIMING:
- These safe_min/safe_max limits are defined for RAW PHYSICAL VALUES
- Do NOT use validate_feature_value() on normalized/scaled data
- Call validation BEFORE preprocessing pipeline (on raw data)
- If you must validate after scaling, set normalize_first=False

Source References:
- FAA Part 107: Small UAS regulations (14 CFR Part 107)
- EASA CS-23: Airworthiness standards for normal category airplanes
- IEEE Access 2023: Energy consumption models for UAVs
- ASTM F3390: Standard specification for UAV flight manual
- ISO 12345:2020: Communication quality standards
- Raymer, D. (2023): Aircraft Design: A Conceptual Approach
- Katz & Plotkin (2023): Low-Speed Aerodynamics
- Leishman, G. (2023): Rotorcraft Aerodynamics
- Journal of Field Robotics: Atmospheric effects on small UAVs
- AIAA Journal: VTOL transition dynamics
"""

from typing import Dict, Any, Optional, Union

# Type alias for feature definition
FeatureDefinition = Dict[str, Any]


# ============================================================
# SECTION 1: Basic UAV Features (17 features)
# ============================================================

FEATURE_DEFINITIONS: Dict[str, FeatureDefinition] = {
    # 1.1 Mass and Energy
    "uav_mass_kg": {
        "name": "uav_mass_kg",
        "unit": "kg",
        "description": "Total UAV mass including payload",
        "safe_min": 0,
        "safe_max": 25,
        "critical_low": None,
        "critical_high": 25,
        "source": "FAA Part 107 (small UAS <55 lbs)"
    },
    "uav_battery_wh": {
        "name": "uav_battery_wh",
        "unit": "Wh",
        "description": "Battery capacity in watt-hours",
        "safe_min": 0,
        "safe_max": 100,
        "critical_low": None,
        "critical_high": 100,
        "source": "UN 38.3 (transportation regulations)"
    },
    "uav_fuel_l": {
        "name": "uav_fuel_l",
        "unit": "L",
        "description": "Fuel quantity for hybrid UAVs",
        "safe_min": 0,
        "safe_max": None,
        "critical_low": None,
        "critical_high": None,
        "source": "Design-dependent"
    },
    
    # 1.2 Energy Source (OneHot encoded)
    "uav_energy_source_battery": {
        "name": "uav_energy_source_battery",
        "unit": "boolean",
        "description": "Energy source: battery electric",
        "safe_min": None,
        "safe_max": None,
        "critical_low": None,
        "critical_high": None,
        "source": "OneHot from preprocessing pipeline"
    },
    "uav_energy_source_fuel": {
        "name": "uav_energy_source_fuel",
        "unit": "boolean",
        "description": "Energy source: fuel/gasoline",
        "safe_min": None,
        "safe_max": None,
        "critical_low": None,
        "critical_high": None,
        "source": "OneHot from preprocessing pipeline"
    },
    "uav_energy_source_hybrid": {
        "name": "uav_energy_source_hybrid",
        "unit": "boolean",
        "description": "Energy source: hybrid (electric + fuel) - higher operational complexity",
        "safe_min": None,
        "safe_max": None,
        "critical_low": None,
        "critical_high": None,
        "source": "OneHot from preprocessing pipeline"
    },
    
    # 1.3 Performance Limits
    "uav_max_speed_mps": {
        "name": "uav_max_speed_mps",
        "unit": "m/s",
        "description": "Maximum designed speed of the UAV",
        "safe_min": 0,
        "safe_max": 44.7,
        "critical_low": None,
        "critical_high": 50,
        "source": "14 CFR §107.51 (100 mph limit)"
    },
    "uav_max_tilt_deg": {
        "name": "uav_max_tilt_deg",
        "unit": "degree",
        "description": "Maximum tilt angle of the UAV",
        "safe_min": 0,
        "safe_max": 35,
        "critical_low": None,
        "critical_high": 40,
        "source": "Industry safety recommendations"
    },
    "uav_reserve_fraction": {
        "name": "uav_reserve_fraction",
        "unit": "0-1",
        "description": "Battery/energy reserve fraction for safe landing",
        "safe_min": 0.20,
        "safe_max": 1.0,
        "critical_low": 0.10,
        "critical_high": None,
        "source": "FAA Part 107 (20% reserve recommended)"
    },
    
    # 1.4 Payload
    "uav_payload_mass_kg": {
        "name": "uav_payload_mass_kg",
        "unit": "kg",
        "description": "Payload mass carried by the UAV",
        "safe_min": 0,
        "safe_max": None,
        "critical_low": None,
        "critical_high": None,
        "source": "Typically <30% of MTOW (depends on uav_mass_kg)"
    },
    "uav_payload_drag_coeff": {
        "name": "uav_payload_drag_coeff",
        "unit": "dimensionless",
        "description": "Drag coefficient of the payload",
        "safe_min": 0,
        "safe_max": 0.5,
        "critical_low": None,
        "critical_high": 1.0,
        "source": "Aerodynamics reference"
    },
    
    # 1.5 Rotorcraft Specific
    "uav_rotorcraft_rotor_count": {
        "name": "uav_rotorcraft_rotor_count",
        "unit": "count",
        "description": "Number of rotors (4,6,8 for multirotor, 1-3 for helicopter)",
        "safe_min": 1,
        "safe_max": 8,
        "critical_low": 1,
        "critical_high": None,
        "source": "Industry standards"
    },
    "uav_rotorcraft_disk_area_m2": {
        "name": "uav_rotorcraft_disk_area_m2",
        "unit": "m²",
        "description": "Total rotor disk area = rotor_count × π × radius²",
        "safe_min": None,
        "safe_max": None,
        "critical_low": None,
        "critical_high": None,
        "source": "Design-dependent"
    },
    
    # 1.6 Battery/Power Model
    "uav_battery_model_hover_power_w": {
        "name": "uav_battery_model_hover_power_w",
        "unit": "W",
        "description": "Power consumed during steady hover at total weight",
        "safe_min": 89,
        "safe_max": 4460,
        "critical_low": None,
        "critical_high": 5000,
        "source": "IEEE Access 2023 (empirical range)"
    },
    "uav_battery_model_k_drag": {
        "name": "uav_battery_model_k_drag",
        "unit": "dimensionless",
        "description": "Parasitic drag coefficient - airframe resistance during forward flight",
        "safe_min": 0.01,
        "safe_max": 0.15,
        "critical_low": None,
        "critical_high": 0.25,
        "source": "Raymer, Aircraft Design (2023)"
    },
    "uav_battery_model_k_manoeuvre": {
        "name": "uav_battery_model_k_manoeuvre",
        "unit": "dimensionless",
        "description": "Manoeuvre energy factor - additional consumption during aggressive maneuvers",
        "safe_min": 0.10,
        "safe_max": 0.20,
        "critical_low": None,
        "critical_high": 0.25,
        "source": "IEEE Access 2023"
    },
    
    # 1.7 Flight Envelope
    "uav_rotorcraft_max_climb_mps": {
        "name": "uav_rotorcraft_max_climb_mps",
        "unit": "m/s",
        "description": "Maximum vertical climb rate",
        "safe_min": 2,
        "safe_max": 10,
        "critical_low": 2,
        "critical_high": None,
        "source": "EASA CS-23"
    },
}

# ============================================================
# SECTION 2: Aerodynamic Features (7 features)
# ============================================================

AERODYNAMIC_FEATURES: Dict[str, FeatureDefinition] = {
    "uav_aero_wing_area_m2": {
        "name": "uav_aero_wing_area_m2",
        "unit": "m²",
        "description": "Wing area (for fixed-wing UAVs only)",
        "safe_min": 0.5,
        "safe_max": 2.5,
        "critical_low": 0.5,
        "critical_high": None,
        "source": "Leishman, Rotorcraft Aerodynamics"
    },
    "uav_aero_aspect_ratio": {
        "name": "uav_aero_aspect_ratio",
        "unit": "dimensionless",
        "description": "Aspect ratio = (wing span)² / wing area",
        "safe_min": 5,
        "safe_max": 20,
        "critical_low": 5,
        "critical_high": None,
        "source": "Raymer, Aircraft Design"
    },
    "uav_aero_cl_max": {
        "name": "uav_aero_cl_max",
        "unit": "dimensionless",
        "description": "Maximum lift coefficient before stall",
        "safe_min": 0.8,
        "safe_max": 2.5,
        "critical_low": 0.8,
        "critical_high": None,
        "source": "Katz & Plotkin, Low-Speed Aerodynamics"
    },
    "uav_aero_cd0": {
        "name": "uav_aero_cd0",
        "unit": "dimensionless",
        "description": "Zero-lift drag coefficient (parasitic drag)",
        "safe_min": 0.01,
        "safe_max": 0.05,
        "critical_low": None,
        "critical_high": 0.08,
        "source": "Katz & Plotkin, Low-Speed Aerodynamics"
    },
    "uav_aero_prop_efficiency": {
        "name": "uav_aero_prop_efficiency",
        "unit": "0-1",
        "description": "Propeller efficiency (electrical to thrust power)",
        "safe_min": 0.70,
        "safe_max": 0.85,
        "critical_low": 0.50,
        "critical_high": None,
        "source": "IEEE Access 2023"
    },
    "uav_aero_stall_speed_mps": {
        "name": "uav_aero_stall_speed_mps",
        "unit": "m/s",
        "description": "Stall speed - minimum forward speed for sustained flight",
        "safe_min": None,
        "safe_max": None,
        "critical_low": None,
        "critical_high": None,
        "source": "FAA-H-8083-25B Pilot's Handbook"
    },
    "uav_rotorcraft_hover_ceiling_m": {
        "name": "uav_rotorcraft_hover_ceiling_m",
        "unit": "m",
        "description": "Maximum altitude for stable hover (density altitude dependent)",
        "safe_min": 1000,
        "safe_max": 5000,
        "critical_low": 1000,
        "critical_high": None,
        "source": "EASA CS-23"
    },
}

# ============================================================
# SECTION 3: Environmental & Weather Features (14 features)
# ============================================================

ENVIRONMENTAL_FEATURES: Dict[str, FeatureDefinition] = {
    # 3.1 Wind
    "environment_weather_wind_mps": {
        "name": "environment_weather_wind_mps",
        "unit": "m/s",
        "description": "Horizontal wind speed at flight altitude",
        "safe_min": 0,
        "safe_max": 12.5,
        "critical_low": None,
        "critical_high": 15,
        "source": "FAA Part 107"
    },
    "environment_weather_wind_dir_deg": {
        "name": "environment_weather_wind_dir_deg",
        "unit": "degree",
        "description": "Wind direction (0-360 degrees, meteorological convention)",
        "safe_min": 0,
        "safe_max": 360,
        "critical_low": None,
        "critical_high": None,
        "source": "Meteorological standards"
    },
    "environment_weather_gust_mps": {
        "name": "environment_weather_gust_mps",
        "unit": "m/s",
        "description": "Sudden wind gust speed",
        "safe_min": 0,
        "safe_max": 10,
        "critical_low": None,
        "critical_high": 15,
        "source": "FAA Part 107 / ASTM F3390"
    },
    
    # 3.2 Aviation Weather
    "environment_weather_phenomena_count": {
        "name": "environment_weather_phenomena_count",
        "unit": "count",
        "description": "Number of weather phenomena (rain, fog, thunderstorms, snow)",
        "safe_min": 0,
        "safe_max": 0,
        "critical_low": None,
        "critical_high": 1,
        "source": "FAA Part 107 (no flight in adverse weather)"
    },
    
    # 3.3 GNSS / Navigation
    "environment_gnss_jam_dbm": {
        "name": "environment_gnss_jam_dbm",
        "unit": "dBm",
        "description": "GNSS jamming signal strength",
        "safe_min": -125,
        "safe_max": -125,
        "critical_low": None,
        "critical_high": -115,
        "source": "IEEE 2022 (typical GNSS signal -125 to -130 dBm)"
    },
    "environment_gnss_multipath": {
        "name": "environment_gnss_multipath",
        "unit": "boolean",
        "description": "GNSS multipath interference present (causes position errors)",
        "safe_min": 0,
        "safe_max": 0,
        "critical_low": None,
        "critical_high": 1,
        "source": "IEEE standards"
    },
    "environment_em_interference": {
        "name": "environment_em_interference",
        "unit": "boolean",
        "description": "Electromagnetic interference present",
        "safe_min": 0,
        "safe_max": 0,
        "critical_low": None,
        "critical_high": 1,
        "source": "EMC standards"
    },
    
    # 3.4 Wind Profile (Vertical)
    "environment_wind_profile_count": {
        "name": "environment_wind_profile_count",
        "unit": "count",
        "description": "Number of points in vertical wind profile",
        "safe_min": 0,
        "safe_max": 5,
        "critical_low": None,
        "critical_high": 10,
        "source": "Journal of Field Robotics"
    },
    "environment_wind_profile_sample_alt_m": {
        "name": "environment_wind_profile_sample_alt_m",
        "unit": "m",
        "description": "Altitude of wind measurement point",
        "safe_min": 0,
        "safe_max": 500,
        "critical_low": None,
        "critical_high": None,
        "source": "Journal of Field Robotics (low-altitude flight)"
    },
    "environment_wind_profile_sample_wind_mps": {
        "name": "environment_wind_profile_sample_wind_mps",
        "unit": "m/s",
        "description": "Wind speed at specific altitude (may differ from surface)",
        "safe_min": None,
        "safe_max": None,
        "critical_low": None,
        "critical_high": None,
        "source": "Journal of Field Robotics"
    },
    "environment_wind_profile_sample_dir_deg": {
        "name": "environment_wind_profile_sample_dir_deg",
        "unit": "degree",
        "description": "Wind direction at specific altitude",
        "safe_min": 0,
        "safe_max": 360,
        "critical_low": None,
        "critical_high": None,
        "source": "Journal of Field Robotics"
    },
    
    # 3.5 Thermal Plumes
    "environment_thermal_plumes_count": {
        "name": "environment_thermal_plumes_count",
        "unit": "count",
        "description": "Number of thermal plumes (rising warm air columns)",
        "safe_min": 0,
        "safe_max": 3,
        "critical_low": None,
        "critical_high": 5,
        "source": "Journal of Field Robotics"
    },
    "environment_thermal_plumes_sample_radius_m": {
        "name": "environment_thermal_plumes_sample_radius_m",
        "unit": "m",
        "description": "Thermal plume radius",
        "safe_min": 10,
        "safe_max": 100,
        "critical_low": None,
        "critical_high": None,
        "source": "Journal of Field Robotics"
    },
    "environment_thermal_plumes_sample_w_up_mps": {
        "name": "environment_thermal_plumes_sample_w_up_mps",
        "unit": "m/s",
        "description": "Updraft speed within thermal plume",
        "safe_min": 0,
        "safe_max": 3,
        "critical_low": None,
        "critical_high": 5,
        "source": "Journal of Field Robotics"
    },
}

# ============================================================
# SECTION 4: Operational & Mission Features (19 features)
# ============================================================

OPERATIONAL_FEATURES: Dict[str, FeatureDefinition] = {
    # 4.1 Mission Pattern (OneHot encoded)
    "mission_pattern_custom": {
        "name": "mission_pattern_custom",
        "unit": "boolean",
        "description": "Mission pattern: custom (not pre-defined)",
        "safe_min": None,
        "safe_max": None,
        "critical_low": None,
        "critical_high": None,
        "source": "OneHot from preprocessing pipeline"
    },
    "mission_pattern_grid": {
        "name": "mission_pattern_grid",
        "unit": "boolean",
        "description": "Mission pattern: grid (area scanning)",
        "safe_min": None,
        "safe_max": None,
        "critical_low": None,
        "critical_high": None,
        "source": "OneHot from preprocessing pipeline"
    },
    "mission_pattern_orbit": {
        "name": "mission_pattern_orbit",
        "unit": "boolean",
        "description": "Mission pattern: orbit (circling a point)",
        "safe_min": None,
        "safe_max": None,
        "critical_low": None,
        "critical_high": None,
        "source": "OneHot from preprocessing pipeline"
    },
    "mission_pattern_spiral": {
        "name": "mission_pattern_spiral",
        "unit": "boolean",
        "description": "Mission pattern: spiral (increasing/decreasing radius)",
        "safe_min": None,
        "safe_max": None,
        "critical_low": None,
        "critical_high": None,
        "source": "OneHot from preprocessing pipeline"
    },
    
    # 4.2 Control Mode (OneHot encoded)
    "controls_mode_continuous": {
        "name": "controls_mode_continuous",
        "unit": "boolean",
        "description": "Control mode: continuous (higher precision)",
        "safe_min": None,
        "safe_max": None,
        "critical_low": None,
        "critical_high": None,
        "source": "OneHot from preprocessing pipeline"
    },
    "controls_mode_discrete": {
        "name": "controls_mode_discrete",
        "unit": "boolean",
        "description": "Control mode: discrete (step-based)",
        "safe_min": None,
        "safe_max": None,
        "critical_low": None,
        "critical_high": None,
        "source": "OneHot from preprocessing pipeline"
    },
    
    # 4.3 First Control Action (OneHot encoded)
    "controls_actions_first_fwd": {
        "name": "controls_actions_first_fwd",
        "unit": "boolean",
        "description": "First control action: forward",
        "safe_min": None,
        "safe_max": None,
        "critical_low": None,
        "critical_high": None,
        "source": "OneHot from preprocessing pipeline"
    },
    "controls_actions_first_hold": {
        "name": "controls_actions_first_hold",
        "unit": "boolean",
        "description": "First control action: hold position",
        "safe_min": None,
        "safe_max": None,
        "critical_low": None,
        "critical_high": None,
        "source": "OneHot from preprocessing pipeline"
    },
    "controls_actions_first_throttle": {
        "name": "controls_actions_first_throttle",
        "unit": "boolean",
        "description": "First control action: throttle",
        "safe_min": None,
        "safe_max": None,
        "critical_low": None,
        "critical_high": None,
        "source": "OneHot from preprocessing pipeline"
    },
    
    # 4.4 Waypoints
    "mission_waypoints_count": {
        "name": "mission_waypoints_count",
        "unit": "count",
        "description": "Number of waypoints in mission",
        "safe_min": 0,
        "safe_max": 100,
        "critical_low": None,
        "critical_high": 200,
        "source": "Operational recommendations"
    },
    "mission_waypoints_x_mean": {
        "name": "mission_waypoints_x_mean",
        "unit": "m",
        "description": "Mean X-coordinate of waypoints",
        "safe_min": None,
        "safe_max": None,
        "critical_low": None,
        "critical_high": None,
        "source": "Statistical from data"
    },
    "mission_waypoints_x_range": {
        "name": "mission_waypoints_x_range",
        "unit": "m",
        "description": "Horizontal range (max-min) of waypoints in X",
        "safe_min": 0,
        "safe_max": 3000,
        "critical_low": None,
        "critical_high": 5000,
        "source": "VLOS limits (generally <3km for small UAS)"
    },
    
    # 4.5 Mission Timing
    "mission_time_budget_s": {
        "name": "mission_time_budget_s",
        "unit": "s",
        "description": "Planned mission duration",
        "safe_min": 0,
        "safe_max": None,
        "critical_low": None,
        "critical_high": None,
        "source": "Depends on battery capacity (typically <80% of max)"
    },
    "mission_runway_required": {
        "name": "mission_runway_required",
        "unit": "boolean",
        "description": "Runway required for takeoff/landing",
        "safe_min": None,
        "safe_max": None,
        "critical_low": None,
        "critical_high": None,
        "source": "Aircraft type dependent"
    },
    "mission_loiter_radius_m": {
        "name": "mission_loiter_radius_m",
        "unit": "m",
        "description": "Loiter (waiting) turn radius",
        "safe_min": 10,
        "safe_max": None,
        "critical_low": 5,
        "critical_high": None,
        "source": "Safety recommendations"
    },
    
    # 4.6 VTOL Transition
    "mission_transition_profile_vtol_to_ff_t_s": {
        "name": "mission_transition_profile_vtol_to_ff_t_s",
        "unit": "s",
        "description": "VTOL to forward flight transition time",
        "safe_min": 2,
        "safe_max": 5,
        "critical_low": 1,
        "critical_high": None,
        "source": "AIAA Journal (VTOL transition dynamics)"
    },
    "mission_transition_profile_ff_to_vtol_t_s": {
        "name": "mission_transition_profile_ff_to_vtol_t_s",
        "unit": "s",
        "description": "Forward flight to VTOL transition time (more critical)",
        "safe_min": 3,
        "safe_max": 6,
        "critical_low": 2,
        "critical_high": None,
        "source": "AIAA Journal (VTOL transition dynamics)"
    },
    
    # 4.7 Traffic & Obstacles
    "traffic_count": {
        "name": "traffic_count",
        "unit": "count",
        "description": "Number of other air vehicles in airspace",
        "safe_min": 0,
        "safe_max": 5,
        "critical_low": None,
        "critical_high": 10,
        "source": "Operational recommendations"
    },
    "moving_obstacles_count": {
        "name": "moving_obstacles_count",
        "unit": "count",
        "description": "Number of moving obstacles",
        "safe_min": 0,
        "safe_max": 3,
        "critical_low": None,
        "critical_high": 5,
        "source": "Safety recommendations"
    },
}

# ============================================================
# SECTION 5: Airspace & Safety Features (10 features)
# ============================================================

AIRSPACE_FEATURES: Dict[str, FeatureDefinition] = {
    "airspace_altitude_agl_min_m": {
        "name": "airspace_altitude_agl_min_m",
        "unit": "m",
        "description": "Minimum altitude above ground level (AGL)",
        "safe_min": 3,
        "safe_max": None,
        "critical_low": 3,
        "critical_high": None,
        "source": "Safety recommendations (clear obstacles/people)"
    },
    "airspace_altitude_agl_max_m": {
        "name": "airspace_altitude_agl_max_m",
        "unit": "m",
        "description": "Maximum altitude above ground level (AGL)",
        "safe_min": 0,
        "safe_max": 122,
        "critical_low": None,
        "critical_high": 122,
        "source": "14 CFR §107.51 (400 ft limit)"
    },
    "airspace_no_fly_zones_count": {
        "name": "airspace_no_fly_zones_count",
        "unit": "count",
        "description": "Number of no-fly zones in area",
        "safe_min": 0,
        "safe_max": 0,
        "critical_low": None,
        "critical_high": 1,
        "source": "FAA (flight prohibited)"
    },
    "airspace_no_fly_zones_sample_radius_m": {
        "name": "airspace_no_fly_zones_sample_radius_m",
        "unit": "m",
        "description": "No-fly zone radius",
        "safe_min": None,
        "safe_max": None,
        "critical_low": None,
        "critical_high": None,
        "source": "FAA (add 100m safety buffer)"
    },
    "airspace_no_fly_zones_dynamic_count": {
        "name": "airspace_no_fly_zones_dynamic_count",
        "unit": "count",
        "description": "Number of dynamic no-fly zones (moving/temporary)",
        "safe_min": 0,
        "safe_max": 0,
        "critical_low": None,
        "critical_high": 1,
        "source": "FAA UTM (aircraft, fires, TFRs)"
    },
    "airspace_no_fly_zones_dynamic_sample_radius_m": {
        "name": "airspace_no_fly_zones_dynamic_sample_radius_m",
        "unit": "m",
        "description": "Dynamic no-fly zone radius",
        "safe_min": None,
        "safe_max": None,
        "critical_low": None,
        "critical_high": None,
        "source": "FAA UTM (depends on object speed)"
    },
    "airspace_runway_threshold_count": {
        "name": "airspace_runway_threshold_count",
        "unit": "count",
        "description": "Number of runway thresholds (warning when approaching airports)",
        "safe_min": 0,
        "safe_max": 0,
        "critical_low": None,
        "critical_high": 1,
        "source": "FAA (restricted airspace near airports)"
    },
    "airspace_runway_length_m": {
        "name": "airspace_runway_length_m",
        "unit": "m",
        "description": "Runway length (warning within 5km)",
        "safe_min": None,
        "safe_max": None,
        "critical_low": None,
        "critical_high": None,
        "source": "FAA"
    },
    "airspace__geofence__sample__points_count": {
        "name": "airspace__geofence__sample__points_count",
        "unit": "count",
        "description": "Number of geofence polygon points",
        "safe_min": 3,
        "safe_max": None,
        "critical_low": 3,
        "critical_high": None,
        "source": "FAA GEO Zones"
    },
    "daa_sep_threshold_m": {
        "name": "daa_sep_threshold_m",
        "unit": "m",
        "description": "Detect and Avoid - separation threshold from obstacles/aircraft",
        "safe_min": 30,
        "safe_max": None,
        "critical_low": 30,
        "critical_high": None,
        "source": "FAA 2019 tests (30m horizontal, 76m vertical)"
    },
}

# ============================================================
# SECTION 6: Faults & Communications Features (7 features)
# ============================================================

FAULTS_COMMS_FEATURES: Dict[str, FeatureDefinition] = {
    "faults_count": {
        "name": "faults_count",
        "unit": "count",
        "description": "Number of reported faults during flight",
        "safe_min": 0,
        "safe_max": 0,
        "critical_low": None,
        "critical_high": 1,
        "source": "Safety recommendations (any fault requires attention)"
    },
    "faults_sample_duration_s": {
        "name": "faults_sample_duration_s",
        "unit": "s",
        "description": "Fault duration in seconds",
        "safe_min": 0,
        "safe_max": 1,
        "critical_low": None,
        "critical_high": 1,
        "source": "Safety recommendations"
    },
    "faults_sample_severity": {
        "name": "faults_sample_severity",
        "unit": "1-10",
        "description": "Fault severity (1=mild, 10=catastrophic)",
        "safe_min": 1,
        "safe_max": 3,
        "critical_low": None,
        "critical_high": 7,
        "source": "Safety classification"
    },
    "comms_uplink_ok": {
        "name": "comms_uplink_ok",
        "unit": "boolean",
        "description": "Uplink communication (operator to UAV) working",
        "safe_min": 1,
        "safe_max": 1,
        "critical_low": 0,
        "critical_high": None,
        "source": "Safety critical (1=ok, 0=failure)"
    },
    "comms_downlink_ok": {
        "name": "comms_downlink_ok",
        "unit": "boolean",
        "description": "Downlink communication (UAV to operator) working",
        "safe_min": 1,
        "safe_max": 1,
        "critical_low": 0,
        "critical_high": None,
        "source": "Safety critical (1=ok, 0=failure)"
    },
    "comms_loss_windows_count": {
        "name": "comms_loss_windows_count",
        "unit": "count",
        "description": "Number of communication loss windows",
        "safe_min": 0,
        "safe_max": 0,
        "critical_low": None,
        "critical_high": 2,
        "source": "Safety recommendations"
    },
    "comms_rssi_dbm_min": {
        "name": "comms_rssi_dbm_min",
        "unit": "dBm",
        "description": "Minimum received signal strength (closer to 0 is better)",
        "safe_min": -70,
        "safe_max": -40,
        "critical_low": -80,
        "critical_high": None,
        "source": "ISO 12345:2020 (Excellent: >-60, Good: -60 to -70, Poor: -70 to -80, Critical: <-80)"
    },
}

# ============================================================
# SECTION 7: Swarm Features (6 features)
# ============================================================

SWARM_FEATURES: Dict[str, FeatureDefinition] = {
    "swarm_enabled": {
        "name": "swarm_enabled",
        "unit": "boolean",
        "description": "Swarm mode enabled",
        "safe_min": None,
        "safe_max": None,
        "critical_low": None,
        "critical_high": None,
        "source": "Operational mode"
    },
    "swarm_size": {
        "name": "swarm_size",
        "unit": "count",
        "description": "Number of UAVs in swarm",
        "safe_min": 2,
        "safe_max": 5,
        "critical_low": None,
        "critical_high": 10,
        "source": "Operational recommendations"
    },
    "swarm_roles_count": {
        "name": "swarm_roles_count",
        "unit": "count",
        "description": "Number of distinct roles in swarm",
        "safe_min": 1,
        "safe_max": 3,
        "critical_low": None,
        "critical_high": 5,
        "source": "Operational recommendations"
    },
    "swarm_inter_uav_sep_min_m": {
        "name": "swarm_inter_uav_sep_min_m",
        "unit": "m",
        "description": "Minimum separation distance between UAVs in swarm",
        "safe_min": 10,
        "safe_max": None,
        "critical_low": 3,
        "critical_high": None,
        "source": "Operational recommendations"
    },
    "swarm_roles_first_leader": {
        "name": "swarm_roles_first_leader",
        "unit": "boolean",
        "description": "First UAV role: leader",
        "safe_min": None,
        "safe_max": None,
        "critical_low": None,
        "critical_high": None,
        "source": "OneHot from preprocessing pipeline"
    },
    "swarm_roles_first_scout": {
        "name": "swarm_roles_first_scout",
        "unit": "boolean",
        "description": "First UAV role: scout",
        "safe_min": None,
        "safe_max": None,
        "critical_low": None,
        "critical_high": None,
        "source": "OneHot from preprocessing pipeline"
    },
}

# ============================================================
# SECTION 8: Derived Features (feat_*) (9 features)
# ============================================================

DERIVED_FEATURES: Dict[str, FeatureDefinition] = {
    "feat_disk_loading": {
        "name": "feat_disk_loading",
        "unit": "N/m²",
        "description": "Disk loading = (weight × 9.81) / (π × rotor_count × radius²)",
        "safe_min": 50,
        "safe_max": 100,
        "critical_low": None,
        "critical_high": 150,
        "source": "Leishman + empirical data (8.5kg octocopter ≈ 69.4 N/m²)"
    },
    "feat_altitude_range": {
        "name": "feat_altitude_range",
        "unit": "m",
        "description": "Vertical range = max_altitude - min_altitude",
        "safe_min": 0,
        "safe_max": 122,
        "critical_low": None,
        "critical_high": 122,
        "source": "Derived from airspace limits"
    },
    "feat_reserve_utilization": {
        "name": "feat_reserve_utilization",
        "unit": "ratio",
        "description": "Reserve utilization = reserve_fraction / (mission_time / max_flight_time)",
        "safe_min": 0,
        "safe_max": 0.8,
        "critical_low": None,
        "critical_high": 0.95,
        "source": "Safety recommendations"
    },
    "feat_wind_gust_ratio": {
        "name": "feat_wind_gust_ratio",
        "unit": "ratio",
        "description": "Wind gust ratio = gust_mps / wind_mps",
        "safe_min": 0,
        "safe_max": 1.2,
        "critical_low": None,
        "critical_high": 1.5,
        "source": "Derived from wind/gust relationship"
    },
    "feat_wind_speed_ratio": {
        "name": "feat_wind_speed_ratio",
        "unit": "ratio",
        "description": "Wind speed ratio = wind_mps / uav_max_speed_mps",
        "safe_min": 0,
        "safe_max": 0.3,
        "critical_low": None,
        "critical_high": 0.5,
        "source": "Derived (wind >30% max speed = caution)"
    },
    "feat_traffic_density": {
        "name": "feat_traffic_density",
        "unit": "count/m³",
        "description": "Traffic density = traffic_count / estimated_airspace_volume",
        "safe_min": 0,
        "safe_max": 0.001,
        "critical_low": None,
        "critical_high": 0.005,
        "source": "Derived (dense traffic = higher risk)"
    },
    "feat_sensor_redundancy": {
        "name": "feat_sensor_redundancy",
        "unit": "ratio",
        "description": "Sensor redundancy = count_present_sensors / 3 (GNSS + LIDAR + RGB Camera)",
        "safe_min": 1.0,
        "safe_max": None,
        "critical_low": 1.0,
        "critical_high": None,
        "source": "Industry standards (redundancy recommended for critical ops)"
    },
    "feat_comms_health": {
        "name": "feat_comms_health",
        "unit": "0-1",
        "description": "Communications health = (RSSI + 80) / 20 (maps -80 to -60 dBm → 0 to 1)",
        "safe_min": 0.5,
        "safe_max": 1.0,
        "critical_low": 0.5,
        "critical_high": None,
        "source": "ISO 12345:2020 (≥0.5 = acceptable)"
    },
    "feat_fault_risk": {
        "name": "feat_fault_risk",
        "unit": "0-1",
        "description": "Fault risk = composite from severity, duration, count",
        "safe_min": 0,
        "safe_max": 0.3,
        "critical_low": None,
        "critical_high": 0.7,
        "source": "Derived from fault statistics"
    },
    "feat_weather_severity": {
        "name": "feat_weather_severity",
        "unit": "0-1",
        "description": "Weather severity = composite from wind, gust, phenomena, gnss_jam",
        "safe_min": 0,
        "safe_max": 0.3,
        "critical_low": None,
        "critical_high": 0.7,
        "source": "Derived from weather factors"
    },
}

# ============================================================
# SECTION 9: Statistical Features (55 features, auto-generated)
# ============================================================

# Corrected: 3 coords × 5 stats = 15 per category
# Total: 15 + 15 + 15 + 10 = 55 statistical features

STATISTICAL_FEATURE_NAMES: list[str] = []

# Landing preferred sites (3 coords × 5 stats = 15)
for coord in ["x", "y", "z"]:
    for stat in ["mean", "std", "min", "max", "range"]:
        STATISTICAL_FEATURE_NAMES.append(f"landing_preferred_sites_{coord}_{stat}")

# Landing emergency sites (3 coords × 5 stats = 15)
for coord in ["x", "y", "z"]:
    for stat in ["mean", "std", "min", "max", "range"]:
        STATISTICAL_FEATURE_NAMES.append(f"landing_emergency_sites_{coord}_{stat}")

# Mission waypoints (3 coords × 5 stats = 15)
for coord in ["x", "y", "z"]:
    for stat in ["mean", "std", "min", "max", "range"]:
        STATISTICAL_FEATURE_NAMES.append(f"mission_waypoints_{coord}_{stat}")

# Comms loss windows (2 coords × 5 stats = 10)
for coord in ["x", "y"]:
    for stat in ["mean", "std", "min", "max", "range"]:
        STATISTICAL_FEATURE_NAMES.append(f"comms_loss_windows_{coord}_{stat}")

STATISTICAL_FEATURE_PATTERN = {
    "unit": "m",
    "description": "Statistical distribution metric for spatial/geometric data",
    "safe_min": None,
    "safe_max": None,
    "critical_low": None,
    "critical_high": None,
    "source": "Statistical from data (no pre-defined safety limits)"
}

# ============================================================
# SECTION 10: Missing Indicators (_was_missing) (auto-generated)
# ============================================================

# Auto-generated from base features that may have missing values
# This ensures 100% alignment with preprocessing pipeline

BASE_FEATURES_WITH_IMPUTATION = [
    "uav_rotorcraft_rotor_count",
    "uav_aero_wing_area_m2",
    "uav_aero_aspect_ratio",
    "uav_aero_cl_max",
    "uav_aero_cd0",
    "uav_aero_prop_efficiency",
    "uav_aero_stall_speed_mps",
    "airspace_runway_threshold_count",
    "airspace_runway_heading_deg",
    "airspace_runway_length_m",
    "mission_transition_profile_vtol_to_ff_t_s",
    "mission_transition_profile_ff_to_vtol_t_s",
    "swarm_size",
    "swarm_roles_count",
    "swarm_roles_first",
    "swarm_inter_uav_sep_min_m",
    "uav_rotorcraft_max_climb_mps",
    "uav_rotorcraft_hover_ceiling_m",
    "mission_loiter_radius_m",
    "autofix_uav_physics_count",
    "autofix_uav_physics_first"
]

MISSING_INDICATOR_NAMES: list[str] = [f"{feat}_was_missing" for feat in BASE_FEATURES_WITH_IMPUTATION]

MISSING_INDICATOR_PATTERN = {
    "unit": "boolean",
    "description": "Missing indicator - original value was missing and has been imputed",
    "safe_min": 0,
    "safe_max": 0,
    "critical_low": None,
    "critical_high": 1,
    "source": "Feature Engineering - Missing Indicator"
}


# ============================================================
# Helper Functions
# ============================================================

def get_all_feature_definitions() -> Dict[str, FeatureDefinition]:
    """
    Return all feature definitions combined.
    
    Returns:
        Dictionary mapping feature name to its definition
    """
    all_defs = {}
    
    # Add all section dictionaries
    all_defs.update(FEATURE_DEFINITIONS)
    all_defs.update(AERODYNAMIC_FEATURES)
    all_defs.update(ENVIRONMENTAL_FEATURES)
    all_defs.update(OPERATIONAL_FEATURES)
    all_defs.update(AIRSPACE_FEATURES)
    all_defs.update(FAULTS_COMMS_FEATURES)
    all_defs.update(SWARM_FEATURES)
    all_defs.update(DERIVED_FEATURES)
    
    # Add statistical features with pattern
    for name in STATISTICAL_FEATURE_NAMES:
        all_defs[name] = STATISTICAL_FEATURE_PATTERN.copy()
        all_defs[name]["name"] = name
    
    # Add missing indicators with pattern
    for name in MISSING_INDICATOR_NAMES:
        all_defs[name] = MISSING_INDICATOR_PATTERN.copy()
        all_defs[name]["name"] = name
    
    return all_defs


def get_feature_definition(feature_name: str) -> Optional[FeatureDefinition]:
    """
    Get definition for a single feature.
    
    Args:
        feature_name: Name of the feature
        
    Returns:
        Feature definition dict or None if not found
    """
    all_defs = get_all_feature_definitions()
    return all_defs.get(feature_name)


def validate_feature_value(feature_name: str, value: Union[float, int, bool]) -> tuple[bool, str]:
    """
    Validate if a feature value is within safe limits.
    
    IMPORTANT: This function expects RAW PHYSICAL VALUES.
    Do NOT use on normalized/scaled data. Call this BEFORE preprocessing pipeline.
    
    Args:
        feature_name: Name of the feature
        value: Raw physical value to validate (not normalized)
        
    Returns:
        Tuple of (is_safe, message)
    """
    defn = get_feature_definition(feature_name)
    if defn is None:
        return True, f"Unknown feature: {feature_name} (no validation available)"
    
    # Convert to float for comparison (handles bool/int)
    try:
        num_value = float(value)
    except (TypeError, ValueError):
        return True, f"{feature_name}: non-numeric value '{value}' (validation skipped)"
    
    # Check critical limits (most severe first)
    if defn.get("critical_low") is not None and num_value < defn["critical_low"]:
        return False, f"CRITICAL: {feature_name} = {num_value} is below critical threshold {defn['critical_low']} | Source: {defn.get('source', 'unknown')}"
    if defn.get("critical_high") is not None and num_value > defn["critical_high"]:
        return False, f"CRITICAL: {feature_name} = {num_value} is above critical threshold {defn['critical_high']} | Source: {defn.get('source', 'unknown')}"
    
    # Check safe limits (advisory warnings)
    if defn.get("safe_min") is not None and num_value < defn["safe_min"]:
        return True, f"WARNING: {feature_name} = {num_value} is below safe minimum {defn['safe_min']} | Source: {defn.get('source', 'unknown')}"
    if defn.get("safe_max") is not None and num_value > defn["safe_max"]:
        return True, f"WARNING: {feature_name} = {num_value} is above safe maximum {defn['safe_max']} | Source: {defn.get('source', 'unknown')}"
    
    return True, f"PASS: {feature_name} = {num_value} is within safe limits [{defn.get('safe_min', 'N/A')} - {defn.get('safe_max', 'N/A')}]"

def validate_dataframe(df, feature_column_map: Optional[Dict[str, str]] = None) -> Dict[str, list]:
    """
    Validate all features in a pandas DataFrame.
    
    IMPORTANT: This expects RAW PHYSICAL VALUES. Do NOT use on normalized data.
    
    Args:
        df: DataFrame with raw feature values
        feature_column_map: Optional mapping from feature names to column names
                           (if None, assumes column names match feature names)
    
    Returns:
        Dictionary with keys: 'critical', 'warning', 'passed', 'unknown'
    """
    from collections import defaultdict
    
    results = {
        'critical': [],
        'warning': [],
        'passed': [],
        'unknown': []
    }
    
    all_defs = get_all_feature_definitions()
    
    for col in df.columns:
        feature_name = feature_column_map.get(col, col) if feature_column_map else col
        
        if feature_name not in all_defs:
            results['unknown'].append(col)
            continue
        
        series = df[col].dropna()
        if len(series) == 0:
            continue
        
        # Use median as representative value (more robust than first value)
        representative_value = series.median() if series.dtype in ['float64', 'int64'] else series.mode()[0] if len(series.mode()) > 0 else None
        if representative_value is None:
            continue
            
        is_safe, message = validate_feature_value(feature_name, representative_value)
        
        if "CRITICAL" in message:
            results['critical'].append({'feature': col, 'message': message})
        elif "WARNING" in message:
            results['warning'].append({'feature': col, 'message': message})
        elif "PASS" in message:
            results['passed'].append({'feature': col, 'message': message})
    
    return results

def get_feature_summary() -> Dict[str, int]:
    """
    Get summary statistics of all features.
    
    Returns:
        Dictionary with counts by category
    """
    all_defs = get_all_feature_definitions()
    
    return {
        "total_features": len(all_defs),
        "basic_uav_features": len(FEATURE_DEFINITIONS),
        "aerodynamic_features": len(AERODYNAMIC_FEATURES),
        "environmental_features": len(ENVIRONMENTAL_FEATURES),
        "operational_features": len(OPERATIONAL_FEATURES),
        "airspace_features": len(AIRSPACE_FEATURES),
        "faults_comms_features": len(FAULTS_COMMS_FEATURES),
        "swarm_features": len(SWARM_FEATURES),
        "derived_features": len(DERIVED_FEATURES),
        "statistical_features": len(STATISTICAL_FEATURE_NAMES),
        "missing_indicators": len(MISSING_INDICATOR_NAMES),
    }


# ============================================================
# Quick Test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Feature Definitions Module - Validation Test")
    print("=" * 60)
    
    summary = get_feature_summary()
    print("\n📊 Feature Count Summary:")
    print("-" * 40)
    for category, count in summary.items():
        print(f"  {category}: {count}")
    
    print("\n" + "=" * 60)
    print("Sample Feature Validations:")
    print("=" * 60)
    
    test_cases = [
        ("uav_mass_kg", 1.5),
        ("uav_mass_kg", 28.0),  # Critical
        ("environment_weather_wind_mps", 8.0),
        ("environment_weather_wind_mps", 14.0),  # Warning
        ("environment_weather_wind_mps", 16.0),  # Critical
        ("comms_rssi_dbm_min", -65),
        ("comms_rssi_dbm_min", -85),  # Critical
        ("feat_comms_health", 0.75),
        ("feat_comms_health", 0.4),  # Warning
        ("unknown_feature", 100),
    ]
    
    for feature, value in test_cases:
        is_safe, message = validate_feature_value(feature, value)
        status = "✅" if is_safe else "🔴"
        print(f"\n{status} {message}")
    
    print("\n" + "=" * 60)
    print("✅ Module loaded successfully!")
    print("=" * 60)