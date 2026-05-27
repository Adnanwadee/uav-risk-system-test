"""
Module: uav_risk.ml.feature_defs
Purpose: Packaging-grade single source of truth for the authoritative 198 feature registry,
         68 strict primary core definitions, and agent semantic categories.
Dependencies: Fully standalone core constitution module matching SOLID principles.
Source References: FAA Part 107, EASA CS-23, ISO 12345:2020, ASTM F3390, Raymer (2023).
"""

from __future__ import annotations
import logging
from typing import Dict, Any, Optional, List, Tuple

# تدوين اسم مستعار موحد لهياكل بيانات خصائص الميزات
FeatureDefinition = Dict[str, Any]
logger = logging.getLogger(__name__)

# =====================================================================
# 🌐 Matrix Constitution: 198 Official Features in Exact Order of the PKL Binary
# =====================================================================
OFFICIAL_198_FEATURE_ORDER: List[str] = [
    "uav_energy_source_fuel", "uav_energy_source_hybrid", "mission_pattern_custom",
    "mission_pattern_grid", "mission_pattern_orbit", "mission_pattern_spiral",
    "controls_mode_discrete", "controls_actions_first_hold", "controls_actions_first_throttle",
    "swarm_roles_first_relay", "swarm_roles_first_scout", "swarm_roles_first_single",
    "swarm_roles_first_solo", "sim_duration_steps", "sim_policy_frequency",
    "uav_mass_kg", "uav_battery_wh", "uav_fuel_l", "uav_max_speed_mps",
    "uav_max_tilt_deg", "uav_reserve_fraction", "uav_battery_model_hover_power_w",
    "uav_battery_model_k_drag", "uav_battery_model_k_manoeuvre", "uav_rotorcraft_rotor_count",
    "uav_rotorcraft_disk_area_m2", "uav_payload_mass_kg", "uav_payload_drag_coeff",
    "environment_weather_wind_mps", "environment_weather_wind_dir_deg", "environment_weather_gust_mps",
    "environment_weather_phenomena_count", "airspace_altitude_agl_min_m", "airspace_altitude_agl_max_m",
    "airspace_no_fly_zones_sample_radius_m", "airspace_no_fly_zones_sample_floor_m",
    "airspace_no_fly_zones_sample_ceiling_m", "spawn_xyz_first", "spawn_yaw_deg",
    "landing_preferred_sites_count", "landing_preferred_sites_x_mean", "landing_preferred_sites_x_std",
    "landing_preferred_sites_x_min", "landing_preferred_sites_x_max", "landing_preferred_sites_x_range",
    "landing_preferred_sites_y_mean", "landing_preferred_sites_y_std", "landing_preferred_sites_y_min",
    "landing_preferred_sites_y_max", "landing_preferred_sites_y_range", "landing_preferred_sites_z_mean",
    "landing_preferred_sites_z_std", "landing_preferred_sites_z_min", "landing_preferred_sites_z_max",
    "landing_preferred_sites_z_range", "landing_emergency_sites_count", "landing_emergency_sites_x_mean",
    "landing_emergency_sites_x_std", "landing_emergency_sites_x_min", "landing_emergency_sites_x_max",
    "landing_emergency_sites_x_range", "landing_emergency_sites_y_mean", "landing_emergency_sites_y_std",
    "landing_emergency_sites_y_min", "landing_emergency_sites_y_max", "landing_emergency_sites_y_range",
    "landing_emergency_sites_z_mean", "landing_emergency_sites_z_std", "landing_emergency_sites_z_min",
    "landing_emergency_sites_z_max", "landing_emergency_sites_z_range", "mission_waypoints_count",
    "mission_waypoints_x_mean", "mission_waypoints_x_std", "mission_waypoints_x_min",
    "mission_waypoints_x_max", "mission_waypoints_x_range", "mission_waypoints_y_mean",
    "mission_waypoints_y_std", "mission_waypoints_y_min", "mission_waypoints_y_max",
    "mission_waypoints_y_range", "mission_waypoints_z_mean", "mission_waypoints_z_std",
    "mission_waypoints_z_min", "mission_waypoints_z_max", "mission_waypoints_z_range",
    "mission_time_budget_s", "traffic_count", "traffic_sample_speed_mps",
    "traffic_sample_heading_deg", "moving_obstacles_sample_radius_m", "controls_actions_count",
    "daa_sep_threshold_m", "daa_ttc_threshold_s", "faults_count",
    "faults_sample_t_s", "faults_sample_duration_s", "faults_sample_severity",
    "comms_loss_windows_count", "comms_loss_windows_x_mean", "comms_loss_windows_x_std",
    "comms_loss_windows_x_min", "comms_loss_windows_x_max", "comms_loss_windows_x_range",
    "comms_loss_windows_y_mean", "comms_loss_windows_y_std", "comms_loss_windows_y_min",
    "comms_loss_windows_y_max", "comms_loss_windows_y_range", "comms_rssi_dbm_min",
    "airspace_no_fly_zones_dynamic_count", "airspace_no_fly_zones_dynamic_sample_radius_m",
    "airspace_no_fly_zones_dynamic_sample_floor_m", "airspace_no_fly_zones_dynamic_sample_ceiling_m",
    "autofix_uav_physics_count", "uav_aero_wing_area_m2", "uav_aero_aspect_ratio",
    "uav_aero_cl_max", "uav_aero_cd0", "uav_aero_prop_efficiency",
    "uav_aero_stall_speed_mps", "airspace_runway_threshold_first", "airspace_runway_heading_deg",
    "airspace_runway_length_m", "environment_wind_profile_count", "environment_wind_profile_sample_alt_m",
    "environment_wind_profile_sample_wind_mps", "environment_wind_profile_sample_dir_deg", "environment_thermal_plumes_count",
    "environment_thermal_plumes_sample_radius_m", "environment_thermal_plumes_sample_w_up_mps", "environment_gnss_jam_dbm",
    "mission_transition_profile_vtol_to_ff_t_s", "mission_transition_profile_ff_to_vtol_t_s", "swarm_size",
    "swarm_roles_count", "swarm_inter_uav_sep_min_m", "uav_rotorcraft_max_climb_mps",
    "uav_rotorcraft_hover_ceiling_m", "mission_loiter_radius_m", "airspace__geofence__sample__points_count",
    "uav_sensors_gnss", "uav_sensors_lidar", "uav_sensors_radar",
    "uav_sensors_camera_rgb", "uav_sensors_camera_thermal", "airspace_no_fly_zones_count",
    "mission_runway_required", "moving_obstacles_count", "swarm_enabled",
    "comms_uplink_ok", "comms_downlink_ok", "airspace_runway_threshold_count",
    "environment_gnss_multipath", "environment_em_interference", "airspace__no__fly__zones__sample__center_count",
    "landing__preferred__sites__sample_count", "landing__emergency__sites__sample_count", "mission__waypoints__sample_count",
    "traffic__sample__spawn_count", "moving__obstacles__sample__center_count", "moving__obstacles__sample__vel_count",
    "comms__loss__windows__sample_count", "airspace__no__fly__zones__dynamic__sampl_count", "environment__thermal__plumes__sample__ce_count",
    "uav_rotorcraft_rotor_count_was_missing", "autofix_uav_physics_count_was_missing", "autofix_uav_physics_first_was_missing",
    "uav_aero_wing_area_m2_was_missing", "uav_aero_aspect_ratio_was_missing", "uav_aero_cl_max_was_missing",
    "uav_aero_cd0_was_missing", "uav_aero_prop_efficiency_was_missing", "uav_aero_stall_speed_mps_was_missing",
    "airspace_runway_threshold_count_was_missing", "airspace_runway_threshold_first_was_missing", "airspace_runway_heading_deg_was_missing",
    "airspace_runway_length_m_was_missing", "mission_transition_profile_vtol_to_ff_t_s_was_missing", "mission_transition_profile_ff_to_vtol_t_s_was_missing",
    "swarm_size_was_missing", "swarm_roles_count_was_missing", "swarm_roles_first_was_missing",
    "swarm_inter_uav_sep_min_m_was_missing", "uav_rotorcraft_max_climb_mps_was_missing", "uav_rotorcraft_hover_ceiling_m_was_missing",
    "mission_loiter_radius_m_was_missing", "feat_disk_loading", "feat_altitude_range",
    "feat_reserve_utilization", "feat_wind_gust_ratio", "feat_wind_speed_ratio",
    "feat_sensor_redundancy", "feat_comms_health", "feat_traffic_density",
    "feat_fault_risk", "feat_weather_severity"
]

# قاموس التعيين المسبق الصارم للوحدات غير القياسية لزيادة مناعة النظام
EXPLICIT_UNIT_MAP: Dict[str, str] = {
    "swarm_inter_uav_sep_min_m": "meter",
    "comms_rssi_dbm_min": "dBm",
    "environment_gnss_jam_dbm": "dBm",
    "uav_battery_model_hover_power_w": "watt",
    "sim_policy_frequency": "Hz"
}

# Legacy processed-validator bounds. Production raw structural hard veto lives in
# core.data_validator.run_structural_hard_veto() and must not apply universal
# mass/wind/altitude policy thresholds.
BOUNDS_REGISTRY: Dict[str, Dict[str, float]] = {
    "uav_mass_kg": {"safe_min": 0.5, "safe_max": 24.5, "critical_high": 25.0},
    "environment_weather_wind_mps": {"safe_min": 0.0, "safe_max": 12.0, "critical_high": 15.0},
    "airspace_altitude_agl_max_m": {"safe_min": 10.0, "safe_max": 121.0, "critical_high": 122.0}
}


def get_all_feature_names() -> list[str]:
    """المصدر المطلق والجامد لأسماء وترتيب مصفوفة الـ 198 لمنع أي انزياح فهارس."""
    return list(OFFICIAL_198_FEATURE_ORDER)


def get_core_features() -> list[str]:
    """Legacy processed 68-feature list. Do not use as the production user contract.

    The raw-first contract is defined in uav_risk.ml.raw_schema via
    PROFILE_DERIVED_RAW_FEATURES and SCENARIO_REQUIRED_RAW_FEATURES.
    """
    return [
        "uav_energy_source_fuel", "uav_energy_source_hybrid", "mission_pattern_custom",
        "mission_pattern_grid", "mission_pattern_orbit", "mission_pattern_spiral",
        "controls_mode_discrete", "swarm_enabled", "swarm_size",
        "swarm_inter_uav_sep_min_m", "swarm_roles_first_relay", "swarm_roles_first_scout",
        "swarm_roles_first_single", "swarm_roles_first_solo", "uav_mass_kg",
        "uav_battery_wh", "uav_fuel_l", "uav_payload_mass_kg",
        "uav_max_speed_mps", "uav_max_tilt_deg", "uav_reserve_fraction",
        "uav_rotorcraft_rotor_count", "uav_rotorcraft_max_climb_mps", "uav_rotorcraft_hover_ceiling_m",
        "uav_aero_prop_efficiency", "uav_sensors_gnss", "uav_sensors_lidar",
        "uav_sensors_radar", "uav_sensors_camera_rgb", "uav_sensors_camera_thermal",
        "environment_weather_wind_mps", "environment_weather_wind_dir_deg", "environment_weather_gust_mps",
        "environment_weather_phenomena_count", "environment_gnss_jam_dbm", "environment_gnss_multipath",
        "environment_em_interference", "airspace_altitude_agl_min_m", "airspace_altitude_agl_max_m",
        "airspace_no_fly_zones_count", "airspace_no_fly_zones_sample_radius_m", "airspace_no_fly_zones_sample_floor_m",
        "airspace_no_fly_zones_sample_ceiling_m", "airspace_no_fly_zones_dynamic_count", "mission_runway_required",
        "airspace_runway_length_m", "spawn_xyz_first", "spawn_yaw_deg",
        "landing_preferred_sites_count", "landing_preferred_sites_z_mean", "landing_emergency_sites_count",
        "mission_waypoints_count", "mission_waypoints_z_mean", "mission_time_budget_s",
        "mission_loiter_radius_m", "traffic_count", "traffic_sample_speed_mps",
        "moving_obstacles_count", "moving_obstacles_sample_radius_m", "daa_sep_threshold_m",
        "daa_ttc_threshold_s", "comms_uplink_ok", "comms_downlink_ok",
        "comms_rssi_dbm_min", "comms_loss_windows_count", "faults_count",
        "faults_sample_severity", "faults_sample_duration_s"
    ]


def get_safe_value(feature_name: str) -> float:
    """درع حظر التعبئة التلقائية. يمنع برمجياً فرض أي قيم وهمية خارج الـ DAG."""
    logger.critical(f"Deterministic Protocol Violation: Imputation is strictly disabled for feature '{feature_name}'.")
    raise ValueError(f"Feature '{feature_name}' must be explicitly resolved by the DAG or user. Safe value fallbacks are destroyed.")


def is_critical_value(feature_name: str, value: float) -> bool:
    """فحص ما إذا كانت قيمة الميزة تجاوزت حد الخطر الحرج (Hard Veto Trigger)."""
    if feature_name not in BOUNDS_REGISTRY:
        return False
    
    bounds = BOUNDS_REGISTRY[feature_name]
    critical_high = bounds.get("critical_high")
    
    if critical_high is not None and value >= critical_high:
        logger.warning(f"CRITICAL BREACH: '{feature_name}' = {value} exceeds critical_high threshold {critical_high}")
        return True
    
    return False


def get_feature_definition(feature_name: str) -> FeatureDefinition:
    """بناء الشروحات الفنية الدقيقة والوحدات الفيزيائية حركياً لجميع الـ 198 ميزة دون استثناء."""
    if feature_name not in OFFICIAL_198_FEATURE_ORDER:
        raise KeyError(f"Feature '{feature_name}' is outside the verified 198 model registry.")

    # 1. الاستنباط من القاموس الصريح أولاً لضمان النقاء التشغيلي
    if feature_name in EXPLICIT_UNIT_MAP:
        unit = EXPLICIT_UNIT_MAP[feature_name]
    elif feature_name.endswith("_kg"): unit = "kg"
    elif feature_name.endswith("_wh"): unit = "Wh"
    elif feature_name.endswith("_l"): unit = "L"
    elif feature_name.endswith("_mps"): unit = "m/s"
    elif feature_name.endswith("_deg"): unit = "degree"
    elif feature_name.endswith("_m2"): unit = "m²"
    elif feature_name.endswith("_m"): unit = "meter"
    elif feature_name.endswith("_s"): unit = "second"
    elif feature_name.endswith("_hz") or feature_name.endswith("_frequency"): unit = "Hz"
    elif feature_name.endswith("_dbm") or feature_name.endswith("_dbm_min"): unit = "dBm"
    elif feature_name.endswith("_count") or feature_name.endswith("_steps"): unit = "count"
    elif any(w in feature_name for w in ["ok", "enabled", "required", "multipath", "interference", "_was_missing"]):
        unit = "boolean"
    elif any(w in feature_name for w in ["ratio", "fraction", "efficiency", "density", "redundancy", "health", "risk", "severity"]):
        unit = "ratio/index"
    else:
        unit = "dimensionless"

    # 2. صياغة الشروح النصية الدلالية الموجهة لفهم وسياق الوكيل اللغوي المساعد ReAct
    if feature_name.endswith("_was_missing"):
        description = f"Data integrity metadata flag tracking if the feature asset '{feature_name.replace('_was_missing', '')}' was missing from the operator's telemetry input stream."
    elif "landing_preferred_sites_" in feature_name:
        parts = feature_name.split("_")
        description = f"Statistical {parts[-1]} value of the spatial {parts[-2].upper()} matrix component calculated across preferred landing field coordinates site distribution parameters."
    elif "landing_emergency_sites_" in feature_name:
        parts = feature_name.split("_")
        description = f"Statistical {parts[-1]} value of the spatial {parts[-2].upper()} matrix component calculated across contingency emergency landing spot zone configurations."
    elif "mission_waypoints_" in feature_name and any(w in feature_name for w in ["_mean", "_std", "_min", "_max", "_range"]):
        parts = feature_name.split("_")
        description = f"Statistical {parts[-1]} structural distribution metric for the flight route waypoint coordinate sequence path envelope on the {parts[-2].upper()} spatial matrix dimension."
    elif "comms_loss_windows_" in feature_name:
        parts = feature_name.split("_")
        description = f"Geometric spatial distribution {parts[-1]} boundary marker tracking radio blackout packet drop slots sequence behavior context across the {parts[-2].upper()} terrain flight path layout."
    elif feature_name.startswith("feat_"):
        description = f"High-level composite derived physics indicator mapping multi-variable interaction properties: {feature_name.replace('feat_', '').replace('_', ' ').title()}."
    else:
        description = f"Authoritative aviation system telemetry profile feature parameter tracking operational status element: {feature_name.replace('_', ' ').title()}."

    definition: FeatureDefinition = {
        "name": feature_name,
        "unit": unit,
        "description": description,
        "is_core": feature_name in get_core_features()
    }
    
    if feature_name in BOUNDS_REGISTRY:
        definition.update(BOUNDS_REGISTRY[feature_name])
        
    return definition


def get_all_feature_definitions() -> Dict[str, FeatureDefinition]:
    """تجميع وتصنيف القاموس الكامل لـ 198 ميزة دلالية متسقة بدون أي نقصان أو Placeholders."""
    return {name: get_feature_definition(name) for name in OFFICIAL_198_FEATURE_ORDER}


def get_features_by_category(category: str) -> list[str]:
    """تقسيم الميزات الـ 198 دلالياً لبناء فضاء سياق نظيف وعالي النزاهة لصالح الوكيل المساعد ReAct."""
    cat_clean = category.lower().strip()
    all_features = OFFICIAL_198_FEATURE_ORDER
    
    if cat_clean == "aerodynamic":
        return [f for f in all_features if any(w in f for w in ["aero", "tilt", "speed", "mass", "rotorcraft", "disk_area", "wing", "aspect", "cl_max", "cd0", "stall"])]
    if cat_clean == "environmental":
        return [f for f in all_features if any(w in f for w in ["weather", "wind", "gust", "phenomena", "jam", "multipath", "interference", "thermal", "plumes"])]
    if cat_clean == "battery":
        return [f for f in all_features if any(w in f for w in ["battery", "power", "energy"]) and "fuel" not in f]
    if cat_clean == "mission":
        return [f for f in all_features if any(w in f for w in ["mission", "waypoints", "pattern", "time_budget", "loiter", "controls", "action"])]
    if cat_clean == "gps":
        return [f for f in all_features if any(w in f for w in ["gps", "spawn", "landing", "airspace", "geofence", "runway"])]
    if cat_clean == "comms":
        return [f for f in all_features if any(w in f for w in ["comms", "rssi", "uplink", "downlink", "faults"])]
    if cat_clean == "operator":
        return [f for f in all_features if any(w in f for w in ["operator_", "traffic", "obstacles", "daa_"])]
        
    # الباقي الصافي (مؤشرات الـ missing والأعلام الإحصائية البحتة لـ feat_) يحجز للـ Fallback منعاً لتلوث الحقول التشغيلية
    return [f for f in all_features if "feat_" in f or "_was_missing" in f]


def validate_core_feature_ranges(feature_names: list[str], feature_vector: list[float], strict: bool = True) -> tuple[bool, str]:
    """فحص المتجه عيار 198 والتأكد النيوتني التام من سلامة الحدود الصارمة لبيانات الحظر."""
    if len(feature_names) != len(feature_vector):
        return False, "Feature dimension and vector shape mismatch."
        
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    
    for core in ["uav_mass_kg", "environment_weather_wind_mps", "airspace_altitude_agl_max_m"]:
        if core in name_to_idx:
            idx = name_to_idx[core]
            val = float(feature_vector[idx])
            defn = BOUNDS_REGISTRY[core]
            
            if defn.get("critical_high") is not None and val >= defn["critical_high"]:
                if strict:
                    return False, f"HARD BOUND VIOLATION: {core} = {val} breaks aeronautical constraints limit."
                    
    return True, "All active core constraints cleared successfully."


def validate_feature_registry_against_artifact(feature_names_from_bundle: List[str]) -> Tuple[bool, str]:
    """إبطال مفعول القنبلة الموقوتة: فحص جنائي يكسر المنظومة فوراً إذا اختلف ترتيب الكود عن ملف الـ pkl الثنائي."""
    if len(feature_names_from_bundle) != len(OFFICIAL_198_FEATURE_ORDER):
        return False, f"Dimension Mismatch: Model expects {len(feature_names_from_bundle)} columns, code locks 198."
        
    for idx, (bundle_name, code_name) in enumerate(zip(feature_names_from_bundle, OFFICIAL_198_FEATURE_ORDER)):
        if bundle_name != code_name:
            return False, f"Index Shift Blocked at node [{idx}]: Binary has '{bundle_name}' vs Code '{code_name}'."
            
    return True, "Registry is strictly aligned with ML binary topology."

# =====================================================================
# Architectural Registry Block:
# This file serves as the Single Source of Truth for features constitution metadata.
# This file depends on: None (Unified Packaged Constitution Layer).
# Files depending on this file: src/uav_risk/core/data_validator.py, src/uav_risk/core/feature_router.py, src/uav_risk/ml/loader.py
# =====================================================================