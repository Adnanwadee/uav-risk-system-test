from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping

from uav_risk.ml import feature_defs


RAW_FEATURE_NAMES: tuple[str, ...] = (
    "sim_duration_steps", "sim_policy_frequency", "uav_type", "uav_mass_kg",
    "uav_battery_wh", "uav_fuel_l", "uav_energy_source", "uav_max_speed_mps",
    "uav_max_tilt_deg", "uav_reserve_fraction", "uav_battery_model_hover_power_w",
    "uav_battery_model_k_drag", "uav_battery_model_k_manoeuvre", "uav_rotorcraft_rotor_count",
    "uav_rotorcraft_disk_area_m2", "uav_sensors_gnss", "uav_sensors_lidar",
    "uav_sensors_radar", "uav_sensors_camera_rgb", "uav_sensors_camera_thermal",
    "uav_payload_mass_kg", "uav_payload_drag_coeff", "environment_airspace",
    "environment_weather_wind_mps", "environment_weather_wind_dir_deg",
    "environment_weather_gust_mps", "environment_weather_visibility",
    "environment_weather_phenomena_count", "environment_weather_phenomena_first",
    "airspace_altitude_agl_min_m", "airspace_altitude_agl_max_m",
    "airspace_no_fly_zones_count", "airspace_no_fly_zones_sample_radius_m",
    "airspace_no_fly_zones_sample_floor_m", "airspace_no_fly_zones_sample_ceiling_m",
    "spawn_xyz_first", "spawn_yaw_deg", "landing_preferred_sites_count",
    "landing_preferred_sites_x_mean", "landing_preferred_sites_x_std",
    "landing_preferred_sites_x_min", "landing_preferred_sites_x_max",
    "landing_preferred_sites_x_range", "landing_preferred_sites_y_mean",
    "landing_preferred_sites_y_std", "landing_preferred_sites_y_min",
    "landing_preferred_sites_y_max", "landing_preferred_sites_y_range",
    "landing_preferred_sites_z_mean", "landing_preferred_sites_z_std",
    "landing_preferred_sites_z_min", "landing_preferred_sites_z_max",
    "landing_preferred_sites_z_range", "landing_emergency_sites_count",
    "landing_emergency_sites_x_mean", "landing_emergency_sites_x_std",
    "landing_emergency_sites_x_min", "landing_emergency_sites_x_max",
    "landing_emergency_sites_x_range", "landing_emergency_sites_y_mean",
    "landing_emergency_sites_y_std", "landing_emergency_sites_y_min",
    "landing_emergency_sites_y_max", "landing_emergency_sites_y_range",
    "landing_emergency_sites_z_mean", "landing_emergency_sites_z_std",
    "landing_emergency_sites_z_min", "landing_emergency_sites_z_max",
    "landing_emergency_sites_z_range", "mission_type", "mission_waypoints_count",
    "mission_waypoints_x_mean", "mission_waypoints_x_std", "mission_waypoints_x_min",
    "mission_waypoints_x_max", "mission_waypoints_x_range", "mission_waypoints_y_mean",
    "mission_waypoints_y_std", "mission_waypoints_y_min", "mission_waypoints_y_max",
    "mission_waypoints_y_range", "mission_waypoints_z_mean", "mission_waypoints_z_std",
    "mission_waypoints_z_min", "mission_waypoints_z_max", "mission_waypoints_z_range",
    "mission_pattern", "mission_time_budget_s", "mission_runway_required", "traffic_count",
    "traffic_sample_speed_mps", "traffic_sample_heading_deg", "moving_obstacles_count",
    "moving_obstacles_sample_radius_m", "swarm_enabled", "controls_mode",
    "controls_actions_count", "controls_actions_first", "daa_sep_threshold_m",
    "daa_ttc_threshold_s", "faults_count", "faults_sample_t_s", "faults_sample_type",
    "faults_sample_duration_s", "faults_sample_severity", "comms_uplink_ok",
    "comms_downlink_ok", "comms_loss_windows_count", "comms_loss_windows_x_mean",
    "comms_loss_windows_x_std", "comms_loss_windows_x_min", "comms_loss_windows_x_max",
    "comms_loss_windows_x_range", "comms_loss_windows_y_mean", "comms_loss_windows_y_std",
    "comms_loss_windows_y_min", "comms_loss_windows_y_max", "comms_loss_windows_y_range",
    "comms_rssi_dbm_min", "airspace_no_fly_zones_dynamic_count",
    "airspace_no_fly_zones_dynamic_sample_radius_m",
    "airspace_no_fly_zones_dynamic_sample_floor_m",
    "airspace_no_fly_zones_dynamic_sample_ceiling_m", "autofix_uav_physics_count",
    "autofix_uav_physics_first", "uav_aero_wing_area_m2", "uav_aero_aspect_ratio",
    "uav_aero_cl_max", "uav_aero_cd0", "uav_aero_prop_efficiency",
    "uav_aero_stall_speed_mps", "airspace_runway_threshold_count",
    "airspace_runway_threshold_first", "airspace_runway_heading_deg", "airspace_runway_length_m",
    "environment_wind_profile_count", "environment_wind_profile_sample_alt_m",
    "environment_wind_profile_sample_wind_mps", "environment_wind_profile_sample_dir_deg",
    "environment_thermal_plumes_count", "environment_thermal_plumes_sample_radius_m",
    "environment_thermal_plumes_sample_w_up_mps", "environment_gnss_multipath",
    "environment_gnss_jam_dbm", "environment_em_interference",
    "mission_transition_profile_vtol_to_ff_t_s", "mission_transition_profile_ff_to_vtol_t_s",
    "swarm_size", "swarm_roles_count", "swarm_roles_first", "swarm_inter_uav_sep_min_m",
    "uav_rotorcraft_max_climb_mps", "uav_rotorcraft_hover_ceiling_m", "mission_loiter_radius_m",
    "airspace__geofence__sample__points_count", "airspace__no__fly__zones__sample__center_count",
    "landing__preferred__sites__sample_count", "landing__emergency__sites__sample_count",
    "mission__waypoints__sample_count", "traffic__sample__spawn_count",
    "moving__obstacles__sample__center_count", "moving__obstacles__sample__vel_count",
    "comms__loss__windows__sample_count", "airspace__no__fly__zones__dynamic__sampl_count",
    "environment__thermal__plumes__sample__ce_count", "feat_disk_loading", "feat_altitude_range",
    "feat_reserve_utilization", "feat_wind_gust_ratio", "feat_wind_speed_ratio",
    "feat_sensor_redundancy", "feat_comms_health", "feat_traffic_density", "feat_fault_risk",
    "feat_weather_severity", "uav_rotorcraft_rotor_count_was_missing",
    "autofix_uav_physics_count_was_missing", "autofix_uav_physics_first_was_missing",
    "uav_aero_wing_area_m2_was_missing", "uav_aero_aspect_ratio_was_missing",
    "uav_aero_cl_max_was_missing", "uav_aero_cd0_was_missing",
    "uav_aero_prop_efficiency_was_missing", "uav_aero_stall_speed_mps_was_missing",
    "airspace_runway_threshold_count_was_missing", "airspace_runway_threshold_first_was_missing",
    "airspace_runway_heading_deg_was_missing", "airspace_runway_length_m_was_missing",
    "mission_transition_profile_vtol_to_ff_t_s_was_missing",
    "mission_transition_profile_ff_to_vtol_t_s_was_missing", "swarm_size_was_missing",
    "swarm_roles_count_was_missing", "swarm_roles_first_was_missing",
    "swarm_inter_uav_sep_min_m_was_missing", "uav_rotorcraft_max_climb_mps_was_missing",
    "uav_rotorcraft_hover_ceiling_m_was_missing", "mission_loiter_radius_m_was_missing",
)

PROCESSED_FEATURE_NAMES: tuple[str, ...] = tuple(feature_defs.get_all_feature_names())

RAW_CATEGORICAL_FEATURES: dict[str, tuple[str, ...]] = {
    "uav_energy_source": ("battery", "fuel", "hybrid"),
    "mission_pattern": ("corridor", "custom", "grid", "orbit", "spiral"),
    "controls_mode": ("continuous", "discrete"),
    "controls_actions_first": ("fwd", "hold", "throttle"),
    "swarm_roles_first": ("leader", "relay", "scout", "single", "solo"),
}

FORBIDDEN_USER_FEATURES: tuple[str, ...] = (
    "uav_energy_source_fuel", "uav_energy_source_hybrid", "mission_pattern_custom",
    "mission_pattern_grid", "mission_pattern_orbit", "mission_pattern_spiral",
    "controls_mode_discrete", "controls_actions_first_hold", "controls_actions_first_throttle",
    "swarm_roles_first_relay", "swarm_roles_first_scout", "swarm_roles_first_single",
    "swarm_roles_first_solo",
)

PROCESSED_ONEHOT_FEATURES = FORBIDDEN_USER_FEATURES

DROPPED_RAW_METADATA_FEATURES: tuple[str, ...] = (
    "uav_type", "environment_airspace", "environment_weather_visibility",
    "environment_weather_phenomena_first", "mission_type", "faults_sample_type",
    "autofix_uav_physics_first",
)

PROFILE_IDENTITY_FIELDS: tuple[str, ...] = ("user_id", "profile_id", "profile_name")

PROFILE_DERIVED_RAW_FEATURES: tuple[str, ...] = (
    "uav_mass_kg", "uav_battery_wh", "uav_fuel_l", "uav_energy_source",
    "uav_max_speed_mps", "uav_max_tilt_deg", "uav_reserve_fraction",
    "uav_rotorcraft_rotor_count", "uav_sensors_gnss", "uav_sensors_lidar",
    "uav_sensors_radar", "uav_sensors_camera_rgb", "uav_sensors_camera_thermal",
    "uav_aero_prop_efficiency", "uav_rotorcraft_max_climb_mps",
    "uav_rotorcraft_hover_ceiling_m",
)



PROFILE_CAPABILITY_FIELDS: tuple[str, ...] = (
    "max_payload_kg", "max_takeoff_mass_kg", "runway_capable",
    "swarm_capable", "max_swarm_size",
)

SCENARIO_REQUIRED_RAW_FEATURES: tuple[str, ...] = (
    "uav_payload_mass_kg", "environment_weather_wind_mps", "environment_weather_wind_dir_deg",
    "environment_weather_gust_mps", "environment_weather_phenomena_count",
    "environment_gnss_multipath", "environment_gnss_jam_dbm", "environment_em_interference",
    "airspace_altitude_agl_min_m", "airspace_altitude_agl_max_m",
    "airspace_no_fly_zones_count", "airspace_no_fly_zones_sample_radius_m",
    "airspace_no_fly_zones_sample_floor_m", "airspace_no_fly_zones_sample_ceiling_m",
    "airspace_no_fly_zones_dynamic_count", "airspace_runway_length_m", "spawn_xyz_first",
    "spawn_yaw_deg", "landing_preferred_sites_count", "landing_preferred_sites_z_mean",
    "landing_emergency_sites_count", "mission_waypoints_count", "mission_waypoints_z_mean",
    "mission_pattern", "mission_time_budget_s", "mission_runway_required", "mission_loiter_radius_m",
    "traffic_count", "traffic_sample_speed_mps", "moving_obstacles_count",
    "moving_obstacles_sample_radius_m", "swarm_enabled", "swarm_size", "swarm_roles_first",
    "swarm_inter_uav_sep_min_m", "controls_mode", "daa_sep_threshold_m", "daa_ttc_threshold_s",
    "faults_count", "faults_sample_duration_s", "faults_sample_severity",
    "comms_uplink_ok", "comms_downlink_ok", "comms_loss_windows_count", "comms_rssi_dbm_min",
)

_PARTITIONED_RAW_FEATURES = (
    set(PROFILE_DERIVED_RAW_FEATURES)
    | set(SCENARIO_REQUIRED_RAW_FEATURES)
    | set(DROPPED_RAW_METADATA_FEATURES)
)

GENERATED_RAW_FEATURES: tuple[str, ...] = tuple(
    name for name in RAW_FEATURE_NAMES if name not in _PARTITIONED_RAW_FEATURES
)

INTERNAL_ONLY_RAW_FEATURES: tuple[str, ...] = tuple(
    name
    for name in GENERATED_RAW_FEATURES
    if name.endswith("_was_missing")
    or name
    in {
        "sim_policy_frequency",
        "autofix_uav_physics_count",
        "controls_actions_count",
        "airspace__geofence__sample__points_count",
        "airspace__no__fly__zones__sample__center_count",
        "landing__preferred__sites__sample_count",
        "landing__emergency__sites__sample_count",
        "mission__waypoints__sample_count",
        "traffic__sample__spawn_count",
        "moving__obstacles__sample__center_count",
        "moving__obstacles__sample__vel_count",
        "comms__loss__windows__sample_count",
        "airspace__no__fly__zones__dynamic__sampl_count",
        "environment__thermal__plumes__sample__ce_count",
    }
)

OPTIONAL_RAW_OVERRIDE_FEATURES: tuple[str, ...] = tuple(
    name for name in GENERATED_RAW_FEATURES if name not in set(INTERNAL_ONLY_RAW_FEATURES)
)


DROPPED_RAW_DEFAULTS: dict[str, Any] = {
    "uav_type": "unknown",
    "environment_airspace": "unknown",
    "environment_weather_visibility": 0.0,
    "environment_weather_phenomena_first": "none",
    "mission_type": "unknown",
    "faults_sample_type": "none",
    "autofix_uav_physics_first": 0.0,
}


@dataclass(frozen=True)
class RawSchema:
    raw_feature_names: list[str]
    processed_feature_names: list[str]
    processed_onehot_feature_names: list[str]
    onehot_input_columns: list[str]
    onehot_categories: dict[str, list[Any]]


def get_raw_feature_names(bundle: Any) -> list[str]:
    preprocessor = getattr(bundle, "preprocessor", None)
    if preprocessor is None or not hasattr(preprocessor, "feature_names_in_"):
        raise ValueError("Stage-1 bundle is missing a fitted preprocessor with feature_names_in_.")
    return [str(name) for name in preprocessor.feature_names_in_]


def get_processed_feature_names(bundle: Any) -> list[str]:
    return [str(name) for name in getattr(bundle, "feature_names")]


def get_onehot_categories(bundle: Any) -> dict[str, list[Any]]:
    preprocessor = getattr(bundle, "preprocessor", None)
    if preprocessor is None:
        raise ValueError("Stage-1 bundle is missing a fitted preprocessor.")

    for name, transformer, columns in getattr(preprocessor, "transformers_", []):
        if name != "onehot":
            continue
        return {
            str(column): list(categories)
            for column, categories in zip(columns, transformer.categories_)
        }
    return {}


def get_raw_schema(bundle: Any) -> RawSchema:
    categories = get_onehot_categories(bundle)
    return RawSchema(
        raw_feature_names=get_raw_feature_names(bundle),
        processed_feature_names=get_processed_feature_names(bundle),
        processed_onehot_feature_names=list(PROCESSED_ONEHOT_FEATURES),
        onehot_input_columns=list(categories.keys()),
        onehot_categories=categories,
    )


def reject_processed_onehot_inputs(input_mapping: Mapping[str, Any]) -> None:
    forbidden = [name for name in PROCESSED_ONEHOT_FEATURES if name in input_mapping]
    if forbidden:
        raise ValueError(
            "Processed one-hot model features are not accepted as user inputs: "
            + ", ".join(forbidden)
        )


def categorical_to_processed_flags(
    input_mapping: Mapping[str, Any],
    bundle: Any,
) -> dict[str, float]:
    """Legacy compatibility helper for the processed/mixed 68-feature bridge."""
    categories_by_column = get_onehot_categories(bundle)
    flags: dict[str, float] = {}

    for raw_column, categories in categories_by_column.items():
        if raw_column not in input_mapping or input_mapping[raw_column] is None:
            if raw_column == "controls_actions_first":
                continue
            raise ValueError(f"Missing required categorical raw feature: {raw_column}")

        value = str(input_mapping[raw_column]).strip().lower()
        normalized_categories = [str(category).strip().lower() for category in categories]
        if value not in normalized_categories:
            allowed = ", ".join(str(category) for category in categories)
            raise ValueError(
                f"Invalid category for {raw_column}: {input_mapping[raw_column]!r}. "
                f"Allowed values: {allowed}"
            )

        canonical_value = str(categories[normalized_categories.index(value)])
        for category in categories[1:]:
            processed_name = f"{raw_column}_{category}"
            flags[processed_name] = 1.0 if str(category) == canonical_value else 0.0

    return flags


def infer_controls_action(raw_mapping: Mapping[str, Any], processed_map: Mapping[str, Any]) -> str:
    """Legacy compatibility helper for reconstructing raw controls_actions_first."""
    if raw_mapping.get("controls_actions_first") is not None:
        return str(raw_mapping["controls_actions_first"])
    if float(processed_map.get("controls_actions_first_hold", 0.0)) == 1.0:
        return "hold"
    if float(processed_map.get("controls_actions_first_throttle", 0.0)) == 1.0:
        return "throttle"
    return "fwd"


def build_raw_feature_map(
    input_mapping: Mapping[str, Any],
    processed_physical_map: Mapping[str, Any],
    bundle: Any,
) -> dict[str, Any]:
    """Legacy compatibility only. Reconstruct raw 197 from processed/mixed bridge output."""
    raw_feature_names = get_raw_feature_names(bundle)
    onehot_columns = set(get_onehot_categories(bundle))
    raw_map: dict[str, Any] = {}

    for name in raw_feature_names:
        if name in onehot_columns:
            if name == "controls_actions_first":
                raw_map[name] = infer_controls_action(input_mapping, processed_physical_map)
            elif name in input_mapping and input_mapping[name] is not None:
                raw_map[name] = input_mapping[name]
            else:
                raise ValueError(f"Missing required categorical raw feature: {name}")
        elif name in processed_physical_map:
            raw_map[name] = processed_physical_map[name]
        elif name in input_mapping and input_mapping[name] is not None:
            raw_map[name] = input_mapping[name]
        elif name in DROPPED_RAW_DEFAULTS:
            raw_map[name] = DROPPED_RAW_DEFAULTS[name]
        else:
            raise KeyError(f"Raw preprocessor feature '{name}' was not generated or provided.")

    return raw_map


def get_secondary_raw_override_names(bundle: Any, required_names: Iterable[str]) -> list[str]:
    required = set(required_names)
    onehot = set(get_onehot_categories(bundle))
    return [
        name
        for name in get_raw_feature_names(bundle)
        if name not in required and name not in onehot
    ]
