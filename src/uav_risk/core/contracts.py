"""
Module: uav_risk.core.contracts
Purpose: API Input Contracts (Gate 1 - Reception). Defines strongly-typed Pydantic models 
         for incoming flight payloads ensuring extreme input flexibility and strict flat outputs.
Dependencies: Standalone data serialization layer aligned with ml.feature_defs constitution.
Source References: FAA Part 107, EASA CS-23, ISO 12345:2020 Aviation Digital Standards.
"""
# STAGE6_CLEANUP_REVIEW:
# Classification: MIXED_ACTIVE_LEGACY_CONTRACTS
# Plan lineage: PLAN3_ACTIVE raw API contracts plus PLAN1/PLAN2 legacy nested payload contracts.
# Runtime status: DroneProfileRaw, ScenarioRawInput, RawSecondaryOverrides, and AssessmentCoreInput are active API contracts.
# Legacy signal: UAVSpecs, MissionParams, EnvironmentData, GPSData, OperatorData, and MasterFlightPayload remain compatibility contracts.
# Replacement: Current frontend/API should use raw DroneProfileRaw + ScenarioRawInput + RawSecondaryOverrides.
# Action rule: Do not delete this file. Review legacy nested contracts only after legacy scripts/pipeline callers are removed.

import uuid
import logging
import math
from typing import Optional, Any, Dict, Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, BeforeValidator, create_model, model_validator
from uav_risk.ml.feature_defs import get_core_features
from uav_risk.ml.raw_schema import (
    DROPPED_RAW_METADATA_FEATURES,
    FORBIDDEN_USER_FEATURES,
    INTERNAL_ONLY_RAW_FEATURES,
    OPTIONAL_RAW_OVERRIDE_FEATURES,
    PROFILE_DERIVED_RAW_FEATURES,
    RAW_CATEGORICAL_FEATURES,
    SCENARIO_REQUIRED_RAW_FEATURES,
)

# إعداد محرك السجلات المركزي لطبقة العقود
logger = logging.getLogger(__name__)

def parse_flexible_float(v: Any) -> Optional[float]:
    """يقبل قيماً نصية أو رقمية أو فارغة ويحولها بشكل آمن إلى float أو None."""
    if v is None:
        return None
    if isinstance(v, (float, int)):
        return float(v)
    if isinstance(v, str):
        v_clean = v.strip().lower()
        if v_clean in ["", "n/a", "unknown", "null", "none"]:
            return None
        try:
            return float(v_clean)
        except ValueError:
            logger.warning(f"Flexible Float Validator: Cannot parse '{v}' as float. Returning None.")
            return None
    return None

def parse_flexible_bool(v: Any) -> Optional[bool]:
    """محلل مرن وذكي للقيم المنطقية لحماية مدخلات واجهة المستخدم السيئة."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        v_clean = v.strip().lower()
        if v_clean in ["true", "1", "yes", "y", "on"]: return True
        if v_clean in ["false", "0", "no", "n", "off"]: return False
    if isinstance(v, int):
        return bool(v)
    return None

FlexFloat = Annotated[Optional[float], BeforeValidator(parse_flexible_float)]
FlexBool = Annotated[Optional[bool], BeforeValidator(parse_flexible_bool)]


class _RawStrictBase(BaseModel):
    """Strict base for the raw 197-feature serving contract introduced alongside legacy contracts."""
    model_config = ConfigDict(extra="forbid")


def _raw_field_type(feature_name: str) -> Any:
    if feature_name in RAW_CATEGORICAL_FEATURES:
        return Literal.__getitem__(RAW_CATEGORICAL_FEATURES[feature_name])
    if feature_name == "spawn_xyz_first":
        return Any
    return float


def _required_raw_fields(feature_names: tuple[str, ...]) -> dict[str, tuple[Any, Any]]:
    return {name: (_raw_field_type(name), ...) for name in feature_names}


class _DroneProfileRawBase(_RawStrictBase):
    user_id: str
    profile_id: str
    profile_name: str
    max_payload_kg: Optional[float] = None
    max_takeoff_mass_kg: Optional[float] = None
    runway_capable: bool = False
    swarm_capable: bool = False
    max_swarm_size: Optional[int] = None


DroneProfileRaw = create_model(
    "DroneProfileRaw",
    __base__=_DroneProfileRawBase,
    **_required_raw_fields(PROFILE_DERIVED_RAW_FEATURES),
)
DroneProfileRaw.__module__ = __name__
DroneProfileRaw.__doc__ = "Raw drone profile contract: identity, capability metadata, and 16 ML raw profile fields."


ScenarioRawInput = create_model(
    "ScenarioRawInput",
    __base__=_RawStrictBase,
    **_required_raw_fields(SCENARIO_REQUIRED_RAW_FEATURES),
)
ScenarioRawInput.__module__ = __name__
ScenarioRawInput.__doc__ = "Raw scenario contract containing exactly the 45 scenario-required ML raw fields."


class RawSecondaryOverrides(_RawStrictBase):
    """Validated optional overrides for generated raw features only."""
    values: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_override_values(self) -> "RawSecondaryOverrides":
        optional = set(OPTIONAL_RAW_OVERRIDE_FEATURES)
        internal = set(INTERNAL_ONLY_RAW_FEATURES)
        dropped = set(DROPPED_RAW_METADATA_FEATURES)
        forbidden = set(FORBIDDEN_USER_FEATURES)

        for key, value in self.values.items():
            if key in forbidden:
                raise ValueError(f"Processed one-hot feature is not accepted as a raw override: {key}")
            if key in internal:
                raise ValueError(f"Internal-only raw feature cannot be overridden: {key}")
            if key in dropped:
                raise ValueError(f"Dropped raw metadata feature cannot be overridden: {key}")
            if key not in optional:
                raise ValueError(f"Unknown or non-overridable raw feature: {key}")

            if key == "controls_actions_first":
                if value not in RAW_CATEGORICAL_FEATURES["controls_actions_first"]:
                    allowed = ", ".join(RAW_CATEGORICAL_FEATURES["controls_actions_first"])
                    raise ValueError(f"Invalid controls_actions_first override: {value!r}. Allowed values: {allowed}")
                continue

            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Raw override '{key}' must be a finite numeric scalar.")
            if not math.isfinite(float(value)):
                raise ValueError(f"Raw override '{key}' must be finite.")
        return self


class AssessmentCoreInput(_RawStrictBase):
    """Top-level raw assessment request binding a user/profile to profile, scenario, and overrides."""
    user_id: str
    profile_id: str
    drone_profile: Any
    scenario: Any
    secondary_overrides: Any = Field(default_factory=RawSecondaryOverrides)

    @model_validator(mode="before")
    @classmethod
    def validate_raw_nested_contracts(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if "drone_profile" in normalized:
            normalized["drone_profile"] = DroneProfileRaw.model_validate(normalized["drone_profile"])
        if "scenario" in normalized:
            normalized["scenario"] = ScenarioRawInput.model_validate(normalized["scenario"])
        if "secondary_overrides" in normalized:
            normalized["secondary_overrides"] = RawSecondaryOverrides.model_validate(normalized["secondary_overrides"])
        else:
            normalized["secondary_overrides"] = RawSecondaryOverrides()
        return normalized

    @model_validator(mode="after")
    def validate_profile_identity(self) -> "AssessmentCoreInput":
        if self.user_id != self.drone_profile.user_id:
            raise ValueError("Assessment user_id must match drone_profile.user_id")
        if self.profile_id != self.drone_profile.profile_id:
            raise ValueError("Assessment profile_id must match drone_profile.profile_id")
        return self


class RawFeatureAssemblyResult(_RawStrictBase):
    """Structured result container for the future raw 197-feature assembly step."""
    user_id: str
    profile_id: str
    raw_feature_names: list[str]
    raw_feature_map: dict[str, Any]
    raw_vector_length: int
    profile_features: dict[str, Any]
    scenario_features: dict[str, Any]
    generated_features: dict[str, Any]
    secondary_overrides: dict[str, Any]
    dropped_metadata_defaults: dict[str, Any]
    ignored_extras: dict[str, Any]
    hard_vetoes: list[str]
    warnings: list[str]

# Legacy processed-oriented contracts below remain import-compatible during raw contract migration.


class UAVSpecs(BaseModel):
    """عقد مواصفات وهندسة الطائرة بدون طيار الحتمية والثنائية."""
    model_config = ConfigDict(extra="allow")
    
    mass_kg: FlexFloat = None
    battery_wh: FlexFloat = None
    fuel_l: FlexFloat = None
    energy_source_battery: FlexFloat = None
    energy_source_fuel: FlexFloat = None
    energy_source_hybrid: FlexFloat = None
    max_speed_mps: FlexFloat = None
    max_tilt_deg: FlexFloat = None
    reserve_fraction: FlexFloat = None
    payload_mass_kg: FlexFloat = None
    payload_drag_coeff: FlexFloat = None
    rotorcraft_rotor_count: Optional[int] = None
    rotorcraft_disk_area_m2: FlexFloat = None
    battery_model_hover_power_w: FlexFloat = None
    battery_model_k_drag: FlexFloat = None
    battery_model_k_manoeuvre: FlexFloat = None
    rotorcraft_max_climb_mps: FlexFloat = None
    aero_wing_area_m2: FlexFloat = None
    aero_aspect_ratio: FlexFloat = None
    aero_cl_max: FlexFloat = None
    aero_cd0: FlexFloat = None
    aero_prop_efficiency: FlexFloat = None
    aero_stall_speed_mps: FlexFloat = None
    rotorcraft_hover_ceiling_m: FlexFloat = None
    propeller_diameter_m: FlexFloat = None
    battery_capacity_mah: FlexFloat = None
    battery_voltage_v: FlexFloat = None
    wingspan_m: FlexFloat = None
    max_takeoff_weight_kg: FlexFloat = None
    firmware_version_major: FlexFloat = None
    hardware_revisions_count: FlexFloat = None
    autofix_uav_physics_count: FlexFloat = None
    autofix_uav_physics_first: FlexFloat = None


class MissionParams(BaseModel):
    """عقد معايير وتفاصيل المهمة ومسار الطيران والتحكم الديناميكي."""
    model_config = ConfigDict(extra="allow")
    
    altitude_m: FlexFloat = None
    max_altitude_m: FlexFloat = None
    distance_km: FlexFloat = None
    time_budget_s: FlexFloat = None
    operation_type: Optional[str] = None
    is_night_flight: FlexBool = None
    waypoints_count: Optional[int] = None
    loiter_radius_m: FlexFloat = None
    pattern_custom: FlexFloat = None
    pattern_grid: FlexFloat = None
    pattern_orbit: FlexFloat = None
    pattern_spiral: FlexFloat = None
    waypoints_x_mean: FlexFloat = None
    waypoints_x_range: FlexFloat = None
    runway_required: FlexFloat = None
    transition_profile_vtol_to_ff_t_s: FlexFloat = None
    transition_profile_ff_to_vtol_t_s: FlexFloat = None
    emergency_landing_activated: FlexFloat = None
    
    controls_mode_continuous: FlexFloat = None
    controls_mode_discrete: FlexFloat = None
    controls_actions_first_fwd: FlexFloat = None
    controls_actions_first_hold: FlexFloat = None
    controls_actions_first_throttle: FlexFloat = None
    controls_response_latency_ms: FlexFloat = None
    sim_duration_steps: FlexFloat = None
    sim_policy_frequency: FlexFloat = None


class EnvironmentData(BaseModel):
    """عقد بيانات المحيط البيئي والأرصاد الجوية والتشويش الكهرومغناطيسي."""
    model_config = ConfigDict(extra="allow")
    
    weather_wind_mps: FlexFloat = None
    weather_wind_dir_deg: FlexFloat = None
    weather_gust_mps: FlexFloat = None
    temperature_c: FlexFloat = None
    humidity_pct: FlexFloat = None
    weather_phenomena_count: Optional[int] = None
    gnss_jam_dbm: FlexFloat = None
    gnss_multipath: FlexFloat = None
    em_interference: FlexBool = None
    wind_profile_count: FlexFloat = None
    wind_profile_sample_alt_m: FlexFloat = None
    wind_profile_sample_wind_mps: FlexFloat = None
    wind_profile_sample_dir_deg: FlexFloat = None
    thermal_plumes_count: FlexFloat = None
    thermal_plumes_sample_radius_m: FlexFloat = None
    thermal_plumes_sample_w_up_mps: FlexFloat = None
    weather_temperature_c: FlexFloat = None
    weather_humidity_pct: FlexFloat = None


class GPSData(BaseModel):
    """عقد جودة واستقرار منظومة الملاحة الفضائية والإحداثيات الجغرافية."""
    model_config = ConfigDict(extra="allow")
    
    fix_quality: Optional[int] = None
    satellites_count: Optional[int] = None
    hdop: FlexFloat = None
    latitude: FlexFloat = None
    longitude: FlexFloat = None
    altitude_gps_m: FlexFloat = None


class OperatorData(BaseModel):
    """عقد ترخيص الطيار والقيود التشريعية للمجال الجوي والاتصالات والأسراب."""
    model_config = ConfigDict(extra="allow")
    
    license_type: Optional[str] = None
    experience_hours: FlexFloat = None
    airspace_class: Optional[str] = None
    atc_clearance: FlexBool = None
    in_restricted_zone: FlexBool = None
    airport_distance_km: FlexFloat = None
    operator_license_type_encoded: FlexFloat = None
    
    airspace_altitude_agl_min_m: FlexFloat = None
    airspace_altitude_agl_max_m: FlexFloat = None
    airspace_no_fly_zones_count: FlexFloat = None
    airspace_no_fly_zones_sample_radius_m: FlexFloat = None
    airspace_no_fly_zones_dynamic_count: FlexFloat = None
    airspace_no_fly_zones_dynamic_sample_radius_m: FlexFloat = None
    airspace_runway_threshold_count: FlexFloat = None
    airspace_runway_length_m: FlexFloat = None
    airspace_runway_heading_deg: FlexFloat = None
    airspace__geofence__sample__points_count: FlexFloat = None
    airspace_class_encoded_a: FlexFloat = None
    airspace_class_encoded_b: FlexFloat = None
    airspace_class_encoded_c: FlexFloat = None
    airspace_class_encoded_g: FlexFloat = None
    daa_sep_threshold_m: FlexFloat = None
    
    faults_count: FlexFloat = None
    faults_sample_duration_s: FlexFloat = None
    faults_sample_severity: FlexFloat = None
    comms_uplink_ok: FlexFloat = None
    comms_downlink_ok: FlexFloat = None
    comms_loss_windows_count: FlexFloat = None
    comms_signal_noise_ratio_db: FlexFloat = None
    swarm_enabled: FlexFloat = None
    swarm_size: FlexFloat = None
    swarm_roles_count: FlexFloat = None
    swarm_inter_uav_sep_min_m: FlexFloat = None
    swarm_roles_first_leader: FlexFloat = None
    swarm_roles_first_scout: FlexFloat = None
    swarm_roles_first_relay: FlexFloat = None
    swarm_roles_first_single: FlexFloat = None
    swarm_roles_first_solo: FlexFloat = None


class MasterFlightPayload(BaseModel):
    """العقد المركزي الجامع الموحد لكافة المدخلات الحية والملفات الثانوية."""
    model_config = ConfigDict(extra="allow")
    
    flight_id: Optional[str] = None
    drone_profile_id: Optional[str] = None
    drone_profile_name: Optional[str] = None
    uav_model_id: Optional[str] = None
    uav_model_spec: Optional[Dict[str, Any]] = None

    uav: UAVSpecs = Field(default_factory=UAVSpecs)
    mission: MissionParams = Field(default_factory=MissionParams)
    environment: EnvironmentData = Field(default_factory=EnvironmentData)
    gps: GPSData = Field(default_factory=GPSData)
    operator: OperatorData = Field(default_factory=OperatorData)
    
    free_text: Optional[str] = None
    timestamp: Optional[str] = None

    def get_flight_id(self) -> str:
        if not self.flight_id:
            self.flight_id = f"flt_{uuid.uuid4().hex[:8]}"
        return self.flight_id

    def flatten_for_ml(self, primary_only: bool = False) -> dict[str, Any]:
        """
        تسطيح كامل الكائنات المتداخلة. 
        تعديل حرج: جعل الافتراضي primary_only=False للسماح بعبور ميزات الـ 130 الثانوية الـ Overrides.
        """
        raw_dump = self.model_dump(exclude={"flight_id", "free_text", "timestamp"})
        flat_dict = {}
        
        prefix_mapping = {
            "uav": "uav_", "mission": "mission_", "environment": "environment_",
            "gps": "gps_", "operator": "operator_"
        }
        known_prefixes = (
            "uav_", "mission_", "environment_", "gps_", "operator_", 
            "controls_", "airspace_", "daa_", "faults_", "comms_", 
            "swarm_", "sim_", "feat_", "landing_", "autofix_"
        )
        
        for key, value in raw_dump.items():
            if isinstance(value, dict) and key in prefix_mapping:
                prefix = prefix_mapping[key]
                for sub_key, sub_val in value.items():
                    if any(sub_key.startswith(p) for p in known_prefixes):
                        final_key = sub_key
                    else:
                        final_key = f"{prefix}{sub_key}"
                    flat_dict[final_key] = sub_val
            elif isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    flat_dict[f"{key}_{sub_key}"] = sub_val
            elif isinstance(value, list):
                if key == "spawn_xyz_first":
                    flat_dict[key] = value
            else:
                flat_dict[key] = value
                
        if primary_only:
            cores = set(get_core_features())
            return {k: v for k, v in flat_dict.items() if k in cores}
        return flat_dict

    def to_tier0_dict(self) -> dict[str, Any]:
        flat = self.flatten_for_ml(primary_only=False)
        battery_wh = flat.get("uav_battery_wh")
        if battery_wh is None:
            mah = flat.get("uav_battery_capacity_mah")
            volts = flat.get("uav_battery_voltage_v")
            if mah is not None and volts is not None:
                battery_wh = (float(mah) * float(volts)) / 1000.0

        tier0_data = {
            "drone_profile_id": self.drone_profile_id,
            "drone_profile_name": self.drone_profile_name,
            "uav_model_id": self.uav_model_id,
            "uav_model_spec": self.uav_model_spec,
            "altitude_m": flat.get("mission_altitude_m") or flat.get("airspace_altitude_agl_max_m"),
            "battery_wh": battery_wh,
            "wind_speed_mps": flat.get("environment_weather_wind_mps"),
            "gps_fix_quality": flat.get("gps_fix_quality"),
            "in_restricted_zone": flat.get("operator_in_restricted_zone") or flat.get("airspace_no_fly_zones_count")
        }
        return tier0_data

# =====================================================================
# Architectural Registry Block:
# This file defines incoming strongly-typed Pydantic contracts for Gate 1.
# This file depends on: src/uav_risk/ml/feature_defs.py
# Files depending on this file: src/uav_risk/core/data_validator.py
# =====================================================================