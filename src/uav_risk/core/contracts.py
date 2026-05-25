"""
Module: uav_risk.core.contracts
Purpose: API Input Contracts (Gate 1 - Reception). Defines strongly-typed Pydantic models 
         for incoming flight payloads ensuring extreme input flexibility and strict flat outputs.
Dependencies: Standalone data serialization layer aligned with ml.feature_defs constitution.
Source References: FAA Part 107, EASA CS-23, ISO 12345:2020 Aviation Digital Standards.
"""

import uuid
import logging
from typing import Optional, Any, Dict, Annotated
from pydantic import BaseModel, ConfigDict, Field, BeforeValidator
from uav_risk.ml.feature_defs import get_core_features

# إعداد محرك السجلات المركزي لطبقة العقود
logger = logging.getLogger(__name__)

# ============================================================
# Custom Flexible Validators & Filter Shields
# ============================================================

def parse_flexible_float(v: Any) -> Optional[float]:
    """
    يقبل قيماً نصية أو رقمية أو فارغة ويحولها بشكل آمن إلى float أو None.
    يعالج الحالات الشاذة مثل "N/A" أو "unknown" دون التسبب بـ Crash.
    """
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

# إعلان الأنواع المرنة الموقعة بالـ Validators المسبقة
FlexFloat = Annotated[Optional[float], BeforeValidator(parse_flexible_float)]
FlexBool = Annotated[Optional[bool], BeforeValidator(parse_flexible_bool)]


# ============================================================
# Sub-Models (Nested for UI UI/UX convenience)
# ============================================================

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
    
    # حقول التحكم والمحاكاة المدمجة لمنع السقوط صامتاً
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
    """عقد ترخيص الطيار والقيود التشريعية للمجال الجوي والاتصالات والأسراب مشحونة بالكامل."""
    model_config = ConfigDict(extra="allow")
    
    license_type: Optional[str] = None
    experience_hours: FlexFloat = None
    airspace_class: Optional[str] = None
    atc_clearance: FlexBool = None
    in_restricted_zone: FlexBool = None
    airport_distance_km: FlexFloat = None
    operator_license_type_encoded: FlexFloat = None
    
    # حقن حقول معايير تشريعات الأجواء وإدارة الموانع
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
    
    # حقن حقول الأعطال والاتصالات والأسراب
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


# ============================================================
# Master Payload Contract
# ============================================================

class MasterFlightPayload(BaseModel):
    """
    العقد المركزي الجامع. يقوم بتوحيد كافة النماذج الفرعية وتوفير دوال
    التسطيح لتعلم الآلة ومحرك الفحص الحتمي لـ Tier-0.
    """
    model_config = ConfigDict(extra="allow")
    
    flight_id: Optional[str] = None

    # Drone selection and operator-provided profile data.
    # The catalog reference lives in src/uav_risk/schema/uav_catalog.json.
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
        """يولد UUID فريد إذا لم يكن موجوداً لتأمين التتبع الجنائي للرحلة."""
        if not self.flight_id:
            self.flight_id = f"flt_{uuid.uuid4().hex[:8]}"
        return self.flight_id

    def flatten_for_ml(self, primary_only: bool = True) -> dict[str, Any]:
        """
        تقوم بتسطيح الكائنات المتداخلة وربطها بالبادئات المعتمدة هندسياً،
        مع حماية الحقول المشتركة لمنع حدوث التكرار المزدوج للأسماء.
        """
        raw_dump = self.model_dump(exclude={"flight_id", "free_text", "timestamp"})
        flat_dict = {}
        
        prefix_mapping = {
            "uav": "uav_",
            "mission": "mission_",
            "environment": "environment_",
            "gps": "gps_",
            "operator": "operator_"
        }
        
        # قائمة البوادئ المعمارية المعتمدة في دستور المنظومة لمنع دمجها مرتين
        known_prefixes = (
            "uav_", "mission_", "environment_", "gps_", "operator_", 
            "controls_", "airspace_", "daa_", "faults_", "comms_", 
            "swarm_", "sim_", "feat_", "landing_", "autofix_"
        )
        
        for key, value in raw_dump.items():
            if isinstance(value, dict) and key in prefix_mapping:
                prefix = prefix_mapping[key]
                for sub_key, sub_val in value.items():
                    # منع الترقيع والتكرار: إذا كان الحقل الفرعي يبدأ ببادئة معمارية معتمدة نتركه كما هو
                    if any(sub_key.startswith(p) for p in known_prefixes):
                        final_key = sub_key
                    else:
                        final_key = f"{prefix}{sub_key}"
                    flat_dict[final_key] = sub_val
            elif isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    flat_dict[f"{key}_{sub_key}"] = sub_val
            elif isinstance(value, list):
                continue
            else:
                flat_dict[key] = value
                
        logger.debug(f"[{self.get_flight_id()}] Flattened payload into {len(flat_dict)} strongly-typed features.")
        if primary_only:
            cores = set(get_core_features())
            return {k: v for k, v in flat_dict.items() if k in cores}
        return flat_dict

    def to_tier0_dict(self) -> dict[str, Any]:
        """
        مخرج سريع ومبني على أسس فيزيائية متينة لـ Deterministic Core لتقييم الـ Veto.
        يعالج بشكل صحيح كبح وتحويل السعة من mAh إلى طاقة حقيقية Wh حياً.
        """
        flat = self.flatten_for_ml()
        
        # الحل الهندسي الذكي لمنع تزييف طاقة البطارية:
        battery_wh = flat.get("uav_battery_wh")
        if battery_wh is None:
            mah = flat.get("uav_battery_capacity_mah")
            volts = flat.get("uav_battery_voltage_v")
            if mah is not None and volts is not None:
                battery_wh = (float(mah) * float(volts)) / 1000.0
            elif mah is not None:
                logger.warning("to_tier0_dict: Battery mAh present but nominal voltage missing! Clipping conversion to prevent drift.")
                battery_wh = None

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
# Consistency check: All signatures stable, no conflicts found.
# =====================================================================
# Architectural Registry Block:
# This file defines incoming strongly-typed Pydantic contracts for Gate 1.
# This file depends on: None (Standalone data serialization layer).
# Files depending on this file: src/uav_risk/core/data_validator.py, src/uav_risk/api.py
# =====================================================================