"""
API Input Contracts (Gate 1 - Reception)
Defines the Pydantic models for incoming flight payloads.
Ensures extreme flexibility on input and strict, flat formatting for output (ML ready).
"""

import uuid
import logging
from typing import Optional, Any, Dict, Annotated
from pydantic import BaseModel, ConfigDict, Field, BeforeValidator

# إعداد اللوجر الخاص بالـ Contracts
logger = logging.getLogger(__name__)

# ============================================================
# Custom Flexible Validators
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
    """محلل مرن للقيم المنطقية"""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        v_clean = v.strip().lower()
        if v_clean in ["true", "1", "yes", "y"]: return True
        if v_clean in ["false", "0", "no", "n"]: return False
    if isinstance(v, int):
        return bool(v)
    return None

# Custom Types
FlexFloat = Annotated[Optional[float], BeforeValidator(parse_flexible_float)]
FlexBool = Annotated[Optional[bool], BeforeValidator(parse_flexible_bool)]


# ============================================================
# Sub-Models (Nested for UI convenience)
# ============================================================

class UAVSpecs(BaseModel):
    model_config = ConfigDict(extra="allow")
    
    mass_kg: FlexFloat = None
    wingspan_m: FlexFloat = None
    max_speed_mps: FlexFloat = None  # لاحظ تعديل الاسم ليتوافق مع feature_defs (mps بدلا من ms)
    battery_wh: FlexFloat = None     # يتوافق مع uav_battery_wh
    battery_capacity_mah: FlexFloat = None
    battery_voltage_v: FlexFloat = None
    rotorcraft_rotor_count: Optional[int] = None
    payload_mass_kg: FlexFloat = None
    max_takeoff_weight_kg: FlexFloat = None
    aero_wing_area_m2: FlexFloat = None


class MissionParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    
    altitude_m: FlexFloat = None
    max_altitude_m: FlexFloat = None
    distance_km: FlexFloat = None
    time_budget_s: FlexFloat = None  # يتوافق مع mission_time_budget_s
    operation_type: Optional[str] = None # "VLOS", "BVLOS", "Indoor"
    is_night_flight: FlexBool = None
    waypoints_count: Optional[int] = None
    loiter_radius_m: FlexFloat = None


class EnvironmentData(BaseModel):
    model_config = ConfigDict(extra="allow")
    
    weather_wind_mps: FlexFloat = None  # يتوافق مع environment_weather_wind_mps
    weather_wind_dir_deg: FlexFloat = None
    weather_gust_mps: FlexFloat = None
    temperature_c: FlexFloat = None
    humidity_pct: FlexFloat = None
    weather_phenomena_count: Optional[int] = None
    gnss_jam_dbm: FlexFloat = None
    em_interference: FlexBool = None


class GPSData(BaseModel):
    model_config = ConfigDict(extra="allow")
    
    fix_quality: Optional[int] = None # 0=none, 1=GPS, 2=DGPS
    satellites_count: Optional[int] = None
    hdop: FlexFloat = None
    latitude: FlexFloat = None
    longitude: FlexFloat = None
    altitude_gps_m: FlexFloat = None


class OperatorData(BaseModel):
    model_config = ConfigDict(extra="allow")
    
    license_type: Optional[str] = None # "A1","A2","A3","STS","none"
    experience_hours: FlexFloat = None
    airspace_class: Optional[str] = None # "A"-"G"
    atc_clearance: FlexBool = None
    in_restricted_zone: FlexBool = None
    airport_distance_km: FlexFloat = None


# ============================================================
# Master Payload Contract
# ============================================================

class MasterFlightPayload(BaseModel):
    """
    The main contract. Aggregates all nested models and provides 
    flattening utilities for ML and Tier0 processing.
    """
    model_config = ConfigDict(extra="allow")
    
    flight_id: Optional[str] = None
    uav: UAVSpecs = Field(default_factory=UAVSpecs)
    mission: MissionParams = Field(default_factory=MissionParams)
    environment: EnvironmentData = Field(default_factory=EnvironmentData)
    gps: GPSData = Field(default_factory=GPSData)
    operator: OperatorData = Field(default_factory=OperatorData)
    
    free_text: Optional[str] = None
    timestamp: Optional[str] = None

    def get_flight_id(self) -> str:
        """يولد UUID إذا لم يكن موجوداً لضمان وجود معرف دائم للرحلة"""
        if not self.flight_id:
            self.flight_id = f"flt_{uuid.uuid4().hex[:8]}"
        return self.flight_id

    def flatten_for_ml(self) -> dict[str, Any]:
        """
        يقوم بتسطيح الكائنات المتداخلة وربطها بالـ Prefix المناسب
        لتتطابق مفاتيحها تماماً مع أسماء الميزات في `feature_defs.py`.
        """
        raw_dump = self.model_dump(exclude={"flight_id", "free_text", "timestamp"})
        flat_dict = {}
        
        # خريطة البوادئ (Prefixes) التي تحول uav.mass_kg إلى uav_mass_kg
        prefix_mapping = {
            "uav": "uav_",
            "mission": "mission_",
            "environment": "environment_",
            "gps": "gps_",
            "operator": "operator_"
        }
        
        for key, value in raw_dump.items():
            if isinstance(value, dict) and key in prefix_mapping:
                prefix = prefix_mapping[key]
                for sub_key, sub_val in value.items():
                    # تجنب تكرار الـ prefix إذا كان المستخدم قد أرسله جاهزاً
                    final_key = f"{prefix}{sub_key}" if not sub_key.startswith(prefix) else sub_key
                    flat_dict[final_key] = sub_val
            elif isinstance(value, dict):
                # لتمكين تمرير قواميس إضافية مسطحة
                for sub_key, sub_val in value.items():
                    flat_dict[f"{key}_{sub_key}"] = sub_val
            elif isinstance(value, list):
                # حسب الخطة: يتم تجاهل القوائم (Sensors arrays, etc.)
                continue
            else:
                # Top-level extra fields (e.g., custom flags)
                flat_dict[key] = value
                
        logger.debug(f"[{self.get_flight_id()}] Flattened payload into {len(flat_dict)} initial features.")
        return flat_dict

    def to_tier0_dict(self) -> dict[str, Any]:
        """
        مخرج سريع وبسيط لـ Deterministic Core ليقوم بالفحص السريع (Veto Check).
        """
        # جلب القيم إما من الأماكن الرسمية أو من الحقول الإضافية (Extra)
        flat = self.flatten_for_ml()
        
        # تجميع المتطلبات لـ Tier 0
        tier0_data = {
            "altitude_m": flat.get("mission_altitude_m") or flat.get("airspace_altitude_agl_max_m"),
            "battery_wh": flat.get("uav_battery_wh") or flat.get("uav_battery_capacity_mah"),
            "wind_speed_ms": flat.get("environment_weather_wind_mps"),
            "gps_fix_quality": flat.get("gps_fix_quality"),
            "in_restricted_zone": flat.get("operator_in_restricted_zone")
        }
        return tier0_data