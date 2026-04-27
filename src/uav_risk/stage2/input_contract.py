"""
API Input Contract (V2.3 - Clean Code & ML-Synchronized)
========================================================
Fixes applied from Architectural Audit:
1. Pure Functions: Flattening logic refactored to prevent outer-scope side effects.
2. Dynamic Sensors: Removed hardcoded sensor lists. Maps any sensor -> `has_{sensor_name}`.
3. Audit Trail: Injected logging into `flexible_float` to track corrupted telemetry points.
4. Single Responsibility: Removed Feature Engineering (`feat_obstacle_count`), leaving it to Stage1Bridge.

Author: Stage 2 — ACE System
"""

import math
import logging
from pydantic import BaseModel, Field, ConfigDict, BeforeValidator, AliasChoices
from typing import Optional, List, Dict, Any
from typing_extensions import Annotated

logger = logging.getLogger("InputContract")

# ─── 1. The Auditable Validator Hook ───
def flexible_float(v: Any) -> Optional[float]:
    """يحول الأرقام بأمان، ويسجل محاولات إدخال البيانات الفاسدة للحفاظ على أثر التدقيق."""
    if v is None or v == "": 
        return None
    try:
        val = float(v)
        if math.isfinite(val):
            return val
        else:
            # [FIX] أثر التدقيق للقيم غير المنتهية
            logger.debug(f"Audit: Rejected non-finite value '{v}' in input contract.")
            return None
    except (ValueError, TypeError):
        # [FIX] أثر التدقيق للنصوص الفاسدة
        logger.debug(f"Audit: Failed to parse '{v}' as float. Passed as None to Stage 1 Imputer.")
        return None

FlexFloat = Annotated[Optional[float], BeforeValidator(flexible_float)]


class BaseAviationContract(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

# ─── 2. Models (Using FlexFloat) ───

class WeatherInput(BaseAviationContract):
    wind_mps: FlexFloat = Field(None, validation_alias=AliasChoices("wind", "wind_mps"), ge=0.0)
    gust_mps: FlexFloat = Field(None, ge=0.0)
    visibility: Optional[str] = None
    phenomena: List[str] = Field(default_factory=list)

class EnvironmentInput(BaseAviationContract):
    weather: WeatherInput = Field(default_factory=WeatherInput)
    gnss_jam_dbm: FlexFloat = Field(None, le=0.0)
    gnss_multipath: Optional[bool] = None
    em_interference: Optional[bool] = None

class BatteryModelInput(BaseAviationContract):
    hover_power_W: FlexFloat = Field(None, ge=0.0)
    capacity_wh: FlexFloat = Field(None, ge=0.0)

class UavInput(BaseAviationContract):
    mass_kg: FlexFloat = Field(None, ge=0.1)
    max_speed_mps: FlexFloat = Field(None, ge=0.0)
    type: Optional[str] = None
    sensors: Dict[str, bool] = Field(default_factory=dict)
    battery_model: BatteryModelInput = Field(default_factory=BatteryModelInput)
    
    max_thrust_n: FlexFloat = Field(None, ge=0.0)
    drag_coefficient: FlexFloat = Field(None, ge=0.0, le=2.0)
    frontal_area_m2: FlexFloat = Field(None, ge=0.0)
    structural_load_limit_n: FlexFloat = Field(None, ge=0.0)
    max_wind_tolerance_ms: FlexFloat = Field(None, ge=0.0)

class MissionInput(BaseAviationContract):
    type: Optional[str] = None
    pattern: Optional[str] = None
    runway_required: Optional[bool] = None
    waypoints: List[List[float]] = Field(default_factory=list)

class CommsInput(BaseAviationContract):
    uplink_ok: Optional[bool] = None
    downlink_ok: Optional[bool] = None
    uplink_status: str = Field(default="NOMINAL")

class DaaInput(BaseAviationContract):
    sep_threshold_m: FlexFloat = Field(None, ge=0.0)
    ttc_threshold_s: FlexFloat = Field(None, ge=0.0)

class AirspaceInput(BaseAviationContract):
    altitude_agl_max_m: FlexFloat = Field(None, ge=0.0, le=11000.0)
    geofence: Optional[Any] = None

class DynamicTelemetryInput(BaseAviationContract):
    battery_state_of_charge_pct: FlexFloat = Field(None, ge=0.0, le=100.0)
    battery_drain_rate_pct_per_min: FlexFloat = Field(None, ge=0.0)
    altitude_m: FlexFloat = Field(None, ge=0.0, le=11000.0)
    temperature_c: FlexFloat = Field(None, ge=-60.0, le=60.0)
    wind_speed_mps: FlexFloat = Field(None, ge=0.0)
    wind_direction_deg: FlexFloat = Field(None, ge=0.0, le=360.0)
    uav_heading_deg: FlexFloat = Field(None, ge=0.0, le=360.0)
    distance_remaining_m: FlexFloat = Field(None, ge=0.0)
    speed_mps: FlexFloat = Field(None, ge=0.0)
    estimated_flight_time_min: FlexFloat = Field(None, ge=0.0)
    population_density: str = Field(default="SPARSE")

class MasterFlightPayload(BaseAviationContract):
    uav: UavInput = Field(default_factory=UavInput)
    environment: EnvironmentInput = Field(default_factory=EnvironmentInput)
    mission: MissionInput = Field(default_factory=MissionInput)
    comms: CommsInput = Field(default_factory=CommsInput)
    daa: DaaInput = Field(default_factory=DaaInput)
    airspace: AirspaceInput = Field(default_factory=AirspaceInput)
    moving_obstacles: List[Dict[str, Any]] = Field(default_factory=list)
    telemetry: DynamicTelemetryInput = Field(default_factory=DynamicTelemetryInput)

    # ── واجهات التكامل ──
    
    def to_tier0_dict(self) -> Dict[str, Any]:
        """يستخرج البيانات الحيوية المطلوبة لدرع الـ Tier-0."""
        tel = self.telemetry.model_dump()
        env = self.environment.model_dump()
        wea = env.get("weather", {}) or {}
        
        wind = tel.get("wind_speed_mps")
        if wind is None:
            wind = wea.get("wind_mps")
            
        return {
            "battery_state_of_charge_pct": tel.get("battery_state_of_charge_pct"),
            "altitude_m": tel.get("altitude_m"),
            "wind_speed_mps": wind,
            "comms_uplink_status": self.comms.uplink_status,
            "environment_gnss_jam_dbm": env.get("gnss_jam_dbm"),
            "population_density": tel.get("population_density", "SPARSE"),
        }

    def flatten_for_ml(self) -> Dict[str, Any]:
        """
        تسطيح دلالي ونقي (Pure Semantic Flattening).
        لا يعتمد على تعديل المتغيرات الخارجية، ويتوافق تماماً مع Headers مرحلة الـ ML.
        """
        dump = self.model_dump()
        
        # [FIX] دالة نقية (Pure Function) ترجع قاموساً ولا تلعب بنطاق خارجي
        def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
            items = []
            for k, v in d.items():
                if isinstance(v, list) or k == "sensors":
                    continue # نتخطى القوائم ومفتاح الحساسات الخام لكي لا نكسر XGBoost
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        flat_data = flatten_dict(dump)
        
        sensors = dump.get("uav", {}).get("sensors", {})
        for sensor_name, is_present in sensors.items():
            flat_data[f"has_{sensor_name}"] = is_present
            
        
        return flat_data
    