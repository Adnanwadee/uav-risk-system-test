# src/uav_risk/stage1/canonicalize.py
from __future__ import annotations
import pandas as pd
import logging
from typing import Dict, Any, Tuple
from uav_risk.stage1.utils import calc_power_to_weight, calc_effective_wind_gust

logger = logging.getLogger(__name__)

# الحدود الفيزيائية للطيران (Safety Envelopes)
PHYSICAL_LIMITS = {
    "uav_mass_kg": (0.1, 150.0),             # من درون صغير إلى درون شحن كبير
    "environment_weather_wind_mps": (0.0, 45.0), # إعصار مدمر = رفض فوري
    "environment_gnss_jam_dbm": (-140.0, 0.0), 
    "airspace_altitude_agl_m": (0.0, 500.0)  # حدود الـ EASA والـ FAA
}

def validate_physical_bounds(data: Dict[str, Any]) -> bool:
    """يتحقق من منطقية الأرقام قبل معالجتها."""
    for key, (low, high) in PHYSICAL_LIMITS.items():
        val = data.get(key)
        if val is not None:
            if not (low <= float(val) <= high):
                logger.error(f"[PHYSICS VIOLATION] {key}={val} is outside safe envelope ({low}-{high})")
                return False
    return True

def canonicalize_scenario(flat_data: Dict[str, Any], policy: Dict[str, Any], expected_columns: list) -> Tuple[pd.DataFrame | None, str]:
    try:
        # 1. التدقيق الفيزيائي (Physics Sanity Check)
        if not validate_physical_bounds(flat_data):
            return None, "OUT_OF_BOUNDS"

        # 2. فحص جودة البيانات (Data Quality)
        core_sensors = ["uav_mass_kg", "environment_weather_wind_mps", "environment_gnss_jam_dbm"]
        present_count = sum(1 for k in core_sensors if flat_data.get(k) is not None)
        if (present_count / len(core_sensors)) < policy.get("min_dq_core_present", 0.75):
            return None, "DATA_INSUFFICIENT"

        # 3. هندسة الميزات مع القص (Feature Engineering with Clipping)
        engineered_row = {
            "uav.mass_kg": flat_data.get("uav_mass_kg", 2.0),
            "environment.weather.wind_mps": flat_data.get("environment_weather_wind_mps", 0.0),
            "feat_power_to_weight": calc_power_to_weight(
                flat_data.get("uav_battery_model_hover_power_W"), 
                flat_data.get("uav_mass_kg")
            ),
            "environment.weather.gust_mps": calc_effective_wind_gust(
                flat_data.get("environment_weather_wind_mps"),
                flat_data.get("environment_weather_gust_mps")
            )
        }

        final_row = {col: engineered_row.get(col, 0.0) for col in expected_columns}
        return pd.DataFrame([final_row]), "OK"

    except Exception as e:
        logger.error(f"[CANONICALIZE FATAL] {e}")
        return None, "SYSTEM_ERROR"