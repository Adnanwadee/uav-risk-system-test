"""
Module: src.uav_risk.core.imputation_strategy
Purpose: Rigorous and precise context-aware physical imputation strategy computing derived features from live inputs.
Dependencies: uav_risk.ml.feature_defs for safe registry fallbacks.
"""

import logging
from typing import Dict, Tuple, Any
from uav_risk.ml.feature_defs import get_safe_value

logger = logging.getLogger(__name__)


class ImputationStrategy:
    """The advanced aerodynamic math engine solving mathematical formulas from clean telemetry data."""
    
    def __init__(self) -> None:
        pass

    def get_imputed_value(
        self, 
        feature_name: str, 
        available_features: Dict[str, float], 
        raw_inputs: Dict[str, Any] = None
    ) -> Tuple[float, str]:
        """
        Computes dynamic mathematical derivations or fetches calibrated fallbacks for missing data.
        """
        raw = raw_inputs or {}
        
        # 1. اشتقاق طاقة البطارية (Wh) مع تحصين كامل ضد الـ Prefixes لمنع الفشل الصامت
        if feature_name == "uav_battery_wh":
            mah = (
                raw.get("uav_battery_capacity_mah") or 
                raw.get("battery_capacity_mah") or 
                available_features.get("uav_battery_capacity_mah")
            )
            volts = (
                raw.get("uav_battery_voltage_v") or 
                raw.get("battery_voltage_v") or 
                available_features.get("uav_battery_voltage_v")
            )
            if mah and volts:
                try:
                    val = (float(mah) * float(volts)) / 1000.0
                    return val, f"Derived physics: ({mah} mAh * {volts} V) / 1000"
                except (ValueError, TypeError):
                    pass

        # 2. حساب حمولة القرص المروحي بناءً على مساحة دوران مروحة الدرون الفعلي
        elif feature_name == "feat_disk_loading":
            mass = available_features.get("uav_mass_kg")
            disk_area = available_features.get("uav_rotorcraft_disk_area_m2")
            if mass and disk_area and disk_area > 0:
                val = (mass * 9.81) / disk_area
                return float(val), f"Derived physics: (mass {mass} kg * 9.81) / disk area {disk_area} m²"

        # 3. اشتقاق هبات الرياح الحية بناءً على سرعة الرياح المستمرة
        elif feature_name == "environment_weather_gust_mps":
            wind = available_features.get("environment_weather_wind_mps")
            if wind is not None and wind > 0:
                val = wind * 1.4
                return float(val), f"Aviation gust factor: wind {wind} m/s * 1.4"

        # 4. الحساب المحصن لسلامة الاتصالات وكبح النسبة رياضياً داخل النطاق [0.0 - 1.0]
        elif feature_name == "feat_comms_health":
            rssi = available_features.get("comms_rssi_dbm_min")
            if rssi is not None:
                ratio = (rssi + 80.0) / 20.0
                final_ratio = max(0.0, min(1.0, ratio))
                return float(final_ratio), f"Bound ratio: ({rssi} dBm + 80) / 20 capped inside [0, 1]"

        # 5. الحساب التراكمي لخطورة الطقس المركب من معطيات الرياح والهبات والتشويش حياً
        elif feature_name == "feat_weather_severity":
            wind = available_features.get("environment_weather_wind_mps", 0.0)
            gust = available_features.get("environment_weather_gust_mps", 0.0)
            jam = available_features.get("environment_gnss_jam_dbm", -125.0)
            
            wind_severity = min(1.0, wind / 12.2) if wind > 0 else 0.0
            gust_severity = min(1.0, gust / 14.9) if gust > 0 else 0.0
            jam_severity = max(0.0, min(1.0, (jam + 150.0) / 34.0))
            
            composite = (wind_severity * 0.4) + (gust_severity * 0.4) + (jam_severity * 0.2)
            return float(composite), "Assembled dynamically from live wind, gust, and gnss telemetry vectors."

        # 6. اشتقاق النسبة الباعية للجناح للطائرات الثابتة الجناح
        elif feature_name == "uav_aero_aspect_ratio":
            wingspan = raw.get("uav_wingspan_m") or raw.get("wingspan_m")
            wing_area = available_features.get("uav_aero_wing_area_m2")
            if wingspan and wing_area and wing_area > 0:
                try:
                    val = (float(wingspan) ** 2) / wing_area
                    return float(val), f"Derived physics: (wingspan {wingspan} m)² / wing area {wing_area} m²"
                except (ValueError, TypeError):
                    pass

        # 7. حساب مساحة القرص المروحي الإجمالية من حقول المحركات وقطر المروحة الخام
        elif feature_name == "uav_rotorcraft_disk_area_m2":
            rotors = available_features.get("uav_rotorcraft_rotor_count")
            prop_dia = raw.get("uav_propeller_diameter_m") or raw.get("propeller_diameter_m")
            if rotors and prop_dia:
                try:
                    radius = float(prop_dia) / 2.0
                    val = float(rotors) * 3.1415926535 * (radius ** 2)
                    return float(val), f"Calculated area: {rotors} rotors * pi * (diameter {prop_dia} m / 2)²"
                except (ValueError, TypeError):
                    pass

        # شبكة الأمان المطلقة والمدروسة للميزات الفرعية والمحاكاة المتبقية
        fallback_val = get_safe_value(feature_name)
        return float(fallback_val), "Used static safe fallback configuration from registry."

# =====================================================================
# Architectural Registry Block:
# This file depends on: src/uav_risk/ml/feature_defs.py
# This file is explicitly imported and used by: src/uav_risk/core/data_validator.py
# =====================================================================