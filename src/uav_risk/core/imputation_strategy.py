"""
Module: uav_risk.core.imputation_strategy
Purpose: Rigorous and precise context-aware physical imputation strategy computing 
         derived aerodynamic features and handling execution order races reactively.
Dependencies: uav_risk.ml.feature_defs for safe registry fallbacks.
Source References: Leishman (Rotorcraft Aerodynamics), FAA Part 107 Risk Frameworks.
"""

import logging
from typing import Dict, Tuple, Any
from uav_risk.ml.feature_defs import get_safe_value

# إعداد محرك السجلات المركزي لطبقة الاشتقاق الفيزيائي
logger = logging.getLogger(__name__)


class ImputationStrategy:
    """المحرك الرياضي الحركي المتقدم لحساب الاشتقاقات المترابطة وحل المشاكل التلازمية حياً."""
    
    def __init__(self) -> None:
        pass

    def get_imputed_value(
        self, 
        feature_name: str, 
        available_features: Dict[str, float], 
        raw_inputs: Dict[str, Any] = None
    ) -> Tuple[float, str]:
        """
        تقوم بفك وحساب المعادلات الفيزيائية الحركية ديناميكياً مع حل مشكلة سباق التنفيذ ذاتياً.
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
                    logger.error(f"Imputation Engine: Failed numeric conversion for battery capacity={mah} or volts={volts}")

        # 2. حساب مساحة القرص المروحي الإجمالية من حقول المحركات وقطر المروحة الخام
        elif feature_name == "uav_rotorcraft_disk_area_m2":
            rotors = available_features.get("uav_rotorcraft_rotor_count")
            prop_dia = (
                raw.get("uav_propeller_diameter_m") or 
                raw.get("propeller_diameter_m") or 
                available_features.get("uav_propeller_diameter_m")
            )
            if rotors and prop_dia:
                try:
                    radius = float(prop_dia) / 2.0
                    val = float(rotors) * 3.1415926535 * (radius ** 2)
                    return float(val), f"Calculated area: {rotors} rotors * pi * (diameter {prop_dia} m / 2)²"
                except (ValueError, TypeError):
                    logger.error(f"Imputation Engine: Failed disk area calculation for rotors={rotors}, diameter={prop_dia}")

        # 3. حساب حمولة القرص المروحي (مع حل ذكي ومستقل لمشكلة سباق التنفيذ الحلقي)
        elif feature_name == "feat_disk_loading":
            mass = available_features.get("uav_mass_kg")
            disk_area = available_features.get("uav_rotorcraft_disk_area_m2")
            
            # 🎯 حل المشكلة الجوهرية: إذا غابت المساحة بسبب الترتيب الأبجدي للحلقة، نشتقها هنا فوراً وتلازمياً!
            if not disk_area or disk_area <= 0:
                derived_area_val, _ = self.get_imputed_value("uav_rotorcraft_disk_area_m2", available_features, raw)
                if derived_area_val > 0:
                    disk_area = derived_area_val
                    logger.info(f"Pipeline Race Condition Intercepted: Dynamically solved missing disk_area as {disk_area} m²")
            
            if mass and disk_area and disk_area > 0:
                val = (float(mass) * 9.81) / float(disk_area)
                return float(val), f"Derived physics with race-shield: (mass {mass} kg * 9.81) / disk area {disk_area} m²"

        # 4. اشتقاق هبات الرياح الحية بناءً على سرعة الرياح المستمرة
        elif feature_name == "environment_weather_gust_mps":
            wind = available_features.get("environment_weather_wind_mps")
            if wind is not None and wind > 0:
                val = wind * 1.4
                return float(val), f"Aviation gust factor: wind {wind} m/s * 1.4"

        # 5. الحساب المحصن لسلامة الاتصالات وكبح النسبة رياضياً داخل النطاق المستقر [0.0 - 1.0]
        elif feature_name == "feat_comms_health":
            rssi = available_features.get("comms_rssi_dbm_min") or available_features.get("operator_comms_rssi_dbm_min")
            if rssi is not None:
                ratio = (float(rssi) + 80.0) / 20.0
                final_ratio = max(0.0, min(1.0, ratio))
                return float(final_ratio), f"Bound ratio: ({rssi} dBm + 80) / 20 capped inside [0, 1]"

        # 6. الحساب التراكمي لخطورة الطقس المركب (مع حل ذكي لحالة غياب ميزة الهبات حياً)
        elif feature_name == "feat_weather_severity":
            wind = available_features.get("environment_weather_wind_mps", 0.0)
            gust = available_features.get("environment_weather_gust_mps")
            
            # إذا لم تكن الهبات متوفرة بعد في المصفوفة، نقوم باشتقاقها فورياً لحماية المؤشر المركب من العمى
            if gust is None or gust <= 0:
                derived_gust, _ = self.get_imputed_value("environment_weather_gust_mps", available_features, raw)
                gust = derived_gust if derived_gust > 0 else wind
                
            jam = available_features.get("environment_gnss_jam_dbm", -125.0)
            
            wind_severity = min(1.0, wind / 12.2) if wind > 0 else 0.0
            gust_severity = min(1.0, gust / 14.9) if gust > 0 else 0.0
            jam_severity = max(0.0, min(1.0, (float(jam) + 150.0) / 34.0))
            
            composite = (wind_severity * 0.4) + (gust_severity * 0.4) + (jam_severity * 0.2)
            return float(composite), "Assembled dynamically from live wind, reactive gust shield, and gnss telemetry vectors."

        # 7. اشتقاق النسبة الباعية للجناح للطائرات الثابتة الجناح
        elif feature_name == "uav_aero_aspect_ratio":
            wingspan = raw.get("uav_wingspan_m") or raw.get("wingspan_m") or available_features.get("uav_wingspan_m")
            wing_area = available_features.get("uav_aero_wing_area_m2")
            if wingspan and wing_area and wing_area > 0:
                try:
                    val = (float(wingspan) ** 2) / float(wing_area)
                    return float(val), f"Derived physics: (wingspan {wingspan} m)² / wing area {wing_area} m²"
                except (ValueError, TypeError):
                    logger.error(f"Imputation Engine: Failed aspect ratio check for wingspan={wingspan}, area={wing_area}")

        # شبكة الأمان المطلقة والمدروسة للميزات الفرعية والمحاكاة المتبقية من الدستور
        fallback_val = get_safe_value(feature_name)
        return float(fallback_val), "Used static safe fallback configuration from registry."

# =====================================================================
# Consistency check: All signatures stable, no conflicts found.
# =====================================================================
# Architectural Registry Block:
# This file serves as the context-aware math derivation engine for physics.
# This file depends on: src/uav_risk/ml/feature_defs.py
# Files depending on this file: src/uav_risk/core/data_validator.py
# =====================================================================