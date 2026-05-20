"""
Imputation Strategy Engine (Context-Aware Fallbacks)
Responsible for determining the smartest and safest fallback value for missing features
using physical derivations or the official Safe Values Registry.
"""

import logging
from typing import Dict, Tuple, Any

# استدعاء دالة جلب القيمة الآمنة من الدستور
from uav_risk.ml.feature_defs import get_safe_value

logger = logging.getLogger(__name__)

class ImputationStrategy:
    """
    The brain behind filling missing data. 
    It looks at the context (available features) before falling back to static defaults.
    """
    
    def __init__(self):
        # يمكن تمرير أي إعدادات للمستقبل هنا (مثلاً: Strict Mode vs Lenient Mode)
        pass

    def get_imputed_value(
        self, 
        feature_name: str, 
        available_features: Dict[str, float], 
        raw_inputs: Dict[str, Any] = None
    ) -> Tuple[float, str]:
        """
        يستقبل اسم الميزة الناقصة، ويحاول إيجاد قيمة لها بالاشتقاق.
        إذا فشل الاشتقاق، يعود للقيمة الآمنة الصارمة.
        
        Returns:
            Tuple[float, str]: (The imputed value, The reason/formula used)
        """
        raw = raw_inputs or {}
        
        # ==========================================
        # 🧠 الذكاء الفيزيائي: محاولات الاشتقاق (Derivations)
        # ==========================================
        
        # 1. اشتقاق طاقة البطارية (Wh)
        if feature_name == "uav_battery_wh":
            # نبحث في raw inputs أو available features
            mah = raw.get("battery_capacity_mah") or raw.get("uav_battery_capacity_mah")
            volts = raw.get("battery_voltage_v") or raw.get("uav_battery_voltage_v")
            if mah is not None and volts is not None:
                try:
                    val = (float(mah) * float(volts)) / 1000.0
                    return val, f"Derived from capacity ({mah}mAh) and voltage ({volts}V)"
                except (ValueError, TypeError):
                    pass

        # 2. اشتقاق حمولة الجناح / القرص (Disk/Wing Loading)
        elif feature_name == "feat_disk_loading":
            mass = available_features.get("uav_mass_kg")
            wing_area = available_features.get("uav_aero_wing_area_m2")
            if mass and wing_area and wing_area > 0:
                val = (mass * 9.81) / wing_area
                return val, f"Derived physics: (mass {mass}kg * 9.81) / area {wing_area}m²"

        # 3. اشتقاق الباعية (Aspect Ratio)
        elif feature_name == "uav_aero_aspect_ratio":
            wingspan = raw.get("wingspan_m") or raw.get("uav_wingspan_m")
            wing_area = available_features.get("uav_aero_wing_area_m2")
            if wingspan and wing_area and wing_area > 0:
                try:
                    val = (float(wingspan) ** 2) / wing_area
                    return val, f"Derived physics: (wingspan {wingspan}m)² / area {wing_area}m²"
                except (ValueError, TypeError):
                    pass

        # 4. [إضافة إبداعية] اشتقاق هبات الرياح (Gust Estimation)
        elif feature_name == "environment_weather_gust_mps":
            wind = available_features.get("environment_weather_wind_mps")
            if wind is not None and wind > 0:
                # في الطيران، عادة ما تُقدر الهبات بزيادة 40% عن الرياح المستمرة
                val = wind * 1.4
                return val, f"Estimated aviation gust factor (wind {wind}m/s * 1.4)"

        # 5. [إضافة إبداعية] اشتقاق مساحة القرص المروحي لطائرات الدرون
        elif feature_name == "uav_rotorcraft_disk_area_m2":
            rotors = available_features.get("uav_rotorcraft_rotor_count")
            prop_dia = raw.get("propeller_diameter_m") or raw.get("uav_propeller_diameter_m")
            if rotors and prop_dia:
                try:
                    radius = float(prop_dia) / 2.0
                    val = float(rotors) * 3.14159 * (radius ** 2)
                    return val, f"Calculated disk area from {rotors} rotors and {prop_dia}m dia"
                except (ValueError, TypeError):
                    pass

        # ==========================================
        # 🛡️ شبكة الأمان: القيمة الافتراضية الصارمة (Fallback)
        # ==========================================
        
        # إذا لم تتطابق أي قاعدة اشتقاق، نلجأ للدستور
        try:
            fallback_val = get_safe_value(feature_name)
            return fallback_val, "Used static safe value from registry."
        except KeyError:
            # حالة طوارئ قصوى: الميزة غير موجودة في الدستور
            logger.error(f"Critical: Feature {feature_name} has no safe value defined!")
            return 0.0, "UNKNOWN FEATURE - Forced to 0.0 fallback."