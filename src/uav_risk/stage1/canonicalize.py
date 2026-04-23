# src/uav_risk/stage1/canonicalize.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# ============================================================
# 1. Feature Map (يجب أن يطابق ترتيب أعمدة التدريب 100%)
# ============================================================
TRAINING_FEATURES = [
    "uav_mass_kg",
    "uav_max_speed_mps",
    "uav_battery_model_hover_power_W",
    "environment_weather_wind_mps",
    "environment_weather_gust_mps",
    "environment_weather_visibility_m",
    "environment_gnss_jam_dbm",
    "airspace_altitude_agl_m"
]

# قيم افتراضية محافظة ومتوافقة إحصائياً مع بيانات التدريب
# تجنبنا 0.0 لأنه يشوه التوزيع الإحصائي للنموذج
SAFE_IMPUTATION_DEFAULTS = {
    "uav_battery_model_hover_power_W": 180.0,  # متوسط صناعي آمن للـ Hover
    "environment_weather_gust_mps": None,      # يُحسب ديناميكياً أدناه
    "environment_weather_visibility_m": 5000.0, # 5km (متوسط محافظ للـ VLOS)
    "environment_gnss_jam_dbm": -90.0,          # إشارة GNSS طبيعية ضعيفة
    "airspace_altitude_agl_m": 30.0             # ارتفاع تحليق نموذجي
}

def canonicalize_scenario(scenario: Dict[str, Any]) -> pd.DataFrame:
    """
    Converts a UAVScenario dict into a flat DataFrame aligned with Stage-1 training features.
    ⚠️ Mلاحظة: هذه الخطوة مخصصة حصرياً للـ ML Oracle. لا تستخدمها للفحص الفيزيائي الحتمي.
    """
    try:
        # 1. استخراج القيم الأساسية فقط
        flat_data = {feat: scenario.get(feat) for feat in TRAINING_FEATURES}

        # 2. معالجة القيم المفقودة بمنطق إحصائي آمن (بدون تشويه)
        sustained_wind = flat_data.get("environment_weather_wind_mps", 0.0)
        
        if flat_data["environment_weather_gust_mps"] is None:
            # مطابقة منطق InputContractEngine: gust = 1.5 * wind عند الغياب
            flat_data["environment_weather_gust_mps"] = sustained_wind * 1.5
            
        if flat_data["environment_weather_visibility_m"] is None:
            flat_data["environment_weather_visibility_m"] = SAFE_IMPUTATION_DEFAULTS["environment_weather_visibility_m"]

        # 3. ملء الحقول الاختيارية المتبقية بقيم آمنة إحصائياً
        for feat, val in flat_data.items():
            if val is None:
                flat_data[feat] = SAFE_IMPUTATION_DEFAULTS.get(feat, 0.0)

        # 4. تحويل إلى DataFrame وإعادة الترتيب الصارم
        df = pd.DataFrame([flat_data]).reindex(columns=TRAINING_FEATURES)

        # 5. تأكيد نوع البيانات ومنع NaN نهائيًا قبل الـ Preprocessor
        df = df.astype(float).fillna(0.0) # Fallback أخير فقط للطوارئ
        return df

    except Exception as e:
        logger.error(f"[CANONICALIZE ERROR] Feature alignment failed: {e}")
        # إرجاع إطار فارغ بمتوسطات آمنة لمنع انهيار XGBoost
        safe_fallback = {k: v if v is not None else 0.0 for k, v in SAFE_IMPUTATION_DEFAULTS.items()}
        safe_fallback["uav_mass_kg"] = 1.5
        safe_fallback["uav_max_speed_mps"] = 15.0
        safe_fallback["environment_weather_wind_mps"] = 5.0
        return pd.DataFrame([safe_fallback]).reindex(columns=TRAINING_FEATURES).astype(float)