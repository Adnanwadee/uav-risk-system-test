"""
Stage 1 Canonicalization (V13.0 - ACE Integrated)
==================================================
التعديلات:
1. توحيد المفاتيح: التحول لنسق النقاط (uav.mass_kg) ليتوافق مع مخرجات input_contract.
2. السيادة للوكلاء: تحويل "الرفض القاطع" إلى تحذير (Warning) للسماح للوكلاء بالتحليل.
3. معالجة الفقدان: ملء البيانات المفقودة بقيم محايدة إحصائياً لضمان عدم انهيار النموذج.
"""

from __future__ import annotations
import pandas as pd
import logging
from typing import Dict, Any, Tuple
from uav_risk.stage1.utils import calc_power_to_weight, calc_effective_wind_gust

logger = logging.getLogger(__name__)

# [تحديث]: الحدود الفيزيائية للطيران (Safety Envelopes) بنسق النقاط
# تم رفع الحد الأقصى للوزن ليدعم داتا سيت شحن الطرود (حتى 150 كجم)
PHYSICAL_LIMITS = {
    "uav.mass_kg": (0.1, 150.0),             
    "environment.weather.wind_mps": (0.0, 45.0),
    "environment.gnss_jam_dbm": (-140.0, 0.0), 
    "airspace.altitude_agl_m": (0.0, 500.0)  
}

def validate_physical_bounds(data: Dict[str, Any]) -> bool:
    """
    يتحقق من منطقية الأرقام. 
    [إصلاح]: لم يعد يعيد False لإيقاف النظام، بل يسجل تحذيراً للتدقيق الجنائي فقط.
    """
    for key, (low, high) in PHYSICAL_LIMITS.items():
        val = data.get(key)
        if val is not None:
            try:
                num_val = float(val)
                if not (low <= num_val <= high):
                    logger.warning(f"[OUT_OF_BOUNDS] {key}={num_val} is outside reference envelope ({low}-{high})")
            except (ValueError, TypeError):
                continue
    return True # السماح دائماً بالمرور للوكلاء

def canonicalize_scenario(flat_data: Dict[str, Any], policy: Dict[str, Any], expected_columns: list) -> Tuple[pd.DataFrame | None, str]:
    """تحويل القاموس الخام إلى DataFrame جاهز لنموذج XGBoost."""
    try:
        # 1. التدقيق الفيزيائي (لأغراض التسجيل والتحذير فقط)
        validate_physical_bounds(flat_data)

        # 2. هندسة الميزات (Feature Engineering) بنسق المفاتيح الجديد
        # ملاحظة: نستخدم get مع قيم افتراضية آمنة لمنع الـ NaNs من إفساد النموذج
        engineered_row = {
            "uav.mass_kg": float(flat_data.get("uav.mass_kg", 2.0)),
            "environment.weather.wind_mps": float(flat_data.get("environment.weather.wind_mps", 0.0)),
            "environment.gnss_jam_dbm": float(flat_data.get("environment.gnss_jam_dbm", -90.0)),
            "airspace.altitude_agl_m": float(flat_data.get("airspace.altitude_agl_m", 10.0)),
            
            # ميزات مركبة (Derived Features)
            "feat_power_to_weight": calc_power_to_weight(
                flat_data.get("uav.battery_model.hover_power_W"), 
                flat_data.get("uav.mass_kg")
            ),
            "environment.weather.gust_mps": calc_effective_wind_gust(
                flat_data.get("environment.weather.wind_mps"),
                flat_data.get("environment.weather.gust_mps")
            )
        }
        
        # إضافة بقية الحقول من الـ 50 عاموداً إذا كانت موجودة في expected_columns
        for col in expected_columns:
            if col not in engineered_row:
                engineered_row[col] = flat_data.get(col, 0.0)

        # 3. بناء الـ DataFrame مع إعادة الترتيب الصارم للأعمدة
        df = pd.DataFrame([engineered_row]).reindex(columns=expected_columns, fill_value=0.0)
        
        return df, "OK"
        
    except Exception as e:
        logger.error(f"Canonicalization Crash: {e}", exc_info=True)
        return None, "CANONICAL_ERROR"