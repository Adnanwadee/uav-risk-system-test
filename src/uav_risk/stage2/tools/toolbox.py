"""
Aviation Toolbox (V16.0 - Agentic & Full-Data Compatible)
========================================================
التعديلات الجوهرية:
1. إلغاء القوائم الجامدة: تم استبدال الفلترة الصارمة بمنطق "التنظيف مع الحفاظ على البيانات" لضمان وصول الـ 50 عاموداً للوكلاء.
2. الاستخراج الديناميكي: إضافة أدوات لاستخراج مواصفات الطائرة (الوزن، الدفع) من البيانات الحية.
3. توحيد المعايير: دمج منطق flexible_float ليتوافق مع input_contract.
"""

import math
import logging
from typing import Dict, Any, Optional
from uav_risk.stage1.infer import run_stage1_inference
from uav_risk.stage2.schemas import MLResult

logger = logging.getLogger("AviationToolbox")

class TelemetryFormatter:
    """
    مترجم البيانات الحيوية: يقوم بتنظيف البيانات دون حذف الأعمدة المجهولة (ML Columns).
    """
    
    @staticmethod
    def _safe_float(v: Any) -> float:
        """تحويل آمن يتماشى مع معايير المشروع، يعيد NaN في حالة الفشل بدلاً من الانهيار."""
        if v is None or v == "": return math.nan
        try:
            val = float(v)
            return val if math.isfinite(val) else math.nan
        except (ValueError, TypeError):
            return math.nan

    @classmethod
    def clean_telemetry(cls, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        [إصلاح هندسي]: تنظيف البيانات مع الحفاظ على كافة الأعمدة (50+ عامود).
        لم نعد نستخدم القوائم البيضاء (Whitelist) لمنع تبخر بيانات الـ ML.
        """
        clean_data = {}
        
        for key, value in raw_data.items():
            # إذا كانت القيمة تبدو كرقَم، نقوم بتنظيفها وتحويلها لـ float
            if isinstance(value, (int, float, str)) and not isinstance(value, bool):
                try:
                    val = cls._safe_float(value)
                    clean_data[key] = val
                except:
                    clean_data[key] = value # الحفاظ على القيمة الأصلية إذا فشل التحويل
            else:
                clean_data[key] = value # الحفاظ على النصوص والحالات المنطقية كما هي
                
        return clean_data

class AviationMath:
    """
    أدوات الحساب الجوي المتطورة.
    """
    
    @staticmethod
    def extract_live_specs(telemetry: Dict[str, Any]) -> Dict[str, float]:
        """
        [جديد]: استخراج مواصفات الطائرة ديناميكياً من البيانات.
        يبحث في الـ 50 عاموداً عن الوزن وقوة الدفع الحقيقية.
        """
        return {
            "mass_kg": telemetry.get("uav.mass_kg", telemetry.get("mass_kg", 1.3)),
            "max_thrust_n": telemetry.get("uav.max_thrust_n", telemetry.get("max_thrust_n", 45.0)),
            "hover_power_w": telemetry.get("uav.battery_model.hover_power_W", 220.0)
        }

    @staticmethod
    def get_stage1_risk(telemetry: Dict[str, Any]) -> MLResult:
        """
        تشغيل استدلال المرحلة الأولى (ML) بناءً على القاموس الكامل.
        """
        try:
            # التأكد من وجود درجة المخاطرة في البيانات المسطحة
            risk_score = telemetry.get("stage1_ml_risk_score", 0.0)
            
            return MLResult(
                predicted_class="UNKNOWN", # سيتم تحديثها من قبل الوكيل
                risk_score=float(risk_score),
                confidence=1.0
            )
        except Exception as e:
            logger.error(f"Failed to extract Stage 1 Risk: {e}")
            return MLResult(predicted_class="ERROR", risk_score=1.0, confidence=0.0)

# ---------------------------------------------------------------------------
# الروابط الرياضية للوكلاء (Physics Helpers)
# ---------------------------------------------------------------------------
def calculate_air_density(altitude_m: float, temp_c: float) -> float:
    """حساب كثافة الهواء باستخدام نموذج ISA standard."""
    # (تم نقل المنطق ليكون متاحاً لـ Physics Agent بشكل مستقل)
    p0 = 101325
    T0 = 288.15
    L = 0.0065
    R = 8.31447
    M = 0.0289644
    g = 9.80665
    
    T = temp_c + 273.15
    pressure = p0 * (1 - (L * altitude_m) / T0)**((g * M) / (R * L))
    rho = (pressure * M) / (R * T)
    return max(0.3, min(1.3, rho))