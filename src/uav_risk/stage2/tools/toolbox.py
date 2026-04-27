"""
Aviation Toolbox (V15 - Flight-Ready Deterministic Core)
========================================================
ARCHITECTURAL PIVOT:
LangChain autonomous `@tool` functions have been DEPRECATED in the ACE System.

Fixes in V15 (Aviation Grade Runtime Hardening):
- Runtime NaN Shield: Explicit fail-fast input validation prevents NaN propagation.
- Unified Sentinel Semantics: `math.nan` replaces `-1.0` for all invalid/critical states.
- The "None" String Trap: Fixed handling of `None` values to prevent "NONE" string bugs.
- Performance: Hot-path mathematical functions optimized for Real-Time loops.
- Semantic Typing: `NewType` strictly guides developers, backed by runtime validation.

Author: Stage 2 — ACE System
"""

import math
import logging
from typing import Dict, Any, Set, NewType

from uav_risk.stage1.infer import run_stage1_inference
from uav_risk.stage2.schemas import MLResult

# Note for Production: In a real-time flight control system, configure this logger
# with a `logging.handlers.QueueHandler` to move I/O operations to a background thread.
logger = logging.getLogger("AviationToolbox")

# ---------------------------------------------------------------------------
# Semantic Physical Types (Static Analysis Guidance)
# ---------------------------------------------------------------------------
MetersPerSecond = NewType('MetersPerSecond', float)
Meters = NewType('Meters', float)
Degrees = NewType('Degrees', float)
Percentage = NewType('Percentage', float)

# ---------------------------------------------------------------------------
# Core Libraries
# ---------------------------------------------------------------------------

class AviationMath:
    """مكتبة حسابات فيزيائية قطعية (Stateless). مصممة للأداء العالي في الحلقات."""
    
    # القيمة الحارسة الموحدة (Unified Sentinel Value) الدالة على حالة غير صالحة
    INVALID_STATE_SENTINEL = math.nan
    
    @staticmethod
    def _are_inputs_finite(*args: float) -> bool:
        """[FIX] حارس وقت التشغيل (Runtime Guard): يمنع انتشار الـ NaN والـ Infinity."""
        return all(math.isfinite(arg) for arg in args)

    @staticmethod
    def calculate_crosswind_component(
        wind_speed: MetersPerSecond, 
        wind_direction: Degrees, 
        uav_heading: Degrees
    ) -> float: 
        """
        حساب مركبة الرياح الجانبية.
        يعيد math.nan إذا كانت المدخلات فاسدة.
        """
        if not AviationMath._are_inputs_finite(wind_speed, wind_direction, uav_heading):
            return AviationMath.INVALID_STATE_SENTINEL
            
        angle_diff = math.radians(wind_direction - uav_heading)
        crosswind = abs(wind_speed * math.sin(angle_diff))
        
        return max(0.0, crosswind)

    @staticmethod
    def project_battery_survival(
        current_pct: Percentage, 
        drain_rate_pct_per_min: float, 
        distance_remaining: Meters, 
        speed: MetersPerSecond
    ) -> float:
        """
        توقع نسبة البطارية عند الوصول للهدف.
        [FIX] يعيد math.nan في الحالات الحرجة (سقوط الطائرة أو مدخلات فاسدة).
        """
        # 1. منع انتشار الـ NaN من البداية
        if not AviationMath._are_inputs_finite(current_pct, drain_rate_pct_per_min, distance_remaining, speed):
            return AviationMath.INVALID_STATE_SENTINEL

        # 2. حماية فيزيائية من القسمة على صفر أو الوقوف التام في الجو
        if speed <= 0.1: 
            return AviationMath.INVALID_STATE_SENTINEL
            
        time_to_target_min = (distance_remaining / speed) / 60.0
        
        # 3. منع الشحن السحري (Negative Drain Rate)
        effective_drain_rate = max(0.0, drain_rate_pct_per_min)
        
        battery_consumed = time_to_target_min * effective_drain_rate
        battery_remaining = current_pct - battery_consumed
        
        # 4. تصحيح الحدود
        if battery_remaining > 100.0:
            return 100.0
            
        if battery_remaining < 0.0:
            # البطارية ستنفد قبل الوصول (سقوط الطائرة)
            return AviationMath.INVALID_STATE_SENTINEL
            
        return battery_remaining


class Stage1Bridge:
    """جسر العبور للمرحلة الأولى. آمن ومعزول."""
    
    @staticmethod
    def extract_ml_risk(scenario_data: Dict[str, Any]) -> float:
        try:
            ml_result: MLResult = run_stage1_inference(scenario_data)
            risk_score = float(ml_result.risk_score)
            
            if not math.isfinite(risk_score):
                return 1.0 # خطورة قصوى (Fail-Safe)
                
            return max(0.0, min(1.0, risk_score))
        except Exception as e:
            logger.error(f"Stage 1 Bridge Error. Forcing MAX RISK. Details: {e}")
            return 1.0 


class TelemetryFormatter:
    """
    بوابة الجحيم للبيانات (The Telemetry Gatekeeper).
    لا تسمح بمرور أي قيمة غير معرفة أو فاسدة إلى داخل النظام.
    """
    
    NUMERIC_KEYS: Set[str] = {
        "battery_level_pct", "altitude_m", "wind_speed_ms", # تم التوحيد
        "wind_direction_deg", "uav_heading_deg", "planned_distance_m", 
        "speed_mps", "environment_gnss_jam_dbm", "stage1_ml_risk_score",
        "uav_max_speed_mps", "uav_mass_kg", "uav_max_thrust_n"
    }
    
    STRING_KEYS: Set[str] = {
        "comms_uplink_status", "population_density", "mission_type"
    }
    
    # المفاتيح التي سيؤدي غيابها لرفض الرحلة فوراً
    REQUIRED_KEYS: Set[str] = {
        "battery_level_pct", "altitude_m", "wind_speed_ms", # تم التوحيد
        "comms_uplink_status", "environment_gnss_jam_dbm"
    }
    
    @staticmethod
    def _safe_float(value: Any) -> float:
        """يحول الأرقام بأمان، ويرد NaN في حال الفشل."""
        try:
            val = float(value)
            return val if math.isfinite(val) else math.nan
        except (ValueError, TypeError):
            return math.nan
    
    @staticmethod
    def sanitize_and_normalize(raw_data: Dict[str, Any], strict: bool = True) -> Dict[str, Any]:
        """
        تأمين البيانات. strict=True يضمن عدم إقلاع الطائرة إذا كانت الحساسات معطلة.
        """
        clean_data = {}
        
        for key, value in raw_data.items():
            if key in TelemetryFormatter.NUMERIC_KEYS:
                val = TelemetryFormatter._safe_float(value)
                clean_data[key] = val
                
            elif key in TelemetryFormatter.STRING_KEYS:
                # [FIX] The "None" String Trap Avoidance
                if value is None:
                    clean_data[key] = "UNKNOWN"
                else:
                    clean_str = str(value).strip().upper()
                    if len(clean_str) > 200:
                        logger.warning(f"Truncated overly long string for telemetry key: {key}")
                    clean_data[key] = clean_str[:200] if clean_str else "UNKNOWN"
                    
        # الفحص الإلزامي (Fail-Fast)
        if strict:
            missing = TelemetryFormatter.REQUIRED_KEYS - set(clean_data.keys())
            if missing:
                raise ValueError(f"CRITICAL: Missing mandatory telemetry keys: {missing}")
                
            for req_key in TelemetryFormatter.REQUIRED_KEYS.intersection(TelemetryFormatter.NUMERIC_KEYS):
                if math.isnan(clean_data.get(req_key, math.nan)):
                    raise ValueError(f"CRITICAL: Sensor failure (NaN) detected on critical key '{req_key}'")

        # التطبيع الفيزيائي الإضافي (Clamping)
        if "battery_state_of_charge_pct" in clean_data and not math.isnan(clean_data["battery_state_of_charge_pct"]):
            clean_data["battery_state_of_charge_pct"] = max(0.0, min(100.0, clean_data["battery_state_of_charge_pct"]))
            
        return clean_data