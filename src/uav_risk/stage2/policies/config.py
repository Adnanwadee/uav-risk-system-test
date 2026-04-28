"""
Aviation Regulatory Thresholds (V12.0 - Agentic Context)
=================================================================
Role: Defines reference thresholds. These are NO LONGER used for hard vetoes at Tier 0.
They are kept here to be passed down as context for Legal and Physics agents if needed.
"""

import os
from typing import Dict, Any
from pydantic import BaseModel, Field

# ── Policy Registry (سجل الأسانيد القانونية والهندسية للتدقيق) ──
POLICY_REGISTRY = {
    "SYS_ERR_00": "Critical telemetry parsing failure or missing required sensor data.",
    # تم الإبقاء على باقي السجل كمرجع لوكلاء القانون (Legal Agent) وليس كفيتو حتمي
    "HW_COMMS_01": "Critical loss of Command & Control (C2) link. Violates FAA Part 107.31.",
    "HW_NAV_02": "GNSS Jamming exceeds -65dBm. High risk of autonomous fly-away.",
    "HW_BATT_03": "Battery below safe dispatch minimum. Violates Pre-flight energy reserves.",
    "ML_PREDICTIVE_VETO_01": "Stage 1 Machine Learning model predicts imminent catastrophic failure.",
    "ENV_ALT_01": "Altitude exceeds global absolute ceiling.",
    "ENV_POP_02": "Low altitude flight over dense population. Violates ground risk constraints."
}

class SafetyThresholds(BaseModel):
    """
    [تحديث هندسي]: تم إزالة قيود Pydantic الصارمة (ge, le) لمنع انهيار النظام (ValidationError)
    إذا جاءت قراءات استثنائية. الوكلاء في LangGraph هم من سيقررون مدى خطورة هذه القراءات.
    """
    max_altitude_agl_m: float = Field(default=float(os.getenv("POLICY_MAX_ALTITUDE_M", "120.0")))
    dense_pop_max_alt_m: float = Field(default=float(os.getenv("POLICY_DENSE_POP_MAX_ALT_M", "10.0")))
    critical_jamming_dbm: float = Field(default=float(os.getenv("POLICY_CRIT_JAMMING_DBM", "-65.0")))
    min_dispatch_battery_pct: float = Field(default=float(os.getenv("POLICY_MIN_DISPATCH_BATTERY", "15.0")))
    max_stage1_ml_risk_score: float = Field(default=float(os.getenv("POLICY_MAX_STAGE1_RISK", "1.0")))

    def get_active_policies(self) -> Dict[str, Any]:
        """تصدير السياسات الفعالة الحالية للمراقبة التشغيلية."""
        return self.model_dump()

# Global instance
THRESHOLDS = SafetyThresholds()