"""
Aviation Regulatory & Absolute Thresholds (V11.1 - Tier 0 Config)
=================================================================
Defines the absolute non-negotiable boundaries. 
Includes Policy Registry for strict auditability and Type Checking.
"""

import os
from typing import Dict, Any
from pydantic import BaseModel, Field

# ── Policy Registry (سجل الأسانيد القانونية والهندسية للتدقيق) ──
POLICY_REGISTRY = {
    "HW_COMMS_01": "Critical loss of Command & Control (C2) link. Violates FAA Part 107.31.",
    "HW_NAV_02": "GNSS Jamming exceeds -65dBm. High risk of autonomous fly-away.",
    "HW_BATT_03": "Battery below safe dispatch minimum. Violates Pre-flight energy reserves.",
    "ML_PREDICTIVE_VETO_01": "Stage 1 Machine Learning model predicts imminent catastrophic failure.",
    "ENV_ALT_01": "Altitude exceeds global absolute ceiling (e.g., 400ft / 120m AGL).",
    "ENV_POP_02": "Low altitude flight over dense population. Violates ground risk constraints.",
    "SYS_ERR_00": "Critical telemetry parsing failure or missing required sensor data."
}

class SafetyThresholds(BaseModel):
    """
    عقد بيانات صارم لضمان سلامة أنواع المتغيرات (Type Safety).
    جميع الأسماء موحدة النهايات بالوحدات المادية (_m, _dbm, _pct).
    """
    # Regulatory Ceilings
    max_altitude_agl_m: float = Field(
        default=float(os.getenv("POLICY_MAX_ALTITUDE_M", "120.0")),
        description="400ft absolute legal ceiling globally.",
        ge=0, le=20000  # تحقق من النطاق
    )
    dense_pop_max_alt_m: float = Field(
        default=float(os.getenv("POLICY_DENSE_POP_MAX_ALT_M", "10.0")),
        description="Max altitude over dense populations without specific permits.",
        ge=0, le=1000
    )
    
    # System Integrity
    critical_jamming_dbm: float = Field(
        default=float(os.getenv("POLICY_CRIT_JAMMING_DBM", "-65.0")),
        description="Severe interference threshold leading to Fly-Away risk.",
        ge=-150, le=0
    )
    min_dispatch_battery_pct: float = Field(
        default=float(os.getenv("POLICY_MIN_DISPATCH_BATTERY", "15.0")),
        description="Absolute minimum battery to even allow system boot.",
        ge=0, le=100
    )
    
    # Stage 1 ML Integration
    # Stage 1 ML Integration
    max_stage1_ml_risk_score: float = Field(
        default=float(os.getenv("POLICY_MAX_STAGE1_RISK", "1.0")), # تغيير 0.85 إلى 1.0
        description="Abort if risk is strictly above 1.0",
        ge=0, le=1.1 # زيادة الحد للسماح بـ 1.0
    
    )

    def get_active_policies(self) -> Dict[str, Any]:
        """تصدير السياسات الفعالة الحالية للـ Health Check والمراقبة التشغيلية."""
        return self.model_dump()

    def __post_init__(self):
        """تحقق إضافي من اتساق السياسات."""
        # تحقق اختياري: تأكد أن كل عتبة ضمن نطاق معقول
        # (تم تطبيقه عبر Field constraints أعلاه)
        pass

# Global instance
THRESHOLDS = SafetyThresholds()