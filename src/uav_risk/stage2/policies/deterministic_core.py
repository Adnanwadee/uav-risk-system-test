"""
Deterministic Pre-Flight Gateway Shield (V12.0 - Agentic Empowerment)
==================================================================
Role: Intercepts the API request BEFORE routing to the ACE LangGraph.
Fixes Applied: 
- Removed arbitrary physical limits (altitude, battery) to allow AI agents to decide.
- Removed ML Veto to transfer authority to the Consensus Agent.
- Focuses strictly on Data Schema Validation (preventing parsing crashes).

Author: Stage 2 — ACE System
"""

import logging
from typing import Optional, Dict, Any
from pydantic import BaseModel, ValidationError, Field
from .config import THRESHOLDS, POLICY_REGISTRY

logger = logging.getLogger("Tier0_Gateway")

# ── Data Contracts ──

class VetoResult(BaseModel):
    """عقد إرجاع نتيجة الرفض."""
    is_veto: bool
    reason: Optional[str] = None
    policy_id: Optional[str] = None
    policy_reference: Optional[str] = None

class Tier0TelemetryContract(BaseModel):
    """
    [تحديث هندسي]: تم جعل جميع الحقول اختيارية (Optional) وبدون قيود رياضية صارمة.
    الهدف الوحيد هنا هو التأكد من أن الهيكل العام للبيانات سليم (No JSON Crash).
    """
    stage1_ml_risk_score: Optional[float] = Field(default=0.0)
    comms_uplink_status: Optional[str] = Field(default="UNKNOWN")
    environment_gnss_jam_dbm: Optional[float] = None
    battery_level_pct: Optional[float] = None
    altitude_m: Optional[float] = None
    population_density: Optional[str] = Field(default="UNKNOWN")

# ── The Core Bouncer ──

class DeterministicCore:
    """The Pre-Flight Bouncer. Executing ONLY schema validation now."""

    @classmethod
    def _create_veto(cls, policy_id: str, dynamic_reason: str) -> VetoResult:
        if policy_id not in POLICY_REGISTRY:
            logger.warning(f"Policy ID '{policy_id}' not found in registry. Using default reference.")
        return VetoResult(
            is_veto=True,
            reason=dynamic_reason,
            policy_id=policy_id,
            policy_reference=POLICY_REGISTRY.get(policy_id, "Unknown Policy Reference")
        )

    @classmethod
    def pre_flight_veto_check(cls, raw_telemetry: Dict[str, Any]) -> VetoResult:
        """
        The Master Tier-0 Check Sequence.
        Now strictly focuses on Data Parsing. Let the AI Agents handle the physical/legal logic.
        """
        logger.info("Executing Tier-0 Deterministic Shield (Schema Validation Only)...")
        
        # 1. التحقق من هيكل البيانات فقط (Data Parsing Valid)
        try:
            validated_data = Tier0TelemetryContract(**raw_telemetry)
        except ValidationError as e:
            logger.error(f"Telemetry Validation Failed: {e}")
            return cls._create_veto(
                "SYS_ERR_00",
                f"Invalid telemetry format. Cannot parse data safely: {e}"
            )
        
        # 2. تمرير جميع السيناريوهات للذكاء الاصطناعي 
        logger.info("Tier-0 Shield Passed. Data is fundamentally viable. Routing to ACE LangGraph.")
        return VetoResult(is_veto=False)

    @classmethod
    def get_health_status(cls) -> Dict[str, Any]:
        """فحص جاهزية البوابة المخففة للمراقبة التشغيلية."""
        return {
            "gateway_status": "healthy_and_liberated",
            "policy_registry_size": len(POLICY_REGISTRY)
        }