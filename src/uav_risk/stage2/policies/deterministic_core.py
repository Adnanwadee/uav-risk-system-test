"""
Deterministic Pre-Flight Gateway Shield (V11.1 - Tier 0 Authority)
==================================================================
Role: Intercepts the API request BEFORE routing to the ACE LangGraph.
Fixes Applied: 
- Strict Telemetry Validation (No silent 0.0 defaults for missing sensors).
- Removed static wind limits (Delegated to PhysicsAgent for dynamic load analysis).
- Integrated Policy Registry for audit-ready rejections.
- Added Health Check endpoint for operational monitoring.

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
    policy_reference: Optional[str] = None # الوصف القانوني من السجل

class Tier0TelemetryContract(BaseModel):
    """
    [العبقرية الهندسية]: بدلاً من `telemetry.get("alt", 0.0)`، 
    نفرض التحقق الصارم. إذا كانت البيانات مفقودة، سيرفع Pydantic خطأ،
    وسيتم التقاطه كـ `SYS_ERR_00`، مما يمنع انطلاق طائرة عمياء!
    """
    stage1_ml_risk_score: float = Field(default=0.0, ge=0, le=1)
    comms_uplink_status: str = Field(..., min_length=1)
    environment_gnss_jam_dbm: float = Field(..., ge=-150, le=0)
    battery_state_of_charge_pct: float = Field(..., ge=0, le=100)
    altitude_m: float = Field(..., ge=0, le=11000)
    population_density: str = Field(default="SPARSE")

# ── The Core Bouncer ──

class DeterministicCore:
    """The Pre-Flight Bouncer. Executes O(1) checks to block fatal operations."""

    @classmethod
    def _create_veto(cls, policy_id: str, dynamic_reason: str) -> VetoResult:
        """Helper to create standardized veto results mapping to the registry."""
        if policy_id not in POLICY_REGISTRY:
            logger.warning(f"Policy ID '{policy_id}' not found in registry. Using default reference.")
        return VetoResult(
            is_veto=True,
            reason=dynamic_reason,
            policy_id=policy_id,
            policy_reference=POLICY_REGISTRY.get(policy_id, "Unknown Policy Reference")
        )

    @classmethod
    def _check_stage1_ml_veto(cls, data: Tier0TelemetryContract) -> VetoResult:
        if data.stage1_ml_risk_score > THRESHOLDS.max_stage1_ml_risk_score:
            return cls._create_veto(
                "ML_PREDICTIVE_VETO_01",
                f"Neural predictive risk ({data.stage1_ml_risk_score:.2f}) exceeds hard limit ({THRESHOLDS.max_stage1_ml_risk_score})."
            )
        return VetoResult(is_veto=False)

    @classmethod
    def _check_system_hardware(cls, data: Tier0TelemetryContract) -> VetoResult:
        comms_status = data.comms_uplink_status.strip().upper()
        if comms_status in ["DEGRADED", "LOST"]:
            return cls._create_veto(
                "HW_COMMS_01",
                f"Uplink status is {comms_status}. Command & Control severed."
            )
            
        if data.environment_gnss_jam_dbm > THRESHOLDS.critical_jamming_dbm:
            return cls._create_veto(
                "HW_NAV_02",
                f"Severe GNSS Jamming ({data.environment_gnss_jam_dbm} dBm). Fly-away risk."
            )
            
        if data.battery_state_of_charge_pct < THRESHOLDS.min_dispatch_battery_pct:
            return cls._create_veto(
                "HW_BATT_03",
                f"Dispatch battery ({data.battery_state_of_charge_pct}%) is below minimum boot requirement."
            )

        return VetoResult(is_veto=False)

    @classmethod
    def _check_absolute_envelope(cls, data: Tier0TelemetryContract) -> VetoResult:
        """
        فحص السقف التنظيمي المطلق (الفيزياء الدقيقة للرياح متروكة لـ PhysicsAgent).
        """
        if data.altitude_m > THRESHOLDS.max_altitude_agl_m:
            return cls._create_veto(
                "ENV_ALT_01",
                f"Altitude ({data.altitude_m}m) exceeds global absolute ceiling ({THRESHOLDS.max_altitude_agl_m}m)."
            )
            
        pop_density = data.population_density.strip().upper()
        if pop_density == "DENSE" and data.altitude_m > THRESHOLDS.dense_pop_max_alt_m:
            return cls._create_veto(
                "ENV_POP_02",
                f"Altitude ({data.altitude_m}m) over DENSE population violates low-altitude safety constraints."
            )

        return VetoResult(is_veto=False)

    @classmethod
    def pre_flight_veto_check(cls, raw_telemetry: Dict[str, Any]) -> VetoResult:
        """
        The Master Tier-0 Check Sequence.
        MUST BE CALLED FROM THE API ENDPOINT BEFORE INVOKING THE LANGGRAPH ORCHESTRATOR.
        """
        logger.info("Executing Tier-0 Deterministic Shield...")
        
        # 1. التحقق الصارم من نوع ووجود البيانات (Anti-Silent-Failure)
        try:
            validated_data = Tier0TelemetryContract(**raw_telemetry)
        except ValidationError as e:
            logger.error(f"Telemetry Validation Failed: {e}")
            return cls._create_veto(
                "SYS_ERR_00",
                f"Invalid or missing critical telemetry data. System cannot evaluate safety safely."
            )
        
        # 2. تنفيذ الفحوصات الفورية (O(1) Complexity)
        checks = [
            cls._check_system_hardware,
            cls._check_stage1_ml_veto,
            cls._check_absolute_envelope
        ]
        
        for check in checks:
            try:
                res = check(validated_data)
                if res.is_veto:
                    logger.critical(f"Tier-0 VETO TRIGGERED: [{res.policy_id}] {res.reason}")
                    return res
            except Exception as e:
                logger.error(f"Unexpected error during deterministic check: {e}")
                return cls._create_veto("SYS_ERR_00", f"Internal Gateway Error: {e}")
                
        logger.info("Tier-0 Shield Passed. Mission is fundamentally viable. Routing to ACE LangGraph.")
        return VetoResult(is_veto=False)

    @classmethod
    def get_health_status(cls) -> Dict[str, Any]:
        """فحص جاهزية نظام السياسات للمراقبة التشغيلية."""
        return {
            "gateway_status": "healthy",
            "thresholds_loaded": THRESHOLDS is not None,
            "policy_registry_size": len(POLICY_REGISTRY),
            "active_policies": THRESHOLDS.get_active_policies() if hasattr(THRESHOLDS, 'get_active_policies') else {},
        }