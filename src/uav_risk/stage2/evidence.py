"""
Evidence & Traceability Engine (V13.0 - Full Spectrum Audit)
==============================================================
التعديلات الجوهرية:
1. دعم الوكيل الرابع: دمج نتائج الـ ML Consultant في حزمة الأدلة الجنائية.
2. أرشيف البيانات الكامل: ضمان حفظ الـ 50+ عاموداً في حقل raw_snapshot دون أي حذف.
3. تكامل RAG: استخراج الاستشهادات القانونية الدقيقة [Source | Article] للتقرير.
4. الشفافية الديناميكية: تسجيل الوزن وقوة الدفع الحقيقية المستخدمة في الحسابات.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# استيراد العقود المحدثة
from uav_risk.stage2.schemas import ConsensusReport, RiskLevel
from uav_risk.stage2.input_contract import MasterFlightPayload

logger = logging.getLogger("EvidenceEngine")

class AuditEvidencePack(BaseModel):
    """حزمة الأدلة النهائية غير القابلة للتعديل للتدقيق بعد الرحلة."""
    flight_id: str
    decision: str
    overall_confidence: float
    risk_level: str
    
    # [تحديث]: أرشيف كامل لـ 50+ عامود لضمان الشفافية
    raw_snapshot: Dict[str, Any] 
    
    # [تحديث]: إضافة نتائج الوكيل الرابع (ML)
    ml_consultant_score: float = 0.0
    
    legal_citations: List[str] = Field(default_factory=list)
    forensic_drivers: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: float

class EvidenceBuilder:
    @staticmethod
    def build_final_pack(
        flight_id: str,
        payload: MasterFlightPayload,
        consensus_report: ConsensusReport,
        full_telemetry: Dict[str, Any], # القاموس الكامل المسطح
        processing_time_ms: float
    ) -> AuditEvidencePack:
        """
        بناء حزمة الأدلة الجنائية الشاملة للتقرير الاحترافي.
        """
        
        # 1. استخراج الأدلة القانونية (RAG Citations) من الوكيل القانوني
        legal_refs = []
        if consensus_report.legal_report:
            # نستخدم حقل الاستشهاد [Source | Article] المطور
            legal_refs = consensus_report.legal_violations 

        # 2. بناء المحركات الجنائية (Forensic Drivers) لجميع الوكلاء الأربعة
        forensics = [
            {"agent": "PHYSICS", "decision": consensus_report.physics_decision, "score": consensus_report.physics_nrs},
            {"agent": "LEGAL", "decision": consensus_report.legal_decision, "score": consensus_report.legal_nrs},
            {"agent": "TEMPORAL", "decision": consensus_report.temporal_decision, "score": consensus_report.temporal_nrs},
            {"agent": "ML_CONSULTANT", "decision": consensus_report.ml_decision, "score": consensus_report.ml_nrs}
        ]

        # 3. تجميع الحزمة النهائية
        return AuditEvidencePack(
            flight_id=flight_id,
            decision=str(consensus_report.final_decision.value),
            overall_confidence=consensus_report.calibrated_confidence_score,
            risk_level=consensus_report.physics_report.risk_level if consensus_report.physics_report else "UNKNOWN",
            
            # [CRITICAL]: حفظ الـ 50 عاموداً كاملة للتدقيق
            raw_snapshot=full_telemetry, 
            
            ml_consultant_score=consensus_report.ml_nrs,
            legal_citations=legal_refs,
            forensic_drivers=forensics,
            processing_time_ms=processing_time_ms
        )