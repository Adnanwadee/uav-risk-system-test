"""
Evidence & Traceability Engine (V12 - Hardened Mission Critical)
===============================================================
Role: Builds an immutable, JSON-safe Evidence Pack for post-flight auditing.

Fixes in V12:
- Dynamic Schema Mapping: Replaced hardcoded field counts with Pydantic introspection.
- Severity Enforcement: Implemented Enum-based severity mapping with keyword sets.
- Legal Context+: Enhanced citations with relevance scores and better parsing.
- Audit Transparency: Explicit handling and logging for missing sub-reports.
- Unit Standardization: All time-based fields documented in milliseconds (ms).

Author: Stage 2 — ACE System
"""

from __future__ import annotations
import re
import math
import logging
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# استيراد العقود المعتمدة
from uav_risk.stage2.schemas import ConsensusReport, RuntimeFlightData, Decision, RiskLevel
from uav_risk.stage2.input_contract import MasterFlightPayload

logger = logging.getLogger("EvidenceEngine")

# =================================================================
# 1. Enums & Data Contracts
# =================================================================

class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"

class DataQualityProfile(BaseModel):
    """تحليل جودة البيانات ومدى اكتمال المستشعرات."""
    completeness_ratio: float = Field(..., ge=0.0, le=1.0)
    missing_critical_fields: List[str]
    is_ml_reliable: bool
    data_freshness_ms: float = Field(..., description="Data latency in milliseconds (ms)")

class ForensicRiskDriver(BaseModel):
    """يمثل الدليل الجنائي لسبب وجود خطر معين."""
    agent: str # "PHYSICS", "LEGAL", "TEMPORAL"
    driver: str
    severity: Severity 
    evidence_text: str

class AuditEvidencePack(BaseModel):
    """حزمة الأدلة الشاملة للتدقيق البشري والآلي."""
    flight_id: str
    decision: Decision
    overall_confidence: float
    risk_level: RiskLevel
    quality_profile: DataQualityProfile
    forensic_drivers: List[ForensicRiskDriver]
    legal_citations: List[str]
    raw_snapshot: Dict[str, Any]


# =================================================================
# 2. Evidence Builder (The Hardened Engine)
# =================================================================

class EvidenceBuilder:
    """محرك بناء الأدلة - النسخة المصفحة (V12)."""

    # كلمات مفتاحية لتحديد الشدة (Observation #2)
    CRITICAL_KEYWORDS = {"FATAL", "CRITICAL", "CATASTROPHIC", "FAILURE", "LOSS", "VETO"}
    WARNING_KEYWORDS = {"WARNING", "CAUTION", "RISK", "DEGRADED", "DEVIATION"}

    @staticmethod
    def _deep_sanitize(obj: Any) -> Any:
        """تطهير عودي (Recursive) لضمان توافق JSON."""
        if isinstance(obj, dict):
            return {k: EvidenceBuilder._deep_sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [EvidenceBuilder._deep_sanitize(i) for i in obj]
        elif isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        return obj

    @staticmethod
    def _assess_quality(payload: MasterFlightPayload, elapsed_ms: float) -> DataQualityProfile:
        """
        تقييم جودة البيانات.
        [FIX Observation #1]: استخدام عدد الحقول الفعلي من الـ Schema بدلاً من 15.
        """
        missing = []
        tel = payload.telemetry
        
        # الفحص الصارم للحقول الأساسية
        critical_checks = {
            "battery_pct": tel.battery_state_of_charge_pct,
            "wind_speed": tel.wind_speed_mps,
            "gps_jamming": payload.environment.gnss_jam_dbm,
            "uav_mass": payload.uav.mass_kg
        }
        
        for name, val in critical_checks.items():
            if val is None or (isinstance(val, float) and not math.isfinite(val)):
                missing.append(name)

        # حساب إجمالي الحقول ديناميكياً من Pydantic model_fields
        total_fields = len(tel.model_fields) + len(payload.uav.model_fields)
        present = total_fields - len(missing)
        
        return DataQualityProfile(
            completeness_ratio=present / max(total_fields, 1),
            missing_critical_fields=missing,
            is_ml_reliable=len(missing) <= 1,
            data_freshness_ms=elapsed_ms # Documented in ms (Observation #5)
        )

    @staticmethod
    def _determine_severity(text: str) -> Severity:
        """[FIX Observation #2]: تحديد الشدة بناءً على الكلمات المفتاحية."""
        upper_text = text.upper()
        if any(kw in upper_text for kw in EvidenceBuilder.CRITICAL_KEYWORDS):
            return Severity.CRITICAL
        if any(kw in upper_text for kw in EvidenceBuilder.WARNING_KEYWORDS):
            return Severity.WARNING
        return Severity.INFO

    @staticmethod
    def _extract_forensics(report: ConsensusReport) -> List[ForensicRiskDriver]:
        """تحويل تحذيرات الوكلاء إلى أدلة جنائية منظمة."""
        forensics = []
        
        # 1. Physics Forensics
        if report.physics_report:
            p = report.physics_report
            for warn in p.warnings:
                forensics.append(ForensicRiskDriver(
                    agent="PHYSICS",
                    driver="Aero-Dynamics Failure",
                    severity=EvidenceBuilder._determine_severity(warn),
                    evidence_text=warn
                ))
        
        # 2. Legal Forensics
        if report.legal_report:
            l = report.legal_report
            for viol in l.hard_violations:
                forensics.append(ForensicRiskDriver(
                    agent="LEGAL",
                    driver="Regulatory Violation",
                    severity=Severity.CRITICAL,
                    evidence_text=viol
                ))

        # 3. Temporal Forensics
        if report.temporal_report:
            t = report.temporal_report
            for warn in t.temporal_warnings:
                forensics.append(ForensicRiskDriver(
                    agent="TEMPORAL",
                    driver="Sensor Trend Analysis",
                    severity=EvidenceBuilder._determine_severity(warn),
                    evidence_text=warn
                ))

        return forensics

    @staticmethod
    def compile_audit_pack(
        flight_id: str,
        payload: MasterFlightPayload,
        consensus_report: ConsensusReport,
        processing_time_ms: float
    ) -> AuditEvidencePack:
        """[المايسترو]: تجميع حزمة الأدلة النهائية."""
        
        # [FIX Observation #4]: معالجة غياب تقرير الفيزياء صراحةً
        if not consensus_report.physics_report:
            logger.warning(f"[{flight_id}] Audit Warning: Physics Report missing in consensus.")
            risk_lvl = RiskLevel.CRITICAL # Fail-safe: غياب التقرير يعني خطورة قصوى
        else:
            risk_lvl = consensus_report.physics_report.risk_level

        # 1. تطهير البيانات عودياً
        raw_data = EvidenceBuilder._deep_sanitize(payload.model_dump())
        
        # 2. تقييم الجودة بناءً على الحقول الديناميكية
        quality = EvidenceBuilder._assess_quality(payload, processing_time_ms)
        
        # 3. استخراج الأدلة الجنائية مع تصنيف الشدة المطور
        forensics = EvidenceBuilder._extract_forensics(consensus_report)
        
        # 4. [FIX Observation #3]: استخراج الاستشهادات مع سياق المصدر
        legal_refs = []
        if consensus_report.legal_report:
            for article in consensus_report.legal_report.matched_articles:
                # دمج المصدر مع درجة المطابقة (Relevance) لزيادة السياق
                citation = f"[{article.source_document}] ID: {article.chunk_id} (Relevance: {article.relevance_score:.2f})"
                legal_refs.append(citation)

        return AuditEvidencePack(
            flight_id=flight_id,
            decision=consensus_report.final_decision,
            overall_confidence=consensus_report.calibrated_confidence_score,
            risk_level=risk_lvl,
            quality_profile=quality,
            forensic_drivers=forensics,
            legal_citations=legal_refs,
            raw_snapshot=raw_data
        )