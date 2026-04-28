"""
ACE System Master Pipeline (V15.0 - Absolute Apex / Mission Critical)
===================================================================
الدور: المايسترو الذي يربط المرحلة الأولى (ML) والمرحلة الثانية (Agents) في تدفق واحد متكامل.

التكاملات العبقرية في V15:
1. الاستيعاب الكامل: يمرر الـ 50 عاموداً كاملة لجميع الوكلاء (No Data Evaporation).
2. استشارة الـ ML: يستدعي المرحلة الأولى كمستشار بنسبة 10% فقط.
3. التدقيق الجنائي: يبني حزمة الأدلة (Evidence Pack) ويفعل نظام التقارير المطور.
4. الصلابة: معالجة شاملة للأخطاء (Circuit Breaker Aware) وتطهير نهائي للبيانات.
"""

import os
import time
import asyncio
import logging
from typing import Dict, Any, Optional

# ─── 1. استيراد المحركات والعقود ───
from uav_risk.stage2.input_contract import MasterFlightPayload
from uav_risk.stage2.schemas import ConsensusReport
from uav_risk.stage2.policies.deterministic_core import DeterministicCore
from uav_risk.stage1.infer import run_stage1_inference
from uav_risk.stage2.evidence import EvidenceBuilder
from uav_risk.utils.json_sanitize import sanitize_for_json

logger = logging.getLogger("ACE_Master_Pipeline")

async def run_ace_pipeline(
    flight_id: str,
    payload: MasterFlightPayload,
    full_telemetry: Dict[str, Any], # القاموس الكامل للـ 50 عاموداً
    graph_app: Any,                # LangGraph Compiled App
    report_writer: Any             # SafetyReportWriter
) -> Dict[str, Any]:
    """
    تشغيل السلسلة الكاملة لتقييم المخاطر (End-to-End).
    """
    t_start = time.perf_counter()
    
    try:
        # ── الخطوة 1: البوابة الحتمية (Tier-0 Gateway) ──
        # التحقق من سلامة هيكل البيانات فقط (Schema Validation)
        veto_check = DeterministicCore.pre_flight_veto_check(full_telemetry)
        if veto_check.is_veto:
            logger.warning(f"[{flight_id}] Tier-0 Veto Triggered: {veto_check.reason}")
            return _build_veto_response(flight_id, veto_check)

        # ── الخطوة 2: استشارة الـ ML (Stage 1 Inference) ──
        # تشغيل الموديل الإحصائي كمستشار بنسبة 10%
        # نقوم بحقن النتيجة داخل القاموس لكي يراها وكيل الإجماع (Consensus Agent)
        ml_consultant_result = run_stage1_inference(full_telemetry)
        full_telemetry["stage1_ml_risk_score"] = ml_consultant_result.risk_score
        full_telemetry["stage1_ml_confidence"] = ml_consultant_result.confidence

        # ── الخطوة 3: تشغيل مجلس الوكلاء (ACE LangGraph) ──
        # إدخال الـ 50 عاموداً في "عروق" النظام (AgentState)
        initial_state = {
            "flight_id": flight_id,
            "telemetry": full_telemetry,
            "messages": [],
            "iteration_count": 0,
            "graph_start_time_ms": t_start * 1000
        }

        # تشغيل الغراف مع نظام الـ Thread ID لمنع تداخل الرحلات
        config = {"configurable": {"thread_id": flight_id}}
        final_graph_state = await graph_app.ainvoke(initial_state, config=config)
        
        consensus: ConsensusReport = final_graph_state.get("consensus_report")
        if not consensus:
            raise RuntimeError("Council failed to reach a consensus report.")

        # ── الخطوة 4: بناء حزمة الأدلة والتقرير الاحترافي ──
        pipeline_time_ms = (time.perf_counter() - t_start) * 1000
        
        # بناء الأدلة الجنائية (Evidence Pack) التي طلبناها
        evidence_pack = EvidenceBuilder.build_final_pack(
            flight_id=flight_id,
            payload=payload,
            consensus_report=consensus,
            full_telemetry=full_telemetry,
            processing_time_ms=pipeline_time_ms
        )

        # توليد التقارير الملحمية (Markdown & JSON)
        report_md = await report_writer.generate_comprehensive_report(
            report=consensus,
            flight_id=flight_id,
            full_telemetry=full_telemetry
        )

        # ── الخطوة 5: تجميع النتيجة النهائية المطهرة ──
        final_output = {
            "status": "SUCCESS",
            "flight_id": flight_id,
            "decision": consensus.final_decision.value,
            "report_markdown": report_md,
            "structured_data": evidence_pack.model_dump(),
            "metrics": {
                "total_time_ms": pipeline_time_ms,
                "confidence": consensus.calibrated_confidence_score,
                "ml_consultant_contribution": "10%"
            }
        }

        # التطهير الأخير لضمان سلامة الـ JSON
        return sanitize_for_json(final_output)

    except Exception as e:
        logger.critical(f"[{flight_id}] Pipeline Catastrophic Failure: {e}", exc_info=True)
        return _build_error_fallback(flight_id, str(e))

def _build_veto_response(flight_id: str, veto: Any) -> Dict[str, Any]:
    """بناء رد سريع في حالة فشل البوابة الأولية."""
    return {
        "status": "VETO",
        "flight_id": flight_id,
        "decision": "NO-GO",
        "reason": veto.reason,
        "policy_reference": veto.policy_reference
    }

def _build_error_fallback(flight_id: str, error_msg: str) -> Dict[str, Any]:
    """رد الطوارئ في حالة حدوث خطأ غير متوقع في النظام."""
    return {
        "status": "SYSTEM_ERROR",
        "flight_id": flight_id,
        "decision": "NO-GO",
        "error_details": error_msg,
        "instructions": "Manual override required. Check ACE logs."
    }