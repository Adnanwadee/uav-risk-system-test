"""
ACE Safety Report Writer (V4.1.1 - Apex Certified)
=================================================
الدور: ربط البيانات بالذكاء الاصطناعي مع ضمان الموثوقية.
المميزات: إعادة محاولة تلقائية (Tenacity)، قياس أداء كل مرحلة، وتحقق مسبق.
"""

import logging
import time
from typing import Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .prompts import SYSTEM_PROMPT, build_agentic_report_prompt
from .verifier import ReportVerifier
from uav_risk.stage2.evidence import EvidenceBuilder
from uav_risk.stage2.schemas import ConsensusReport

logger = logging.getLogger("ReportWriter")

class SafetyReportWriter:
    def __init__(self, llm_client: Any):
        self.llm = llm_client

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError, Exception)),
        reraise=True
    )
    async def _safe_generate(self, user_prompt: str) -> str:
        """توليد التقرير مع حماية ضد أخطاء الشبكة العابرة."""
        start_time = time.perf_counter()
        try:
            return await self.llm.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.0
            )
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(f"LLM Generation Phase completed in {duration_ms:.1f}ms")

    async def generate_comprehensive_report(
        self, 
        flight_id: str, 
        payload: Any, 
        consensus: ConsensusReport, 
        total_pipeline_time_ms: float
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        stage_times = {}

        # 1. بناء حزمة الأدلة الجنائية
        builder = EvidenceBuilder()
        evidence_pack = builder.build_final_pack(
            flight_id=flight_id,
            payload=payload,
            consensus_report=consensus,
            processing_time_ms=total_pipeline_time_ms
        )
        stage_times["evidence_building_ms"] = (time.perf_counter() - t_start) * 1000

        # [FIX] تحقق مسبق من كفاية الأدلة
        if not evidence_pack.forensic_drivers and evidence_pack.decision != "DATA_INSUFFICIENT":
            logger.warning(f"[{flight_id}] Critical warning: Minimal forensic data available.")

        # 2. التوليد عبر الـ LLM مع Retry
        evidence_json = evidence_pack.model_dump_json(indent=2)
        report_md = "REPORT_GENERATION_FAILED"
        is_valid = False
        
        try:
            user_prompt = build_agentic_report_prompt(evidence_json)
            gen_start = time.perf_counter()
            report_md = await self._safe_generate(user_prompt)
            stage_times["llm_generation_ms"] = (time.perf_counter() - gen_start) * 1000
            
            # 3. التحقق العميق (Deep Grounding Guard)
            ver_start = time.perf_counter()
            is_valid, audit_errors = ReportVerifier.verify_grounding(report_md, evidence_pack)
            stage_times["verification_ms"] = (time.perf_counter() - ver_start) * 1000
            
            if not is_valid:
                logger.error(f"Audit Fail for {flight_id}: {audit_errors}")
                report_md += f"\n\n---\n⚠️ **INTEGRITY NOTICE:** This report failed automated grounding check: {audit_errors}"

        except ValueError as ve:
            logger.error(f"[{flight_id}] Logic Error: {ve}")
            report_md = f"# VALIDATION ERROR\nData validation failed during report generation: {str(ve)}"
        except Exception as e:
            logger.critical(f"Report Generation Crash: {e}", exc_info=True)
            report_md = f"# EMERGENCY FALLBACK\nVerdict: {consensus.final_decision}\nStatus: Cognitive Engine Offline."

        total_gen_time = (time.perf_counter() - t_start) * 1000
        
        return {
            "markdown": report_md,
            "structured_data": evidence_pack.model_dump(),
            "metadata": {
                "audit_passed": is_valid,
                "generation_time_ms": round(total_gen_time, 1),
                "stage_breakdown": stage_times,
                "request_id": flight_id
            }
        }