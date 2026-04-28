"""
ACE Safety Report Writer (V5.0.0 - Comprehensive)
"""

import logging
import time
from typing import Dict, Any
from .prompts import SYSTEM_PROMPT, build_agentic_report_prompt
from .verifier import ReportVerifier
from uav_risk.stage2.schemas import ConsensusReport

logger = logging.getLogger("ReportWriter")

class SafetyReportWriter:
    def __init__(self, llm_client: Any):
        self.llm = llm_client

    async def generate_comprehensive_report(
        self, 
        flight_id: str, 
        telemetry_dict: Dict[str, Any], # الـ 50+ عامود كاملة
        consensus: ConsensusReport, 
        total_pipeline_time_ms: float
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        
        # 1. تجهيز البيانات للـ LLM (تحويل كل شيء لـ JSON)
        evidence_pack = {
            "flight_id": flight_id,
            "decision": consensus.final_decision,
            "confidence": consensus.calibrated_confidence_score,
            "agents": {
                "physics": consensus.physics_decision,
                "legal": consensus.legal_decision,
                "temporal": consensus.temporal_decision,
                "ml": consensus.ml_decision
            },
            "legal_rag_citations": consensus.legal_violations, # معلومات RAG
            "raw_snapshot": telemetry_dict, # عرض كل البيانات (50+ عامود)
            "pipeline_latency": total_pipeline_time_ms
        }

        # 2. التوليد عبر الـ LLM
        evidence_json = str(evidence_pack)
        user_prompt = build_agentic_report_prompt(evidence_json)
        
        report_md = await self.llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1
        )

        # 3. التدقيق البعدي (Verifier)
        is_valid, errors = ReportVerifier.verify_grounding(report_md, evidence_pack)

        return {
            "markdown": report_md,
            "audit_passed": is_valid,
            "audit_errors": [str(e) for e in errors],
            "metadata": {
                "flight_id": flight_id,
                "generation_time_ms": (time.perf_counter() - t_start) * 1000
            }
        }