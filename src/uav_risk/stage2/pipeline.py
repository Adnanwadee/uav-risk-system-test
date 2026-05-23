# File Path: src/uav_risk/stage2/pipeline.py
import asyncio
import json
import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import structlog
from pydantic import BaseModel, Field

# الاستيرادات المطلقة لجميع المراحل الخمسة للنظام الجوي لضمان الامتثال المعماري
from uav_risk.core.contracts import MasterFlightPayload
from uav_risk.core.data_validator import validate_and_enrich, ValidationResult
from uav_risk.core.imputation_strategy import ImputationStrategy
from uav_risk.core.feature_router import FeatureRouter
from uav_risk.ml.loader import Stage1Bundle
from uav_risk.ml.inference import run_stage1_inference
from uav_risk.ml.schemas import MLResult
from uav_risk.stage2.rag.rag_core import AsyncRAGCore
from uav_risk.stage2.rag.groq_llm import GroqLLM
from uav_risk.stage2.agent.agent_schemas import AgentDecision
from uav_risk.stage2.agent.ace_agent import ACEReActAgent
from uav_risk.stage2.evidence import EvidenceBuilder, AuditEvidencePack
from uav_risk.stage2.llm.report_writer import ReportWriter

logger = structlog.get_logger(__name__)


class VetoResult(BaseModel):
    vetoed: bool = Field(..., description="مؤشر يوضح ما إذا تم تفعيل حظر الطيران القطعي أم لا")
    reason: str = Field(..., description="السبب التنظيمي أو الفيزيائي المفصل للحظر")


class DeterministicCore:
    def pre_flight_veto_check(self, tier0_data: dict) -> VetoResult:
        logger.info("tier0_deterministic_veto_check_initiated", components=list(tier0_data.keys()))
        if tier0_data.get("in_restricted_zone") and float(tier0_data["in_restricted_zone"]) > 0:
            return VetoResult(vetoed=True, reason="Absolute Veto: UAV trajectory breaches designated national no-fly restriction zones.")
        if tier0_data.get("altitude_m") and float(tier0_data["altitude_m"]) > 400.0:
            return VetoResult(vetoed=True, reason="Absolute Veto: Operational altitude ceiling breached strict physical emergency threshold (> 400m).")
        return VetoResult(vetoed=False, reason="Cleared Tier-0 constraints.")


def sanitize_for_json(data: Any) -> Any:
    """مطهّر هيكلي عميق يحول كافة كتل البيانات وأنواع NumPy والـ Enums إلى أنواع JSON متوافقة بدقة."""
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(v) for v in data]
    elif isinstance(data, np.ndarray):
        return sanitize_for_json(data.tolist())
    elif isinstance(data, (np.float64, np.float32, np.float16)):
        return float(data)
    elif isinstance(data, (np.int64, np.int32, np.int16, np.int8)):
        return int(data)
    elif isinstance(data, (np.bool_)):
        return bool(data)
    elif hasattr(data, 'value'):
        return data.value
    elif hasattr(data, '__dict__'):
        try:
            return sanitize_for_json(data.__dict__)
        except Exception:
            return str(data)
    else:
        return data


def _build_veto_response(flight_id: str, veto_result: VetoResult, total_time_ms: float) -> dict:
    return sanitize_for_json({
        "flight_id": flight_id,
        "decision": "NO-GO",
        "risk_score": 1.0,
        "ml_risk_class": "High Risk",
        "ml_risk_score": 1.0,
        "confidence": 1.0,
        "critical_findings": [veto_result.reason],
        "recommendations": ["IMMEDIATE GROUNDING MANDATED BY TIER-0 DETERMINISTIC VETO CORE."],
        "shap_explanation": [],
        "legal_citations": [],
        "report_markdown": f"# ⛔ تقرير حظر طيران فوري قطعي\n\n**السبب:** {veto_result.reason}",
        "audit_passed": True,
        "data_quality_score": 1.0,
        "features_examined_by_agent": 0,
        "rag_queries_made": 0,
        "processing_time_ms": total_time_ms,
        "pipeline_version": "ace_v4.5_veto"
    })


def _build_error_fallback_response(flight_id: str, reason: str, total_time_ms: float) -> dict:
    return sanitize_for_json({
        "flight_id": flight_id,
        "decision": "NO-GO",
        "risk_score": 0.99,
        "ml_risk_class": "High Risk",
        "ml_risk_score": 0.99,
        "confidence": 1.0,
        "critical_findings": [f"Pipeline Execution Failure Sequence Activated: {reason}"],
        "recommendations": ["CRITICAL ADAPTIVE INTELLIGENCE FAILURE."],
        "shap_explanation": [],
        "legal_citations": [],
        "report_markdown": f"# ⚠️ تنبيه خط الأنابيب المركزي: وضع الطوارئ الحركي\n\n**طبيعة العطل:** {reason}",
        "audit_passed": False,
        "data_quality_score": 0.0,
        "features_examined_by_agent": 0,
        "rag_queries_made": 0,
        "processing_time_ms": total_time_ms,
        "pipeline_version": "ace_v4.5_fallback"
    })


async def run_ace_pipeline(
    flight_id: str,
    payload: MasterFlightPayload,
    full_telemetry: dict,
    stage1_bundle: Stage1Bundle,
    rag_core: AsyncRAGCore,
    groq_llm: GroqLLM,
    feature_defs: dict,
    report_writer: ReportWriter
    , precomputed_feature_vector: Optional[np.ndarray] = None
    , precomputed_validation_result: Optional[ValidationResult] = None
) -> dict:
    logger.info("run_ace_pipeline_sequence_initiated", flight_id=flight_id)
    start_time = time.perf_counter()
    
    try:
        # STEP 1 — Tier-0 Veto Verification
        tier0_inputs = payload.to_tier0_dict()
        veto_checker = DeterministicCore()
        veto = veto_checker.pre_flight_veto_check(tier0_inputs)
        
        if veto.vetoed:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return _build_veto_response(flight_id, veto, elapsed_ms)
            
        logger.info("pipeline_gate_tier0_cleared", flight_id=flight_id)
        
        # STEP 2 — Production Data Validation
        if precomputed_validation_result is not None:
            validation_result = precomputed_validation_result
            flat_features = validation_result.validated_features
        else:
            flat_features = payload.flatten_for_ml()
            validation_result = validate_and_enrich(flat_features)
            if not validation_result.is_usable:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return _build_error_fallback_response(flight_id, f"Validation failed: {validation_result.errors}", elapsed_ms)
        
        # STEP 3 — Feature Routing (use precomputed vector if provided)
        if precomputed_feature_vector is not None:
            feature_vector = precomputed_feature_vector
            context_pool = {}
        else:
            feature_mapping = stage1_bundle.feature_mapping
            router = FeatureRouter(feature_defs, feature_mapping)
            feature_vector = router.route_to_vector(validation_result.validated_features)
            context_pool = router.route_to_context_pool(validation_result.validated_features)
        
        # STEP 4 — Machine Learning Inference
        ml_result = run_stage1_inference(bundle=stage1_bundle, feature_vector=feature_vector, feature_names=stage1_bundle.feature_names, compute_shap=True)
        logger.info("pipeline_gate_ml_inference_success", risk_score=ml_result.risk_score)
        
        # STEP 5 — Adaptive ACE ReAct Agent
        agent_engine = ACEReActAgent(llm_client=groq_llm, rag_core=rag_core, feature_defs=feature_defs, config_json=None)
        try:
            agent_decision = await asyncio.wait_for(agent_engine.run(validated_features=validation_result.validated_features, ml_result=ml_result, free_text=payload.free_text), timeout=60.0)
            logger.info("pipeline_gate_agent_reasoning_success", decision=agent_decision.decision)
        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return _build_error_fallback_response(flight_id, "Sovereign ReAct Agent Thinking Loop Timeout.", elapsed_ms)
            
        # STEP 6 — Secure Digital Evidence Building
        evidence_pack = EvidenceBuilder.build_final_pack(flight_id=flight_id, validated_features=validation_result.validated_features, validation_result=validation_result, ml_result=ml_result, agent_decision=agent_decision, raw_telemetry=full_telemetry)
        
        # STEP 7 — Structurally Grounded Report Generation
        current_elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        try:
            report_outcome = await asyncio.wait_for(report_writer.generate_comprehensive_report(flight_id=flight_id, telemetry_dict=full_telemetry, agent_decision=agent_decision, ml_result=ml_result, evidence_pack=evidence_pack, total_pipeline_time_ms=current_elapsed_ms), timeout=25.0)
        except Exception as report_exc:
            logger.error("report_generation_failed_using_inline_markdown_error", error=str(report_exc))
            report_outcome = {"report_markdown": f"# ⛔ خطأ في توليد التقرير السحابي\n\n**تفاصيل العطل:** {str(report_exc)}", "audit_passed": False}
            
        # STEP 8 — Master Payload Consolidation
        total_pipeline_time_ms = (time.perf_counter() - start_time) * 1000.0
        
        citations_unpacked = []
        if agent_decision.legal_citations:
            for citation in agent_decision.legal_citations:
                if hasattr(citation, '__dict__'):
                    citations_unpacked.append(citation.__dict__)
                elif isinstance(citation, dict):
                    citations_unpacked.append(citation)
                    
        shap_clean = []
        if hasattr(ml_result, 'top_features') and ml_result.top_features:
            for f in ml_result.top_features[:5]:
                shap_clean.append({
                    "feature_name": getattr(f, 'feature_name', str(f)),
                    "shap_value": getattr(f, 'shap_value', 0.0),
                    "direction": str(getattr(f, 'direction', 'UNKNOWN'))
                })
        
        # ✅ التصحيح الحتمي: قياس طول المصفوفة بدلاً من التحويل الأعمى لمنع الـ TypeError
        queries_count = len(agent_decision.rag_queries_made) if isinstance(agent_decision.rag_queries_made, list) else int(agent_decision.rag_queries_made or 0)

        master_output = {
            "flight_id": flight_id,
            "decision": str(agent_decision.decision),
            "risk_score": float(agent_decision.overall_risk_score),
            "ml_risk_class": str(ml_result.risk_class.value if hasattr(ml_result.risk_class, 'value') else ml_result.risk_class),
            "ml_risk_score": float(ml_result.risk_score),
            "confidence": float(agent_decision.confidence),
            "critical_findings": agent_decision.critical_findings,
            "recommendations": agent_decision.recommendations,
            "shap_explanation": shap_clean,
            "legal_citations": citations_unpacked,
            "report_markdown": report_outcome["report_markdown"],
            "audit_passed": bool(report_outcome.get("audit_passed", False)),
            "data_quality_score": float(validation_result.overall_data_quality_score),
            "features_examined_by_agent": len(agent_decision.feature_assessments) if agent_decision.feature_assessments else 0,
            "rag_queries_made": queries_count,
            "processing_time_ms": float(total_pipeline_time_ms),
            "pipeline_version": "ACE-v4.5.0-Production-CoreMaster"
        }
        
        return sanitize_for_json(master_output)
        
    except Exception as pipeline_fatal_exc:
        total_failed_time_ms = (time.perf_counter() - start_time) * 1000.0
        return _build_error_fallback_response(flight_id, f"Fatal Master Pipeline Interruption: {str(pipeline_fatal_exc)}", total_failed_time_ms)

# =====================================================================
# Consumed by: src/uav_risk/api/main.py
# =====================================================================