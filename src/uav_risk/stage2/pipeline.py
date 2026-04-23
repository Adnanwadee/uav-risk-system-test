# src/uav_risk/stage2/pipeline.py
from __future__ import annotations
import uuid
import asyncio
import logging
from typing import List
from .schemas import (
    Stage2Request, Stage2Response, EvidenceSnippet, 
    RegulationChunk, AgentState
)
from .graph.safety_agent import safety_agent_app

logger = logging.getLogger(__name__)

# عتبة زمنية آمنة: 15 ثانية كافية للتفكير + استدعاء الأدوات + الصياغة
GRAPH_TIMEOUT_SECONDS = 15.0

async def run_stage2_pipeline(request: Stage2Request) -> Stage2Response:
    """
    The Main Entry Point.
    Passes the request to the LangGraph Agentic Brain with strict safety & timeout controls.
    """
    # 1. تهيئة حالة البداية صراحةً (تتطابق مع AgentState)
    initial_state: AgentState = {
        "flight_id": request.flight_id,
        "scenario": request.scenario,
        "messages": [],
        "reasoning_chain": [],
        "iteration_count": 0,
        "data_quality_profile": None,
        "deterministic_veto_pre": None,
        "deterministic_veto_post": None,
        "agent_decision": None,
        "confidence_score": 0.0,
        "guardrail_passed": True,
        "ml_prediction": None,
        "retrieved_regulations": []
    }

    # 2. إعداد الصندوق الأسود (Audit Trail / Checkpointer)
    thread_id = f"flight_{request.flight_id}_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # 3. تشغيل العقل المدبر مع حماية زمنية صارمة
        final_state = await asyncio.wait_for(
            safety_agent_app.ainvoke(initial_state, config=config),
            timeout=GRAPH_TIMEOUT_SECONDS
        )

        # 4. استخراج الأدلة بشكل آمن (Mapping Regulations -> EvidenceSnippet)
        raw_regs: List[RegulationChunk] = final_state.get("retrieved_regulations", [])
        evidence_list: List[EvidenceSnippet] = [
            EvidenceSnippet(
                source_article=r.article_id,
                content=r.content,
                citation=f"[{r.article_id}]"
            ) for r in raw_regs
        ]

        # 5. تغليف المخرجات في القالب النهائي
        return Stage2Response(
            status="OK",
            flight_id=final_state["flight_id"],
            final_decision=final_state.get("agent_decision") or "NO_GO",
            report_md="\n\n".join(final_state.get("reasoning_chain", [])) or "Decision based on deterministic physics & regulatory compliance.",
            evidence=evidence_list,
            observability_log_id=thread_id
        )

    except asyncio.TimeoutError:
        logger.warning(f"[PIPELINE TIMEOUT] Flight {request.flight_id} exceeded {GRAPH_TIMEOUT_SECONDS}s limit.")
        return Stage2Response(
            status="ERROR_FALLBACK",
            flight_id=request.flight_id,
            final_decision="NO_GO",
            report_md="SYSTEM HALTED: Response timeout exceeded. Physical verification incomplete. Abort flight.",
            evidence=[],
            observability_log_id=thread_id
        )

    except Exception as e:
        logger.error(f"[PIPELINE CRASH] Flight {request.flight_id} failed: {e}")
        return Stage2Response(
            status="ERROR_FALLBACK",
            flight_id=request.flight_id,
            final_decision="NO_GO",
            report_md=f"SYSTEM HALTED: Critical pipeline failure. Error: {str(e)}",
            evidence=[],
            observability_log_id=thread_id
        )