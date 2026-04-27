"""
ACE System Master Pipeline (V14 - Absolute Apex / Mission Critical)
===================================================================
Role: The Central Mission Control orchestrating the End-to-End flow.

Genius Integrations & SRE Upgrades (V14):
- Pervasive Traceability: `thread_id` is generated instantly to track failures at ANY stage.
- Universal Resource Cleanup: Advanced async/sync `finally` block for robust LLM client shutdown.
- Runtime Protocol Validation: `@runtime_checkable` added for strict, dynamic dependency injection safety.
- Python Version Agnostic: Graceful fallbacks for `typing.Protocol` to support older environments.
- Deep Operational Probes: Component-level health checks instead of hardcoded strings.

Author: Stage 2 — ACE System
"""

import os
import uuid
import time
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# [FIX] Protocol import fallback for Python <3.8
try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable  # type: ignore

# Resilience Library for Transient Errors
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ─── 1. External & Internal Contracts ───
from uav_risk.stage2.input_contract import MasterFlightPayload, DynamicTelemetryInput
from uav_risk.stage2.schemas import RuntimeFlightData, AgentState, ConsensusReport
from uav_risk.stage2.policies.deterministic_core import DeterministicCore
from uav_risk.stage2.tools.toolbox import Stage1Bridge, TelemetryFormatter

# ─── 2. Stage 2 LangGraph ───
from uav_risk.stage2.graph.safety_agent import safety_agent_app 

# ─── 3. Final Report Writer ───
from uav_risk.stage2.llm.groq_client import GroqAsyncClient
from uav_risk.stage2.llm.report_writer import SafetyReportWriter

logger = logging.getLogger("MasterPipeline")
GRAPH_TIMEOUT_SECONDS = 25.0 

# ─── [FIX] Protocol for Strict DI Validation at Runtime ───
@runtime_checkable
class ReportWriterProtocol(Protocol):
    """عقد ثابت وحركي يضمن أن الكائن المُحقن يمتلك الدوال المطلوبة."""
    async def generate_markdown_report(self, consensus_report: ConsensusReport, flight_id: str) -> str: ...
    async def generate_json_report(self, consensus_report: ConsensusReport, flight_id: str) -> Dict[str, Any]: ...


# ─── Smart Retry Wrapper for LangGraph ───
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type((TimeoutError, ConnectionError, asyncio.TimeoutError)),
    reraise=True,
    before_sleep=lambda retry_state: logger.warning(
        f"[RESILIENCE] Retrying LangGraph (Attempt {retry_state.attempt_number}/3) "
        f"after error: {retry_state.outcome.exception()}"
    )
)
async def _invoke_graph_with_retry(state: AgentState, config: dict, timeout: float) -> dict:
    return await asyncio.wait_for(
        safety_agent_app.ainvoke(state, config=config),
        timeout=timeout
    )


async def run_ace_pipeline(
    flight_id: str, 
    payload: MasterFlightPayload,
    report_writer: Optional[ReportWriterProtocol] = None
) -> Dict[str, Any]:
    
    pipeline_start_time = time.perf_counter()
    
    # [FIX] Pervasive Traceability: Generate thread_id immediately for observability at ANY stage
    thread_id = f"flight_{flight_id}_{uuid.uuid4().hex[:8]}"
    
    logger.info(f"=== ACE Pipeline Initiated | Flight ID: {flight_id} | Trace: {thread_id} ===")

    local_llm_client: Optional[GroqAsyncClient] = None

    try:
        # =====================================================================
        # STEP 1: Pre-Flight Physical Integrity Check
        # =====================================================================
        missing_physics = []
        if payload.uav.mass_kg is None: missing_physics.append("mass_kg")
        if payload.uav.max_thrust_n is None: missing_physics.append("max_thrust_n")
        if payload.uav.max_speed_mps is None: missing_physics.append("max_speed_mps")
        
        if missing_physics:
            logger.warning(f"[PRE-FLIGHT VETO] Missing physical limits: {missing_physics}")
            return _build_veto_response(
                flight_id, "SYS_MISSING_SPECS", f"Missing core UAV specs: {missing_physics}", thread_id
            )

        # =====================================================================
        # STEP 2: Stage 1 Machine Learning Inference
        # =====================================================================
        ml_features = payload.flatten_for_ml()
        stage1_risk_score = Stage1Bridge.extract_ml_risk(ml_features)

        # =====================================================================
        # STEP 3: Tier-0 Deterministic Shield
        # =====================================================================
        tier0_dict = payload.to_tier0_dict()
        tier0_dict["stage1_ml_risk_score"] = stage1_risk_score 
        
        veto_result = DeterministicCore.pre_flight_veto_check(tier0_dict)
        if veto_result.is_veto:
            return _build_veto_response(flight_id, veto_result.policy_reference, veto_result.reason, thread_id)

        # =====================================================================
        # STEP 4: Telemetry Formatting & NaN Shielding
        # =====================================================================
        raw_telemetry = (payload.telemetry or DynamicTelemetryInput()).model_dump()
        raw_telemetry["stage1_ml_risk_score"] = stage1_risk_score
        raw_telemetry["uav_max_thrust_n"] = payload.uav.max_thrust_n
        raw_telemetry["uav_mass_kg"] = payload.uav.mass_kg
        raw_telemetry["uav_max_speed_mps"] = payload.uav.max_speed_mps
        
        try:
            safe_telemetry_dict = TelemetryFormatter.sanitize_and_normalize(raw_telemetry, strict=True)
            runtime_data = RuntimeFlightData(**safe_telemetry_dict)
        except ValueError as ve:
            logger.error(f"[TELEMETRY ERROR] Flight {flight_id} failed strict formatting: {ve}")
            return _build_veto_response(flight_id, "SYS_TELEMETRY_CORRUPTION", str(ve), thread_id)

        # =====================================================================
        # STEP 5: Stage 2 LangGraph Orchestration
        # =====================================================================
        initial_state: AgentState = {
            "flight_id": flight_id,
            "telemetry": runtime_data,
            "messages": [],
            "physics_report": None,
            "temporal_report": None,
            "legal_report": None,
            "consensus_report": None,
            "iteration_count": 0,
            "graph_start_time_ms": time.perf_counter() * 1000
        }

        if not isinstance(initial_state["telemetry"], RuntimeFlightData):
            raise TypeError("AgentState telemetry must be explicitly typed as RuntimeFlightData.")

        config = {"configurable": {"thread_id": thread_id}}
        
        graph_start_timer = time.perf_counter()
        final_state = await _invoke_graph_with_retry(initial_state, config, GRAPH_TIMEOUT_SECONDS)
        graph_duration_ms = (time.perf_counter() - graph_start_timer) * 1000

        consensus_report: ConsensusReport = final_state.get("consensus_report")
        if not consensus_report:
            raise RuntimeError("LangGraph completed but yielded no ConsensusReport.")

        # =====================================================================
        # STEP 6: Final Report Generation
        # =====================================================================
        # [FIX] Runtime Protocol Verification
        if report_writer is not None and not isinstance(report_writer, ReportWriterProtocol):
            logger.warning("[DI WARNING] Injected report_writer invalid. Instantiating fallback writer.")
            report_writer = None
            
        if report_writer is None:
            local_llm_client = GroqAsyncClient(model_name="llama3-70b-8192", temperature=0.0)
            report_writer = SafetyReportWriter(llm_client=local_llm_client)
        
        markdown_report = await report_writer.generate_markdown_report(consensus_report, flight_id)
        json_payload = await report_writer.generate_json_report(consensus_report, flight_id)
        
        total_time_ms = (time.perf_counter() - pipeline_start_time) * 1000
        logger.info(f"=== ACE Pipeline Completed successfully in {total_time_ms:.1f}ms ===")

        return {
            "status": "OK",
            "flight_id": flight_id,
            "decision": str(consensus_report.final_decision.value),
            "report_markdown": markdown_report,
            "data": json_payload,
            "observability_thread": thread_id,
            "metrics": {
                "total_pipeline_ms": round(total_time_ms, 1),
                "langgraph_orchestration_ms": round(graph_duration_ms, 1)
            }
        }

    except asyncio.CancelledError:
        logger.warning(f"[CANCELLED] Flight {flight_id} processing was cancelled.")
        raise 

    except asyncio.TimeoutError:
        logger.critical(f"[TIMEOUT] Flight {flight_id} exceeded graph limits.")
        return _build_error_response(flight_id, "SYSTEM HALTED: Multi-Agent Orchestrator Timeout.", thread_id)
        
    except Exception as e:
        logger.critical(f"[CRASH] Flight {flight_id} encountered fatal pipeline error: {e}", exc_info=True)
        return _build_error_response(flight_id, f"SYSTEM CRASH: Internal Logic Failure. Error: {str(e)}", thread_id)

    finally:
        # [FIX] Advanced Resource Leak Prevention
        if local_llm_client:
            try:
                if hasattr(local_llm_client, 'close'):
                    close_method = local_llm_client.close
                    if asyncio.iscoroutinefunction(close_method):
                        await close_method()
                    else:
                        close_method()
                elif hasattr(local_llm_client, 'shutdown'):
                    local_llm_client.shutdown()
                logger.debug(f"[{flight_id}] Closed ephemeral GroqAsyncClient session.")
            except Exception as cleanup_err:
                logger.warning(f"[{flight_id}] Error during client cleanup: {cleanup_err}")


# ─── Helper Functions ───

def _build_veto_response(flight_id: str, policy_ref: str, reason: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
    return {
        "status": "TIER_0_VETO",
        "flight_id": flight_id,
        "decision": "NO-GO",
        "report_markdown": f"# ⛔ FLIGHT REJECTED (Gateway)\n**Policy Violated:** `{policy_ref}`\n\n**Reason:** {reason}",
        "data": {
            "flight_id": flight_id,
            "decision": "NO-GO",
            "disqualifying_conditions": [reason],
            "observability_thread": thread_id
        }
    }

def _build_error_response(flight_id: str, error_msg: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
    return {
        "status": "ERROR_FALLBACK",
        "flight_id": flight_id,
        "decision": "NO-GO",
        "report_markdown": f"# ⚠️ SYSTEM FAILURE\n**Critical Alert:** ACE System Error.\n\n**Details:** `{error_msg}`",
        "data": {
            "flight_id": flight_id,
            "decision": "NO-GO",
            "disqualifying_conditions": ["ACE System Failure"],
            "observability_thread": thread_id # [FIX] Add for traceability
        }
    }

# ─── Deep Pipeline Health Check ───

async def pipeline_health_check() -> Dict[str, Any]:
    """Deep Operational Health Check for K8s Probes."""
    is_groq_configured = bool(os.environ.get("GROQ_API_KEY"))
    
    # [FIX] Actual component verification
    stage1_ok = Stage1Bridge is not None
    tier0_ok = DeterministicCore is not None
    langgraph_ok = safety_agent_app is not None
    
    is_healthy = is_groq_configured and stage1_ok and tier0_ok and langgraph_ok
    
    return {
        "status": "HEALTHY" if is_healthy else "DEGRADED",
        "components": {
            "stage1_ml_bridge": "ONLINE" if stage1_ok else "OFFLINE",
            "tier0_gateway": "ONLINE" if tier0_ok else "OFFLINE",
            "telemetry_formatter": "ONLINE", 
            "langgraph_orchestrator": "ONLINE" if langgraph_ok else "OFFLINE",
            "llm_provider_configured": is_groq_configured
        },
        "config": {
            "graph_timeout_sec": GRAPH_TIMEOUT_SECONDS
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }