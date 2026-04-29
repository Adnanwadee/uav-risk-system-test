"""
ACE System Orchestrator (V17.5 - Hybrid Data Extractor)
===========================================================
- إصلاح الكارثة: محرك _deep_get أصبح هجيناً يدعم القواميس المسطحة والمتداخلة لتفادي Pydantic Flattening.
"""

from __future__ import annotations
import asyncio
import logging
import operator
from typing import Optional, Literal, TypedDict, List, Annotated, Any, Dict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from uav_risk.stage2.agents.physics_agent import PhysicsRiskReport, RuntimeFlightData
from uav_risk.stage2.agents.temporal_agent import TemporalStateEstimate, SensorReading
from uav_risk.stage2.agents.legal_agent import LegalRiskReport, ComplianceStatus, GoNoGo, ArgumentNode
from uav_risk.stage2.agents.consensus_agent import ConsensusReport, FinalDecision, DeliberationMetrics

logger = logging.getLogger("ACE_Orchestrator")

class ACEGraphState(TypedDict, total=False):
    telemetry: Dict[str, Any]
    sensor_history: List[SensorReading]
    physics_report: PhysicsRiskReport
    temporal_estimate: TemporalStateEstimate
    legal_report: LegalRiskReport
    consensus_report: ConsensusReport
    human_override_decision: Literal["APPROVED", "REJECTED"]
    audit_trail: Annotated[List[str], operator.add]

class ACESafetyGraph:
    def __init__(self, physics_agent, temporal_agent, legal_agent, consensus_agent):
        self.physics = physics_agent
        self.temporal = temporal_agent
        self.legal = legal_agent
        self.consensus = consensus_agent
        self.timeouts = {"physics": 15.0, "temporal": 5.0, "legal": 10.0, "consensus": 5.0}
        self.graph_app = self._build_graph()

    async def _execute_with_resilience(self, func, *args, timeout: float, retries: int = 1, **kwargs):
        for attempt in range(retries + 1):
            try:
                return await asyncio.wait_for(func(*args), timeout=timeout)
            except Exception as e:
                if attempt == retries: raise e
                logger.warning(f"Transient error (attempt {attempt+1}): {e}")
                await asyncio.sleep(0.2)

    # [الإصلاح الجذري المطلق]: محرك بيانات هجين لا يقهر!
    def _deep_get(self, d: dict, keys: str, default: Any = None) -> Any:
        """محرك استخراج ذكي: يدعم القواميس المسطحة (Flattened) والمتداخلة (Nested)"""
        if not isinstance(d, dict):
            return default
            
        # 1. الخطة أ: هل المفتاح موجود كـ Flat String؟ (مثال: "uav.mass_kg")
        if keys in d:
            return d[keys]
            
        # 2. الخطة ب: إذا كانت البيانات متداخلة Nested JSON، نقوم بالغوص فيها
        current = d
        for key in keys.split('.'):
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                # 3. الخطة ج: ربما تم تحويل النقاط إلى شرطة سفلية (مثال: "uav_mass_kg")
                fallback_key = keys.replace('.', '_')
                if fallback_key in d:
                    return d[fallback_key]
                return default
        return current

    def _adapt_telemetry_for_physics(self, telemetry: Dict[str, Any]) -> RuntimeFlightData:
        wind_val = self._deep_get(telemetry, "environment.weather.wind_mps", 0.0)
        batt_val = self._deep_get(telemetry, "telemetry.battery_level_pct", 100.0)
        alt_val = self._deep_get(telemetry, "telemetry.altitude_m", 0.0)
        
        mass = self._deep_get(telemetry, "uav.mass_kg", 0.1) 
        thrust = self._deep_get(telemetry, "uav.max_thrust_n", 0.1)
        power = self._deep_get(telemetry, "uav.hover_power_w", 250.0)
        uav_type = str(self._deep_get(telemetry, "uav.type", "quadrotor")).lower()
        
        flight_time = self._deep_get(telemetry, "mission.estimated_flight_time_min", 15.0)
        distance = self._deep_get(telemetry, "mission.planned_distance_m", 2000.0)
        drain_rate = self._deep_get(telemetry, "telemetry.battery_drain_rate_pct_per_min", 2.5)

        rd = RuntimeFlightData(
            wind_speed_ms=float(wind_val), wind_direction_deg=0.0,
            battery_level_pct=float(batt_val), battery_drain_rate_pct_per_min=float(drain_rate),
            altitude_m=float(alt_val), temperature_c=25.0,
            planned_distance_m=float(distance), estimated_flight_time_min=float(flight_time),
            mass_kg=float(mass), max_thrust_n=float(thrust), hover_power_w=float(power)
        )
        setattr(rd, 'uav_type', uav_type)
        return rd

    def _generate_physics_failsafe(self) -> PhysicsRiskReport:
        return PhysicsRiskReport(go_no_go="NO-GO", risk_level="CRITICAL", mc_failure_probability=1.0, mc_confidence_interval=(0.99, 1.0), mc_samples=1, thrust_margin_ratio=0.0, battery_margin_pct=-100.0, structural_load_ratio=2.0, wind_tolerance_ratio=2.0, projected_risk_level="CRITICAL", warnings=["PHYSICS_CRASH"], execution_time_ms=0.0)
    
    def _generate_temporal_failsafe(self) -> TemporalStateEstimate:
        return TemporalStateEstimate(
            wind_speed_ms=0.0, wind_speed_variance=0.0, wind_trend_ms_per_min=0.0, 
            battery_pct=0.0, battery_variance=0.0, battery_drain_rate_pct_per_min=1.0, 
            projected_wind_ms=99.0, projected_battery_pct=0.0, 
            wind_increasing=False, battery_draining_fast=True, horizon_min=5.0, 
            wind_trend_p_value=1.0, battery_trend_p_value=1.0, temporal_warnings=["TEMPORAL_CRASH"],
            estimation_time_ms=0.0
        )
        
    def _generate_legal_failsafe(self) -> LegalRiskReport:
        return LegalRiskReport(compliance_status=ComplianceStatus.UNCERTAIN, go_no_go=GoNoGo.NO_GO, primary_argument=ArgumentNode(claim="Agent Crash", is_defeated=True), critical_violations=["LEGAL_CRASH"], required_mitigations=["Manual Clearance"])
    
    def _generate_consensus_failsafe(self) -> ConsensusReport:
        dummy_metrics = DeliberationMetrics(physics_nrs=1.0, temporal_nrs=1.0, legal_nrs=1.0, ml_nrs=1.0, weighted_risk_score=1.0, effective_weights={}, raw_entropy=0.0, max_entropy=1.0, normalized_entropy=0.0, calibrated_confidence_score=0.0, decision_method="FAILSAFE", hitl_triggered=True, hitl_reason="System Crash")
        return ConsensusReport(final_decision=FinalDecision.NO_GO, decision_confidence="LOW", mission_authorized=False, calibrated_confidence_score=0.0, risk_summary="FAILSAFE: Component failure.", physics_decision="NO-GO", physics_nrs=1.0, physics_warnings=["FAILSAFE"], temporal_decision="NO-GO", temporal_nrs=1.0, temporal_warnings=["FAILSAFE"], legal_decision="NO-GO", legal_nrs=1.0, legal_violations=["FAILSAFE"], ml_decision="NO-GO", ml_nrs=1.0, ml_warnings=["FAILSAFE"], metrics=dummy_metrics, hitl_required=True, hitl_reason="Crash Event", all_warnings=["CONSENSUS_CRASH"], required_mitigations=["System Reboot"], disqualifying_conditions=["SYSTEM_FAILURE"], deliberation_steps=["[FAILSAFE]"], total_time_ms=0.0, agent_times_ms={}, physics_report=None, temporal_report=None, legal_report=None)

    async def node_physics(self, state: ACEGraphState) -> ACEGraphState:
        try:
            rd = self._adapt_telemetry_for_physics(state.get("telemetry", {}))
            report = await self._execute_with_resilience(self.physics.analyze, rd, timeout=self.timeouts["physics"])
            return {"physics_report": report}
        except Exception: return {"physics_report": self._generate_physics_failsafe()}

    async def node_temporal(self, state: ACEGraphState) -> ACEGraphState:
        try:
            rd = self._adapt_telemetry_for_physics(state.get("telemetry", {}))
            report = await self._execute_with_resilience(self.temporal.analyze, rd, state.get("sensor_history", []), timeout=self.timeouts["temporal"])
            return {"temporal_estimate": report}
        except Exception as e:
            logger.error(f"Temporal node error: {e}")
            return {"temporal_estimate": self._generate_temporal_failsafe()}

    async def node_legal(self, state: ACEGraphState) -> ACEGraphState:
        try:
            report = await self._execute_with_resilience(self.legal.analyze, state.get("telemetry", {}), timeout=self.timeouts["legal"])
            return {"legal_report": report}
        except Exception: return {"legal_report": self._generate_legal_failsafe()}

    async def node_consensus(self, state: ACEGraphState) -> ACEGraphState:
        try:
            report = self.consensus.deliberate(state.get("physics_report"), state.get("temporal_estimate"), state.get("legal_report"), state.get("telemetry", {}))
            return {"consensus_report": report}
        except Exception as e:
            logger.critical(f"CONSENSUS CRASH: {e}")
            return {"consensus_report": self._generate_consensus_failsafe()}

    def _build_graph(self):
        workflow = StateGraph(ACEGraphState)
        workflow.add_node("node_physics", self.node_physics)
        workflow.add_node("node_temporal", self.node_temporal)
        workflow.add_node("node_legal", self.node_legal)
        workflow.add_node("node_consensus", self.node_consensus)
        workflow.add_edge(START, "node_physics")
        workflow.add_edge(START, "node_temporal")
        workflow.add_edge(START, "node_legal")
        workflow.add_edge("node_physics", "node_consensus")
        workflow.add_edge("node_temporal", "node_consensus")
        workflow.add_edge("node_legal", "node_consensus")
        workflow.add_edge("node_consensus", END)
        return workflow

    def compile(self, checkpointer: Optional[BaseCheckpointSaver] = None):
        return self.graph_app.compile(checkpointer=checkpointer)