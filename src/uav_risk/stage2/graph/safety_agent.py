"""
ACE System Orchestrator (V15.5 - Apex Synchronized)
===========================================================
1. حل خطأ TypeError: إضافة **kwargs لدعم المعاملات الممررة.
2. حل خطأ Consensus: إعادة تمرير المدخلات الأربعة (Physics, Temporal, Legal, Telemetry).
3. اكتمال الـ Failsafes: إضافة دوال التعافي المفقودة لضمان عدم توقف النظام.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
import dataclasses
import operator
from typing import Optional, Literal, TypedDict, List, Annotated, Any, Dict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from uav_risk.stage2.agents.physics_agent import PhysicsAgent, PhysicsRiskReport, RuntimeFlightData
from uav_risk.stage2.agents.temporal_agent import TemporalAgent, TemporalStateEstimate, SensorReading
from uav_risk.stage2.agents.legal_agent import LegalAgent, LegalRiskReport, ComplianceStatus, GoNoGo, ArgumentNode
from uav_risk.stage2.agents.consensus_agent import ConsensusAgent, ConsensusReport, FinalDecision, DeliberationMetrics

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

    # ── محرك المرونة المطور ──
    async def _execute_with_resilience(self, func, *args, timeout: float, retries: int = 1, **kwargs):
        """تم إضافة **kwargs لحل خطأ 'unexpected keyword argument is_math'."""
        for attempt in range(retries + 1):
            try:
                return await asyncio.wait_for(func(*args), timeout=timeout)
            except Exception as e:
                if attempt == retries: raise e
                logger.warning(f"Transient error (attempt {attempt+1}): {e}")
                await asyncio.sleep(0.2)

    def _adapt_telemetry_for_physics(self, telemetry: Dict[str, Any]) -> RuntimeFlightData:
        wind_val = telemetry.get("environment_weather_wind_mps", telemetry.get("wind_speed_ms", 0.0))
        batt_val = telemetry.get("battery_state_of_charge_pct", telemetry.get("battery_level_pct", 100.0))
        return RuntimeFlightData(
            wind_speed_ms=float(wind_val), wind_direction_deg=0.0,
            battery_level_pct=float(batt_val), battery_drain_rate_pct_per_min=1.0,
            altitude_m=float(telemetry.get("altitude_m", 0.0)), temperature_c=25.0,
            planned_distance_m=1000.0, estimated_flight_time_min=10.0,
            mass_kg=float(telemetry.get("uav_mass_kg", 1.5)),
            max_thrust_n=float(telemetry.get("uav_max_thrust_n", 50.0)),
            hover_power_w=float(telemetry.get("uav_battery_model_hover_power_W", 250.0))
        )

    # ── دوال التعافي الكاملة (Failsafes) ──
    def _generate_physics_failsafe(self) -> PhysicsRiskReport:
        return PhysicsRiskReport(
            go_no_go="NO-GO", risk_level="CRITICAL", mc_failure_probability=1.0, 
            mc_confidence_interval=(0.99, 1.0), mc_samples=1, thrust_margin_ratio=0.0, 
            battery_margin_pct=-100.0, structural_load_ratio=2.0, wind_tolerance_ratio=2.0,
            projected_risk_level="CRITICAL", warnings=["PHYSICS_CRASH"], execution_time_ms=0.0
        )

    def _generate_temporal_failsafe(self) -> TemporalStateEstimate:
        return TemporalStateEstimate(
            wind_speed_ms=0.0, wind_speed_variance=0.0, wind_trend_ms_per_min=0.0,
            battery_pct=0.0, battery_variance=0.0, battery_drain_rate_pct_per_min=99.0,
            wind_increasing=False, battery_draining_fast=True, horizon_min=5.0,
            projected_wind_ms=99.0, projected_battery_pct=0.0, temporal_warnings=["TEMPORAL_CRASH"]
        )

    def _generate_legal_failsafe(self) -> LegalRiskReport:
        dummy_arg = ArgumentNode(claim="Agent Crash", is_defeated=True)
        return LegalRiskReport(
            compliance_status=ComplianceStatus.UNCERTAIN, go_no_go=GoNoGo.NO_GO,
            primary_argument=dummy_arg, critical_violations=["LEGAL_CRASH"],
            required_mitigations=["Manual Clearance Required"]
        )

    def _generate_consensus_failsafe(self) -> ConsensusReport:
        """إضافة مفقودة: تحل خطأ 'no attribute _generate_consensus_failsafe'."""
        dummy_metrics = DeliberationMetrics(
            physics_nrs=1.0, temporal_nrs=1.0, legal_nrs=1.0, ml_nrs=1.0, 
            weighted_risk_score=1.0, effective_weights={}, raw_entropy=0, 
            max_entropy=1, normalized_entropy=0, calibrated_confidence_score=0.0,
            decision_method="FAILSAFE", hitl_triggered=True, hitl_reason="System Crash"
        )
        return ConsensusReport(
            final_decision=FinalDecision.NO_GO, mission_authorized=False, 
            risk_summary="FAILSAFE: System component failure.", metrics=dummy_metrics,
            all_warnings=["CONSENSUS_CRASH"], total_time_ms=0.0, agent_times_ms={}
        )

    # ── العقد البرمجية ──
    async def node_physics(self, state: ACEGraphState) -> ACEGraphState:
        try:
            rd = self._adapt_telemetry_for_physics(state.get("telemetry", {}))
            report = await self._execute_with_resilience(self.physics.analyze, rd, timeout=self.timeouts["physics"])
            return {"physics_report": report}
        except Exception as e:
            return {"physics_report": self._generate_physics_failsafe()}

    async def node_temporal(self, state: ACEGraphState) -> ACEGraphState:
        try:
            rd = self._adapt_telemetry_for_physics(state.get("telemetry", {}))
            report = await self._execute_with_resilience(self.temporal.analyze, rd, state.get("sensor_history", []), timeout=self.timeouts["temporal"])
            return {"temporal_estimate": report}
        except Exception as e:
            return {"temporal_estimate": self._generate_temporal_failsafe()}

    async def node_legal(self, state: ACEGraphState) -> ACEGraphState:
        try:
            report = await self._execute_with_resilience(self.legal.analyze, state.get("telemetry", {}), timeout=self.timeouts["legal"])
            return {"legal_report": report}
        except Exception as e:
            return {"legal_report": self._generate_legal_failsafe()}

    async def node_consensus(self, state: ACEGraphState) -> ACEGraphState:
        try:
            # [إصلاح حاسم]: تمرير المعاملات الأربعة المنفصلة لحل خطأ 'missing arguments temporal and legal'
            report = await self._execute_with_resilience(
                self.consensus.deliberate,
                state.get("physics_report"),
                state.get("temporal_estimate"),
                state.get("legal_report"),
                state.get("telemetry", {}),
                timeout=self.timeouts["consensus"],
                is_math=True # المعامل الآن مقبول بفضل **kwargs
            )
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