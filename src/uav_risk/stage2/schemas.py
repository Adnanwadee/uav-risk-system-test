"""
Internal System Schemas (V12 - Certified Aviation Grade)
========================================================
Fixes in V12 (Micro-Audits):
- Real-World Physics: `battery_drain_rate_pct_per_min` now allows negative values for regenerative braking/solar charging.
- Statistical Integrity: `mc_samples` mathematically restricted to `ge=1` to prevent Div-by-Zero in variance calculations.
- LangGraph Bootstrapping: Added explicit `INITIAL_AGENT_STATE` documentation.

Author: Stage 2 — ACE System
"""

import operator
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, ConfigDict, computed_field, model_validator
from langchain_core.messages import BaseMessage

# ---------------------------------------------------------------------------
# 1. Core Enumerations
# ---------------------------------------------------------------------------

class Decision(str, Enum):
    GO = "GO"
    CAUTION = "CAUTION"
    NO_GO = "NO-GO"
    DATA_INSUFFICIENT = "DATA_INSUFFICIENT"

class RiskLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

# ---------------------------------------------------------------------------
# 2. Stage 1 & Runtime Foundations
# ---------------------------------------------------------------------------

class MLResult(BaseModel):
    predicted_class: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)


class RuntimeFlightData(BaseModel):
    # [تعديل] تغيير forbid إلى ignore لضمان عدم الانهيار عند وجود حقول إضافية
    model_config = ConfigDict(extra="ignore", frozen=True) 

    battery_level_pct: float = Field(..., ge=0.0, le=100.0) # المسمى الموحد
    battery_drain_rate_pct_per_min: float = Field(...) 
    altitude_m: float = Field(..., ge=0.0, le=20000.0)
    temperature_c: float = Field(..., ge=-80.0, le=80.0)
    wind_speed_ms: float = Field(..., ge=0.0, le=150.0) # المسمى الموحد
    wind_direction_deg: float = Field(..., ge=0.0, le=360.0)
    uav_heading_deg: float = Field(..., ge=0.0, le=360.0)
    planned_distance_m: float = Field(..., ge=0.0)
    speed_mps: float = Field(..., ge=0.0)
    estimated_flight_time_min: float = Field(..., ge=0.0)
    
    projected_wind_ms: Optional[float] = Field(None, ge=0.0)
    projected_battery_pct: Optional[float] = Field(None, ge=0.0, le=100.0)
    stage1_ml_risk_score: float = Field(0.0, ge=0.0, le=1.0)
# ---------------------------------------------------------------------------
# 3. Agent Output Contracts
# ---------------------------------------------------------------------------

class PhysicsRiskReport(BaseModel):
    risk_level: RiskLevel
    go_no_go: Decision
    thrust_margin_ratio: float = Field(..., ge=0.0)
    structural_load_ratio: float = Field(..., ge=0.0)
    battery_margin_pct: float
    wind_tolerance_ratio: float = Field(..., ge=0.0)
    mc_failure_probability: float = Field(..., ge=0.0, le=1.0)
    mc_confidence_interval: Tuple[float, float]
    # [FIX] يجب أن تكون العينة أكبر من الصفر لمنع القسمة على صفر إحصائياً
    mc_samples: int = Field(..., ge=1) 
    projected_risk_level: Optional[RiskLevel] = None
    projected_failure_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    calculation_time_ms: float = Field(..., ge=0.0)
    warnings: List[str] = Field(default_factory=list)
    equations_used: List[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_physics_consistency(self) -> 'PhysicsRiskReport':
        ci_low, ci_high = self.mc_confidence_interval
        if ci_low > ci_high:
            raise ValueError(f"Confidence interval mathematically invalid: {ci_low} > {ci_high}")
        if self.risk_level == RiskLevel.CRITICAL and self.go_no_go not in (Decision.NO_GO, Decision.DATA_INSUFFICIENT):
            raise ValueError("CRITICAL risk level must map to NO-GO or DATA_INSUFFICIENT.")
        return self

class TemporalStateEstimate(BaseModel):
    wind_speed_ms: float = Field(..., ge=0.0)
    wind_speed_variance: float = Field(..., ge=0.0)
    wind_trend_ms_per_min: float
    battery_pct: float = Field(..., ge=0.0, le=100.0)
    battery_variance: float = Field(..., ge=0.0)
    battery_drain_rate_pct_per_min: float
    wind_increasing: bool
    battery_draining_fast: bool
    horizon_min: float = Field(..., ge=0.0)
    projected_wind_ms: float = Field(..., ge=0.0)
    projected_battery_pct: float = Field(..., ge=0.0, le=100.0)
    wind_trend_p_value: float = Field(..., ge=0.0, le=1.0)
    battery_trend_p_value: float = Field(..., ge=0.0, le=1.0)
    temporal_warnings: List[str] = Field(default_factory=list)
    estimation_time_ms: float = Field(0.0, ge=0.0)

class LegalEvidence(BaseModel):
    source_document: str = Field(..., alias="source")
    chunk_id: str
    content: str
    relevance_score: float = Field(..., alias="score", ge=0.0, le=1.0)
    model_config = ConfigDict(populate_by_name=True)

class LegalRiskReport(BaseModel):
    legal_decision: Decision
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    matched_articles: List[LegalEvidence] = Field(default_factory=list)
    hard_violations: List[str] = Field(default_factory=list)
    legal_warnings: List[str] = Field(default_factory=list)
    required_mitigations: List[str] = Field(default_factory=list)
    search_latency_ms: float = Field(0.0, ge=0.0)

    @computed_field
    @property
    def is_compliant(self) -> bool:
        return self.legal_decision in (Decision.GO, Decision.CAUTION)

# ---------------------------------------------------------------------------
# 4. Final Consensus
# ---------------------------------------------------------------------------

class ConsensusMetrics(BaseModel):
    normalized_entropy: float = Field(..., ge=0.0, le=1.0)
    max_divergence: float = Field(..., ge=0.0, le=1.0)
    temporal_degradation_factor: float = Field(..., ge=0.0, le=1.0)

class ConsensusReport(BaseModel):
    final_decision: Decision
    calibrated_confidence_score: float = Field(..., ge=0.0, le=1.0)
    physics_nrs: float = Field(..., ge=0.0, le=1.0)
    legal_decision: Decision
    metrics: ConsensusMetrics
    disqualifying_conditions: List[str] = Field(default_factory=list)
    physics_warnings: List[str] = Field(default_factory=list)
    legal_violations: List[str] = Field(default_factory=list)
    all_warnings: List[str] = Field(default_factory=list)
    required_mitigations: List[str] = Field(default_factory=list)
    deliberation_steps: List[str] = Field(default_factory=list)
    total_time_ms: float = Field(0.0, ge=0.0)

# ---------------------------------------------------------------------------
# 5. LangGraph State Machine
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """
    مساحة الذاكرة العالمية لـ LangGraph.
    
    [CRITICAL INTEGRATION NOTE]
    In LangGraph, TypedDict does NOT enforce defaults at runtime. 
    You MUST pass INITIAL_AGENT_STATE when starting the graph to avoid KeyErrors.
    
    Example usage in pipeline.py:
    INITIAL_AGENT_STATE = {
        "flight_id": "UAV-123",
        "telemetry": runtime_data,
        "messages": [],
        "physics_report": None,
        "temporal_report": None,
        "legal_report": None,
        "consensus_report": None,
        "iteration_count": 0,
        "graph_start_time_ms": time.time() * 1000
    }
    graph.invoke(INITIAL_AGENT_STATE)
    """
    flight_id: str
    telemetry: RuntimeFlightData
    messages: Annotated[list[BaseMessage], operator.add]
    physics_report: Optional[PhysicsRiskReport]
    temporal_report: Optional[TemporalStateEstimate]
    legal_report: Optional[LegalRiskReport]
    consensus_report: Optional[ConsensusReport]
    iteration_count: int
    graph_start_time_ms: float