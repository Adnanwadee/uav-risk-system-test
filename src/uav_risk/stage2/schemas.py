"""
Internal System Schemas (V15.1 - Apex Unified & Fixed)
========================================================
Fixes in V15.1:
- Added 'projected_risk_level' to PhysicsRiskReport to prevent consensus crash.
- Added explicit report references to ConsensusReport for the Evidence Engine.
- Maintained 50+ column support and dynamic specs.
"""

import operator
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.messages import BaseMessage

# ---------------------------------------------------------------------------
# 1. التعدادات الأساسية (Core Enums)
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

class FinalDecision(str, Enum):
    GO = "GO"
    CAUTION = "CAUTION"
    NO_GO = "NO-GO"

# ---------------------------------------------------------------------------
# 2. نتائج الذكاء الاصطناعي وبيانات الرحلة (Foundation)
# ---------------------------------------------------------------------------

class MLResult(BaseModel):
    predicted_class: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    drift_score: Optional[float] = 0.0

class RuntimeFlightData(BaseModel):
    """بيانات الرحلة التشغيلية مع دعم المواصفات الديناميكية."""
    model_config = ConfigDict(extra="allow")

    battery_level_pct: float
    battery_drain_rate_pct_per_min: float
    altitude_m: float
    temperature_c: float
    wind_speed_ms: float
    wind_direction_deg: float
    planned_distance_m: float
    estimated_flight_time_min: float
    
    mass_kg: Optional[float] = None
    max_thrust_n: Optional[float] = None
    hover_power_w: Optional[float] = None

# ---------------------------------------------------------------------------
# 3. عقود تقارير الوكلاء (Agent Output Contracts)
# ---------------------------------------------------------------------------

class PhysicsRiskReport(BaseModel):
    go_no_go: str
    risk_level: str
    # [FIX]: إضافة الحقل المفقود الذي تسبب في انهيار وكيل الإجماع
    projected_risk_level: Optional[str] = "LOW" 
    mc_failure_probability: float
    mc_confidence_interval: Tuple[float, float]
    mc_samples: int
    thrust_margin_ratio: float
    battery_margin_pct: float
    structural_load_ratio: float
    wind_tolerance_ratio: float
    warnings: List[str] = Field(default_factory=list)
    execution_time_ms: float = 0.0

class TemporalStateEstimate(BaseModel):
    wind_speed_ms: float
    battery_pct: float
    projected_wind_ms: float
    projected_battery_pct: float
    wind_increasing: bool
    battery_draining_fast: bool
    temporal_warnings: List[str] = Field(default_factory=list)
    estimation_time_ms: float = 0.0

class LegalEvidence(BaseModel):
    source_document: str
    chunk_id: str
    exact_quote: str
    relevance_score: float

class LegalRiskReport(BaseModel):
    compliance_status: Any
    go_no_go: Any
    critical_violations: List[str] = Field(default_factory=list)
    required_mitigations: List[str] = Field(default_factory=list)
    # [إضافة]: حقل المقالات المطابقة لدعم محرك الأدلة
    matched_articles: List[LegalEvidence] = Field(default_factory=list)
    execution_time_ms: float = 0.0

# ---------------------------------------------------------------------------
# 4. تقرير الإجماع النهائي (Consensus)
# ---------------------------------------------------------------------------

class DeliberationMetrics(BaseModel):
    normalized_entropy: float
    hitl_triggered: bool
    hitl_reason: Optional[str] = None

class ConsensusReport(BaseModel):
    final_decision: FinalDecision
    calibrated_confidence_score: float
    physics_decision: str
    physics_nrs: float
    legal_decision: str
    legal_nrs: float
    temporal_decision: str
    temporal_nrs: float
    ml_decision: str
    ml_nrs: float
    
    # [FIX]: إضافة مراجع التقارير الفرعية لكي يتمكن محرك الأدلة (Evidence Engine) من العثور عليها
    physics_report: Optional[PhysicsRiskReport] = None
    legal_report: Optional[LegalRiskReport] = None
    temporal_report: Optional[TemporalStateEstimate] = None
    
    legal_violations: List[str] = Field(default_factory=list)
    all_warnings: List[str] = Field(default_factory=list)
    required_mitigations: List[str] = Field(default_factory=list)
    disqualifying_conditions: List[str] = Field(default_factory=list)
    metrics: DeliberationMetrics
    total_time_ms: float

# ---------------------------------------------------------------------------
# 5. حالة الذاكرة لـ LangGraph (AgentState)
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    flight_id: str
    telemetry: Dict[str, Any] 
    messages: Annotated[list[BaseMessage], operator.add]
    physics_report: Optional[PhysicsRiskReport]
    temporal_report: Optional[TemporalStateEstimate]
    legal_report: Optional[LegalRiskReport]
    consensus_report: Optional[ConsensusReport]
    iteration_count: int
    graph_start_time_ms: float