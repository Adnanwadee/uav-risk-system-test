# src/uav_risk/stage2/schemas.py

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, TypedDict, Annotated
from pydantic import BaseModel, Field, ConfigDict, field_validator
from langchain_core.messages import BaseMessage
import operator

# ==========================================
# 1. Extended Sensors & Directives
# ==========================================

class SensorReading(BaseModel):
    value: float
    unit: str
    confidence: Optional[float] = 1.0

class TacticalDirective(BaseModel):
    rule_id: str
    action: Literal["ASSUME_WORST_CASE", "ENFORCE_VLOS", "IGNORE_ML", "REQUEST_DATA", "LIMIT_MANEUVERS"]
    parameter: str
    rationale: str

class DataQualityProfile(BaseModel):
    confidence_score: float
    missing_fields: List[str]
    tactical_directives: List[TacticalDirective]
    is_ml_reliable: bool

# ==========================================
# 2. Input Layer: Strict Aviation Schema
# ==========================================

class UAVScenario(BaseModel):
    model_config = ConfigDict(extra='forbid') # Strict Mode

    # --- AVIATION CONTEXT & MISSION ---
    mission_type: Literal["VLOS", "BVLOS"] = Field(..., description="Visual or Beyond Visual Line of Sight")
    population_density: Literal["REMOTE", "SPARSE", "DENSE"] = Field(..., description="Ground risk factor")
    airspace_altitude_agl_m: float = Field(..., description="Intended flight altitude above ground level")

    # --- CRITICAL PHYSICS & COMMS (Hard Reject if missing) ---
    uav_mass_kg: float = Field(...)
    uav_max_speed_mps: float = Field(...)
    environment_weather_wind_mps: float = Field(...)
    environment_gnss_jam_dbm: float = Field(...)
    comms_uplink_status: Literal["OK", "DEGRADED", "LOST"] = Field(...)
    
    # --- IMPORTANT FIELDS (Agent penalty if missing) ---
    environment_weather_visibility_m: Optional[float] = Field(None, description="Visibility in meters")
    environment_weather_gust_mps: Optional[float] = Field(None, description="Wind gust speed in m/s")
    uav_battery_model_hover_power_W: Optional[float] = Field(None, description="Hover power required in Watts")
    
    # --- EXTENDED SENSORS ---
    extended_sensors: Optional[Dict[str, SensorReading]] = Field(default_factory=dict)

    @field_validator('uav_mass_kg', 'uav_max_speed_mps', 'environment_weather_wind_mps', 'airspace_altitude_agl_m')
    def must_be_positive(cls, v):
        if v < 0: raise ValueError("Physical metrics cannot be negative.")
        return v

class Stage2Request(BaseModel):
    flight_id: str = Field(...)
    scenario: UAVScenario

# ==========================================
# 3. Agent State & Memory (LangGraph)
# ==========================================

class MLResult(BaseModel):
    predicted_class: str
    confidence: float
    risk_score: float

class RegulationChunk(BaseModel):
    article_id: str
    content: str
    numeric_thresholds: Dict[str, float] = Field(default_factory=dict)

class AgentState(TypedDict):
    # LangGraph State Machine Memory
    messages: Annotated[list[BaseMessage], operator.add]
    flight_id: str
    scenario: UAVScenario
    
    # Context & Quality
    data_quality_profile: Optional[DataQualityProfile]
    reasoning_chain: List[str]
    
    # Tool Outputs
    ml_prediction: Optional[MLResult]
    retrieved_regulations: List[RegulationChunk]
    
    # Decisions
    deterministic_veto_pre: Optional[str]
    deterministic_veto_post: Optional[str]
    agent_decision: Optional[Literal["GO", "CAUTION", "NO_GO", "DATA_INSUFFICIENT"]]
    confidence_score: float
    
    # Engine Controls
    guardrail_passed: bool
    iteration_count: int

# ==========================================
# 4. Output Layer
# ==========================================

class EvidenceSnippet(BaseModel):
    source_article: str
    content: str
    citation: str

class FinalDecision(BaseModel):
    decision: Literal["GO", "CAUTION", "NO_GO"]
    justification: str = Field(...)
    confidence: float = Field(..., ge=0.0, le=1.0)
    cited_evidence_ids: List[str]

class Stage2Response(BaseModel):
    status: Literal["OK", "REJECTED_VETO", "ERROR_FALLBACK"]
    flight_id: str
    final_decision: str
    report_md: str
    evidence: List[EvidenceSnippet] = Field(default_factory=list)
    observability_log_id: Optional[str] = None