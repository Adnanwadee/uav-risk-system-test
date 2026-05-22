# File Path: src/uav_risk/stage2/agent/agent_schemas.py


from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from src.uav_risk.stage2.rag.schemas import LegalCitation

class StuckReason(Enum):
    """Reasons for the agent execution loop getting stuck or failing to progress."""
    DUPLICATE_TOOL_CALL = "duplicate_tool_call"
    EMPTY_OBSERVATION = "empty_observation"
    SCHEMA_PARSE_FAILURE = "schema_parse_failure"
    CIRCUIT_BREAKER_TRIPPED = "circuit_breaker_tripped"

@dataclass(frozen=True)
class ConditionalGoConstraint:
    """شرط فيزيائي وتشريعي ملزم يتم استخراجه حياً لتحويل الرحلة لموافقة مشروطة.
    
    Attributes:
        constraint_id: Unique identifier for the constraint.
        description: Explicit description of the operational limitation.
        feature_name: The associated telemetry or configuration feature name.
        required_value_range: The mandated safe boundary range to clear the condition.
        legal_reference: Regulatory clause reference (FAA / SORA).
    """
    constraint_id: str
    description: str
    feature_name: Optional[str]
    required_value_range: Optional[str]
    legal_reference: Optional[str]

@dataclass(frozen=True)
class ToolCall:
    """Encapsulates execution metadata for an isolated tool invocation within the ReAct loop."""
    tool_name: str
    tool_input: Dict[str, Any]
    tool_output: Any
    execution_time_ms: float
    success: bool
    error_message: Optional[str] = None

@dataclass(frozen=True)
class ReasoningStep:
    """Represents a single execution milestone inside the agent's sliding context window."""
    step_number: int
    thought: str
    action: str
    tool_call: Optional[ToolCall]
    observation: str
    features_examined: List[str]

@dataclass(frozen=True)
class FeatureAssessment:
    """Detailed structural validation record for an individual telemetry feature."""
    feature_name: str
    value: float
    status: Literal["SAFE", "WARNING", "CRITICAL"]
    reasoning: str
    rag_consulted: bool
    rag_finding: Optional[str] = None
    action_required: Optional[str] = None
    related_features: Optional[List[str]] = None

@dataclass(frozen=True)
class LoopAction:
    """Structured directive emitted by the LLM defining the next logical step."""
    thought: str
    action: str
    tool_input: Dict[str, Any]
    llm_confidence: float = 1.0

@dataclass(frozen=True)
class AgentDecision:
    """The final sovereign assessment output delivered to the Aviation Compliance Engine."""
    decision: Literal["GO", "NO-GO", "CONDITIONAL-GO"]
    overall_risk_score: float
    confidence: float
    reasoning_chain: List[ReasoningStep]
    feature_assessments: Dict[str, FeatureAssessment]
    critical_findings: List[str]
    recommendations: List[str]
    legal_citations: List[LegalCitation]
    rag_queries_made: List[str]
    total_iterations: int
    processing_time_ms: float
    fallback_degraded_mode: bool = False
    agent_version: str = "v4.5.0-production"
    prompt_hash: str = "sha256_framework_lock_gate6"
    conditional_constraints: List[ConditionalGoConstraint] = field(default_factory=list)

# =====================================================================
# Stage 2 Agent Schemas Submodule Architectural Dependency Report:
#
# Depends on:
#   - src/uav_risk/stage2/rag/schemas.py (LegalCitation)
#
# Consumed by:
#   - src/uav_risk/stage2/agent/fallback.py
#   - src/uav_risk/stage2/agent/agent_memory.py
#   - src/uav_risk/stage2/agent/agent_tools.py
#   - src/uav_risk/stage2/agent/ace_agent.py
#   - tests/unit/test_ace_agent.py
# =====================================================================