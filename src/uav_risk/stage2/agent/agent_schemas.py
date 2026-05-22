"""
ACE UAV Risk Assessment System - Stage 4 (Agent Schemas)
File: src/uav_risk/stage2/agent/agent_schemas.py
Description: Production-grade immutable Pydantic v2 data contracts for the ReAct Agent's 
             reasoning steps, tool execution tracking, and final flight decisions.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
from src.uav_risk.stage2.rag.schemas import LegalCitation


class ToolCall(BaseModel):
    """
    سجل تشريحي وجنائي دقيق لكل أداة يتم استدعاؤها من قبل الوكيل في حلقة الـ ReAct.
    يضمن تتبع المدخلات والمخرجات وحساب زمن المعالجة الفعلي لامتصاص صدمات الفشل.
    """
    model_config = ConfigDict(frozen=True)

    tool_name: str = Field(
        ..., 
        description="The exact name of the executed tactical tool from the agent repository."
    )
    tool_input: Dict[str, Any] = Field(
        default_factory=dict, 
        description="The exact parameters passed by the LLM to the tool invocation handler."
    )
    tool_output: Any = Field(
        ..., 
        description="The raw return value or structural dictionary output produced by the execution."
    )
    execution_time_ms: float = Field(
        ..., 
        description="High-precision delta execution time calculated via performance counters."
    )
    success: bool = Field(
        ..., 
        description="A Boolean flag reflecting whether the tool executed cleanly or encountered errors."
    )
    error_message: Optional[str] = Field(
        None, 
        description="The explicit exception or traceback captured during a degraded tool recovery mode."
    )


class ReasoningStep(BaseModel):
    """
    يمثل خطوة فكرية تاريخية متكاملة داخل حلقة التحكم (Thought -> Action -> Observation).
    يضمن الحفاظ على جينات التتبع الشفاف لخطوات الوكيل لمنع الهلوسة البرمجية.
    """
    model_config = ConfigDict(frozen=True)

    step_number: int = Field(
        ..., 
        description="The chronological index of the current iteration loop within the max budget allocation."
    )
    thought: str = Field(
        ..., 
        description="The metacognitive reflection text formulated by the LLM stating what to evaluate next."
    )
    action: str = Field(
        ..., 
        description="The serialized JSON string representing the structured tool command intended to execute."
    )
    tool_call: Optional[ToolCall] = Field(
        None, 
        description="The structural execution payload associated with this specific step, if a tool was armed."
    )
    observation: str = Field(
        ..., 
        description="The dynamic feedback or empirical physics evidence returned after tool execution."
    )
    features_examined: List[str] = Field(
        default_factory=list, 
        description="A historical snapshot list of specific feature keys examined up to this point."
    )


class FeatureAssessment(BaseModel):
    """
    التقييم التشغيلي النهائي والسياقي لميزة هندسية أو تنظيمية مفردة من الـ 198 ميزة.
    يربط القيم الرقمية الخام بحالة سلامة الطيران الفعلية مع التبرير الفيزيائي.
    """
    model_config = ConfigDict(frozen=True)

    feature_name: str = Field(
        ..., 
        description="The standardized string identifier of the target feature extracted from the constitution."
    )
    value: float = Field(
        ..., 
        description="The cleanly validated, clipped, or physically imputed numerical metric value."
    )
    status: str = Field(
        ..., 
        description="The calibrated safety state assigned. Enforces strictly: SAFE | WARNING | CRITICAL | UNKNOWN."
    )
    reasoning: str = Field(
        ..., 
        description="The localized cross-feature context or analytical engineering justification for the state."
    )
    rag_consulted: bool = Field(
        default=False, 
        description="A dynamic tracker showing if the vector knowledge base was questioned regarding this feature."
    )
    rag_finding: Optional[str] = Field(
        None, 
        description="The raw or synthesized regulatory context extracted from the offline documentation."
    )
    action_required: Optional[str] = Field(
        None, 
        description="The programmatic mitigation protocol or manual recommendation mandated for the pilot."
    )


class AgentDecision(BaseModel):
    """
    الوعاء الحاكم والسيادي النهائي المكمل لكامل خط الأنابيب والمُسلم لمحرك FastAPI.
    يفصل بين الخروقات التشغيلية للـ Core ومؤشرات المحاكاة لضمان الأمان الفولاذي.
    """
    model_config = ConfigDict(frozen=True)

    decision: str = Field(
        ..., 
        description="The absolute sovereign flight determination. Enforces strictly: GO | NO-GO | CONDITIONAL-GO."
    )
    overall_risk_score: float = Field(
        ..., 
        description="The absolute maximum composite risk index merged from LightGBM trees and memory anomalies."
    )
    confidence: float = Field(
        ..., 
        description="The underlying mathematical certainty calculation inherited from tree branch probabilities."
    )
    reasoning_chain: List[ReasoningStep] = Field(
        default_factory=list, 
        description="The complete historic collection of the ReAct loops executed during the investigation."
    )
    feature_assessments: List[FeatureAssessment] = Field(
        default_factory=list, 
        description="The comprehensive 100% full sweep manifest containing evaluations for all 198 metrics."
    )
    critical_findings: List[str] = Field(
        default_factory=list, 
        description="An explicit collection summarizing core hardware breaches and absolute regulatory violations."
    )
    recommendations: List[str] = Field(
        default_factory=list, 
        description="A targeted list of strict, actionable checklists required to mitigate secondary track anomalies."
    )
    legal_citations: List[LegalCitation] = Field(
        default_factory=list, 
        description="The clean pool of un-manipulated legal citations and page numbers fetched from the RAG store."
    )
    rag_queries_made: List[str] = Field(
        default_factory=list, 
        description="The exact query templates engineered dynamically by the agent to target vector pools."
    )
    memory_snapshots: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Cryptographic or factual metadata snapshots of the working memory at critical pivot junctions."
    )
    total_iterations: int = Field(
        ..., 
        description="The total iteration count burned out of the 20 max loop budget allocation."
    )
    processing_time_ms: float = Field(
        ..., 
        description="The comprehensive pipeline delta duration calculated since session initialization."
    )


# ====================================================================================
# Stage 4 Architectural Dependency Block (Consistency Rule 4):
#
# This file: src/uav_risk/stage2/agent/agent_schemas.py
# - Depends on: src/uav_risk/stage2/rag/schemas.py (LegalCitation)
# - Is consumed by:
#   1. src/uav_risk/stage2/agent/agent_memory.py (Working Memory Engine)
#   2. src/uav_risk/stage2/agent/agent_tools.py (Physics & RAG Tactical Tools)
#   3. src/uav_risk/stage2/agent/ace_agent.py (Core ReAct Controller)
#   4. tests/unit/test_ace_agent.py (8-Gate Verification Suite)
# ====================================================================================