from __future__ import annotations

from enum import Enum
from typing import TypeAlias
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

MetadataScalar: TypeAlias = str | int | float | bool | None
MetadataMap: TypeAlias = dict[str, MetadataScalar]


class EvidenceSourceType(str, Enum):
    REGULATION = "regulation"
    ADVISORY_CIRCULAR = "advisory_circular"
    SORA = "sora"
    SPECIAL_CONDITION = "special_condition"
    INTERNAL_DOC = "internal_doc"
    UNKNOWN = "unknown"


class EvidenceSupportStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CONFLICTING = "conflicting"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class EvidenceUse(str, Enum):
    RETRIEVAL_CONTEXT = "retrieval_context"
    OPERATIONAL_SUPPORT = "operational_support"
    LIMITATION = "limitation"
    CONTRADICTION = "contradiction"
    BACKGROUND = "background"


class EvidenceOrigin(str, Enum):
    LOCAL_DOCUMENT = "local_document"
    RETRIEVAL_SYSTEM = "retrieval_system"
    LLM_SYNTHESIS = "llm_synthesis"
    HYDE_GENERATED = "hyde_generated"
    OPERATOR_INPUT = "operator_input"
    UNKNOWN = "unknown"


class EvidenceCitation(BaseModel):
    citation_id: str
    source_id: str
    source_title: str
    source_type: EvidenceSourceType
    origin: EvidenceOrigin
    section: str | None = None
    page: int | None = None
    chunk_id: str | None = None
    quote: str
    retrieval_score: float | None = None
    rerank_score: float | None = None
    metadata: MetadataMap = Field(default_factory=dict)

    @field_validator("citation_id", "source_id", "source_title", "quote")
    @classmethod
    def _validate_non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must be non-empty")
        return value

    @field_validator("page")
    @classmethod
    def _validate_page(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("page must be >= 1")
        return value

    @field_validator("retrieval_score", "rerank_score")
    @classmethod
    def _validate_score_range(cls, value: float | None) -> float | None:
        if value is not None and not (0.0 <= value <= 1.0):
            raise ValueError("score must be within [0.0, 1.0]")
        return value

    @field_validator("origin")
    @classmethod
    def _validate_origin(cls, value: EvidenceOrigin) -> EvidenceOrigin:
        if value in {EvidenceOrigin.LLM_SYNTHESIS, EvidenceOrigin.HYDE_GENERATED}:
            raise ValueError("citation origin cannot be llm_synthesis or hyde_generated")
        return value


class EvidenceClaim(BaseModel):
    claim_id: str
    claim: str
    support_status: EvidenceSupportStatus
    evidence_use: EvidenceUse
    citations: list[EvidenceCitation] = Field(default_factory=list)
    confidence: float
    limitations: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    metadata: MetadataMap = Field(default_factory=dict)

    @field_validator("claim_id", "claim")
    @classmethod
    def _validate_non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must be non-empty")
        return value

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, value: float) -> float:
        if not (0.0 <= value <= 1.0):
            raise ValueError("confidence must be within [0.0, 1.0]")
        return value

    @model_validator(mode="after")
    def _validate_support_requirements(self) -> "EvidenceClaim":
        if self.support_status in {
            EvidenceSupportStatus.SUPPORTED,
            EvidenceSupportStatus.PARTIALLY_SUPPORTED,
        } and not self.citations:
            raise ValueError("supported or partially_supported claims require citations")

        if self.support_status == EvidenceSupportStatus.INSUFFICIENT_EVIDENCE and not [
            item for item in self.limitations if item.strip()
        ]:
            raise ValueError("insufficient_evidence claims require non-empty limitations")

        if self.support_status == EvidenceSupportStatus.CONFLICTING and not [
            item for item in self.conflicts if item.strip()
        ]:
            raise ValueError("conflicting claims require non-empty conflicts")

        return self


class EvidenceBundle(BaseModel):
    bundle_id: str
    query: str
    claims: list[EvidenceClaim] = Field(default_factory=list)
    citations: list[EvidenceCitation] = Field(default_factory=list)
    support_status: EvidenceSupportStatus
    confidence: float
    no_evidence_reason: str | None = None
    metadata: MetadataMap = Field(default_factory=dict)

    @field_validator("bundle_id", "query")
    @classmethod
    def _validate_non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must be non-empty")
        return value

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, value: float) -> float:
        if not (0.0 <= value <= 1.0):
            raise ValueError("confidence must be within [0.0, 1.0]")
        return value

    @model_validator(mode="after")
    def _validate_status_reason(self) -> "EvidenceBundle":
        if self.support_status == EvidenceSupportStatus.INSUFFICIENT_EVIDENCE:
            reason = (self.no_evidence_reason or "").strip()
            if not reason:
                raise ValueError("insufficient_evidence bundles require no_evidence_reason")
            self.no_evidence_reason = reason
        return self


class AgentEvidenceReference(BaseModel):
    claim_id: str
    citation_ids: list[str] = Field(default_factory=list)
    summary: str

    @field_validator("claim_id", "summary")
    @classmethod
    def _validate_non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must be non-empty")
        return value

    @field_validator("citation_ids")
    @classmethod
    def _validate_citation_ids(cls, value: list[str]) -> list[str]:
        for item in value:
            if not item.strip():
                raise ValueError("citation_ids cannot contain empty strings")
        return value


class PublicReasoningTrace(BaseModel):
    observations: list[str] = Field(default_factory=list)
    checks_performed: list[str] = Field(default_factory=list)
    evidence_consulted: list[AgentEvidenceReference] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class Stage2Status(str, Enum):
    SKIPPED = "skipped"
    COMPLETED = "completed"
    DEGRADED = "degraded"
    FAILED = "failed"


class FinalDecision(str, Enum):
    GO = "go"
    CAUTION = "caution"
    NO_GO = "no_go"


class DecisionConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DecisionStageName(str, Enum):
    CORE = "core"
    ML = "ml"
    SHAP = "shap"
    RAG = "rag"
    AGENT = "agent"
    SCENARIO_PROFILE = "scenario_profile"
    LLM = "llm"


class Stage2Error(BaseModel):
    code: str
    message: str
    details: MetadataMap = Field(default_factory=dict)

    @field_validator("code", "message")
    @classmethod
    def _validate_non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must be non-empty")
        return value


class DecisionStageContribution(BaseModel):
    stage: DecisionStageName
    weight: float
    contribution: float
    signal: str
    summary: str
    reasons: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    metadata: MetadataMap = Field(default_factory=dict)

    @field_validator("weight", "contribution")
    @classmethod
    def _validate_score_range(cls, value: float) -> float:
        if not (0.0 <= value <= 1.0):
            raise ValueError("value must be within [0.0, 1.0]")
        return value

    @field_validator("signal", "summary")
    @classmethod
    def _validate_non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must be non-empty")
        return value


class DecisionEngineResult(BaseModel):
    final_decision: FinalDecision
    decision_score: float
    confidence_level: DecisionConfidenceLevel
    stage_weights: dict[str, float] = Field(default_factory=dict)
    stage_contributions: list[DecisionStageContribution] = Field(default_factory=list)
    decision_reasons: list[str] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)
    required_actions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    evidence_refs: list[AgentEvidenceReference] = Field(default_factory=list)
    metadata: MetadataMap = Field(default_factory=dict)

    @field_validator("decision_score")
    @classmethod
    def _validate_decision_score(cls, value: float) -> float:
        if not (0.0 <= value <= 1.0):
            raise ValueError("decision_score must be within [0.0, 1.0]")
        return value

    @field_validator("stage_weights")
    @classmethod
    def _validate_stage_weights(cls, value: dict[str, float]) -> dict[str, float]:
        for key, weight in value.items():
            if not str(key).strip():
                raise ValueError("stage weight keys must be non-empty")
            if not (0.0 <= weight <= 1.0):
                raise ValueError("stage weights must be within [0.0, 1.0]")
        return value


class LLMSynthesisStatus(str, Enum):
    DISABLED = "disabled"
    GENERATED = "generated"
    FALLBACK = "fallback"
    FAILED = "failed"


class LLMSynthesisWarning(BaseModel):
    warning_type: str
    message: str
    related_ids: list[str] = Field(default_factory=list)
    metadata: MetadataMap = Field(default_factory=dict)

    @field_validator("warning_type", "message")
    @classmethod
    def _validate_non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must be non-empty")
        return value

    @field_validator("related_ids")
    @classmethod
    def _validate_related_ids(cls, value: list[str]) -> list[str]:
        for item in value:
            if not item.strip():
                raise ValueError("related_ids cannot contain empty strings")
        return value


class LLMAgentSynthesis(BaseModel):
    status: LLMSynthesisStatus
    executive_summary: str
    operational_interpretation: str
    decision_explanation: str
    key_risk_drivers: list[str] = Field(default_factory=list)
    mitigation_narrative: str
    consistency_warnings: list[LLMSynthesisWarning] = Field(default_factory=list)
    evidence_reference_ids: list[str] = Field(default_factory=list)
    finding_ids: list[str] = Field(default_factory=list)
    action_item_ids: list[str] = Field(default_factory=list)
    limitation_ids: list[str] = Field(default_factory=list)
    model_name: str | None = None
    provider: str | None = None
    metadata: MetadataMap = Field(default_factory=dict)

    @field_validator(
        "executive_summary",
        "operational_interpretation",
        "decision_explanation",
        "mitigation_narrative",
    )
    @classmethod
    def _validate_non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must be non-empty")
        return value

    @field_validator("key_risk_drivers", "evidence_reference_ids", "finding_ids", "action_item_ids", "limitation_ids")
    @classmethod
    def _validate_string_lists(cls, value: list[str]) -> list[str]:
        for item in value:
            if not item.strip():
                raise ValueError("lists cannot contain empty strings")
        return value


class AgentRecommendation(str, Enum):
    GO = "go"
    CAUTION = "caution"
    NO_GO = "no_go"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    DEGRADED = "degraded"


class AgentQuerySourceIntent(str, Enum):
    PART107 = "part107"
    AC107 = "ac107"
    SORA = "sora"
    SORA_ANNEX = "sora_annex"
    SPECIAL_CONDITION = "special_condition"
    EAR_EXPORT = "ear_export"
    GENERAL_UAV = "general_uav"
    UNSUPPORTED_OR_UNKNOWN = "unsupported_or_unknown"


class AgentQueryDerivedFrom(str, Enum):
    SHAP = "shap"
    SCENARIO = "scenario"
    PROFILE = "profile"
    ML = "ml"
    OPERATOR_NOTES = "operator_notes"
    SYSTEM_DEFAULT = "system_default"


class AgentRAGQueryPlan(BaseModel):
    query_id: str
    query_text: str
    query_purpose: str
    source_intent: AgentQuerySourceIntent
    expected_source_family: str | None = None
    derived_from: AgentQueryDerivedFrom
    related_feature_names: list[str] = Field(default_factory=list)
    evidence_required: bool = True
    fallback_if_insufficient: str | None = None
    priority: int
    metadata: MetadataMap = Field(default_factory=dict)

    @field_validator("query_id", "query_text", "query_purpose")
    @classmethod
    def _validate_query_fields(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must be non-empty")
        return value

    @field_validator("priority")
    @classmethod
    def _validate_priority(cls, value: int) -> int:
        if value < 1:
            raise ValueError("priority must be >= 1")
        return value


class AgentFindingType(str, Enum):
    STRUCTURAL = "structural"
    ML_SIGNAL = "ml_signal"
    EVIDENCE_BACKED = "evidence_backed"
    OPERATIONAL_UNCERTAINTY = "operational_uncertainty"
    LIMITATION = "limitation"
    CONFLICT = "conflict"
    TOOL_CHECK = "tool_check"


class AgentFindingSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentFinding(BaseModel):
    finding_id: str
    finding_type: AgentFindingType
    severity: AgentFindingSeverity
    summary: str
    evidence_references: list[AgentEvidenceReference] = Field(default_factory=list)
    requires_evidence: bool
    metadata: MetadataMap = Field(default_factory=dict)

    @field_validator("finding_id", "summary")
    @classmethod
    def _validate_non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must be non-empty")
        return value

    @model_validator(mode="after")
    def _validate_evidence_requirements(self) -> "AgentFinding":
        if self.finding_type == AgentFindingType.EVIDENCE_BACKED and not self.evidence_references:
            raise ValueError("evidence_backed findings require evidence_references")
        if self.requires_evidence and not self.evidence_references:
            raise ValueError("requires_evidence findings require evidence_references")
        return self


class AgentActionItem(BaseModel):
    action_id: str
    summary: str
    priority: AgentFindingSeverity
    evidence_references: list[AgentEvidenceReference] = Field(default_factory=list)
    metadata: MetadataMap = Field(default_factory=dict)

    @field_validator("action_id", "summary")
    @classmethod
    def _validate_non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must be non-empty")
        return value


class AgentInput(BaseModel):
    assessment_id: str | None = None
    scenario_summary: MetadataMap = Field(default_factory=dict)
    ml_prediction: str | None = None
    ml_probabilities: dict[str, float] = Field(default_factory=dict)
    shap_top_features: list[MetadataMap] = Field(default_factory=list)
    evidence_bundles: list[EvidenceBundle] = Field(default_factory=list)
    operator_notes: str | None = None
    metadata: MetadataMap = Field(default_factory=dict)

    @field_validator("ml_probabilities")
    @classmethod
    def _validate_probabilities(cls, value: dict[str, float]) -> dict[str, float]:
        for key, prob in value.items():
            if not (0.0 <= prob <= 1.0):
                raise ValueError(f"ml probability for '{key}' must be within [0.0, 1.0]")
        return value


class AgentResult(BaseModel):
    status: Stage2Status
    recommendation: AgentRecommendation
    confidence: float
    findings: list[AgentFinding] = Field(default_factory=list)
    action_items: list[AgentActionItem] = Field(default_factory=list)
    reasoning_trace: PublicReasoningTrace
    evidence_bundles: list[EvidenceBundle] = Field(default_factory=list)
    errors: list[Stage2Error] = Field(default_factory=list)
    metadata: MetadataMap = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, value: float) -> float:
        if not (0.0 <= value <= 1.0):
            raise ValueError("confidence must be within [0.0, 1.0]")
        return value

    @model_validator(mode="after")
    def _validate_state(self) -> "AgentResult":
        if self.status == Stage2Status.COMPLETED and self.recommendation in {
            AgentRecommendation.GO,
            AgentRecommendation.CAUTION,
            AgentRecommendation.NO_GO,
        } and not self.findings:
            raise ValueError("completed go/caution/no_go results require findings")

        if self.recommendation == AgentRecommendation.INSUFFICIENT_EVIDENCE:
            bundle_has_insufficient = any(
                bundle.support_status == EvidenceSupportStatus.INSUFFICIENT_EVIDENCE
                for bundle in self.evidence_bundles
            )
            finding_explains = any(
                finding.finding_type in {
                    AgentFindingType.OPERATIONAL_UNCERTAINTY,
                    AgentFindingType.LIMITATION,
                }
                for finding in self.findings
            )
            if not (bundle_has_insufficient or finding_explains):
                raise ValueError(
                    "insufficient_evidence recommendation requires insufficient evidence bundle or uncertainty/limitation finding"
                )
        return self


class SHAPFeatureAttribution(BaseModel):
    feature: str
    value: MetadataScalar = None
    importance: float
    direction: str | None = None
    metadata: MetadataMap = Field(default_factory=dict)

    @field_validator("feature")
    @classmethod
    def _validate_feature(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("feature must be non-empty")
        return value


class MLAssessmentSnapshot(BaseModel):
    predicted_class: str
    probabilities: dict[str, float]
    shap_top_features: list[SHAPFeatureAttribution] = Field(default_factory=list)
    raw_feature_count: int | None = None
    processed_feature_count: int | None = None
    metadata: MetadataMap = Field(default_factory=dict)

    @field_validator("predicted_class")
    @classmethod
    def _validate_predicted_class(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("predicted_class must be non-empty")
        return value

    @field_validator("probabilities")
    @classmethod
    def _validate_probabilities(cls, value: dict[str, float]) -> dict[str, float]:
        for key, prob in value.items():
            if not (0.0 <= prob <= 1.0):
                raise ValueError(f"probability for {key} must be within [0.0, 1.0]")
        return value

    @field_validator("raw_feature_count", "processed_feature_count")
    @classmethod
    def _validate_counts(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("feature counts must be >= 0")
        return value


class Stage2AssessmentInput(BaseModel):
    assessment_id: str | None = None
    user_id: str
    profile_id: str
    scenario_summary: MetadataMap = Field(default_factory=dict)
    ml: MLAssessmentSnapshot
    evidence_bundles: list[EvidenceBundle] = Field(default_factory=list)
    operator_notes: str | None = None
    metadata: MetadataMap = Field(default_factory=dict)

    @field_validator("user_id", "profile_id")
    @classmethod
    def _validate_non_empty_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must be non-empty")
        return value


class Stage2AssessmentResult(BaseModel):
    status: Stage2Status
    assessment_id: str | None = None
    evidence_bundles: list[EvidenceBundle] = Field(default_factory=list)
    agent_result: AgentResult | None = None
    decision: DecisionEngineResult | None = None
    llm_synthesis: LLMAgentSynthesis | None = None
    errors: list[Stage2Error] = Field(default_factory=list)
    metadata: MetadataMap = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_status_contract(self) -> "Stage2AssessmentResult":
        if self.status == Stage2Status.COMPLETED and self.agent_result is None:
            raise ValueError("completed status requires agent_result")
        if self.status == Stage2Status.FAILED and not self.errors:
            raise ValueError("failed status requires errors")
        return self



class OperationalReportSectionType(str, Enum):
    METADATA = "metadata"
    PROFILE_SCENARIO_SUMMARY = "profile_scenario_summary"
    ML_SIGNAL = "ml_signal"
    SHAP_EXPLANATION = "shap_explanation"
    EVIDENCE_SUMMARY = "evidence_summary"
    DECISION_ENGINE = "decision_engine"
    LLM_SYNTHESIS = "llm_synthesis"
    AGENT_ASSESSMENT = "agent_assessment"
    LIMITATIONS = "limitations"
    OPERATOR_ACTIONS = "operator_actions"
    ERRORS = "errors"


class OperationalReportSection(BaseModel):
    section_type: OperationalReportSectionType
    title: str
    content: list[str] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)
    metadata: MetadataMap = Field(default_factory=dict)

    @field_validator("title")
    @classmethod
    def _validate_title(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("title must be non-empty")
        return value

    @field_validator("content")
    @classmethod
    def _validate_content(cls, values: list[str]) -> list[str]:
        forbidden = [
            "reasoning_chain",
            "chain_of_thought",
            "thought",
            "scratchpad",
            "internal_reasoning",
            "private_reasoning",
        ]
        for item in values:
            text = str(item)
            lower = text.lower()
            for token in forbidden:
                if token in lower:
                    raise ValueError("content contains forbidden chain-of-thought field name")
        return values


class OperationalReport(BaseModel):
    report_id: str
    assessment_id: str | None = None
    status: Stage2Status
    sections: list[OperationalReportSection]
    evidence_bundles: list[EvidenceBundle] = Field(default_factory=list)
    agent_result: AgentResult | None = None
    errors: list[Stage2Error] = Field(default_factory=list)
    metadata: MetadataMap = Field(default_factory=dict)

    @field_validator("report_id")
    @classmethod
    def _validate_report_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("report_id must be non-empty")
        return value

    @field_validator("sections")
    @classmethod
    def _validate_sections(cls, value: list[OperationalReportSection]) -> list[OperationalReportSection]:
        if not value:
            raise ValueError("sections must not be empty")
        return value

def make_insufficient_evidence_bundle(query: str, reason: str) -> EvidenceBundle:
    normalized_query = query.strip()
    normalized_reason = reason.strip()
    if not normalized_query:
        raise ValueError("query must be non-empty")
    if not normalized_reason:
        raise ValueError("reason must be non-empty")

    return EvidenceBundle(
        bundle_id=f"bundle_{uuid4().hex}",
        query=normalized_query,
        claims=[],
        citations=[],
        support_status=EvidenceSupportStatus.INSUFFICIENT_EVIDENCE,
        confidence=0.0,
        no_evidence_reason=normalized_reason,
        metadata={},
    )


def collect_unique_citations(claims: list[EvidenceClaim]) -> list[EvidenceCitation]:
    seen: set[str] = set()
    unique: list[EvidenceCitation] = []
    for claim in claims:
        for citation in claim.citations:
            if citation.citation_id in seen:
                continue
            seen.add(citation.citation_id)
            unique.append(citation)
    return unique


LegalCitation = EvidenceCitation
