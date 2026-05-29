from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from uav_risk.core.contracts import RawSecondaryOverrides, ScenarioRawInput


class AssessmentRequest(BaseModel):
    scenario: Any
    secondary_overrides: Any = Field(default_factory=RawSecondaryOverrides)
    operator_notes: Optional[str] = None

    @model_validator(mode="after")
    def validate_raw_contracts(self) -> "AssessmentRequest":
        self.scenario = ScenarioRawInput.model_validate(self.scenario)
        self.secondary_overrides = RawSecondaryOverrides.model_validate(self.secondary_overrides)
        return self


class IssueResponse(BaseModel):
    code: str
    field: str | None = None
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class Stage2AssessmentRequest(AssessmentRequest):
    """Backward-compatible extension for Stage2 full assessments.

    Currently identical to `AssessmentRequest` but provided for future
    Stage2-specific extensions and clearer API typing.
    """

    pass


class Stage1MLSection(BaseModel):
    predicted_class: str | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    top_probability: float | None = None
    raw_feature_count: int | None = None
    processed_feature_count: int | None = None


class Stage1SHAPSection(BaseModel):
    top_features: list[dict[str, Any]] = Field(default_factory=list)
    topic_count: int | None = None
    notes: list[str] = Field(default_factory=list)


class Stage1AssessmentSection(BaseModel):
    core: dict[str, Any] | None = None
    ml: Stage1MLSection | None = None
    shap: Stage1SHAPSection | None = None


class Stage2PolicySection(BaseModel):
    policy_name: str | None = None
    policy_version: str | None = None
    weights: dict[str, float] = Field(default_factory=dict)
    go_threshold: float | None = None
    no_go_threshold: float | None = None
    weight_rationales: dict[str, str] = Field(default_factory=dict)


class Stage2RAGSection(BaseModel):
    retrieval_usable: bool = False
    rag_quality_is_proven: bool = False
    evidence_bundle_count: int = 0
    insufficient_evidence_count: int = 0
    scenario_evidence_complete: bool | None = None
    scenario_evidence_status: str | None = None
    corpus_coverage_status: str | None = None
    expected_source_count: int | None = None
    indexed_source_count: int | None = None
    missing_sources_count: int | None = None
    source_ids: list[str] = Field(default_factory=list)
    source_titles: list[str] = Field(default_factory=list)
    missing_sources: list[str] = Field(default_factory=list)
    retrieval_origins: list[str] = Field(default_factory=list)
    synthetic_bundle_count: int | None = None
    reranker_configured: bool | None = None
    reranker_available: bool | None = None
    reranker_used: bool | None = None
    reranker_reason: str | None = None
    evidence_bundle_details: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class Stage2AgentSection(BaseModel):
    recommendation: str | None = None
    findings: list[dict[str, Any]] = Field(default_factory=list)
    action_items: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    system_work_trace: dict[str, Any] | None = None
    working_memory_summary: dict[str, Any] = Field(default_factory=dict)
    top_input_signals: list[dict[str, Any]] = Field(default_factory=list)
    top_feature_assessments: list[dict[str, Any]] = Field(default_factory=list)
    selected_rag_queries: list[str] = Field(default_factory=list)
    skipped_rag_queries: list[str] = Field(default_factory=list)


class Stage2DecisionSection(BaseModel):
    final_decision: str = "caution"
    decision_score: float = 0.0
    confidence_level: str = "low"
    stage_weights: dict[str, float] = Field(default_factory=dict)
    stage_contributions: list[dict[str, Any]] = Field(default_factory=list)
    decision_reasons: list[str] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)
    required_actions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class Stage2LLMSection(BaseModel):
    status: str = "disabled"
    synthesis_status: str = "disabled"
    provider: str | None = None
    model_name: str | None = None
    external_provider_used: bool = False
    executive_summary: str = ""
    operational_interpretation: str = ""
    decision_explanation: str = ""
    key_risk_drivers: list[str] = Field(default_factory=list)
    mitigation_narrative: str = ""
    consistency_warnings: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def sync_synthesis_status(self) -> "Stage2LLMSection":
        if not (self.synthesis_status or "").strip() or (self.synthesis_status == "disabled" and self.status != "disabled"):
            self.synthesis_status = self.status
        return self


class Stage2ReportSection(BaseModel):
    markdown: str | None = None
    sections: list[dict[str, Any]] = Field(default_factory=list)
    generated: bool = False


class Stage2AISection(BaseModel):
    profile_context: dict[str, Any] | None = None
    policy: Stage2PolicySection = Field(default_factory=Stage2PolicySection)
    rag: Stage2RAGSection = Field(default_factory=Stage2RAGSection)
    agent: Stage2AgentSection = Field(default_factory=Stage2AgentSection)
    decision: Stage2DecisionSection = Field(default_factory=Stage2DecisionSection)
    llm_synthesis: Stage2LLMSection = Field(default_factory=Stage2LLMSection)
    report: Stage2ReportSection | None = None


class Stage2DiagnosticsSection(BaseModel):
    path_resolution_status: str | None = None
    index_provenance_status: str | None = None
    persistence_status: str | None = None
    retrieval_usable: bool = False
    rag_quality_is_proven: bool = False
    scenario_evidence_complete: bool | None = None
    corpus_coverage_status: str | None = None
    expected_source_count: int | None = None
    indexed_source_count: int | None = None
    source_ids: list[str] = Field(default_factory=list)
    source_titles: list[str] = Field(default_factory=list)
    reranker_configured: bool | None = None
    reranker_available: bool | None = None
    reranker_used: bool | None = None
    reranker_reason: str | None = None
    llm_mode: str = "disabled"
    external_llm_provider_used: bool = False
    llm_provider: str | None = None
    llm_model_name: str | None = None
    faiss_secret_configured: bool | None = None
    warnings: list[dict[str, Any]] = Field(default_factory=list)


class Stage2AssessmentResponse(BaseModel):
    status: str
    user_id: str
    profile_id: str
    assessment_id: str
    created_at: str | None = None
    persisted: bool | None = None
    persistence_status: str | None = None
    system_work_trace: dict[str, Any] | None = None
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    stage1: Stage1AssessmentSection
    stage2: Stage2AISection
    diagnostics: Stage2DiagnosticsSection


class AssessmentRecord(BaseModel):
    assessment_id: str
    user_id: str
    profile_id: str
    created_at: str
    status: str
    final_decision: str | None = None
    decision_score: float | None = None
    confidence_level: str | None = None
    stage1: dict[str, Any] = Field(default_factory=dict)
    stage2: dict[str, Any] = Field(default_factory=dict)
    report: dict[str, Any] | str | None = None
    system_work_trace: dict[str, Any] | list[dict[str, Any]] | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class AssessmentListItem(BaseModel):
    assessment_id: str
    user_id: str
    profile_id: str
    created_at: str
    status: str
    final_decision: str | None = None
    decision_score: float | None = None
    confidence_level: str | None = None
    summary: str | None = None
