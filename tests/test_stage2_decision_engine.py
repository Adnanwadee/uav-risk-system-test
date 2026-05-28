from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

from uav_risk.stage2.contracts import (
    AgentRecommendation,
    AgentResult,
    DecisionEngineResult,
    DecisionStageContribution,
    DecisionStageName,
    EvidenceBundle,
    EvidenceCitation,
    EvidenceOrigin,
    EvidenceSourceType,
    EvidenceSupportStatus,
    FinalDecision,
    MLAssessmentSnapshot,
    PublicReasoningTrace,
    SHAPFeatureAttribution,
    Stage2AssessmentInput,
    Stage2AssessmentResult,
    Stage2Status,
)
from uav_risk.stage2.decision_engine import WeightedDecisionEngine, evaluate_stage2_decision


def _citation(cid: str = "c1") -> EvidenceCitation:
    return EvidenceCitation(
        citation_id=cid,
        source_id="source1",
        source_title="AC_107-2A",
        source_type=EvidenceSourceType.ADVISORY_CIRCULAR,
        origin=EvidenceOrigin.LOCAL_DOCUMENT,
        page=4,
        chunk_id="chunk1",
        quote="Retrieved operational evidence for safe small UAS planning.",
        metadata={"source_filename": "AC_107-2A.pdf", "page_start": 4},
    )


def _supported_bundle() -> EvidenceBundle:
    return EvidenceBundle(
        bundle_id="b-supported",
        query="AC 107-2A preflight weather assessment small UAS wind conditions",
        claims=[],
        citations=[_citation()],
        support_status=EvidenceSupportStatus.SUPPORTED,
        confidence=0.8,
    )


def _insufficient_bundle() -> EvidenceBundle:
    return EvidenceBundle(
        bundle_id="b-insufficient",
        query="unsupported",
        claims=[],
        citations=[],
        support_status=EvidenceSupportStatus.INSUFFICIENT_EVIDENCE,
        confidence=0.0,
        no_evidence_reason="No sufficient evidence candidates passed retrieval safety checks.",
    )


def _input(
    *,
    predicted_class: str = "Medium Risk",
    probabilities: dict[str, float] | None = None,
    metadata: dict[str, object] | None = None,
    shap: list[SHAPFeatureAttribution] | None = None,
) -> Stage2AssessmentInput:
    probs = probabilities or {"High Risk": 0.2, "Medium Risk": 0.6, "Low Risk": 0.2}
    return Stage2AssessmentInput(
        assessment_id="a1",
        user_id="u1",
        profile_id="p1",
        scenario_summary={"environment_weather_wind_mps": 5.0},
        ml=MLAssessmentSnapshot(
            predicted_class=predicted_class,
            probabilities=probs,
            shap_top_features=shap if shap is not None else [],
            raw_feature_count=197,
            processed_feature_count=198,
        ),
        evidence_bundles=[],
        operator_notes="notes",
        metadata=metadata or {},
    )


def _agent(
    *,
    recommendation: AgentRecommendation = AgentRecommendation.CAUTION,
    severity: str = "medium",
    finding_type: str = "tool_check",
    action: bool = False,
    metadata: dict[str, object] | None = None,
) -> AgentResult:
    return AgentResult(
        status=Stage2Status.COMPLETED,
        recommendation=recommendation,
        confidence=0.7,
        findings=[
            {
                "finding_id": "f1",
                "finding_type": finding_type,
                "severity": severity,
                "summary": "Operational concern requires review.",
                "requires_evidence": False,
                "metadata": metadata or {},
            }
        ],
        action_items=(
            [
                {
                    "action_id": "a1",
                    "summary": "Verify airspace authorization before launch.",
                    "priority": "high",
                    "evidence_references": [],
                }
            ]
            if action
            else []
        ),
        reasoning_trace=PublicReasoningTrace(),
        evidence_bundles=[],
        errors=[],
    )


def _result(*, bundles: list[EvidenceBundle] | None = None, agent: AgentResult | None = None) -> Stage2AssessmentResult:
    return Stage2AssessmentResult(
        status=Stage2Status.COMPLETED,
        assessment_id="a1",
        evidence_bundles=bundles if bundles is not None else [_supported_bundle()],
        agent_result=agent if agent is not None else _agent(),
        errors=[],
    )


def test_hard_veto_metadata_produces_no_go_regardless_of_weights() -> None:
    decision = evaluate_stage2_decision(
        _input(
            predicted_class="Low Risk",
            probabilities={"High Risk": 0.05, "Medium Risk": 0.1, "Low Risk": 0.85},
            metadata={"core_hard_veto": True, "core_blocking_reason": "payload exceeds profile limit"},
        ),
        _result(agent=_agent(recommendation=AgentRecommendation.GO, severity="info")),
    )
    assert decision.final_decision == FinalDecision.NO_GO
    assert decision.decision_score == 1.0
    assert "payload exceeds profile limit" in decision.blocking_reasons


def test_high_ml_risk_plus_severe_agent_finding_reaches_strong_caution_or_no_go() -> None:
    decision = evaluate_stage2_decision(
        _input(predicted_class="High Risk", probabilities={"High Risk": 0.9, "Medium Risk": 0.05, "Low Risk": 0.05}),
        _result(agent=_agent(recommendation=AgentRecommendation.CAUTION, severity="high")),
    )
    assert decision.final_decision in {FinalDecision.CAUTION, FinalDecision.NO_GO}
    assert decision.decision_score >= 0.35


def test_medium_ml_risk_with_supported_evidence_produces_caution() -> None:
    decision = evaluate_stage2_decision(_input(), _result())
    assert decision.final_decision == FinalDecision.CAUTION


def test_low_ml_risk_with_supported_evidence_and_no_major_findings_can_produce_go() -> None:
    decision = evaluate_stage2_decision(
        _input(predicted_class="Low Risk", probabilities={"High Risk": 0.05, "Medium Risk": 0.1, "Low Risk": 0.85}),
        _result(agent=_agent(recommendation=AgentRecommendation.GO, severity="info")),
    )
    assert decision.final_decision == FinalDecision.GO


def test_insufficient_evidence_prevents_overconfident_go() -> None:
    decision = evaluate_stage2_decision(
        _input(predicted_class="Low Risk", probabilities={"High Risk": 0.05, "Medium Risk": 0.1, "Low Risk": 0.85}),
        _result(bundles=[_insufficient_bundle()], agent=_agent(recommendation=AgentRecommendation.GO, severity="info")),
    )
    assert decision.final_decision == FinalDecision.CAUTION
    assert decision.confidence_level.value == "low"
    assert any("No sufficient evidence" in item for item in decision.limitations)


def test_severe_agent_finding_increases_decision_score() -> None:
    stage2_input = _input(predicted_class="Low Risk", probabilities={"High Risk": 0.05, "Medium Risk": 0.1, "Low Risk": 0.85})
    low = evaluate_stage2_decision(stage2_input, _result(agent=_agent(recommendation=AgentRecommendation.GO, severity="low")))
    high = evaluate_stage2_decision(stage2_input, _result(agent=_agent(recommendation=AgentRecommendation.GO, severity="high")))
    assert high.decision_score > low.decision_score


def test_agent_action_items_are_propagated_to_required_actions() -> None:
    decision = evaluate_stage2_decision(_input(), _result(agent=_agent(action=True)))
    assert "Verify airspace authorization before launch." in decision.required_actions


def test_stage_contributions_include_required_stages_and_llm_placeholder() -> None:
    decision = evaluate_stage2_decision(_input(), _result())
    stages = {item.stage for item in decision.stage_contributions}
    assert {
        DecisionStageName.ML,
        DecisionStageName.SHAP,
        DecisionStageName.RAG,
        DecisionStageName.AGENT,
        DecisionStageName.SCENARIO_PROFILE,
        DecisionStageName.LLM,
    }.issubset(stages)
    llm = next(item for item in decision.stage_contributions if item.stage == DecisionStageName.LLM)
    assert llm.contribution == 0.0
    assert llm.metadata["llm_called"] is False


def test_decision_engine_does_not_import_llm_or_groq_modules() -> None:
    import uav_risk.stage2.decision_engine as module

    source = inspect.getsource(module).lower()
    assert "stage2.llm" not in source
    assert "groq" not in source
    assert "hyde" not in source
    assert "prompts" not in source


def test_score_and_contribution_bounds_are_validated() -> None:
    with pytest.raises(ValidationError):
        DecisionStageContribution(
            stage=DecisionStageName.ML,
            weight=1.2,
            contribution=0.1,
            signal="x",
            summary="x",
        )
    with pytest.raises(ValidationError):
        DecisionEngineResult(
            final_decision=FinalDecision.GO,
            decision_score=1.5,
            confidence_level="high",
        )


def test_scenario_profile_metadata_contributes_stage_signal() -> None:
    decision = evaluate_stage2_decision(
        _input(),
        _result(agent=_agent(metadata={"topic": "airspace", "related_scenario_fields": "airspace_altitude_agl_max_m"})),
    )
    contrib = next(item for item in decision.stage_contributions if item.stage == DecisionStageName.SCENARIO_PROFILE)
    assert contrib.signal == "concerns_present"
    assert contrib.contribution > 0.10
