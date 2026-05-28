from __future__ import annotations

import pytest

from uav_risk.stage2.contracts import (
    AgentRecommendation,
    AgentResult,
    DecisionConfidenceLevel,
    DecisionEngineResult,
    DecisionStageContribution,
    DecisionStageName,
    EvidenceBundle,
    EvidenceSupportStatus,
    FinalDecision,
    LLMAgentSynthesis,
    LLMSynthesisStatus,
    PublicReasoningTrace,
    Stage2AssessmentResult,
    Stage2Status,
)


@pytest.mark.asyncio
async def test_smoke_separates_global_quality_from_scenario_sufficiency(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.run_stage2_pipeline_v2_smoke as smoke

    insufficient = EvidenceBundle(
        bundle_id="b1",
        query="q1",
        claims=[],
        citations=[],
        support_status=EvidenceSupportStatus.INSUFFICIENT_EVIDENCE,
        confidence=0.0,
        no_evidence_reason="none",
    )

    supported = EvidenceBundle(
        bundle_id="b2",
        query="q2",
        claims=[],
        citations=[],
        support_status=EvidenceSupportStatus.SUPPORTED,
        confidence=0.8,
    )

    class FakePipeline:
        def __init__(self, *args, **kwargs) -> None:
            assert kwargs.get("llm_orchestrator") is not None

        async def run(self, _):
            return Stage2AssessmentResult(
                status=Stage2Status.DEGRADED,
                assessment_id="a1",
                evidence_bundles=[supported, insufficient],
                agent_result=AgentResult(
                    status=Stage2Status.COMPLETED,
                    recommendation=AgentRecommendation.CAUTION,
                    confidence=0.5,
                    findings=[
                        {
                            "finding_id": "f1",
                            "finding_type": "limitation",
                            "severity": "medium",
                            "summary": "limited",
                            "requires_evidence": False,
                        }
                    ],
                    action_items=[],
                    reasoning_trace=PublicReasoningTrace(),
                    evidence_bundles=[supported, insufficient],
                    errors=[],
                ),
                decision=DecisionEngineResult(
                    final_decision=FinalDecision.CAUTION,
                    decision_score=0.42,
                    confidence_level=DecisionConfidenceLevel.MEDIUM,
                    stage_weights={"ml": 0.22, "llm": 0.05},
                    stage_contributions=[
                        DecisionStageContribution(
                            stage=DecisionStageName.ML,
                            weight=0.22,
                            contribution=0.55,
                            signal="medium_risk",
                            summary="ML signal indicates elevated risk.",
                            reasons=["Medium Risk"],
                        ),
                        DecisionStageContribution(
                            stage=DecisionStageName.LLM,
                            weight=0.05,
                            contribution=0.0,
                            signal="not_configured",
                            summary="LLM synthesis not evaluated in this patch.",
                        ),
                    ],
                    decision_reasons=["Decision requires caution."],
                    blocking_reasons=[],
                    required_actions=["Review operational limitations."],
                    limitations=["Scenario evidence incomplete."],
                    evidence_refs=[],
                ),
                llm_synthesis=LLMAgentSynthesis(
                    status=LLMSynthesisStatus.FALLBACK,
                    executive_summary="Final operational decision is caution.",
                    operational_interpretation="Evidence is incomplete for one topic.",
                    decision_explanation="Decision requires caution.",
                    key_risk_drivers=["medium_risk", "scenario_gap"],
                    mitigation_narrative="Review operational limitations.",
                    consistency_warnings=[
                        {"warning_type": "llm_fallback", "message": "Fallback synthesis used."}
                    ],
                    evidence_reference_ids=[],
                    finding_ids=["f1"],
                    action_item_ids=[],
                    limitation_ids=["llm_fallback"],
                ),
                errors=[],
                metadata={},
            )

    class FakeProv:
        provenance_status = "current"
        index_path = "/tmp/dense_index.faiss"
        sparse_index_path = "/tmp/sparse_index.pkl"
        path_resolution_status = "canonical"

    class FakeDiag:
        metadata = {
            "retrieval_usable": True,
            "rag_quality_is_proven": True,
            "quality_is_proven": True,
        }

    monkeypatch.setattr(smoke, "Stage2PipelineV2", FakePipeline)
    monkeypatch.setattr(smoke, "inspect_rag_index_provenance", lambda: FakeProv())
    async def fake_run_diag(**kwargs):
        return FakeDiag()

    monkeypatch.setattr(smoke, "run_rag_runtime_diagnostic", fake_run_diag)

    result = await smoke._run(use_real_rag=True, use_llm_fallback=True)
    assert result["rag_quality_is_proven"] is True
    assert result["quality_is_proven"] is True
    assert result["scenario_evidence_complete"] is False
    assert result["scenario_evidence_status"] == "incomplete"
    assert "evidence_bundle_details" in result
    assert isinstance(result["evidence_bundle_details"], list)

    assert result["final_decision"] == "caution"
    assert result["decision_score"] == 0.42
    assert result["decision_confidence_level"] == "medium"
    assert result["decision_reasons"] == ["Decision requires caution."]
    assert result["blocking_reasons"] == []
    assert result["required_actions"] == ["Review operational limitations."]
    assert result["decision_limitations"] == ["Scenario evidence incomplete."]
    assert result["stage_weights"] == {"ml": 0.22, "llm": 0.05}
    assert result["stage_contributions"] == [
        {
            "stage": "ml",
            "weight": 0.22,
            "contribution": 0.55,
            "signal": "medium_risk",
            "summary": "ML signal indicates elevated risk.",
        },
        {
            "stage": "llm",
            "weight": 0.05,
            "contribution": 0.0,
            "signal": "not_configured",
            "summary": "LLM synthesis not evaluated in this patch.",
        },
    ]

    assert result["llm_synthesis_status"] == "fallback"
    assert result["llm_executive_summary"] == "Final operational decision is caution."
    assert result["llm_operational_interpretation"] == "Evidence is incomplete for one topic."
    assert result["llm_decision_explanation"] == "Decision requires caution."
    assert result["llm_key_risk_drivers"] == ["medium_risk", "scenario_gap"]
    assert result["llm_mitigation_narrative"] == "Review operational limitations."
    assert result["llm_consistency_warnings"] == [
        {
            "warning_type": "llm_fallback",
            "message": "Fallback synthesis used.",
            "related_ids": [],
        }
    ]
