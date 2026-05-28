from __future__ import annotations

import json

import pytest

from uav_risk.stage2.contracts import (
    AgentActionItem,
    AgentEvidenceReference,
    AgentFinding,
    AgentFindingSeverity,
    AgentFindingType,
    AgentRecommendation,
    AgentResult,
    DecisionConfidenceLevel,
    DecisionEngineResult,
    EvidenceBundle,
    EvidenceCitation,
    EvidenceOrigin,
    EvidenceSourceType,
    EvidenceSupportStatus,
    FinalDecision,
    LLMSynthesisStatus,
    MLAssessmentSnapshot,
    PublicReasoningTrace,
    Stage2AssessmentInput,
    Stage2AssessmentResult,
    Stage2Status,
)
from uav_risk.stage2.llm.orchestrator import (
    FORBIDDEN_OUTPUT_FIELDS,
    LLMOrchestrator,
    LLMOrchestratorConfig,
    build_llm_synthesis_context,
)


def _input() -> Stage2AssessmentInput:
    return Stage2AssessmentInput(
        assessment_id="a1",
        user_id="u1",
        profile_id="p1",
        scenario_summary={"environment_weather_wind_mps": 7.0},
        ml=MLAssessmentSnapshot(
            predicted_class="Medium Risk",
            probabilities={"Medium Risk": 0.6, "High Risk": 0.2, "Low Risk": 0.2},
            shap_top_features=[],
        ),
        operator_notes="Operator reports gusty winds near the site.",
    )


def _result() -> Stage2AssessmentResult:
    citation = EvidenceCitation(
        citation_id="cit1",
        source_id="src1",
        source_title="AC_107-2A",
        source_type=EvidenceSourceType.ADVISORY_CIRCULAR,
        origin=EvidenceOrigin.LOCAL_DOCUMENT,
        page=4,
        quote="Remote pilots should evaluate weather before operation.",
    )
    bundle = EvidenceBundle(
        bundle_id="bundle1",
        query="AC 107-2A preflight weather assessment small UAS wind conditions",
        claims=[],
        citations=[citation],
        support_status=EvidenceSupportStatus.SUPPORTED,
        confidence=0.8,
    )
    ref = AgentEvidenceReference(claim_id="claim1", citation_ids=["cit1"], summary="weather citation")
    agent = AgentResult(
        status=Stage2Status.COMPLETED,
        recommendation=AgentRecommendation.CAUTION,
        confidence=0.7,
        findings=[
            AgentFinding(
                finding_id="finding1",
                finding_type=AgentFindingType.EVIDENCE_BACKED,
                severity=AgentFindingSeverity.MEDIUM,
                summary="Weather and wind conditions require preflight assessment.",
                evidence_references=[ref],
                requires_evidence=True,
            )
        ],
        action_items=[
            AgentActionItem(
                action_id="action1",
                summary="Review weather and define wind limits.",
                priority=AgentFindingSeverity.HIGH,
                evidence_references=[ref],
            )
        ],
        reasoning_trace=PublicReasoningTrace(limitations=[]),
        evidence_bundles=[bundle],
        errors=[],
    )
    decision = DecisionEngineResult(
        final_decision=FinalDecision.CAUTION,
        decision_score=0.44,
        confidence_level=DecisionConfidenceLevel.MEDIUM,
        stage_weights={"ml": 0.22, "rag": 0.28, "agent": 0.25},
        stage_contributions=[],
        decision_reasons=["Weighted decision score requires caution."],
        blocking_reasons=[],
        required_actions=["Review weather and define wind limits."],
        limitations=[],
        evidence_refs=[ref],
    )
    return Stage2AssessmentResult(
        status=Stage2Status.COMPLETED,
        assessment_id="a1",
        evidence_bundles=[bundle],
        agent_result=agent,
        decision=decision,
        errors=[],
    )


class ValidProvider:
    async def generate_json(self, prompt: str, schema_name: str):
        assert schema_name == "LLMAgentSynthesis"
        assert "final decision" in prompt.lower()
        return {
            "executive_summary": "Mission should proceed only with caution.",
            "operational_interpretation": "Wind creates a preflight review concern.",
            "decision_explanation": "Decision Engine remains caution.",
            "key_risk_drivers": ["weather", "ml_signal"],
            "mitigation_narrative": "Review weather and define wind limits.",
            "consistency_warnings": [],
            "evidence_reference_ids": ["cit1"],
            "finding_ids": ["finding1"],
            "action_item_ids": ["action1"],
            "limitation_ids": [],
        }


class UnknownCitationProvider(ValidProvider):
    async def generate_json(self, prompt: str, schema_name: str):
        payload = await super().generate_json(prompt, schema_name)
        payload["evidence_reference_ids"] = ["made_up_citation"]
        return payload


class PrivateReasoningProvider(ValidProvider):
    async def generate_json(self, prompt: str, schema_name: str):
        payload = await super().generate_json(prompt, schema_name)
        payload["chain_of_thought"] = "hidden reasoning"
        return payload


class ChangedDecisionProvider(ValidProvider):
    async def generate_json(self, prompt: str, schema_name: str):
        payload = await super().generate_json(prompt, schema_name)
        payload["final_decision"] = "go"
        return payload


class ExplodingProvider:
    async def generate_json(self, prompt: str, schema_name: str):
        raise RuntimeError("provider unavailable")


@pytest.mark.asyncio
async def test_missing_provider_returns_fallback_synthesis() -> None:
    synthesis = await LLMOrchestrator().synthesize(_input(), _result())
    assert synthesis.status == LLMSynthesisStatus.FALLBACK
    assert synthesis.metadata["llm_called"] is False
    assert synthesis.executive_summary


@pytest.mark.asyncio
async def test_disabled_orchestrator_returns_disabled_synthesis() -> None:
    synthesis = await LLMOrchestrator(config=LLMOrchestratorConfig(enabled=False)).synthesize(_input(), _result())
    assert synthesis.status == LLMSynthesisStatus.DISABLED


@pytest.mark.asyncio
async def test_fake_provider_valid_json_returns_generated_synthesis() -> None:
    synthesis = await LLMOrchestrator(
        provider=ValidProvider(),
        config=LLMOrchestratorConfig(provider_name="fake", model_name="fake-model"),
    ).synthesize(_input(), _result())
    assert synthesis.status == LLMSynthesisStatus.GENERATED
    assert synthesis.provider == "fake"
    assert synthesis.model_name == "fake-model"
    assert synthesis.metadata["llm_called"] is True


@pytest.mark.asyncio
async def test_unknown_citation_id_is_rejected_and_falls_back() -> None:
    synthesis = await LLMOrchestrator(provider=UnknownCitationProvider()).synthesize(_input(), _result())
    assert synthesis.status == LLMSynthesisStatus.FALLBACK
    assert any(w.warning_type == "llm_provider_invalid" for w in synthesis.consistency_warnings)


@pytest.mark.asyncio
async def test_private_reasoning_field_is_rejected_and_falls_back() -> None:
    synthesis = await LLMOrchestrator(provider=PrivateReasoningProvider()).synthesize(_input(), _result())
    assert synthesis.status == LLMSynthesisStatus.FALLBACK


@pytest.mark.asyncio
async def test_provider_cannot_change_final_decision() -> None:
    synthesis = await LLMOrchestrator(provider=ChangedDecisionProvider()).synthesize(_input(), _result())
    assert synthesis.status == LLMSynthesisStatus.FALLBACK


@pytest.mark.asyncio
async def test_provider_exception_falls_back() -> None:
    synthesis = await LLMOrchestrator(provider=ExplodingProvider()).synthesize(_input(), _result())
    assert synthesis.status == LLMSynthesisStatus.FALLBACK


def test_context_includes_decision_findings_evidence_actions_and_limitations() -> None:
    context = build_llm_synthesis_context(_input(), _result())
    assert context["decision"]["final_decision"] == "caution"
    assert context["evidence"][0]["citation_ids"] == ["cit1"]
    assert context["agent"]["findings"][0]["finding_id"] == "finding1"
    assert context["agent"]["action_items"][0]["action_id"] == "action1"
    assert "cit1" in context["allowed_reference_ids"]


def test_context_does_not_include_forbidden_private_reasoning_fields() -> None:
    encoded = json.dumps(build_llm_synthesis_context(_input(), _result())).lower()
    for token in FORBIDDEN_OUTPUT_FIELDS:
        assert token not in encoded
