from __future__ import annotations

import inspect

import pytest

from uav_risk.stage2.contracts import (
    AgentRecommendation,
    AgentResult,
    EvidenceBundle,
    EvidenceSupportStatus,
    MLAssessmentSnapshot,
    PublicReasoningTrace,
    Stage2AssessmentInput,
    Stage2Status,
)
from uav_risk.stage2.pipeline_v2 import Stage2PipelineV2


def _input(evidence_bundles: list[EvidenceBundle] | None = None) -> Stage2AssessmentInput:
    return Stage2AssessmentInput(
        assessment_id="a1",
        user_id="u1",
        profile_id="p1",
        scenario_summary={"environment_weather_wind_mps": 8.0},
        ml=MLAssessmentSnapshot(
            predicted_class="High Risk",
            probabilities={"High Risk": 0.9, "Low Risk": 0.05, "Medium Risk": 0.05},
            shap_top_features=[],
            raw_feature_count=197,
            processed_feature_count=198,
        ),
        evidence_bundles=evidence_bundles or [],
        operator_notes="notes",
    )


@pytest.mark.asyncio
async def test_pipeline_uses_injected_operational_agent_when_provided() -> None:
    class FakeOperationalAgent:
        def __init__(self) -> None:
            self.called = False

        async def run(self, agent_input):
            self.called = True
            return AgentResult(
                status=Stage2Status.COMPLETED,
                recommendation=AgentRecommendation.INSUFFICIENT_EVIDENCE,
                confidence=0.2,
                findings=[
                    {
                        "finding_id": "f1",
                        "finding_type": "limitation",
                        "severity": "high",
                        "summary": "insufficient",
                        "requires_evidence": False,
                    }
                ],
                action_items=[],
                reasoning_trace=PublicReasoningTrace(),
                evidence_bundles=agent_input.evidence_bundles,
                errors=[],
            )

    agent = FakeOperationalAgent()
    result = await Stage2PipelineV2(operational_agent=agent).run(_input())
    assert agent.called is True
    assert result.agent_result is not None
    assert result.decision is not None


@pytest.mark.asyncio
async def test_pipeline_works_without_operational_agent_via_facade_degraded_path() -> None:
    result = await Stage2PipelineV2().run(_input())
    assert result.agent_result is not None
    assert result.agent_result.recommendation in {
        AgentRecommendation.DEGRADED,
        AgentRecommendation.INSUFFICIENT_EVIDENCE,
    }


@pytest.mark.asyncio
async def test_pipeline_passes_agent_input_with_ml_scenario_notes_and_evidence() -> None:
    existing = EvidenceBundle(
        bundle_id="b1",
        query="q",
        claims=[],
        citations=[],
        support_status=EvidenceSupportStatus.INSUFFICIENT_EVIDENCE,
        confidence=0.0,
        no_evidence_reason="none",
    )

    class RecordingAgent:
        def __init__(self) -> None:
            self.last_input = None

        async def run(self, agent_input):
            self.last_input = agent_input
            return AgentResult(
                status=Stage2Status.COMPLETED,
                recommendation=AgentRecommendation.INSUFFICIENT_EVIDENCE,
                confidence=0.1,
                findings=[
                    {
                        "finding_id": "f1",
                        "finding_type": "limitation",
                        "severity": "high",
                        "summary": "insufficient",
                        "requires_evidence": False,
                    }
                ],
                action_items=[],
                reasoning_trace=PublicReasoningTrace(),
                evidence_bundles=agent_input.evidence_bundles,
                errors=[],
            )

    agent = RecordingAgent()
    result = await Stage2PipelineV2(operational_agent=agent).run(_input([existing]))
    assert agent.last_input is not None
    assert agent.last_input.ml_prediction == "High Risk"
    assert "environment_weather_wind_mps" in agent.last_input.scenario_summary
    assert agent.last_input.operator_notes == "notes"
    assert agent.last_input.evidence_bundles
    assert result.evidence_bundles
    assert result.agent_result is not None


@pytest.mark.asyncio
async def test_pipeline_preserves_evidence_bundles() -> None:
    existing = EvidenceBundle(
        bundle_id="existing",
        query="q",
        claims=[],
        citations=[],
        support_status=EvidenceSupportStatus.INSUFFICIENT_EVIDENCE,
        confidence=0.0,
        no_evidence_reason="cached",
    )
    result = await Stage2PipelineV2().run(_input([existing]))
    assert any(item.bundle_id == "existing" for item in result.evidence_bundles)


@pytest.mark.asyncio
async def test_pipeline_status_degraded_when_operational_agent_returns_insufficient_evidence() -> None:
    class InsufficientAgent:
        async def run(self, agent_input):
            return AgentResult(
                status=Stage2Status.COMPLETED,
                recommendation=AgentRecommendation.INSUFFICIENT_EVIDENCE,
                confidence=0.2,
                findings=[
                    {
                        "finding_id": "f1",
                        "finding_type": "limitation",
                        "severity": "high",
                        "summary": "insufficient",
                        "requires_evidence": False,
                    }
                ],
                action_items=[],
                reasoning_trace=PublicReasoningTrace(),
                evidence_bundles=agent_input.evidence_bundles,
                errors=[],
            )

    result = await Stage2PipelineV2(operational_agent=InsufficientAgent()).run(_input())
    assert result.status == Stage2Status.DEGRADED


def test_pipeline_v2_does_not_import_legacy_pipeline() -> None:
    import uav_risk.stage2.pipeline_v2 as module

    source = inspect.getsource(module)
    assert "stage2.pipeline" not in source


def test_pipeline_v2_does_not_reference_feature_router_or_legacy_feature_generation() -> None:
    import uav_risk.stage2.pipeline_v2 as module

    source = inspect.getsource(module)
    assert "FeatureRouter" not in source
    assert "generate_all_features_map" not in source
    assert "MasterFlightPayload" not in source


def test_pipeline_v2_does_not_call_external_llm_clients_directly() -> None:
    import uav_risk.stage2.pipeline_v2 as module

    source = inspect.getsource(module).lower()
    assert "groq(" not in source
    assert "openai(" not in source
    assert "chat.completions" not in source
