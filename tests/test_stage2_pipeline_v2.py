from __future__ import annotations

import asyncio
import inspect

import pytest

from uav_risk.stage2.agent.facade import AgentResultFacade
from uav_risk.stage2.contracts import (
    AgentRecommendation,
    AgentResult,
    DecisionPolicyConfig,
    EvidenceBundle,
    EvidenceSupportStatus,
    MLAssessmentSnapshot,
    PublicReasoningTrace,
    Stage2AssessmentInput,
    Stage2AssessmentResult,
    Stage2Status,
)
from uav_risk.stage2.llm.orchestrator import LLMOrchestrator
from uav_risk.stage2.pipeline_v2 import (
    Stage2PipelineV2,
    build_agent_input,
    build_stage2_evidence_queries,
)


def _stage2_input(evidence_bundles: list[EvidenceBundle] | None = None) -> Stage2AssessmentInput:
    return Stage2AssessmentInput(
        assessment_id="a1",
        user_id="u1",
        profile_id="p1",
        scenario_summary={
            "airspace_altitude_agl_max_m": 120.0,
            "environment_weather_wind_mps": 4.5,
            "comms_uplink_ok": True,
        },
        ml=MLAssessmentSnapshot(
            predicted_class="High Risk",
            probabilities={"High Risk": 0.9, "Low Risk": 0.05, "Medium Risk": 0.05},
            shap_top_features=[],
            raw_feature_count=197,
            processed_feature_count=198,
        ),
        evidence_bundles=evidence_bundles or [],
        operator_notes="Check near controlled zone",
    )


def _completed_agent_result() -> AgentResult:
    return AgentResult(
        status=Stage2Status.COMPLETED,
        recommendation=AgentRecommendation.CAUTION,
        confidence=0.7,
        findings=[
            {
                "finding_id": "f1",
                "finding_type": "tool_check",
                "severity": "medium",
                "summary": "Risk is elevated.",
                "requires_evidence": False,
            }
        ],
        action_items=[],
        reasoning_trace=PublicReasoningTrace(),
        evidence_bundles=[],
        errors=[],
    )


def test_constructor_is_lightweight() -> None:
    pipeline = Stage2PipelineV2()
    assert pipeline is not None


def test_constructor_rejects_negative_max_evidence_queries() -> None:
    with pytest.raises(ValueError):
        Stage2PipelineV2(max_evidence_queries=-1)


@pytest.mark.asyncio
async def test_run_without_rag_and_without_evidence_returns_degraded_with_insufficient_bundle() -> None:
    pipeline = Stage2PipelineV2()
    result = await pipeline.run(_stage2_input())
    assert isinstance(result, Stage2AssessmentResult)
    assert result.status == Stage2Status.DEGRADED
    assert result.evidence_bundles
    assert result.evidence_bundles[0].support_status == EvidenceSupportStatus.INSUFFICIENT_EVIDENCE


@pytest.mark.asyncio
async def test_run_with_fake_rag_calls_retrieve_evidence_for_deterministic_queries() -> None:
    calls: list[str] = []

    class FakeRAG:
        async def retrieve_evidence(self, query: str, *, scenario_context: dict[str, object] | None = None) -> EvidenceBundle:
            calls.append(query)
            return EvidenceBundle(
                bundle_id=f"b_{len(calls)}",
                query=query,
                claims=[],
                citations=[],
                support_status=EvidenceSupportStatus.INSUFFICIENT_EVIDENCE,
                confidence=0.0,
                no_evidence_reason="none",
            )

    pipeline = Stage2PipelineV2(rag_adapter=FakeRAG(), max_evidence_queries=2)
    await pipeline.run(_stage2_input())
    assert len(calls) == 2
    assert calls == build_stage2_evidence_queries(_stage2_input(), 2)


@pytest.mark.asyncio
async def test_run_preserves_preexisting_evidence_bundles() -> None:
    existing = EvidenceBundle(
        bundle_id="existing",
        query="q",
        claims=[],
        citations=[],
        support_status=EvidenceSupportStatus.INSUFFICIENT_EVIDENCE,
        confidence=0.0,
        no_evidence_reason="cached",
    )
    pipeline = Stage2PipelineV2()
    result = await pipeline.run(_stage2_input(evidence_bundles=[existing]))
    assert any(bundle.bundle_id == "existing" for bundle in result.evidence_bundles)


@pytest.mark.asyncio
async def test_run_calls_agent_facade() -> None:
    class RecordingFacade(AgentResultFacade):
        def __init__(self) -> None:
            super().__init__(agent=None)
            self.called = False

        async def run(self, agent_input):  # type: ignore[override]
            self.called = True
            return _completed_agent_result()

    facade = RecordingFacade()
    pipeline = Stage2PipelineV2(agent_facade=facade)
    await pipeline.run(_stage2_input())
    assert facade.called is True


@pytest.mark.asyncio
async def test_run_returns_stage2_assessment_result() -> None:
    pipeline = Stage2PipelineV2()
    result = await pipeline.run(_stage2_input())
    assert isinstance(result, Stage2AssessmentResult)
    assert result.decision is not None


@pytest.mark.asyncio
async def test_run_can_attach_llm_synthesis_when_orchestrator_is_configured() -> None:
    pipeline = Stage2PipelineV2(llm_orchestrator=LLMOrchestrator())
    result = await pipeline.run(_stage2_input())
    assert result.decision is not None
    assert result.llm_synthesis is not None
    assert result.llm_synthesis.status.value == "fallback"
    assert result.llm_synthesis.executive_summary


@pytest.mark.asyncio
async def test_failed_agent_path_is_safe_without_stacktrace_leakage() -> None:
    class ExplodingFacade(AgentResultFacade):
        async def run(self, agent_input):  # type: ignore[override]
            raise RuntimeError("traceback: internal detail")

    pipeline = Stage2PipelineV2(agent_facade=ExplodingFacade())
    result = await pipeline.run(_stage2_input())
    assert result.status == Stage2Status.FAILED
    assert result.errors
    assert "traceback" not in result.errors[0].message.lower()


def test_query_builder_is_deterministic() -> None:
    stage2_input = _stage2_input()
    assert build_stage2_evidence_queries(stage2_input, 3) == build_stage2_evidence_queries(stage2_input, 3)


def test_query_builder_respects_max_queries() -> None:
    queries = build_stage2_evidence_queries(_stage2_input(), 1)
    assert len(queries) == 1


def test_query_builder_no_generic_risk_class_query() -> None:
    queries = build_stage2_evidence_queries(_stage2_input(), 5)
    text = " | ".join(queries).lower()
    assert "uav operational guidance for risk class" not in text
    assert "risk class high risk" not in text


def test_pipeline_v2_does_not_import_legacy_master_payload() -> None:
    import uav_risk.stage2.pipeline_v2 as module

    source = inspect.getsource(module)
    assert "MasterFlightPayload" not in source


def test_pipeline_v2_does_not_import_feature_router() -> None:
    import uav_risk.stage2.pipeline_v2 as module

    source = inspect.getsource(module)
    assert "FeatureRouter" not in source


def test_pipeline_v2_does_not_import_generate_all_features_map() -> None:
    import uav_risk.stage2.pipeline_v2 as module

    source = inspect.getsource(module)
    assert "generate_all_features_map" not in source


def test_pipeline_v2_does_not_call_core_ml_api_feature_generation() -> None:
    import uav_risk.stage2.pipeline_v2 as module

    source = inspect.getsource(module)
    assert "uav_risk.core" not in source
    assert "uav_risk.ml" not in source
    assert "uav_risk.api" not in source


@pytest.mark.asyncio
async def test_pipeline_accepts_decision_policy_and_keeps_behavior_stable() -> None:
    policy = DecisionPolicyConfig(policy_name="test_policy", policy_version="1.1")
    pipeline = Stage2PipelineV2(decision_policy=policy)
    result = await pipeline.run(_stage2_input())
    assert result.decision is not None
    assert result.decision.metadata["policy_name"] == "test_policy"
    assert result.decision.metadata["policy_version"] == "1.1"


@pytest.mark.asyncio
async def test_pipeline_metadata_exposes_rag_status_fields_from_bundles() -> None:
    class FakeRAG:
        async def retrieve_evidence(self, query: str, *, scenario_context=None, retrieval_origin=None):
            from uav_risk.stage2.contracts import EvidenceBundle, EvidenceCitation, EvidenceOrigin, EvidenceSourceType

            citation = EvidenceCitation(
                citation_id="c-meta",
                source_id="src-meta",
                source_title="AC_107-2A",
                source_type=EvidenceSourceType.INTERNAL_DOC,
                origin=EvidenceOrigin.LOCAL_DOCUMENT,
                quote="metadata propagation quote for pipeline diagnostics checks.",
                metadata={"source_filename": "AC_107-2A.pdf", "page_start": 4},
            )
            return EvidenceBundle(
                bundle_id="b-meta",
                query=query,
                claims=[],
                citations=[citation],
                support_status=EvidenceSupportStatus.SUPPORTED,
                confidence=0.8,
                metadata={
                    "retrieval_origin": retrieval_origin or "scenario_driven",
                    "reranker_configured": True,
                    "reranker_available": True,
                    "reranker_used": True,
                    "reranker_reason": "reranker_invoked",
                    "corpus_coverage_status": "complete",
                    "expected_source_count": 9,
                    "indexed_source_count": 9,
                    "missing_sources": "",
                },
            )

    result = await Stage2PipelineV2(rag_adapter=FakeRAG(), max_evidence_queries=1).run(_stage2_input())
    assert result.metadata.get("scenario_evidence_status") in {"grounded", "partial"}
    assert result.metadata.get("reranker_used") is True
    assert result.metadata.get("corpus_coverage_status") == "complete"
