from __future__ import annotations

import asyncio

import pytest

from uav_risk.stage2.agent.facade import AgentResultFacade
from uav_risk.stage2.contracts import (
    AgentInput,
    AgentRecommendation,
    AgentResult,
    EvidenceBundle,
    EvidenceSupportStatus,
    Stage2Status,
)


def _bundle_insufficient() -> EvidenceBundle:
    return EvidenceBundle(
        bundle_id="b1",
        query="q",
        support_status=EvidenceSupportStatus.INSUFFICIENT_EVIDENCE,
        confidence=0.0,
        no_evidence_reason="none",
    )


def _agent_input() -> AgentInput:
    return AgentInput(
        scenario_summary={},
        ml_prediction="High Risk",
        ml_probabilities={"High Risk": 0.9, "Low Risk": 0.05, "Medium Risk": 0.05},
        evidence_bundles=[],
    )


def test_constructor_does_not_instantiate_legacy_agent_by_default() -> None:
    facade = AgentResultFacade()
    assert getattr(facade, "_agent") is None


@pytest.mark.asyncio
async def test_run_without_agent_returns_degraded_with_agent_not_configured() -> None:
    facade = AgentResultFacade()
    result = await facade.run(_agent_input())
    assert result.status == Stage2Status.DEGRADED
    assert result.recommendation == AgentRecommendation.DEGRADED
    assert any(err.code == "agent_not_configured" for err in result.errors)


def test_normalize_maps_go_to_go() -> None:
    result = AgentResultFacade().normalize_result({"decision": "GO", "confidence": 0.8})
    assert result.recommendation == AgentRecommendation.GO


def test_normalize_maps_conditional_go_to_caution() -> None:
    result = AgentResultFacade().normalize_result({"decision": "CONDITIONAL-GO", "confidence": 0.8})
    assert result.recommendation == AgentRecommendation.CAUTION


def test_normalize_maps_no_go_to_no_go() -> None:
    result = AgentResultFacade().normalize_result({"decision": "NO-GO", "confidence": 0.8})
    assert result.recommendation == AgentRecommendation.NO_GO


def test_normalize_maps_unknown_to_degraded() -> None:
    result = AgentResultFacade().normalize_result({"decision": "MAYBE"})
    assert result.recommendation == AgentRecommendation.DEGRADED


def test_normalize_clamps_confidence_high() -> None:
    result = AgentResultFacade().normalize_result({"decision": "GO", "confidence": 9.0})
    assert result.confidence == 1.0


def test_normalize_clamps_confidence_low() -> None:
    result = AgentResultFacade().normalize_result({"decision": "GO", "confidence": -2.0})
    assert result.confidence == 0.0


def test_normalize_does_not_expose_reasoning_chain() -> None:
    result = AgentResultFacade().normalize_result({"decision": "GO", "reasoning_chain": ["secret"]})
    dump = result.model_dump()
    assert "reasoning_chain" not in dump


def test_normalize_does_not_expose_cot_style_fields() -> None:
    result = AgentResultFacade().normalize_result(
        {
            "decision": "GO",
            "chain_of_thought": "x",
            "thought": "y",
            "scratchpad": "z",
            "internal_reasoning": "a",
            "private_reasoning": "b",
        }
    )
    dumped = result.model_dump()
    text = str(dumped)
    for forbidden in ["chain_of_thought", "thought", "scratchpad", "internal_reasoning", "private_reasoning"]:
        assert forbidden not in text


def test_reasoning_chain_only_not_in_public_trace() -> None:
    result = AgentResultFacade().normalize_result({"reasoning_chain": ["private raw chain"]})
    assert "private raw chain" not in result.reasoning_trace.model_dump_json()


def test_completed_primary_recommendation_has_at_least_one_finding() -> None:
    result = AgentResultFacade().normalize_result({"decision": "GO", "status": "completed", "findings": []})
    assert result.findings


def test_evidence_backed_finding_not_fabricated_without_refs() -> None:
    result = AgentResultFacade().normalize_result(
        {
            "decision": "GO",
            "findings": [
                {
                    "finding_id": "f1",
                    "finding_type": "evidence_backed",
                    "severity": "medium",
                    "summary": "claimed evidence-backed",
                    "requires_evidence": True,
                    "evidence_references": [],
                }
            ],
        }
    )
    assert result.findings[0].finding_type != "evidence_backed"


def test_insufficient_bundle_promotes_insufficient_evidence_unless_structural_no_go() -> None:
    facade = AgentResultFacade()
    result = facade.normalize_result(
        {"decision": "GO", "confidence": 0.9},
        evidence_bundles=[_bundle_insufficient()],
    )
    assert result.recommendation == AgentRecommendation.INSUFFICIENT_EVIDENCE


def test_async_agent_run_called() -> None:
    class AsyncAgent:
        def __init__(self) -> None:
            self.called = False

        async def run(self, agent_input: AgentInput) -> dict:
            self.called = True
            return {"decision": "GO", "confidence": 0.7}

    agent = AsyncAgent()
    facade = AgentResultFacade(agent)
    result = asyncio.run(facade.run(_agent_input()))
    assert agent.called is True
    assert result.recommendation == AgentRecommendation.GO


def test_sync_agent_assess_called_when_run_absent() -> None:
    class SyncAgent:
        def __init__(self) -> None:
            self.called = False

        def assess(self, agent_input: AgentInput) -> dict:
            self.called = True
            return {"decision": "NO-GO", "confidence": 0.6}

    agent = SyncAgent()
    facade = AgentResultFacade(agent)
    result = asyncio.run(facade.run(_agent_input()))
    assert agent.called is True
    assert result.recommendation == AgentRecommendation.NO_GO


def test_agent_exception_returns_degraded_without_stacktrace_leak() -> None:
    class BadAgent:
        def run(self, agent_input: AgentInput) -> dict:
            raise RuntimeError("traceback: leaked internal")

    result = asyncio.run(AgentResultFacade(BadAgent()).run(_agent_input()))
    assert result.status == Stage2Status.DEGRADED
    assert "traceback" not in result.errors[0].message.lower()


def test_facade_returns_canonical_agent_result() -> None:
    result = AgentResultFacade().normalize_result({"decision": "GO"})
    assert isinstance(result, AgentResult)


def test_importing_facade_has_no_heavy_side_effects() -> None:
    module = __import__("uav_risk.stage2.agent.facade", fromlist=["AgentResultFacade"])
    assert hasattr(module, "AgentResultFacade")
