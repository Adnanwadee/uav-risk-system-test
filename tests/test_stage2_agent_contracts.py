from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from uav_risk.stage2.contracts import (
    AgentActionItem,
    AgentFinding,
    AgentFindingSeverity,
    AgentFindingType,
    AgentInput,
    AgentRecommendation,
    AgentResult,
    AgentEvidenceReference,
    EvidenceBundle,
    EvidenceSupportStatus,
    PublicReasoningTrace,
    Stage2Status,
)


def _insufficient_bundle() -> EvidenceBundle:
    return EvidenceBundle(
        bundle_id="b1",
        query="q",
        support_status=EvidenceSupportStatus.INSUFFICIENT_EVIDENCE,
        confidence=0.0,
        no_evidence_reason="no local evidence",
    )


def _trace() -> PublicReasoningTrace:
    return PublicReasoningTrace(
        observations=["o"],
        checks_performed=["c"],
        evidence_consulted=[],
        conflicts=[],
        limitations=[],
    )


def _finding() -> AgentFinding:
    return AgentFinding(
        finding_id="f1",
        finding_type=AgentFindingType.STRUCTURAL,
        severity=AgentFindingSeverity.MEDIUM,
        summary="structural check",
        requires_evidence=False,
    )


def test_agent_finding_rejects_empty_summary() -> None:
    with pytest.raises(ValidationError):
        AgentFinding(
            finding_id="f1",
            finding_type=AgentFindingType.STRUCTURAL,
            severity=AgentFindingSeverity.LOW,
            summary=" ",
            requires_evidence=False,
        )


def test_agent_finding_evidence_backed_requires_references() -> None:
    with pytest.raises(ValidationError):
        AgentFinding(
            finding_id="f1",
            finding_type=AgentFindingType.EVIDENCE_BACKED,
            severity=AgentFindingSeverity.LOW,
            summary="needs evidence",
            requires_evidence=False,
            evidence_references=[],
        )


def test_agent_finding_requires_evidence_true_requires_references() -> None:
    with pytest.raises(ValidationError):
        AgentFinding(
            finding_id="f1",
            finding_type=AgentFindingType.TOOL_CHECK,
            severity=AgentFindingSeverity.LOW,
            summary="guard",
            requires_evidence=True,
            evidence_references=[],
        )


def test_agent_input_rejects_probability_below_zero() -> None:
    with pytest.raises(ValidationError):
        AgentInput(ml_probabilities={"High Risk": -0.1})


def test_agent_input_rejects_probability_above_one() -> None:
    with pytest.raises(ValidationError):
        AgentInput(ml_probabilities={"High Risk": 1.1})


def test_agent_result_rejects_confidence_below_zero() -> None:
    with pytest.raises(ValidationError):
        AgentResult(
            status=Stage2Status.DEGRADED,
            recommendation=AgentRecommendation.DEGRADED,
            confidence=-0.1,
            reasoning_trace=_trace(),
        )


def test_agent_result_rejects_confidence_above_one() -> None:
    with pytest.raises(ValidationError):
        AgentResult(
            status=Stage2Status.DEGRADED,
            recommendation=AgentRecommendation.DEGRADED,
            confidence=1.1,
            reasoning_trace=_trace(),
        )


@pytest.mark.parametrize(
    "recommendation",
    [AgentRecommendation.GO, AgentRecommendation.CAUTION, AgentRecommendation.NO_GO],
)
def test_completed_result_requires_findings_for_primary_recommendations(
    recommendation: AgentRecommendation,
) -> None:
    with pytest.raises(ValidationError):
        AgentResult(
            status=Stage2Status.COMPLETED,
            recommendation=recommendation,
            confidence=0.8,
            findings=[],
            reasoning_trace=_trace(),
        )


def test_insufficient_evidence_result_valid_with_insufficient_bundle() -> None:
    result = AgentResult(
        status=Stage2Status.COMPLETED,
        recommendation=AgentRecommendation.INSUFFICIENT_EVIDENCE,
        confidence=0.2,
        findings=[],
        reasoning_trace=_trace(),
        evidence_bundles=[_insufficient_bundle()],
    )
    assert result.recommendation == AgentRecommendation.INSUFFICIENT_EVIDENCE


def test_agent_result_and_trace_expose_no_chain_of_thought_fields() -> None:
    result = AgentResult(
        status=Stage2Status.COMPLETED,
        recommendation=AgentRecommendation.GO,
        confidence=0.8,
        findings=[_finding()],
        reasoning_trace=_trace(),
        action_items=[
            AgentActionItem(
                action_id="a1",
                summary="continue monitoring",
                priority=AgentFindingSeverity.INFO,
            )
        ],
    )
    forbidden = {
        "reasoning_chain",
        "chain_of_thought",
        "thought",
        "scratchpad",
        "internal_reasoning",
    }
    assert forbidden.isdisjoint(result.model_fields)
    assert forbidden.isdisjoint(result.reasoning_trace.model_fields)


def test_agent_contracts_are_json_serializable() -> None:
    result = AgentResult(
        status=Stage2Status.COMPLETED,
        recommendation=AgentRecommendation.GO,
        confidence=0.8,
        findings=[
            AgentFinding(
                finding_id="f1",
                finding_type=AgentFindingType.EVIDENCE_BACKED,
                severity=AgentFindingSeverity.MEDIUM,
                summary="evidence-backed finding",
                requires_evidence=True,
                evidence_references=[
                    AgentEvidenceReference(
                        claim_id="c1",
                        citation_ids=["cit-1"],
                        summary="citation used",
                    )
                ],
            )
        ],
        action_items=[
            AgentActionItem(
                action_id="a1",
                summary="review citation",
                priority=AgentFindingSeverity.LOW,
                evidence_references=[
                    AgentEvidenceReference(
                        claim_id="c1",
                        citation_ids=["cit-1"],
                        summary="same citation",
                    )
                ],
            )
        ],
        reasoning_trace=_trace(),
        evidence_bundles=[_insufficient_bundle()],
    )
    payload = result.model_dump()
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert decoded["recommendation"] == "go"
