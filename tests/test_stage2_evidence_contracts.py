from __future__ import annotations

import pytest
from pydantic import ValidationError

from uav_risk.stage2.contracts import (
    AgentEvidenceReference,
    EvidenceBundle,
    EvidenceCitation,
    EvidenceClaim,
    EvidenceOrigin,
    EvidenceSourceType,
    EvidenceSupportStatus,
    EvidenceUse,
    LegalCitation,
    PublicReasoningTrace,
    Stage2Error,
    collect_unique_citations,
    make_insufficient_evidence_bundle,
)


def _valid_citation(**overrides: object) -> EvidenceCitation:
    payload = {
        "citation_id": "cit-1",
        "source_id": "doc-1",
        "source_title": "FAA Part 107",
        "source_type": EvidenceSourceType.REGULATION,
        "origin": EvidenceOrigin.LOCAL_DOCUMENT,
        "quote": "Operate under visual line of sight.",
    }
    payload.update(overrides)
    return EvidenceCitation(**payload)


def test_valid_evidence_citation_is_accepted() -> None:
    citation = _valid_citation(retrieval_score=0.75, rerank_score=0.6, page=3)
    assert citation.citation_id == "cit-1"


def test_evidence_citation_rejects_empty_quote() -> None:
    with pytest.raises(ValidationError):
        _valid_citation(quote="   ")


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_evidence_citation_rejects_invalid_retrieval_score(value: float) -> None:
    with pytest.raises(ValidationError):
        _valid_citation(retrieval_score=value)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_evidence_citation_rejects_invalid_rerank_score(value: float) -> None:
    with pytest.raises(ValidationError):
        _valid_citation(rerank_score=value)


@pytest.mark.parametrize(
    "origin",
    [EvidenceOrigin.LLM_SYNTHESIS, EvidenceOrigin.HYDE_GENERATED],
)
def test_evidence_citation_rejects_forbidden_origins(origin: EvidenceOrigin) -> None:
    with pytest.raises(ValidationError):
        _valid_citation(origin=origin)


def test_evidence_claim_supported_requires_citations() -> None:
    with pytest.raises(ValidationError):
        EvidenceClaim(
            claim_id="c1",
            claim="Claim",
            support_status=EvidenceSupportStatus.SUPPORTED,
            evidence_use=EvidenceUse.OPERATIONAL_SUPPORT,
            citations=[],
            confidence=0.8,
            limitations=[],
            conflicts=[],
        )


def test_evidence_claim_partially_supported_requires_citations() -> None:
    with pytest.raises(ValidationError):
        EvidenceClaim(
            claim_id="c1",
            claim="Claim",
            support_status=EvidenceSupportStatus.PARTIALLY_SUPPORTED,
            evidence_use=EvidenceUse.OPERATIONAL_SUPPORT,
            citations=[],
            confidence=0.8,
            limitations=[],
            conflicts=[],
        )


def test_evidence_claim_insufficient_requires_limitations() -> None:
    with pytest.raises(ValidationError):
        EvidenceClaim(
            claim_id="c1",
            claim="Claim",
            support_status=EvidenceSupportStatus.INSUFFICIENT_EVIDENCE,
            evidence_use=EvidenceUse.LIMITATION,
            citations=[],
            confidence=0.2,
            limitations=[],
            conflicts=[],
        )


def test_evidence_claim_conflicting_requires_conflicts() -> None:
    with pytest.raises(ValidationError):
        EvidenceClaim(
            claim_id="c1",
            claim="Claim",
            support_status=EvidenceSupportStatus.CONFLICTING,
            evidence_use=EvidenceUse.CONTRADICTION,
            citations=[],
            confidence=0.4,
            limitations=["Has evidence"],
            conflicts=[],
        )


def test_evidence_bundle_insufficient_requires_reason() -> None:
    with pytest.raises(ValidationError):
        EvidenceBundle(
            bundle_id="b1",
            query="query",
            claims=[],
            citations=[],
            support_status=EvidenceSupportStatus.INSUFFICIENT_EVIDENCE,
            confidence=0.0,
            no_evidence_reason=" ",
        )


def test_make_insufficient_evidence_bundle_shape() -> None:
    bundle = make_insufficient_evidence_bundle("Query", "No local docs matched")
    assert bundle.support_status == EvidenceSupportStatus.INSUFFICIENT_EVIDENCE
    assert bundle.confidence == 0.0
    assert bundle.claims == []
    assert bundle.citations == []
    assert bundle.no_evidence_reason == "No local docs matched"


def test_collect_unique_citations_dedupes_and_preserves_order() -> None:
    first = _valid_citation(citation_id="a")
    dup = _valid_citation(citation_id="a", quote="Different quote")
    second = _valid_citation(citation_id="b")
    claims = [
        EvidenceClaim(
            claim_id="c1",
            claim="one",
            support_status=EvidenceSupportStatus.SUPPORTED,
            evidence_use=EvidenceUse.OPERATIONAL_SUPPORT,
            citations=[first, second],
            confidence=0.9,
            limitations=[],
            conflicts=[],
        ),
        EvidenceClaim(
            claim_id="c2",
            claim="two",
            support_status=EvidenceSupportStatus.SUPPORTED,
            evidence_use=EvidenceUse.OPERATIONAL_SUPPORT,
            citations=[dup],
            confidence=0.7,
            limitations=[],
            conflicts=[],
        ),
    ]
    result = collect_unique_citations(claims)
    assert [item.citation_id for item in result] == ["a", "b"]
    assert result[0].quote == "Operate under visual line of sight."


def test_public_reasoning_trace_has_no_chain_of_thought_fields() -> None:
    trace = PublicReasoningTrace(
        observations=["obs"],
        checks_performed=["check"],
        evidence_consulted=[
            AgentEvidenceReference(claim_id="c1", citation_ids=["cit-1"], summary="summary")
        ],
    )
    assert "reasoning_chain" not in trace.model_fields
    assert "chain_of_thought" not in trace.model_fields
    assert "thought" not in trace.model_fields
    assert "scratchpad" not in trace.model_fields
    assert "internal_reasoning" not in trace.model_fields


def test_legal_citation_alias_points_to_evidence_citation() -> None:
    assert LegalCitation is EvidenceCitation


def test_stage2_error_rejects_empty_code() -> None:
    with pytest.raises(ValidationError):
        Stage2Error(code=" ", message="msg")


def test_stage2_error_rejects_empty_message() -> None:
    with pytest.raises(ValidationError):
        Stage2Error(code="ERR", message=" ")
