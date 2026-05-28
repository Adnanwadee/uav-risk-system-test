from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

from uav_risk.stage2.contracts import (
    AgentRecommendation,
    AgentResult,
    EvidenceBundle,
    EvidenceCitation,
    EvidenceOrigin,
    EvidenceSourceType,
    EvidenceSupportStatus,
    LLMAgentSynthesis,
    LLMSynthesisStatus,
    MLAssessmentSnapshot,
    OperationalReport,
    OperationalReportSection,
    OperationalReportSectionType,
    PublicReasoningTrace,
    SHAPFeatureAttribution,
    Stage2AssessmentInput,
    Stage2AssessmentResult,
    Stage2Status,
)
from uav_risk.stage2.decision_engine import evaluate_stage2_decision
from uav_risk.stage2.reporting import build_operational_report, render_markdown_report


def _input() -> Stage2AssessmentInput:
    return Stage2AssessmentInput(
        assessment_id="a1",
        user_id="u1",
        profile_id="p1",
        scenario_summary={"mission_pattern": "grid"},
        ml=MLAssessmentSnapshot(
            predicted_class="Medium Risk",
            probabilities={"High Risk": 0.2, "Low Risk": 0.2, "Medium Risk": 0.6},
            shap_top_features=[
                SHAPFeatureAttribution(feature="wind", value=7.1, importance=0.22, direction="up")
            ],
            raw_feature_count=197,
            processed_feature_count=198,
        ),
        evidence_bundles=[],
        operator_notes="notes",
    )


def _agent_result() -> AgentResult:
    return AgentResult(
        status=Stage2Status.COMPLETED,
        recommendation=AgentRecommendation.CAUTION,
        confidence=0.7,
        findings=[
            {
                "finding_id": "f1",
                "finding_type": "limitation",
                "severity": "medium",
                "summary": "Limited evidence.",
                "requires_evidence": False,
            }
        ],
        action_items=[
            {"action_id": "a1", "summary": "Review route.", "priority": "high", "evidence_references": []}
        ],
        reasoning_trace=PublicReasoningTrace(),
        evidence_bundles=[],
        errors=[],
    )


def test_operational_report_section_rejects_empty_title() -> None:
    with pytest.raises(ValidationError):
        OperationalReportSection(
            section_type=OperationalReportSectionType.METADATA,
            title=" ",
            content=["ok"],
            citation_ids=[],
            metadata={},
        )


def test_operational_report_rejects_empty_sections() -> None:
    with pytest.raises(ValidationError):
        OperationalReport(
            report_id="r1",
            assessment_id="a1",
            status=Stage2Status.COMPLETED,
            sections=[],
        )


def test_section_content_rejects_chain_of_thought_tokens() -> None:
    with pytest.raises(ValidationError):
        OperationalReportSection(
            section_type=OperationalReportSectionType.LIMITATIONS,
            title="x",
            content=["contains chain_of_thought token"],
            citation_ids=[],
            metadata={},
        )


def test_build_operational_report_returns_operational_report() -> None:
    stage2_input = _input()
    stage2_result = Stage2AssessmentResult(
        status=Stage2Status.COMPLETED,
        assessment_id="a1",
        evidence_bundles=[],
        agent_result=_agent_result(),
        errors=[],
    )
    report = build_operational_report(stage2_input, stage2_result)
    assert isinstance(report, OperationalReport)


def test_report_includes_ml_signal_section() -> None:
    report = build_operational_report(
        _input(),
        Stage2AssessmentResult(
            status=Stage2Status.DEGRADED,
            assessment_id="a1",
            evidence_bundles=[],
            agent_result=None,
            errors=[],
        ),
    )
    assert any(section.section_type == OperationalReportSectionType.ML_SIGNAL for section in report.sections)


def test_report_includes_evidence_insufficiency_reason_when_present() -> None:
    bundle = EvidenceBundle(
        bundle_id="b1",
        query="q",
        support_status=EvidenceSupportStatus.INSUFFICIENT_EVIDENCE,
        confidence=0.0,
        no_evidence_reason="No relevant doc chunk",
        claims=[],
        citations=[],
    )
    report = build_operational_report(
        _input(),
        Stage2AssessmentResult(
            status=Stage2Status.DEGRADED,
            assessment_id="a1",
            evidence_bundles=[bundle],
            agent_result=None,
            errors=[],
        ),
    )
    evidence_section = next(s for s in report.sections if s.section_type == OperationalReportSectionType.EVIDENCE_SUMMARY)
    assert any("No relevant doc chunk" in line for line in evidence_section.content)


def test_report_includes_agent_assessment_when_agent_result_exists() -> None:
    report = build_operational_report(
        _input(),
        Stage2AssessmentResult(
            status=Stage2Status.COMPLETED,
            assessment_id="a1",
            evidence_bundles=[],
            agent_result=_agent_result(),
            errors=[],
        ),
    )
    assert any(section.section_type == OperationalReportSectionType.AGENT_ASSESSMENT for section in report.sections)


def test_render_markdown_report_includes_expected_headings() -> None:
    report = build_operational_report(
        _input(),
        Stage2AssessmentResult(
            status=Stage2Status.COMPLETED,
            assessment_id="a1",
            evidence_bundles=[],
            agent_result=_agent_result(),
            errors=[],
        ),
    )
    md = render_markdown_report(report)
    assert "# Operational Report" in md
    assert "## ML Signal" in md


def test_markdown_includes_citation_ids_when_present() -> None:
    citation = EvidenceCitation(
        citation_id="c1",
        source_id="s1",
        source_title="doc",
        source_type=EvidenceSourceType.INTERNAL_DOC,
        origin=EvidenceOrigin.LOCAL_DOCUMENT,
        quote="retrieved quote",
    )
    bundle = EvidenceBundle(
        bundle_id="b1",
        query="q",
        claims=[],
        citations=[citation],
        support_status=EvidenceSupportStatus.SUPPORTED,
        confidence=0.8,
    )
    report = build_operational_report(
        _input(),
        Stage2AssessmentResult(
            status=Stage2Status.COMPLETED,
            assessment_id="a1",
            evidence_bundles=[bundle],
            agent_result=_agent_result(),
            errors=[],
        ),
    )
    md = render_markdown_report(report)
    assert "c1" in md


def test_markdown_excludes_chain_of_thought_tokens() -> None:
    report = build_operational_report(
        _input(),
        Stage2AssessmentResult(
            status=Stage2Status.COMPLETED,
            assessment_id="a1",
            evidence_bundles=[],
            agent_result=_agent_result(),
            errors=[],
        ),
    )
    md = render_markdown_report(report).lower()
    forbidden = [
        "reasoning_chain",
        "chain_of_thought",
        "thought",
        "scratchpad",
        "internal_reasoning",
        "private_reasoning",
    ]
    for token in forbidden:
        assert token not in md


def test_reporting_module_does_not_import_groq_or_llm_clients() -> None:
    import uav_risk.stage2.reporting as module

    source = inspect.getsource(module)
    assert "Groq" not in source
    assert "report_writer" not in source


def test_reporting_module_does_not_create_evidence_citation_from_free_text() -> None:
    import uav_risk.stage2.reporting as module

    source = inspect.getsource(module)
    assert "EvidenceCitation(" not in source


def test_report_preserves_citation_provenance_fields_in_evidence_section() -> None:
    citation = EvidenceCitation(
        citation_id="cprov",
        source_id="sid",
        source_title="doc",
        source_type=EvidenceSourceType.INTERNAL_DOC,
        origin=EvidenceOrigin.LOCAL_DOCUMENT,
        quote="retrieved quote",
        metadata={"source_filename": "src.pdf", "page_start": 4, "page_end": 5, "section_title": "sec"},
    )
    bundle = EvidenceBundle(
        bundle_id="bprov",
        query="q",
        claims=[],
        citations=[citation],
        support_status=EvidenceSupportStatus.SUPPORTED,
        confidence=0.9,
    )
    report = build_operational_report(
        _input(),
        Stage2AssessmentResult(
            status=Stage2Status.COMPLETED,
            assessment_id="a1",
            evidence_bundles=[bundle],
            agent_result=_agent_result(),
            errors=[],
        ),
    )
    evidence_section = next(s for s in report.sections if s.section_type == OperationalReportSectionType.EVIDENCE_SUMMARY)
    joined = "\n".join(evidence_section.content)
    assert "source_filename=src.pdf" in joined
    assert "page_start=4" in joined
    assert "section_title=sec" in joined


def test_report_preserves_insufficient_evidence_reason_with_query_context() -> None:
    bundle = EvidenceBundle(
        bundle_id="bins",
        query="query-text",
        support_status=EvidenceSupportStatus.INSUFFICIENT_EVIDENCE,
        confidence=0.0,
        no_evidence_reason="No sufficient evidence candidates passed retrieval safety checks.",
        claims=[],
        citations=[],
    )
    report = build_operational_report(
        _input(),
        Stage2AssessmentResult(
            status=Stage2Status.DEGRADED,
            assessment_id="a1",
            evidence_bundles=[bundle],
            agent_result=None,
            errors=[],
        ),
    )
    evidence_section = next(s for s in report.sections if s.section_type == OperationalReportSectionType.EVIDENCE_SUMMARY)
    joined = "\n".join(evidence_section.content)
    assert "query='query-text'" in joined
    assert "insufficient_evidence_reason:" in joined


def test_report_includes_detailed_agent_findings_and_evidence_refs() -> None:
    citation = EvidenceCitation(
        citation_id="cf1",
        source_id="sid",
        source_title="AC_107-2A",
        source_type=EvidenceSourceType.INTERNAL_DOC,
        origin=EvidenceOrigin.LOCAL_DOCUMENT,
        quote="retrieved quote",
    )
    bundle = EvidenceBundle(
        bundle_id="bfind",
        query="AC 107-2A preflight weather assessment small UAS wind conditions",
        claims=[],
        citations=[citation],
        support_status=EvidenceSupportStatus.SUPPORTED,
        confidence=0.9,
    )
    agent_result = AgentResult(
        status=Stage2Status.COMPLETED,
        recommendation=AgentRecommendation.CAUTION,
        confidence=0.6,
        findings=[
            {
                "finding_id": "f-weather",
                "finding_type": "evidence_backed",
                "severity": "medium",
                "summary": "Weather and wind conditions require preflight assessment before mission launch.",
                "requires_evidence": True,
                "evidence_references": [
                    {"claim_id": "claim_1", "citation_ids": ["cf1"], "summary": "claim summary"}
                ],
                "metadata": {"support_status": "supported"},
            }
        ],
        action_items=[
            {
                "action_id": "a-weather",
                "summary": "Review local weather and define mission wind limits.",
                "priority": "high",
                "evidence_references": [
                    {"claim_id": "claim_1", "citation_ids": ["cf1"], "summary": "claim summary"}
                ],
            }
        ],
        reasoning_trace=PublicReasoningTrace(),
        evidence_bundles=[bundle],
        errors=[],
    )

    report = build_operational_report(
        _input(),
        Stage2AssessmentResult(
            status=Stage2Status.COMPLETED,
            assessment_id="a1",
            evidence_bundles=[bundle],
            agent_result=agent_result,
            errors=[],
        ),
    )
    agent_section = next(s for s in report.sections if s.section_type == OperationalReportSectionType.AGENT_ASSESSMENT)
    action_section = next(s for s in report.sections if s.section_type == OperationalReportSectionType.OPERATOR_ACTIONS)
    joined_agent = "\n".join(agent_section.content)
    joined_actions = "\n".join(action_section.content)
    assert "finding[f-weather]" in joined_agent
    assert "support_status=supported" in joined_agent
    assert "evidence_ref claim_id=claim_1" in joined_agent
    assert "priority=high" in joined_actions
    assert "evidence_refs=cf1" in joined_actions


def test_report_renders_inspector_topic_and_related_metadata() -> None:
    agent_result = AgentResult(
        status=Stage2Status.COMPLETED,
        recommendation=AgentRecommendation.CAUTION,
        confidence=0.6,
        findings=[
            {
                "finding_id": "f-topic",
                "finding_type": "tool_check",
                "severity": "medium",
                "summary": "Airspace or restriction context requires authorization/proximity review.",
                "requires_evidence": False,
                "metadata": {
                    "topic": "airspace",
                    "support_status": "scenario_derived",
                    "related_feature_names": "airspace_altitude_agl_max_m",
                    "related_scenario_fields": "airspace_altitude_agl_max_m,airspace_restricted_zone",
                },
            }
        ],
        action_items=[],
        reasoning_trace=PublicReasoningTrace(),
        evidence_bundles=[],
        errors=[],
    )

    report = build_operational_report(
        _input(),
        Stage2AssessmentResult(
            status=Stage2Status.COMPLETED,
            assessment_id="a1",
            evidence_bundles=[],
            agent_result=agent_result,
            errors=[],
        ),
    )
    agent_section = next(s for s in report.sections if s.section_type == OperationalReportSectionType.AGENT_ASSESSMENT)
    joined = "\n".join(agent_section.content)
    assert "topic=airspace" in joined
    assert "related_feature_names=airspace_altitude_agl_max_m" in joined
    assert "related_scenario_fields=airspace_altitude_agl_max_m,airspace_restricted_zone" in joined

def test_report_includes_decision_engine_section() -> None:
    stage2_input = _input()
    stage2_result = Stage2AssessmentResult(
        status=Stage2Status.COMPLETED,
        assessment_id="a1",
        evidence_bundles=[],
        agent_result=_agent_result(),
        errors=[],
    )
    decision = evaluate_stage2_decision(stage2_input, stage2_result)
    report = build_operational_report(
        stage2_input,
        stage2_result.model_copy(update={"decision": decision}),
    )
    section = next(s for s in report.sections if s.section_type == OperationalReportSectionType.DECISION_ENGINE)
    joined = "\n".join(section.content)
    assert "final_decision:" in joined
    assert "decision_score:" in joined
    assert "stage[ml]" in joined
    assert "stage[llm]" in joined

def test_report_includes_llm_synthesis_section() -> None:
    synthesis = LLMAgentSynthesis(
        status=LLMSynthesisStatus.FALLBACK,
        executive_summary="Final decision is caution.",
        operational_interpretation="Weather should be reviewed.",
        decision_explanation="Decision Engine score requires caution.",
        key_risk_drivers=["weather", "ml_signal"],
        mitigation_narrative="Review weather before launch.",
        consistency_warnings=[{"warning_type": "llm_fallback", "message": "Fallback used."}],
        evidence_reference_ids=[],
        finding_ids=["f1"],
        action_item_ids=["a1"],
        limitation_ids=["llm_fallback"],
    )
    report = build_operational_report(
        _input(),
        Stage2AssessmentResult(
            status=Stage2Status.COMPLETED,
            assessment_id="a1",
            evidence_bundles=[],
            agent_result=_agent_result(),
            llm_synthesis=synthesis,
            errors=[],
        ),
    )
    section = next(s for s in report.sections if s.section_type == OperationalReportSectionType.LLM_SYNTHESIS)
    joined = "\n".join(section.content)
    assert "LLM-assisted synthesis based only on deterministic pipeline outputs." in joined
    assert "executive_summary: Final decision is caution." in joined
    assert "consistency_warning[llm_fallback]" in joined

