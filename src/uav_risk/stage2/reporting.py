from __future__ import annotations

from uuid import uuid4

from uav_risk.stage2.contracts import (
    AgentFindingType,
    EvidenceBundle,
    EvidenceSupportStatus,
    OperationalReport,
    OperationalReportSection,
    OperationalReportSectionType,
    Stage2AssessmentInput,
    Stage2AssessmentResult,
)


def _collect_citation_ids(evidence_bundles: list[EvidenceBundle]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for bundle in evidence_bundles:
        for citation in bundle.citations:
            cid = citation.citation_id
            if cid in seen:
                continue
            seen.add(cid)
            ordered.append(cid)
    return ordered


def build_operational_report(
    stage2_input: Stage2AssessmentInput,
    stage2_result: Stage2AssessmentResult,
) -> OperationalReport:
    evidence_bundles = list(stage2_result.evidence_bundles)
    citation_ids = _collect_citation_ids(evidence_bundles)

    sections: list[OperationalReportSection] = []
    sections.append(
        OperationalReportSection(
            section_type=OperationalReportSectionType.METADATA,
            title="Assessment Metadata",
            content=[
                f"user_id: {stage2_input.user_id}",
                f"profile_id: {stage2_input.profile_id}",
                f"status: {stage2_result.status.value}",
            ],
            citation_ids=[],
            metadata={},
        )
    )
    sections.append(
        OperationalReportSection(
            section_type=OperationalReportSectionType.PROFILE_SCENARIO_SUMMARY,
            title="Profile And Scenario Summary",
            content=[
                f"scenario_keys: {', '.join(sorted(stage2_input.scenario_summary.keys())) or 'none'}",
                f"operator_notes_present: {bool((stage2_input.operator_notes or '').strip())}",
            ],
            citation_ids=[],
            metadata={},
        )
    )
    sections.append(
        OperationalReportSection(
            section_type=OperationalReportSectionType.ML_SIGNAL,
            title="ML Signal",
            content=[
                f"predicted_class: {stage2_input.ml.predicted_class}",
                f"probabilities: {stage2_input.ml.probabilities}",
            ],
            citation_ids=[],
            metadata={},
        )
    )
    sections.append(
        OperationalReportSection(
            section_type=OperationalReportSectionType.SHAP_EXPLANATION,
            title="SHAP Explanation",
            content=[
                f"{item.feature}: importance={item.importance}, direction={item.direction}, value={item.value}"
                for item in stage2_input.ml.shap_top_features
            ]
            or ["No SHAP features were provided."],
            citation_ids=[],
            metadata={},
        )
    )

    evidence_content: list[str] = []
    evidence_content.append(
        f"evidence_bundle_count: {len(evidence_bundles)}; citations: {len(citation_ids)}"
    )

    for bundle in evidence_bundles:
        evidence_content.append(
            f"bundle[{bundle.bundle_id}] query='{bundle.query}' status={bundle.support_status.value}"
        )
        if bundle.support_status == EvidenceSupportStatus.INSUFFICIENT_EVIDENCE and (bundle.no_evidence_reason or "").strip():
            evidence_content.append(f"insufficient_evidence_reason: {bundle.no_evidence_reason}")

        for citation in bundle.citations[:3]:
            source_filename = citation.metadata.get("source_filename") if isinstance(citation.metadata, dict) else None
            page_start = citation.metadata.get("page_start") if isinstance(citation.metadata, dict) else citation.page
            page_end = citation.metadata.get("page_end") if isinstance(citation.metadata, dict) else None
            section_title = citation.metadata.get("section_title") if isinstance(citation.metadata, dict) else citation.section
            evidence_content.append(
                "citation "
                f"{citation.citation_id}: source_id={citation.source_id}, "
                f"source_filename={source_filename or citation.source_title}, "
                f"page_start={page_start}, page_end={page_end}, section_title={section_title}"
            )
    sections.append(
        OperationalReportSection(
            section_type=OperationalReportSectionType.EVIDENCE_SUMMARY,
            title="Evidence Summary",
            content=evidence_content,
            citation_ids=citation_ids,
            metadata={},
        )
    )

    if stage2_result.decision is not None:
        decision = stage2_result.decision
        decision_content = [
            f"final_decision: {decision.final_decision.value}",
            f"decision_score: {decision.decision_score}",
            f"confidence_level: {decision.confidence_level.value}",
            "stage_weights: " + ", ".join(
                f"{stage}={weight}" for stage, weight in sorted(decision.stage_weights.items())
            ),
        ]
        for contribution in decision.stage_contributions:
            decision_content.append(
                f"stage[{contribution.stage.value}] weight={contribution.weight} "
                f"contribution={contribution.contribution} signal={contribution.signal}: {contribution.summary}"
            )
            for reason in contribution.reasons:
                decision_content.append(f"stage_reason[{contribution.stage.value}]: {reason}")
            for limitation in contribution.limitations:
                decision_content.append(f"stage_limitation[{contribution.stage.value}]: {limitation}")
        for reason in decision.decision_reasons:
            decision_content.append(f"decision_reason: {reason}")
        for reason in decision.blocking_reasons:
            decision_content.append(f"blocking_reason: {reason}")
        for action in decision.required_actions:
            decision_content.append(f"required_action: {action}")
        for limitation in decision.limitations:
            decision_content.append(f"limitation: {limitation}")
        for ref in decision.evidence_refs:
            decision_content.append(
                f"decision_evidence_ref claim_id={ref.claim_id} citation_ids={','.join(ref.citation_ids)}"
            )

        sections.append(
            OperationalReportSection(
                section_type=OperationalReportSectionType.DECISION_ENGINE,
                title="Decision Engine",
                content=decision_content,
                citation_ids=[],
                metadata={},
            )
        )

    if stage2_result.llm_synthesis is not None:
        synthesis = stage2_result.llm_synthesis
        synthesis_content = [
            "LLM-assisted synthesis based only on deterministic pipeline outputs.",
            f"status: {synthesis.status.value}",
            f"provider: {synthesis.provider or 'not_configured'}",
            f"model_name: {synthesis.model_name or 'not_configured'}",
            f"executive_summary: {synthesis.executive_summary}",
            f"operational_interpretation: {synthesis.operational_interpretation}",
            f"decision_explanation: {synthesis.decision_explanation}",
            f"mitigation_narrative: {synthesis.mitigation_narrative}",
        ]
        for driver in synthesis.key_risk_drivers:
            synthesis_content.append(f"key_risk_driver: {driver}")
        for warning in synthesis.consistency_warnings:
            synthesis_content.append(f"consistency_warning[{warning.warning_type}]: {warning.message}")
        if synthesis.evidence_reference_ids:
            synthesis_content.append(
                "evidence_reference_ids: " + ",".join(synthesis.evidence_reference_ids)
            )
        if synthesis.finding_ids:
            synthesis_content.append("finding_ids: " + ",".join(synthesis.finding_ids))
        if synthesis.action_item_ids:
            synthesis_content.append("action_item_ids: " + ",".join(synthesis.action_item_ids))

        sections.append(
            OperationalReportSection(
                section_type=OperationalReportSectionType.LLM_SYNTHESIS,
                title="LLM-Assisted Synthesis",
                content=synthesis_content,
                citation_ids=[],
                metadata={},
            )
        )

    if stage2_result.agent_result is not None:
        findings = stage2_result.agent_result.findings
        assessment_content = [
            f"recommendation: {stage2_result.agent_result.recommendation.value}",
            f"confidence: {stage2_result.agent_result.confidence}",
            f"findings_count: {len(findings)}",
        ]
        for finding in findings:
            support_status = ""
            if isinstance(finding.metadata, dict) and finding.metadata.get("support_status"):
                support_status = f" support_status={finding.metadata.get('support_status')}"
            topic = ""
            related_features = ""
            related_scenario = ""
            related_profile = ""
            if isinstance(finding.metadata, dict):
                if finding.metadata.get("topic"):
                    topic = f" topic={finding.metadata.get('topic')}"
                if finding.metadata.get("related_feature_names"):
                    related_features = f" related_feature_names={finding.metadata.get('related_feature_names')}"
                if finding.metadata.get("related_scenario_fields"):
                    related_scenario = f" related_scenario_fields={finding.metadata.get('related_scenario_fields')}"
                if finding.metadata.get("related_profile_fields"):
                    related_profile = f" related_profile_fields={finding.metadata.get('related_profile_fields')}"
            assessment_content.append(
                f"finding[{finding.finding_id}] severity={finding.severity.value} type={finding.finding_type.value}{support_status}{topic}: {finding.summary}{related_features}{related_scenario}{related_profile}"
            )
            if finding.evidence_references:
                for ref in finding.evidence_references:
                    assessment_content.append(
                        f"evidence_ref claim_id={ref.claim_id} citation_ids={','.join(ref.citation_ids)}"
                    )

        sections.append(
            OperationalReportSection(
                section_type=OperationalReportSectionType.AGENT_ASSESSMENT,
                title="Agent Assessment",
                content=assessment_content,
                citation_ids=[],
                metadata={},
            )
        )

        limitation_lines = [
            finding.summary
            for finding in findings
            if finding.finding_type
            in {AgentFindingType.LIMITATION, AgentFindingType.OPERATIONAL_UNCERTAINTY}
        ]
        sections.append(
            OperationalReportSection(
                section_type=OperationalReportSectionType.LIMITATIONS,
                title="Limitations",
                content=limitation_lines or ["No explicit limitations were provided."],
                citation_ids=[],
                metadata={},
            )
        )

        action_lines = [
            f"{item.summary} (priority={item.priority.value})"
            + (
                f" [evidence_refs={';'.join([','.join(ref.citation_ids) for ref in item.evidence_references])}]"
                if item.evidence_references else ""
            )
            for item in stage2_result.agent_result.action_items
        ]
        sections.append(
            OperationalReportSection(
                section_type=OperationalReportSectionType.OPERATOR_ACTIONS,
                title="Operator Actions",
                content=action_lines or ["No operator action items were provided."],
                citation_ids=[],
                metadata={},
            )
        )
    else:
        sections.append(
            OperationalReportSection(
                section_type=OperationalReportSectionType.LIMITATIONS,
                title="Limitations",
                content=["Agent result is not available."],
                citation_ids=[],
                metadata={},
            )
        )

    if stage2_result.errors:
        sections.append(
            OperationalReportSection(
                section_type=OperationalReportSectionType.ERRORS,
                title="Errors",
                content=[f"{err.code}: {err.message}" for err in stage2_result.errors],
                citation_ids=[],
                metadata={},
            )
        )

    return OperationalReport(
        report_id=f"report_{uuid4().hex}",
        assessment_id=stage2_result.assessment_id or stage2_input.assessment_id,
        status=stage2_result.status,
        sections=sections,
        evidence_bundles=evidence_bundles,
        agent_result=stage2_result.agent_result,
        errors=stage2_result.errors,
        metadata=dict(stage2_result.metadata),
    )


def render_markdown_report(report: OperationalReport) -> str:
    lines: list[str] = []
    lines.append(f"# Operational Report ({report.status.value})")
    if report.assessment_id:
        lines.append(f"Assessment ID: `{report.assessment_id}`")
    lines.append("")

    for section in report.sections:
        lines.append(f"## {section.title}")
        for item in section.content:
            lines.append(f"- {item}")
        if section.citation_ids:
            lines.append(f"Citations: {', '.join(section.citation_ids)}")
        lines.append("")

    if report.errors:
        lines.append("## Error Summary")
        for err in report.errors:
            lines.append(f"- `{err.code}`: {err.message}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"
