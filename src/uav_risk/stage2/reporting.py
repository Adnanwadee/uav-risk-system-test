from __future__ import annotations

import json
from uuid import uuid4

from uav_risk.stage2.contracts import (
    EvidenceBundle,
    EvidenceSupportStatus,
    OperationalReport,
    OperationalReportSection,
    OperationalReportSectionType,
    Stage2AssessmentInput,
    Stage2AssessmentResult,
)


_FORBIDDEN_REPORT_TOKENS = (
    "reasoning_chain",
    "chain_of_thought",
    "thought",
    "scratchpad",
    "internal_reasoning",
    "private_reasoning",
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


def _sanitize_report_text(value: object, *, max_chars: int = 360) -> str:
    """Return a compact public-safe string for report content."""

    if value is None:
        text = "not_available"
    elif isinstance(value, float):
        text = f"{value:.4f}"
    elif isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    else:
        text = str(value)

    text = " ".join(text.replace("\n", " ").split()).strip()

    lowered = text.lower()
    for token in _FORBIDDEN_REPORT_TOKENS:
        if token in lowered:
            text = lowered.replace(token, "[redacted_forbidden_token]")

    if len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text or "not_available"


def _safe_mapping(value: object) -> dict:
    """Return a dictionary when a report input mapping is unavailable or malformed."""

    return value if isinstance(value, dict) else {}


def _format_mapping_lines(prefix: str, mapping: object, *, max_items: int = 14) -> list[str]:
    """Format mapping values as deterministic report lines."""

    safe_mapping = _safe_mapping(mapping)
    lines: list[str] = []
    for key in sorted(safe_mapping)[:max_items]:
        lines.append(f"{prefix}[{key}]: {_sanitize_report_text(safe_mapping.get(key))}")
    if len(safe_mapping) > max_items:
        lines.append(f"{prefix}_truncated_count: {len(safe_mapping) - max_items}")
    return lines


def _metadata_value(metadata: object, key: str) -> object:
    return _safe_mapping(metadata).get(key)


def _append_metadata_lines(
    target: list[str],
    metadata: object,
    keys: tuple[str, ...],
    *,
    prefix: str = "metadata",
) -> None:
    safe_metadata = _safe_mapping(metadata)
    for key in keys:
        if key in safe_metadata:
            target.append(f"{prefix}[{key}]: {_sanitize_report_text(safe_metadata.get(key))}")


def _format_probability_lines(probabilities: dict[str, float]) -> list[str]:
    lines: list[str] = []
    for label, probability in sorted(probabilities.items(), key=lambda item: str(item[0])):
        try:
            lines.append(f"probability[{label}]: {float(probability):.4f}")
        except (TypeError, ValueError):
            lines.append(f"probability[{label}]: {_sanitize_report_text(probability)}")
    return lines


def _build_executive_summary(
    stage2_input: Stage2AssessmentInput,
    stage2_result: Stage2AssessmentResult,
    evidence_bundles: list[EvidenceBundle],
    citation_ids: list[str],
) -> list[str]:
    decision = stage2_result.decision
    agent_result = stage2_result.agent_result
    synthesis = stage2_result.llm_synthesis
    metadata = stage2_result.metadata if isinstance(stage2_result.metadata, dict) else {}

    lines = [
        f"assessment_id: {stage2_result.assessment_id or stage2_input.assessment_id}",
        f"user_id: {stage2_input.user_id}",
        f"profile_id: {stage2_input.profile_id}",
        f"stage2_status: {stage2_result.status.value}",
        f"mission_posture: {decision.final_decision.value if decision else 'decision_unavailable'}",
        f"decision_score: {decision.decision_score if decision else 'not_available'}",
        f"decision_confidence_level: {decision.confidence_level.value if decision else 'not_available'}",
        f"agent_recommendation: {agent_result.recommendation.value if agent_result else 'not_available'}",
        f"evidence_bundle_count: {len(evidence_bundles)}",
        f"citation_count: {len(citation_ids)}",
        f"rag_quality_is_proven: {_metadata_value(metadata, 'rag_quality_is_proven')}",
        f"scenario_evidence_complete: {_metadata_value(metadata, 'scenario_evidence_complete')}",
        f"reranker_used: {_metadata_value(metadata, 'reranker_used')}",
        f"llm_synthesis_status: {synthesis.status.value if synthesis else 'not_generated'}",
        f"llm_provider: {(synthesis.provider or 'not_configured') if synthesis else 'not_configured'}",
    ]

    if decision is not None:
        for reason in decision.decision_reasons[:6]:
            lines.append(f"executive_decision_reason: {reason}")
        for action in decision.required_actions[:6]:
            lines.append(f"executive_required_action: {action}")

    return lines


def _build_input_summary(stage2_input: Stage2AssessmentInput) -> list[str]:
    scenario_summary = _safe_mapping(stage2_input.scenario_summary)
    profile_context = _safe_mapping(stage2_input.profile_context)
    input_metadata = _safe_mapping(stage2_input.metadata)

    lines = [
        f"scenario_keys: {', '.join(sorted(scenario_summary.keys())) or 'none'}",
        f"profile_context_keys: {', '.join(sorted(profile_context.keys())) or 'none'}",
        f"operator_notes_present: {bool((stage2_input.operator_notes or '').strip())}",
    ]

    lines.extend(_format_mapping_lines("scenario", scenario_summary, max_items=20))
    lines.extend(_format_mapping_lines("profile", profile_context, max_items=20))
    lines.extend(_format_mapping_lines("input_metadata", input_metadata, max_items=14))
    return lines


def _build_ml_assessment(stage2_input: Stage2AssessmentInput) -> list[str]:
    return [
        f"predicted_class: {stage2_input.ml.predicted_class}",
        f"probabilities: {_sanitize_report_text(stage2_input.ml.probabilities, max_chars=700)}",
        *_format_probability_lines(stage2_input.ml.probabilities),
        f"raw_feature_count: {stage2_input.ml.raw_feature_count}",
        f"processed_feature_count: {stage2_input.ml.processed_feature_count}",
        f"model_metadata: {_sanitize_report_text(stage2_input.ml.metadata, max_chars=700)}",
        "interpretation_boundary: ML is a learned risk signal, not legal authority and not the final operational decision.",
    ]


def _build_shap_section(stage2_input: Stage2AssessmentInput) -> list[str]:
    lines = [
        "interpretation_boundary: SHAP explains model behavior and risk drivers; it is not causal proof or legal evidence."
    ]

    if not stage2_input.ml.shap_top_features:
        lines.append("No SHAP features were provided.")
        return lines

    for index, item in enumerate(stage2_input.ml.shap_top_features, start=1):
        lines.append(
            f"shap_driver[{index}]: feature={item.feature}, importance={item.importance}, "
            f"direction={item.direction}, value={_sanitize_report_text(item.value)}"
        )
    return lines


def _build_rag_evidence_section(
    stage2_result: Stage2AssessmentResult,
    evidence_bundles: list[EvidenceBundle],
    citation_ids: list[str],
) -> list[str]:
    metadata = stage2_result.metadata if isinstance(stage2_result.metadata, dict) else {}

    grounded_count = sum(
        1
        for bundle in evidence_bundles
        if bundle.support_status in {EvidenceSupportStatus.SUPPORTED, EvidenceSupportStatus.PARTIALLY_SUPPORTED}
    )
    insufficient_count = sum(
        1 for bundle in evidence_bundles if bundle.support_status == EvidenceSupportStatus.INSUFFICIENT_EVIDENCE
    )

    lines = [
        f"evidence_bundle_count: {len(evidence_bundles)}",
        f"citation_count: {len(citation_ids)}",
        f"grounded_evidence_count: {grounded_count}",
        f"insufficient_evidence_count: {insufficient_count}",
        "evidence_boundary: RAG citations are evidence references; LLM synthesis may explain them but cannot invent or replace them.",
    ]

    _append_metadata_lines(
        lines,
        metadata,
        (
            "retrieval_usable",
            "rag_quality_is_proven",
            "quality_is_proven",
            "scenario_evidence_complete",
            "scenario_evidence_status",
            "index_provenance_status",
            "provenance_status",
            "path_resolution_status",
            "corpus_coverage_status",
            "expected_source_count",
            "indexed_source_count",
            "missing_sources_count",
            "source_count",
            "chunk_count",
            "faiss_ntotal",
            "synthetic_bundle_count",
            "agent_requested_query_count",
        ),
        prefix="rag",
    )

    if "reranker_configured" in metadata:
        lines.append(
            "reranker_status: "
            f"configured={metadata.get('reranker_configured')} "
            f"available={metadata.get('reranker_available')} "
            f"used={metadata.get('reranker_used')} "
            f"reason={metadata.get('reranker_reason')}"
        )

    for bundle_index, bundle in enumerate(evidence_bundles, start=1):
        bundle_meta = bundle.metadata if isinstance(bundle.metadata, dict) else {}
        lines.append(
            f"bundle[{bundle_index}:{bundle.bundle_id}]: "
            f"query='{_sanitize_report_text(bundle.query, max_chars=240)}' "
            f"status={bundle.support_status.value} confidence={bundle.confidence} "
            f"claims={len(bundle.claims)} citations={len(bundle.citations)} "
            f"origin={bundle_meta.get('retrieval_origin')} "
            f"evidence_status={bundle_meta.get('evidence_status')} "
            f"source_intent={bundle_meta.get('source_intent')} "
            f"synthetic={bundle_meta.get('synthetic')}"
        )

        if bundle.no_evidence_reason:
            lines.append(f"insufficient_evidence_reason: {bundle.no_evidence_reason}")

        for claim in bundle.claims[:5]:
            lines.append(
                f"claim[{claim.claim_id}]: status={claim.support_status.value} "
                f"evidence_use={claim.evidence_use.value} "
                f"confidence={claim.confidence} "
                f"citations={len(claim.citations)} "
                f"claim={_sanitize_report_text(claim.claim, max_chars=320)}"
            )
            for limitation in claim.limitations:
                lines.append(f"claim_limitation[{claim.claim_id}]: {_sanitize_report_text(limitation, max_chars=260)}")
            for conflict in claim.conflicts:
                lines.append(f"claim_conflict[{claim.claim_id}]: {_sanitize_report_text(conflict, max_chars=260)}")
            if isinstance(claim.metadata, dict) and claim.metadata:
                lines.extend(_format_mapping_lines(f"claim[{claim.claim_id}].metadata", claim.metadata, max_items=8))

        for citation_index, citation in enumerate(bundle.citations[:5], start=1):
            citation_meta = citation.metadata if isinstance(citation.metadata, dict) else {}
            source_filename = citation_meta.get("source_filename") or citation.source_title
            page_start = citation_meta.get("page_start") if citation_meta else citation.page
            page_end = citation_meta.get("page_end") if citation_meta else None
            section_title = citation_meta.get("section_title") if citation_meta else citation.section
            rank = citation_meta.get("rank")
            final_score = citation_meta.get("final_score", citation_meta.get("score"))
            rerank_score = citation_meta.get("rerank_score")
            lines.append(
                f"citation[{bundle_index}.{citation_index}:{citation.citation_id}]: "
                f"source_id={citation.source_id}, source_filename={source_filename}, "
                f"page_start={page_start}, page_end={page_end}, section_title={section_title}, "
                f"rank={rank}, final_score={final_score}, rerank_score={rerank_score}"
            )

    return lines


def _build_decision_section(stage2_result: Stage2AssessmentResult) -> list[str]:
    decision = stage2_result.decision
    if decision is None:
        return ["Decision engine output is not available."]

    lines = [
        f"final_decision: {decision.final_decision.value}",
        f"decision_score: {decision.decision_score}",
        f"confidence_level: {decision.confidence_level.value}",
        "decision_authority: DecisionEngine owns final GO/CAUTION/NO-GO scoring.",
        "stage_weights: " + ", ".join(f"{stage}={weight}" for stage, weight in sorted(decision.stage_weights.items())),
    ]

    for contribution in decision.stage_contributions:
        lines.append(
            f"stage[{contribution.stage.value}] weight={contribution.weight} "
            f"contribution={contribution.contribution} signal={contribution.signal}: {contribution.summary}"
        )
        for reason in contribution.reasons:
            lines.append(f"stage_reason[{contribution.stage.value}]: {reason}")
        for limitation in contribution.limitations:
            lines.append(f"stage_limitation[{contribution.stage.value}]: {limitation}")

    for reason in decision.decision_reasons:
        lines.append(f"decision_reason: {reason}")
    for reason in decision.blocking_reasons:
        lines.append(f"blocking_reason: {reason}")
    for action in decision.required_actions:
        lines.append(f"required_action: {action}")
    for limitation in decision.limitations:
        lines.append(f"limitation: {limitation}")
    for ref in decision.evidence_refs:
        lines.append(f"decision_evidence_ref claim_id={ref.claim_id} citation_ids={','.join(ref.citation_ids)}")

    return lines


def _build_llm_section(stage2_result: Stage2AssessmentResult) -> list[str]:
    synthesis = stage2_result.llm_synthesis
    if synthesis is None:
        return [
            "LLM-assisted synthesis based only on deterministic pipeline outputs.",
            "role: required_post_decision_operational_report_synthesis",
            "scope: structured UAV operational report narrative from ML, SHAP, RAG, Agent, and DecisionEngine outputs only.",
            "authority: does_not_override_decision_engine_or_rag_evidence",
            "status: not_generated",
            "provider: not_configured",
            "model_name: not_configured",
            "executive_summary: ",
            "operational_interpretation: ",
            "decision_explanation: ",
            "mitigation_narrative: ",
        ]

    lines = [
        "LLM-assisted synthesis based only on deterministic pipeline outputs.",
        "role: required_post_decision_operational_report_synthesis",
        "scope: structured UAV operational report narrative from ML, SHAP, RAG, Agent, and DecisionEngine outputs only.",
        "authority: does_not_override_decision_engine_or_rag_evidence",
        f"status: {synthesis.status.value}",
        f"provider: {synthesis.provider or 'not_configured'}",
        f"model_name: {synthesis.model_name or 'not_configured'}",
        f"executive_summary: {synthesis.executive_summary}",
        f"operational_interpretation: {synthesis.operational_interpretation}",
        f"decision_explanation: {synthesis.decision_explanation}",
        f"mitigation_narrative: {synthesis.mitigation_narrative}",
    ]

    for driver in synthesis.key_risk_drivers:
        lines.append(f"key_risk_driver: {driver}")
    for warning in synthesis.consistency_warnings:
        lines.append(f"consistency_warning[{warning.warning_type}]: {warning.message}")
    if synthesis.evidence_reference_ids:
        lines.append("evidence_reference_ids: " + ",".join(synthesis.evidence_reference_ids))
    if synthesis.finding_ids:
        lines.append("finding_ids: " + ",".join(synthesis.finding_ids))
    if synthesis.action_item_ids:
        lines.append("action_item_ids: " + ",".join(synthesis.action_item_ids))
    if synthesis.limitation_ids:
        lines.append("limitation_ids: " + ",".join(synthesis.limitation_ids))
    if isinstance(synthesis.metadata, dict) and synthesis.metadata:
        lines.extend(_format_mapping_lines("llm_metadata", synthesis.metadata, max_items=10))

    return lines


def _build_agent_section(stage2_result: Stage2AssessmentResult) -> list[str]:
    agent_result = stage2_result.agent_result
    if agent_result is None:
        return ["Agent result is not available."]

    findings = agent_result.findings
    lines = [
        f"status: {agent_result.status.value}",
        f"recommendation: {agent_result.recommendation.value}",
        f"confidence: {agent_result.confidence}",
        f"findings_count: {len(findings)}",
        f"action_item_count: {len(agent_result.action_items)}",
        f"tool_trace_count: {len(agent_result.tool_trace)}",
    ]

    for finding in findings:
        finding_meta = finding.metadata if isinstance(finding.metadata, dict) else {}
        lines.append(
            f"finding[{finding.finding_id}] severity={finding.severity.value} "
            f"type={finding.finding_type.value} "
            f"support_status={finding_meta.get('support_status')} "
            f"topic={finding_meta.get('topic')}: "
            f"{_sanitize_report_text(finding.summary, max_chars=340)}"
        )
        for metadata_key in ("related_feature_names", "related_scenario_fields", "related_profile_fields"):
            metadata_value = finding_meta.get(metadata_key)
            if metadata_value:
                lines.append(
                    f"finding[{finding.finding_id}] {metadata_key}="
                    f"{_sanitize_report_text(metadata_value, max_chars=360)}"
                )

        for ref in finding.evidence_references:
            lines.append(
                f"evidence_ref claim_id={ref.claim_id} "
                f"citation_ids={','.join(ref.citation_ids)} "
                f"finding_id={finding.finding_id}"
            )

    if agent_result.working_memory is not None:
        wm = agent_result.working_memory
        lines.append("working_memory_summary: " + (wm.reasoning_summary or "not available"))
        lines.append("working_memory_coverage: " + ", ".join(f"{k}={v}" for k, v in sorted(wm.coverage_summary.items())))
        for signal in wm.input_signals[:10]:
            lines.append(
                f"signal[{signal.signal_id}] source={signal.source.value} topic={signal.topic} "
                f"priority={signal.priority:.2f} risk_relevance={signal.risk_relevance.value}: "
                f"{_sanitize_report_text(signal.value_summary, max_chars=260)}"
            )
        for item in wm.feature_assessments[:10]:
            lines.append(
                f"feature_assessment[{item.assessment_id}] feature={item.feature_name} topic={item.topic} "
                f"priority={item.priority:.2f} evidence_status={item.evidence_status}: "
                f"{_sanitize_report_text(item.conclusion, max_chars=260)}"
            )
        if wm.selected_rag_queries:
            lines.append("selected_rag_queries: " + " | ".join(wm.selected_rag_queries[:10]))
        if wm.skipped_rag_queries:
            lines.append("skipped_rag_queries: " + " | ".join(wm.skipped_rag_queries[:10]))

    return lines


def _build_system_work_trace(stage2_result: Stage2AssessmentResult) -> list[str]:
    agent_result = stage2_result.agent_result
    if agent_result is None:
        return ["No agent tool trace was provided."]

    lines = [f"agent_tool_trace_count: {len(agent_result.tool_trace)}"]
    for index, tool_call in enumerate(agent_result.tool_trace, start=1):
        lines.append(f"tool_step[{index}] tool={tool_call.tool_name.value} status={tool_call.status}: {tool_call.purpose}")
        lines.append(f"tool_step[{index}].input_summary: {tool_call.input_summary}")
        lines.append(f"tool_step[{index}].output_summary: {tool_call.output_summary}")
        if tool_call.related_query_ids:
            lines.append(f"tool_step[{index}].related_query_ids: {','.join(tool_call.related_query_ids)}")
        if tool_call.related_evidence_ids:
            lines.append(f"tool_step[{index}].related_evidence_ids: {','.join(tool_call.related_evidence_ids)}")
        if tool_call.related_finding_ids:
            lines.append(f"tool_step[{index}].related_finding_ids: {','.join(tool_call.related_finding_ids)}")
        if isinstance(tool_call.metadata, dict) and tool_call.metadata:
            lines.extend(_format_mapping_lines(f"tool_step[{index}].metadata", tool_call.metadata, max_items=8))

    return lines


def _build_required_actions(stage2_result: Stage2AssessmentResult) -> list[str]:
    decision = stage2_result.decision
    agent_result = stage2_result.agent_result
    lines: list[str] = []

    if decision is not None:
        for index, action in enumerate(decision.required_actions, start=1):
            lines.append(f"decision_required_action[{index}]: {action}")

    if agent_result is not None:
        for item in agent_result.action_items:
            evidence = ""
            if item.evidence_references:
                evidence = f" [evidence_refs={';'.join([','.join(ref.citation_ids) for ref in item.evidence_references])}]"
            lines.append(f"agent_action[{item.action_id}]: {item.summary} (priority={item.priority.value}){evidence}")

    return lines or ["No operator action items were provided."]


def _build_diagnostics(
    stage2_input: Stage2AssessmentInput,
    stage2_result: Stage2AssessmentResult,
    evidence_bundles: list[EvidenceBundle],
    citation_ids: list[str],
) -> list[str]:
    metadata = stage2_result.metadata if isinstance(stage2_result.metadata, dict) else {}
    synthesis = stage2_result.llm_synthesis

    lines = [
        f"assessment_id: {stage2_result.assessment_id or stage2_input.assessment_id}",
        f"status: {stage2_result.status.value}",
        f"error_count: {len(stage2_result.errors)}",
        f"evidence_bundle_count: {len(evidence_bundles)}",
        f"citation_count: {len(citation_ids)}",
        f"agent_available: {stage2_result.agent_result is not None}",
        f"decision_available: {stage2_result.decision is not None}",
        f"llm_synthesis_available: {synthesis is not None}",
    ]

    if synthesis is not None:
        lines.extend(
            [
                f"llm_status: {synthesis.status.value}",
                f"llm_provider: {synthesis.provider or 'not_configured'}",
                f"llm_model_name: {synthesis.model_name or 'not_configured'}",
                "llm_authority_boundary: post_decision_report_synthesis_only",
            ]
        )

    _append_metadata_lines(
        lines,
        metadata,
        (
            "retrieval_usable",
            "rag_quality_is_proven",
            "quality_is_proven",
            "scenario_evidence_complete",
            "scenario_evidence_status",
            "index_provenance_status",
            "provenance_status",
            "path_resolution_status",
            "corpus_coverage_status",
            "expected_source_count",
            "indexed_source_count",
            "missing_sources_count",
            "source_count",
            "chunk_count",
            "faiss_ntotal",
            "dense_mapping_count",
            "reranker_configured",
            "reranker_available",
            "reranker_used",
            "reranker_reason",
            "synthetic_bundle_count",
            "agent_requested_query_count",
            "retrieval_origins",
        ),
        prefix="diagnostic",
    )

    for err in stage2_result.errors:
        lines.append(f"error[{err.code}]: {err.message}")

    return lines


def build_operational_report(
    stage2_input: Stage2AssessmentInput,
    stage2_result: Stage2AssessmentResult,
) -> OperationalReport:
    evidence_bundles = list(stage2_result.evidence_bundles)
    citation_ids = _collect_citation_ids(evidence_bundles)

    sections: list[OperationalReportSection] = [
        OperationalReportSection(
            section_type=OperationalReportSectionType.METADATA,
            title="Executive Summary",
            content=_build_executive_summary(stage2_input, stage2_result, evidence_bundles, citation_ids),
            citation_ids=[],
            metadata={},
        ),
        OperationalReportSection(
            section_type=OperationalReportSectionType.PROFILE_SCENARIO_SUMMARY,
            title="Input Summary",
            content=_build_input_summary(stage2_input),
            citation_ids=[],
            metadata={},
        ),
        OperationalReportSection(
            section_type=OperationalReportSectionType.ML_SIGNAL,
            title="ML Assessment",
            content=_build_ml_assessment(stage2_input),
            citation_ids=[],
            metadata={},
        ),
        OperationalReportSection(
            section_type=OperationalReportSectionType.SHAP_EXPLANATION,
            title="SHAP Risk Drivers",
            content=_build_shap_section(stage2_input),
            citation_ids=[],
            metadata={},
        ),
        OperationalReportSection(
            section_type=OperationalReportSectionType.EVIDENCE_SUMMARY,
            title="RAG Evidence",
            content=_build_rag_evidence_section(stage2_result, evidence_bundles, citation_ids),
            citation_ids=citation_ids,
            metadata={},
        ),
        OperationalReportSection(
            section_type=OperationalReportSectionType.AGENT_ASSESSMENT,
            title="Agent Operational Analysis",
            content=_build_agent_section(stage2_result),
            citation_ids=[],
            metadata={},
        ),
        OperationalReportSection(
            section_type=OperationalReportSectionType.DECISION_ENGINE,
            title="DecisionEngine Final Decision",
            content=_build_decision_section(stage2_result),
            citation_ids=[],
            metadata={},
        ),
        OperationalReportSection(
            section_type=OperationalReportSectionType.LLM_SYNTHESIS,
            title="LLM Synthesis",
            content=_build_llm_section(stage2_result),
            citation_ids=[],
            metadata={},
        ),
        OperationalReportSection(
            section_type=OperationalReportSectionType.OPERATOR_ACTIONS,
            title="Required Actions",
            content=_build_required_actions(stage2_result),
            citation_ids=[],
            metadata={},
        ),
        OperationalReportSection(
            section_type=OperationalReportSectionType.AGENT_TOOL_TRACE,
            title="System Work Trace",
            content=_build_system_work_trace(stage2_result),
            citation_ids=[],
            metadata={},
        ),
        OperationalReportSection(
            section_type=OperationalReportSectionType.ERRORS,
            title="Diagnostics",
            content=_build_diagnostics(stage2_input, stage2_result, evidence_bundles, citation_ids),
            citation_ids=[],
            metadata={},
        ),
    ]

    section_order = {
        OperationalReportSectionType.METADATA: 1,
        OperationalReportSectionType.PROFILE_SCENARIO_SUMMARY: 2,
        OperationalReportSectionType.ML_SIGNAL: 3,
        OperationalReportSectionType.SHAP_EXPLANATION: 4,
        OperationalReportSectionType.EVIDENCE_SUMMARY: 5,
        OperationalReportSectionType.AGENT_ASSESSMENT: 6,
        OperationalReportSectionType.DECISION_ENGINE: 7,
        OperationalReportSectionType.LLM_SYNTHESIS: 8,
        OperationalReportSectionType.OPERATOR_ACTIONS: 9,
        OperationalReportSectionType.AGENT_TOOL_TRACE: 10,
        OperationalReportSectionType.ERRORS: 11,
    }
    sections = sorted(sections, key=lambda section: section_order.get(section.section_type, 99))

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
