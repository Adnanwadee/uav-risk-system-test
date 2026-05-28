from __future__ import annotations

from typing import Any

from uav_risk.stage2.agent.operational_inspector import inspect_operational_context
from uav_risk.stage2.agent.query_planner import build_agent_rag_query_plan
from uav_risk.stage2.contracts import (
    AgentActionItem,
    AgentEvidenceReference,
    AgentFinding,
    AgentFindingSeverity,
    AgentFindingType,
    AgentInput,
    AgentRAGQueryPlan,
    AgentRecommendation,
    AgentResult,
    EvidenceBundle,
    EvidenceSupportStatus,
    PublicReasoningTrace,
    Stage2Error,
    Stage2Status,
    make_insufficient_evidence_bundle,
)


def build_agent_query_plan(agent_input: AgentInput, max_queries: int) -> list[AgentRAGQueryPlan]:
    if max_queries < 0:
        raise ValueError("max_queries must be >= 0")
    if max_queries == 0:
        return []
    return build_agent_rag_query_plan(agent_input, max_queries=max_queries)


def build_agent_evidence_queries(agent_input: AgentInput, max_queries: int) -> list[str]:
    return [item.query_text for item in build_agent_query_plan(agent_input, max_queries)]


def summarize_ml_signal(agent_input: AgentInput) -> AgentFinding:
    top_class = max(agent_input.ml_probabilities, key=agent_input.ml_probabilities.get) if agent_input.ml_probabilities else "unknown"
    top_prob = float(agent_input.ml_probabilities.get(top_class, 0.0)) if agent_input.ml_probabilities else 0.0
    severity = AgentFindingSeverity.LOW
    if "high" in top_class.lower():
        severity = AgentFindingSeverity.HIGH
    elif "medium" in top_class.lower():
        severity = AgentFindingSeverity.MEDIUM
    return AgentFinding(
        finding_id="ml_signal_summary",
        finding_type=AgentFindingType.ML_SIGNAL,
        severity=severity,
        summary=(
            f"The ML model highlighted '{top_class}' at probability {top_prob:.3f}. "
            "This is a model-indicated signal and requires evidence-grounded operational review."
        ),
        evidence_references=[],
        requires_evidence=False,
        metadata={"ml_prediction": agent_input.ml_prediction or "", "ml_probability": top_prob},
    )


def has_supported_evidence(evidence_bundles: list[EvidenceBundle]) -> bool:
    for bundle in evidence_bundles:
        if bundle.support_status in {EvidenceSupportStatus.SUPPORTED, EvidenceSupportStatus.PARTIALLY_SUPPORTED} and bundle.citations:
            return True
    return False


def _first_ref_from_bundle(bundle: EvidenceBundle, default_claim_id: str) -> AgentEvidenceReference | None:
    for claim in bundle.claims:
        if claim.citations:
            return AgentEvidenceReference(
                claim_id=claim.claim_id,
                citation_ids=[c.citation_id for c in claim.citations],
                summary=claim.claim,
            )
    if bundle.citations:
        return AgentEvidenceReference(
            claim_id=default_claim_id,
            citation_ids=[c.citation_id for c in bundle.citations],
            summary=f"Evidence linked to query '{bundle.query}'.",
        )
    return None


def _topic_from_bundle(bundle: EvidenceBundle) -> str:
    meta = bundle.metadata if isinstance(bundle.metadata, dict) else {}
    source_intent = str(meta.get("source_intent", "")).lower()
    query = bundle.query.lower()

    # Prefer explicit query semantics first; metadata source_intent can be coarse.
    if any(t in query for t in ("command and control", "c2", "link reliability", "uplink", "downlink", "comms")):
        return "c2"
    if any(t in query for t in ("weather", "wind", "gust")):
        return "weather"
    if any(t in query for t in ("airspace", "authorization", "restricted", "altitude", "agl")):
        return "airspace"
    if "visual line of sight" in query:
        return "vlos"
    if source_intent == "special_condition" or "special condition" in query:
        return "ml_review"
    if any(t in query for t in ("ground risk", "operational volume", "adjacent area", "containment")):
        return "sora_volume"

    if source_intent in {"ac107", "part107"}:
        return "airspace"
    if source_intent == "sora":
        return "sora_volume"
    return "general"


def _topic_finding_and_action(topic: str) -> tuple[str, str, AgentFindingSeverity]:
    if topic == "weather":
        return (
            "Weather and wind conditions require preflight assessment before mission launch.",
            "Review local weather, wind direction/speed, gusts, and define mission wind limits before flight.",
            AgentFindingSeverity.MEDIUM,
        )
    if topic == "airspace":
        return (
            "Controlled/restricted airspace context requires authorization or operational review.",
            "Verify airspace class, authorization requirements, and no-fly/restriction proximity before launch.",
            AgentFindingSeverity.HIGH,
        )
    if topic == "vlos":
        return (
            "Visual line-of-sight operating requirements should be reviewed for this mission context.",
            "Confirm remote pilot/visual observer responsibilities and maintain VLOS constraints.",
            AgentFindingSeverity.MEDIUM,
        )
    if topic == "c2":
        return (
            "Command-and-control link reliability is an operational concern for mission safety.",
            "Verify uplink/downlink reliability, contingency procedures, and lost-link behavior.",
            AgentFindingSeverity.HIGH,
        )
    if topic == "ml_review":
        return (
            "The ML risk signal warrants conservative operational review with evidence-backed mitigations.",
            "Review relevant operational limitations and mitigations before accepting mission risk.",
            AgentFindingSeverity.MEDIUM,
        )
    if topic == "sora_volume":
        return (
            "Operational volume and ground-risk context should be evaluated using SORA-style risk controls.",
            "Review operational volume, adjacent area, containment, and mitigation assumptions.",
            AgentFindingSeverity.MEDIUM,
        )
    return (
        "Operational evidence indicates additional UAV mission review is warranted.",
        "Perform targeted operator review for this topic and confirm mission assumptions before launch.",
        AgentFindingSeverity.MEDIUM,
    )


def _related_scenario_fields(agent_input: AgentInput, topic: str) -> list[str]:
    keys = list(agent_input.scenario_summary.keys())
    if topic == "weather":
        return [k for k in keys if any(t in k.lower() for t in ("weather", "wind", "gust", "turbulence"))]
    if topic == "airspace":
        return [k for k in keys if any(t in k.lower() for t in ("airspace", "no_fly", "restricted", "altitude", "agl", "ceiling"))]
    if topic == "vlos":
        return [k for k in keys if any(t in k.lower() for t in ("vlos", "visual", "line_of_sight", "los"))]
    if topic == "c2":
        return [k for k in keys if any(t in k.lower() for t in ("comms", "uplink", "downlink", "c2", "link", "telemetry"))]
    if topic == "sora_volume":
        return [k for k in keys if any(t in k.lower() for t in ("ground_risk", "population", "adjacent_area", "operational_volume", "traffic", "obstacle", "landing"))]
    return keys[:3]


def _related_shap_fields(agent_input: AgentInput, topic: str) -> list[str]:
    names = [str(item.get("feature", "")).strip() for item in agent_input.shap_top_features]
    names = [n for n in names if n]
    if topic == "weather":
        return [n for n in names if any(t in n.lower() for t in ("weather", "wind", "gust", "turbulence"))]
    if topic == "airspace":
        return [n for n in names if any(t in n.lower() for t in ("airspace", "altitude", "agl", "ceiling", "restricted", "no_fly"))]
    if topic == "vlos":
        return [n for n in names if any(t in n.lower() for t in ("vlos", "visual", "line_of_sight", "los"))]
    if topic == "c2":
        return [n for n in names if any(t in n.lower() for t in ("comms", "uplink", "downlink", "c2", "link", "telemetry"))]
    if topic == "ml_review":
        return names[:3]
    if topic == "sora_volume":
        return [n for n in names if any(t in n.lower() for t in ("ground_risk", "population", "adjacent_area", "operational_volume", "traffic", "obstacle", "landing"))]
    return names[:2]


def build_findings_from_evidence(agent_input: AgentInput, evidence_bundles: list[EvidenceBundle]) -> tuple[list[AgentFinding], list[AgentActionItem], list[str]]:
    findings: list[AgentFinding] = []
    action_items: list[AgentActionItem] = []
    limitations: list[str] = []

    for idx, bundle in enumerate(evidence_bundles, start=1):
        meta = bundle.metadata if isinstance(bundle.metadata, dict) else {}
        query_id = str(meta.get("query_id", f"query_{idx}"))
        source_intent = str(meta.get("source_intent", ""))
        derived_from = str(meta.get("derived_from", ""))
        ref = _first_ref_from_bundle(bundle, default_claim_id=f"claim_{idx}")

        if bundle.support_status in {EvidenceSupportStatus.SUPPORTED, EvidenceSupportStatus.PARTIALLY_SUPPORTED} and ref is not None:
            topic = _topic_from_bundle(bundle)
            finding_summary, action_summary, severity = _topic_finding_and_action(topic)
            related_shap = _related_shap_fields(agent_input, topic)
            related_scenario = _related_scenario_fields(agent_input, topic)

            findings.append(
                AgentFinding(
                    finding_id=f"evidence_finding_{idx}",
                    finding_type=AgentFindingType.EVIDENCE_BACKED,
                    severity=severity,
                    summary=finding_summary,
                    evidence_references=[ref],
                    requires_evidence=True,
                    metadata={
                        "bundle_id": bundle.bundle_id,
                        "query_id": query_id,
                        "query_purpose": str(meta.get("query_purpose", "")),
                        "source_intent": source_intent,
                        "derived_from": derived_from,
                        "related_feature_names": ",".join(related_shap),
                        "related_scenario_fields": ",".join(related_scenario),
                        "support_status": bundle.support_status.value,
                        "operator_note_influenced": derived_from == "operator_notes",
                    },
                )
            )
            action_items.append(
                AgentActionItem(
                    action_id=f"action_{idx}",
                    summary=action_summary,
                    priority=severity,
                    evidence_references=[ref],
                    metadata={
                        "related_finding_id": f"evidence_finding_{idx}",
                        "bundle_id": bundle.bundle_id,
                        "query_id": query_id,
                        "source_intent": source_intent,
                    },
                )
            )

        elif bundle.support_status == EvidenceSupportStatus.INSUFFICIENT_EVIDENCE:
            reason = bundle.no_evidence_reason or "Evidence was insufficient for this query."
            limitations.append(f"{bundle.query}: {reason}")
            findings.append(
                AgentFinding(
                    finding_id=f"evidence_gap_{idx}",
                    finding_type=AgentFindingType.OPERATIONAL_UNCERTAINTY,
                    severity=AgentFindingSeverity.HIGH,
                    summary=f"Unresolved operational topic: {reason}",
                    evidence_references=[],
                    requires_evidence=False,
                    metadata={
                        "bundle_id": bundle.bundle_id,
                        "query": bundle.query,
                        "query_id": query_id,
                        "source_intent": source_intent,
                        "derived_from": derived_from,
                        "support_status": bundle.support_status.value,
                    },
                )
            )
            action_items.append(
                AgentActionItem(
                    action_id=f"action_gap_{idx}",
                    summary="Treat this topic as unresolved and seek additional evidence/operator review.",
                    priority=AgentFindingSeverity.HIGH,
                    evidence_references=[],
                    metadata={
                        "related_finding_id": f"evidence_gap_{idx}",
                        "bundle_id": bundle.bundle_id,
                        "query_id": query_id,
                    },
                )
            )

    return findings, action_items, limitations


def _enrich_bundle_with_query_plan(bundle: EvidenceBundle, plan: AgentRAGQueryPlan) -> EvidenceBundle:
    merged = dict(bundle.metadata)
    merged.update(
        {
            "query_id": plan.query_id,
            "query_purpose": plan.query_purpose,
            "source_intent": plan.source_intent.value,
            "derived_from": plan.derived_from.value,
            "related_feature_names": ",".join(plan.related_feature_names),
            "query_priority": plan.priority,
        }
    )
    return bundle.model_copy(update={"metadata": merged})


class OperationalAgentV2:
    def __init__(
        self,
        *,
        rag_adapter: Any | None = None,
        max_queries: int = 5,
    ) -> None:
        if max_queries < 0:
            raise ValueError("max_queries must be >= 0")
        self._rag_adapter = rag_adapter
        self._max_queries = max_queries

    async def run(self, agent_input: AgentInput) -> AgentResult:
        evidence_bundles = list(agent_input.evidence_bundles)
        checks_performed = ["ingest_agent_input", "summarize_ml_signal", "evaluate_evidence"]
        limitations: list[str] = []
        errors: list[Stage2Error] = []

        if self._rag_adapter is None and not evidence_bundles:
            evidence_bundles.append(
                make_insufficient_evidence_bundle(
                    query="operational_agent_evidence",
                    reason="RAG adapter is not configured.",
                )
            )
            limitations.append("No RAG adapter configured; evidence could not be expanded.")
        elif self._rag_adapter is not None:
            plan_items = build_agent_query_plan(agent_input, self._max_queries)
            checks_performed.append("query_rag_adapter")
            for plan in plan_items:
                try:
                    bundle = await self._rag_adapter.retrieve_evidence(
                        plan.query_text,
                        scenario_context=agent_input.scenario_summary,
                    )
                    evidence_bundles.append(_enrich_bundle_with_query_plan(bundle, plan))
                except Exception:
                    limitations.append("RAG adapter query failed for at least one query.")
                    errors.append(
                        Stage2Error(
                            code="rag_query_failed",
                            message="RAG query execution failed.",
                            details={"query": plan.query_text},
                        )
                    )

        findings: list[AgentFinding] = [summarize_ml_signal(agent_input)]
        evidence_findings, evidence_actions, evidence_limitations = build_findings_from_evidence(agent_input, evidence_bundles)
        findings.extend(evidence_findings)
        limitations.extend(evidence_limitations)

        inspector_findings, inspector_actions, inspector_limitations = inspect_operational_context(
            agent_input,
            findings,
        )
        findings.extend(inspector_findings)
        limitations.extend(inspector_limitations)

        if any(bundle.support_status == EvidenceSupportStatus.INSUFFICIENT_EVIDENCE for bundle in evidence_bundles):
            findings.append(
                AgentFinding(
                    finding_id="evidence_limitation",
                    finding_type=AgentFindingType.LIMITATION,
                    severity=AgentFindingSeverity.HIGH,
                    summary="Operational recommendation is constrained by insufficient evidence.",
                    evidence_references=[],
                    requires_evidence=False,
                    metadata={},
                )
            )

        supported = has_supported_evidence(evidence_bundles)
        top_class = max(agent_input.ml_probabilities, key=agent_input.ml_probabilities.get) if agent_input.ml_probabilities else ""
        top_class_l = top_class.lower()
        has_uncertainty = any(
            finding.finding_type in {AgentFindingType.OPERATIONAL_UNCERTAINTY, AgentFindingType.LIMITATION}
            for finding in findings
        )

        if not supported:
            recommendation = AgentRecommendation.INSUFFICIENT_EVIDENCE
            confidence = 0.2
        elif "high" in top_class_l or "medium" in top_class_l:
            recommendation = AgentRecommendation.CAUTION
            confidence = 0.55
        elif "low" in top_class_l and not has_uncertainty:
            recommendation = AgentRecommendation.GO
            confidence = 0.7
        else:
            recommendation = AgentRecommendation.CAUTION
            confidence = 0.5

        if has_uncertainty and recommendation == AgentRecommendation.GO:
            recommendation = AgentRecommendation.CAUTION
            confidence = min(confidence, 0.6)

        refs: list[AgentEvidenceReference] = []
        for finding in findings:
            refs.extend(finding.evidence_references)

        trace = PublicReasoningTrace(
            observations=[
                f"Top ML class: {top_class or 'unknown'}",
                f"Evidence bundle count: {len(evidence_bundles)}",
            ],
            checks_performed=checks_performed,
            evidence_consulted=refs,
            conflicts=[],
            limitations=limitations,
        )

        return AgentResult(
            status=Stage2Status.COMPLETED,
            recommendation=recommendation,
            confidence=confidence,
            findings=findings,
            action_items=evidence_actions + inspector_actions,
            reasoning_trace=trace,
            evidence_bundles=evidence_bundles,
            errors=errors,
            metadata=dict(agent_input.metadata),
        )
