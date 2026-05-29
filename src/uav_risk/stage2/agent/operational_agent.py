from __future__ import annotations

from collections import defaultdict
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
    AgentToolCall,
    AgentToolName,
    AgentWorkingMemory,
    AgentInputSignal,
    AgentFeatureAssessment,
    AgentRiskRelevance,
    AgentSignalSource,
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


def _risk_relevance_from_priority(priority: float) -> AgentRiskRelevance:
    if priority >= 0.85:
        return AgentRiskRelevance.CRITICAL
    if priority >= 0.65:
        return AgentRiskRelevance.HIGH
    if priority >= 0.4:
        return AgentRiskRelevance.MEDIUM
    return AgentRiskRelevance.LOW


def _topic_from_name(name: str) -> str:
    lower = name.lower()
    if any(t in lower for t in ("weather", "wind", "gust", "turbulence", "thermal")):
        return "weather"
    if any(t in lower for t in ("airspace", "no_fly", "restricted", "authorization", "altitude", "agl", "ceiling")):
        return "airspace"
    if any(t in lower for t in ("comms", "uplink", "downlink", "c2", "telemetry", "link")):
        return "c2"
    if any(t in lower for t in ("swarm", "multi_uas", "formation")):
        return "swarm"
    if any(t in lower for t in ("payload", "mass", "weight", "loading")):
        return "payload"
    if any(t in lower for t in ("battery", "reserve", "endurance", "energy", "flight_time")):
        return "energy"
    if any(t in lower for t in ("obstacle", "traffic", "landing", "population", "ground_risk")):
        return "ground_risk"
    if any(t in lower for t in ("vlos", "visual", "line_of_sight", "los")):
        return "vlos"
    return "general"


def _extract_input_signals(agent_input: AgentInput) -> list[AgentInputSignal]:
    signals: list[AgentInputSignal] = []

    ml_probs = agent_input.ml_probabilities
    if ml_probs:
        top_class = max(ml_probs, key=ml_probs.get)
        top_prob = float(ml_probs.get(top_class, 0.0))
        ml_priority = min(1.0, 0.35 + (0.45 if "high" in top_class.lower() else 0.2 if "medium" in top_class.lower() else 0.05) + 0.2 * top_prob)
        signals.append(
            AgentInputSignal(
                signal_id="sig_ml_prediction",
                source=AgentSignalSource.ML,
                name="ml_prediction",
                value_summary=f"predicted_class={top_class} top_probability={top_prob:.3f}",
                topic="ml_signal",
                priority=ml_priority,
                risk_relevance=_risk_relevance_from_priority(ml_priority),
                needs_rag_evidence=False,
                metadata={"predicted_class": top_class, "top_probability": top_prob},
            )
        )

    for idx, shap in enumerate(agent_input.shap_top_features[:8], start=1):
        feature = str(shap.get("feature", "")).strip()
        if not feature:
            continue
        importance = abs(float(shap.get("importance", 0.0) or 0.0))
        topic = _topic_from_name(feature)
        priority = min(1.0, 0.2 + min(0.6, importance) + (0.1 if topic in {"airspace", "c2", "swarm", "weather"} else 0.0))
        signals.append(
            AgentInputSignal(
                signal_id=f"sig_shap_{idx}",
                source=AgentSignalSource.SHAP,
                name=feature,
                value_summary=f"importance={importance:.3f}",
                topic=topic,
                priority=priority,
                risk_relevance=_risk_relevance_from_priority(priority),
                needs_rag_evidence=topic in {"weather", "airspace", "c2", "vlos", "ground_risk", "swarm"},
                related_shap_features=[feature],
                metadata={"importance": importance},
            )
        )

    for key, value in list(agent_input.scenario_summary.items())[:30]:
        k = str(key).strip()
        if not k:
            continue
        topic = _topic_from_name(k)
        base = 0.28 if topic == "general" else 0.5
        if isinstance(value, bool) and topic in {"c2", "swarm"} and value is False:
            base += 0.25
        priority = min(1.0, base)
        signals.append(
            AgentInputSignal(
                signal_id=f"sig_scenario_{k}",
                source=AgentSignalSource.SCENARIO,
                name=k,
                value_summary=f"value={value}",
                topic=topic,
                priority=priority,
                risk_relevance=_risk_relevance_from_priority(priority),
                needs_rag_evidence=topic in {"weather", "airspace", "c2", "payload", "ground_risk", "vlos"},
                related_scenario_fields=[k],
                metadata={},
            )
        )

    profile = agent_input.profile_context
    if profile is not None:
        for name, value in [
            ("max_payload_kg", profile.max_payload_kg),
            ("max_altitude_m", profile.max_altitude_m),
            ("hover_ceiling_m", profile.hover_ceiling_m),
            ("reserve_fraction", profile.reserve_fraction),
            ("swarm_capable", profile.swarm_capable),
            ("detect_and_avoid_available", profile.detect_and_avoid_available),
        ]:
            if value is None:
                continue
            topic = _topic_from_name(name)
            priority = 0.45 if topic != "general" else 0.3
            signals.append(
                AgentInputSignal(
                    signal_id=f"sig_profile_{name}",
                    source=AgentSignalSource.PROFILE,
                    name=name,
                    value_summary=f"value={value}",
                    topic=topic if topic != "general" else "profile_capability",
                    priority=priority,
                    risk_relevance=_risk_relevance_from_priority(priority),
                    needs_rag_evidence=False,
                    related_profile_fields=[name],
                    metadata={},
                )
            )

    note = (agent_input.operator_notes or "").strip()
    if note:
        topic = _topic_from_name(note)
        priority = 0.52 if topic != "general" else 0.35
        signals.append(
            AgentInputSignal(
                signal_id="sig_operator_notes",
                source=AgentSignalSource.OPERATOR_NOTES,
                name="operator_notes",
                value_summary=note[:180],
                topic=topic,
                priority=priority,
                risk_relevance=_risk_relevance_from_priority(priority),
                needs_rag_evidence=topic in {"weather", "airspace", "c2", "vlos", "ground_risk", "swarm"},
                metadata={"untrusted_context": True},
            )
        )

    signals.sort(key=lambda item: item.priority, reverse=True)
    return signals


def _build_working_memory(
    agent_input: AgentInput,
    plan_items: list[AgentRAGQueryPlan],
    evidence_bundles: list[EvidenceBundle],
    findings: list[AgentFinding],
    action_items: list[AgentActionItem],
    limitations: list[str],
) -> AgentWorkingMemory:
    signals = _extract_input_signals(agent_input)
    selected_rag_queries = [item.query_text for item in plan_items]

    selected_topics = {_topic_from_name(item.query_text) for item in plan_items}
    skipped_rag_queries: list[str] = []
    for sig in signals:
        if not sig.needs_rag_evidence:
            continue
        if sig.topic in selected_topics:
            continue
        skipped_rag_queries.append(f"{sig.topic}:{sig.name}")

    bundles_by_query_id: dict[str, list[EvidenceBundle]] = defaultdict(list)
    for bundle in evidence_bundles:
        qid = str((bundle.metadata or {}).get("query_id", "")).strip()
        if qid:
            bundles_by_query_id[qid].append(bundle)

    findings_by_query_id: dict[str, list[AgentFinding]] = defaultdict(list)
    for finding in findings:
        qid = str((finding.metadata or {}).get("query_id", "")).strip()
        if qid:
            findings_by_query_id[qid].append(finding)

    actions_by_query_id: dict[str, list[AgentActionItem]] = defaultdict(list)
    for action in action_items:
        qid = str((action.metadata or {}).get("query_id", "")).strip()
        if qid:
            actions_by_query_id[qid].append(action)

    plan_by_topic: dict[str, AgentRAGQueryPlan] = {}
    for plan in plan_items:
        topic = _topic_from_name(plan.query_text)
        plan_by_topic.setdefault(topic, plan)

    assessments: list[AgentFeatureAssessment] = []
    for idx, sig in enumerate(signals[:12], start=1):
        plan = plan_by_topic.get(sig.topic)
        query_id = plan.query_id if plan else ""
        linked_bundles = bundles_by_query_id.get(query_id, []) if query_id else []
        linked_findings = findings_by_query_id.get(query_id, []) if query_id else []
        linked_actions = actions_by_query_id.get(query_id, []) if query_id else []

        evidence_status = "local_operational_concern"
        if linked_bundles:
            statuses = {item.support_status.value for item in linked_bundles}
            if "insufficient_evidence" in statuses:
                evidence_status = "insufficient_evidence"
            elif "supported" in statuses or "partially_supported" in statuses:
                evidence_status = "supported_concern"
            else:
                evidence_status = sorted(statuses)[0]
        elif sig.needs_rag_evidence:
            evidence_status = "not_queried"

        conclusion = "monitoring_only"
        if sig.priority >= 0.7:
            conclusion = "supported concern" if evidence_status in {"supported_concern", "supported", "partially_supported"} else "local operational concern"
        elif evidence_status == "insufficient_evidence":
            conclusion = "insufficient evidence"
        elif sig.priority < 0.35:
            conclusion = "not risk-significant"

        assessments.append(
            AgentFeatureAssessment(
                assessment_id=f"fa_{idx}",
                signal_id=sig.signal_id,
                feature_name=sig.name,
                source=sig.source,
                topic=sig.topic,
                priority=sig.priority,
                risk_relevance=sig.risk_relevance,
                raw_value_summary=sig.value_summary,
                rag_query=plan.query_text if plan else None,
                evidence_status=evidence_status,
                evidence_bundle_ids=[item.bundle_id for item in linked_bundles],
                finding_ids=[item.finding_id for item in linked_findings],
                action_item_ids=[item.action_id for item in linked_actions],
                conclusion=conclusion,
                limitations=[lim for lim in limitations if sig.topic in lim.lower()][:2],
                metadata={
                    "needs_rag_evidence": sig.needs_rag_evidence,
                    "related_profile_fields": ",".join(sig.related_profile_fields),
                    "related_scenario_fields": ",".join(sig.related_scenario_fields),
                    "related_shap_features": ",".join(sig.related_shap_features),
                },
            )
        )

    coverage_summary = {
        "input_signal_count": len(signals),
        "feature_assessment_count": len(assessments),
        "selected_rag_query_count": len(selected_rag_queries),
        "skipped_rag_query_count": len(skipped_rag_queries),
    }
    reasoning_summary = (
        f"Analyzed {len(signals)} signals across ML/SHAP/scenario/profile/operator context; "
        f"built {len(assessments)} feature assessments and selected {len(selected_rag_queries)} RAG queries."
    )

    return AgentWorkingMemory(
        input_signals=signals,
        feature_assessments=assessments,
        selected_rag_queries=selected_rag_queries,
        skipped_rag_queries=skipped_rag_queries,
        reasoning_summary=reasoning_summary,
        coverage_summary=coverage_summary,
        limitations=limitations[:8],
        metadata={},
    )


def _dedupe_action_items(items: list[AgentActionItem]) -> list[AgentActionItem]:
    seen: set[str] = set()
    out: list[AgentActionItem] = []
    for item in items:
        key = " ".join(item.summary.lower().split())
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out




def _is_supported_bundle(bundle: EvidenceBundle) -> bool:
    return bundle.support_status in {EvidenceSupportStatus.SUPPORTED, EvidenceSupportStatus.PARTIALLY_SUPPORTED}


def _bundle_query_texts(bundles: list[EvidenceBundle]) -> set[str]:
    return {bundle.query.strip().lower() for bundle in bundles if bundle.query.strip()}


def _supported_topics_from_bundles(bundles: list[EvidenceBundle]) -> set[str]:
    topics: set[str] = set()
    for bundle in bundles:
        if not _is_supported_bundle(bundle):
            continue
        topics.add(_topic_from_bundle(bundle))
    return topics


def _build_agent_requested_query_plan(
    agent_input: AgentInput,
    existing_bundles: list[EvidenceBundle],
    *,
    max_queries: int,
) -> tuple[list[AgentRAGQueryPlan], list[str]]:
    candidate_plans = build_agent_query_plan(agent_input, max_queries=max_queries)
    supported_topics = _supported_topics_from_bundles(existing_bundles)
    existing_queries = _bundle_query_texts(existing_bundles)

    selected: list[AgentRAGQueryPlan] = []
    skipped: list[str] = []
    selected_topics: set[str] = set()

    for plan in candidate_plans:
        query_l = plan.query_text.strip().lower()
        topic = _topic_from_name(plan.query_text)

        if query_l in existing_queries:
            skipped.append(f"{plan.query_id}:already_retrieved")
            continue
        if topic in supported_topics:
            skipped.append(f"{plan.query_id}:topic_already_supported")
            continue
        if topic in selected_topics:
            skipped.append(f"{plan.query_id}:duplicate_topic")
            continue

        selected.append(plan)
        selected_topics.add(topic)

    return selected[:max_queries], skipped


async def _retrieve_evidence_with_origin(
    rag_adapter: Any,
    *,
    query_text: str,
    scenario_context: dict[str, Any],
    retrieval_origin: str,
) -> EvidenceBundle:
    try:
        return await rag_adapter.retrieve_evidence(
            query_text,
            scenario_context=scenario_context,
            retrieval_origin=retrieval_origin,
        )
    except TypeError:
        return await rag_adapter.retrieve_evidence(
            query_text,
            scenario_context=scenario_context,
        )


class OperationalAgentV2:
    def __init__(
        self,
        *,
        rag_adapter: Any | None = None,
        max_queries: int = 5,
        max_agent_queries: int = 3,
    ) -> None:
        if max_queries < 0:
            raise ValueError("max_queries must be >= 0")
        if max_agent_queries < 0:
            raise ValueError("max_agent_queries must be >= 0")
        self._rag_adapter = rag_adapter
        self._max_queries = max_queries
        self._max_agent_queries = max_agent_queries

    async def run(self, agent_input: AgentInput) -> AgentResult:
        evidence_bundles = list(agent_input.evidence_bundles)
        checks_performed = ["ingest_agent_input", "summarize_ml_signal", "evaluate_evidence"]
        limitations: list[str] = []
        errors: list[Stage2Error] = []
        plan_items: list[AgentRAGQueryPlan] = []
        skipped_plan_reasons: list[str] = []

        if self._rag_adapter is None and not evidence_bundles:
            evidence_bundles.append(
                make_insufficient_evidence_bundle(
                    query="operational_agent_evidence",
                    reason="RAG adapter is not configured.",
                )
            )
            limitations.append("No RAG adapter configured; evidence could not be expanded.")
        elif self._rag_adapter is not None:
            planned_count = min(self._max_queries, self._max_agent_queries)
            plan_items, skipped_plan_reasons = _build_agent_requested_query_plan(
                agent_input,
                evidence_bundles,
                max_queries=planned_count,
            )
            checks_performed.append("agent_evidence_gap_detection")
            checks_performed.append("agent_requested_rag_retrieval")

            if skipped_plan_reasons:
                limitations.append("Some agent-requested evidence queries were skipped due to existing coverage or dedupe constraints.")

            for plan in plan_items:
                try:
                    bundle = await _retrieve_evidence_with_origin(
                        self._rag_adapter,
                        query_text=plan.query_text,
                        scenario_context=agent_input.scenario_summary,
                        retrieval_origin="agent_requested",
                    )
                    enriched_bundle = _enrich_bundle_with_query_plan(bundle, plan)
                    enriched_meta = dict(enriched_bundle.metadata) if isinstance(enriched_bundle.metadata, dict) else {}
                    enriched_meta["retrieval_origin"] = "agent_requested"
                    enriched = enriched_bundle.model_copy(update={"metadata": enriched_meta})
                    evidence_bundles.append(enriched)
                except Exception:
                    limitations.append("RAG adapter query failed for at least one agent-requested query.")
                    errors.append(
                        Stage2Error(
                            code="rag_query_failed",
                            message="RAG query execution failed.",
                            details={"query": plan.query_text, "retrieval_origin": "agent_requested"},
                        )
                    )

            if skipped_plan_reasons:
                limitations.extend(skipped_plan_reasons[:6])

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

        action_items = _dedupe_action_items(evidence_actions + inspector_actions)

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

        working_memory = _build_working_memory(
            agent_input,
            plan_items,
            evidence_bundles,
            findings,
            action_items,
            limitations,
        )

        query_ids = [str((item.metadata or {}).get("query_id", "")) for item in plan_items]
        query_ids = [item for item in query_ids if item]
        evidence_ids = [bundle.bundle_id for bundle in evidence_bundles]
        finding_ids = [finding.finding_id for finding in findings]

        tool_trace = [
            AgentToolCall(
                tool_name=AgentToolName.SHAP_TOPIC_MAPPER,
                purpose="Derive UAV operational evidence topics from SHAP/scenario/operator context.",
                input_summary=f"shap_features={len(agent_input.shap_top_features)} scenario_keys={len(agent_input.scenario_summary)}",
                output_summary=f"planned_queries={len(plan_items)}",
                status="ok",
                related_query_ids=query_ids,
                related_evidence_ids=[],
                related_finding_ids=[],
                metadata={},
            ),
            AgentToolCall(
                tool_name=AgentToolName.RAG_RETRIEVAL,
                purpose="Retrieve source-grounded evidence bundles for agent-requested evidence gaps.",
                input_summary=f"agent_requested_query_count={len(plan_items)} skipped_query_count={len(skipped_plan_reasons)} adapter_configured={self._rag_adapter is not None}",
                output_summary=f"evidence_bundles={len(evidence_bundles)} selected_queries={len(plan_items)} skipped_queries={len(skipped_plan_reasons)}",
                status="ok" if not errors else "partial",
                related_query_ids=query_ids,
                related_evidence_ids=evidence_ids,
                related_finding_ids=[],
                metadata={
                    "retrieval_origin": "agent_requested",
                    "max_agent_queries": self._max_agent_queries,
                    "selected_queries": " | ".join([item.query_text for item in plan_items[:8]]),
                    "skipped_queries": " | ".join(skipped_plan_reasons[:8]),
                },
            ),
            AgentToolCall(
                tool_name=AgentToolName.SCENARIO_PROFILE_INSPECTOR,
                purpose="Inspect scenario/profile operational constraints and enrich topic-level findings.",
                input_summary=f"profile_context_present={agent_input.profile_context is not None}",
                output_summary=f"total_findings={len(findings)}",
                status="ok",
                related_query_ids=query_ids,
                related_evidence_ids=evidence_ids,
                related_finding_ids=finding_ids,
                metadata={},
            ),
            AgentToolCall(
                tool_name=AgentToolName.FEATURE_RISK_ASSESSOR,
                purpose="Prioritize mission signals and map them to feature-level operational assessments.",
                input_summary=f"input_signals={len(working_memory.input_signals)}",
                output_summary=f"feature_assessments={len(working_memory.feature_assessments)} selected_queries={len(working_memory.selected_rag_queries)}",
                status="ok",
                related_query_ids=query_ids,
                related_evidence_ids=evidence_ids,
                related_finding_ids=finding_ids,
                metadata={},
            ),
        ]

        return AgentResult(
            status=Stage2Status.COMPLETED,
            recommendation=recommendation,
            confidence=confidence,
            findings=findings,
            action_items=action_items,
            reasoning_trace=trace,
            tool_trace=tool_trace,
            working_memory=working_memory,
            evidence_bundles=evidence_bundles,
            errors=errors,
            metadata=dict(agent_input.metadata),
        )
