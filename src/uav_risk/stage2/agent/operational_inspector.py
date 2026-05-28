from __future__ import annotations

from typing import Any

from uav_risk.stage2.contracts import (
    AgentActionItem,
    AgentEvidenceReference,
    AgentFinding,
    AgentFindingSeverity,
    AgentFindingType,
    AgentInput,
)


TOPIC_ORDER = [
    "weather",
    "airspace",
    "altitude",
    "c2",
    "payload",
    "swarm",
    "ground_risk",
    "faults",
    "energy",
    "vlos",
]


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


def _topic_for_text(name: str) -> set[str]:
    lower = name.lower()
    topics: set[str] = set()
    if _contains_any(lower, ("weather", "wind", "gust", "turbulence", "thermal")):
        topics.add("weather")
    if _contains_any(lower, ("airspace", "no_fly", "nofly", "restricted", "geofence", "authorization")):
        topics.add("airspace")
    if _contains_any(lower, ("altitude", "agl", "ceiling")):
        topics.add("altitude")
    if _contains_any(lower, ("comms", "uplink", "downlink", "c2", "link", "telemetry", "lost_link")):
        topics.add("c2")
    if _contains_any(lower, ("payload", "mass", "weight", "loading")):
        topics.add("payload")
    if _contains_any(lower, ("swarm", "multi_uas", "formation")):
        topics.add("swarm")
    if _contains_any(lower, ("obstacle", "traffic", "landing", "landing_site", "population", "ground_risk", "adjacent_area", "operational_volume")):
        topics.add("ground_risk")
    if _contains_any(lower, ("fault", "failure", "degraded", "emergency")):
        topics.add("faults")
    if _contains_any(lower, ("battery", "reserve", "endurance", "fuel", "energy")):
        topics.add("energy")
    if _contains_any(lower, ("vlos", "visual_line_of_sight", "line_of_sight", "los")):
        topics.add("vlos")
    return topics


def _topic_summary_action(topic: str) -> tuple[str, str, AgentFindingSeverity]:
    if topic == "weather":
        return (
            "Weather/wind is an operational concern requiring preflight assessment.",
            "Review weather sources, wind/gust limits, and mission abort criteria.",
            AgentFindingSeverity.MEDIUM,
        )
    if topic == "airspace":
        return (
            "Airspace or restriction context requires authorization/proximity review.",
            "Verify airspace class, authorizations, and restricted/no-fly proximity.",
            AgentFindingSeverity.HIGH,
        )
    if topic == "altitude":
        return (
            "Altitude context should be reviewed against operating limits and aircraft performance margin.",
            "Check mission altitude, route profile, and relevant operating constraints.",
            AgentFindingSeverity.MEDIUM,
        )
    if topic == "c2":
        return (
            "Command-and-control reliability is an operational concern.",
            "Verify C2 link reliability and lost-link contingency behavior.",
            AgentFindingSeverity.HIGH,
        )
    if topic == "payload":
        return (
            "Payload/loading should be reviewed for aircraft performance and mission margin.",
            "Verify payload mass, balance, and expected performance.",
            AgentFindingSeverity.MEDIUM,
        )
    if topic == "swarm":
        return (
            "Multi-UAS/swarm operation increases operational complexity.",
            "Verify coordination, role assignments, separation, and contingency handling.",
            AgentFindingSeverity.HIGH,
        )
    if topic == "ground_risk":
        return (
            "Ground/traffic/landing environment requires operational risk review.",
            "Review landing options, obstacles, traffic, population exposure, and contingency sites.",
            AgentFindingSeverity.MEDIUM,
        )
    if topic == "faults":
        return (
            "Fault/degraded-state indicators require contingency planning.",
            "Verify emergency procedures and degraded-mode handling.",
            AgentFindingSeverity.HIGH,
        )
    if topic == "energy":
        return (
            "Energy/reserve margin should be reviewed for mission duration and contingencies.",
            "Verify reserve assumptions, endurance margins, and contingency energy budget.",
            AgentFindingSeverity.MEDIUM,
        )
    if topic == "vlos":
        return (
            "Visual line-of-sight context should be reviewed for mission execution.",
            "Confirm VLOS responsibilities and any observer requirements.",
            AgentFindingSeverity.MEDIUM,
        )
    return (
        "Operational context requires additional review.",
        "Perform targeted operational review for this topic.",
        AgentFindingSeverity.LOW,
    )


def _severity_to_priority(sev: AgentFindingSeverity) -> AgentFindingSeverity:
    return sev


def _extract_evidence_topics(findings: list[AgentFinding]) -> dict[str, int]:
    found: dict[str, int] = {}
    for idx, finding in enumerate(findings):
        if finding.finding_type != AgentFindingType.EVIDENCE_BACKED:
            continue
        meta = finding.metadata if isinstance(finding.metadata, dict) else {}
        topic = str(meta.get("topic", "")).strip().lower()
        if topic:
            found[topic] = idx
            continue
        summary = finding.summary.lower()
        for t in TOPIC_ORDER:
            if t in summary:
                found[t] = idx
                break
    return found


def _collect_topic_fields(keys: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key in keys:
        topics = _topic_for_text(key)
        for topic in topics:
            out.setdefault(topic, []).append(key)
    return out


def _profile_context_fields(agent_input: AgentInput) -> dict[str, Any]:
    # Profile context is partial in current pipeline: we only inspect agent metadata if present.
    meta = agent_input.metadata if isinstance(agent_input.metadata, dict) else {}
    return {
        str(k): v
        for k, v in meta.items()
        if any(tok in str(k).lower() for tok in (
            "uav_", "profile", "payload", "mass", "battery", "reserve", "swarm", "runway", "ceiling", "sensor", "gnss", "lidar", "radar", "camera"
        ))
    }


def inspect_operational_context(
    agent_input: AgentInput,
    existing_findings: list[AgentFinding],
) -> tuple[list[AgentFinding], list[AgentActionItem], list[str]]:
    new_findings: list[AgentFinding] = []
    action_items: list[AgentActionItem] = []
    limitations: list[str] = []

    evidence_topic_map = _extract_evidence_topics(existing_findings)

    scenario_keys = [str(k) for k in agent_input.scenario_summary.keys()]
    shap_names = [str(item.get("feature", "")).strip() for item in agent_input.shap_top_features if str(item.get("feature", "")).strip()]

    scenario_topics = _collect_topic_fields(scenario_keys)
    shap_topics = _collect_topic_fields(shap_names)

    # Enrich existing evidence-backed findings with scenario/SHAP context to avoid duplication.
    for topic, idx in evidence_topic_map.items():
        if topic not in scenario_topics and topic not in shap_topics:
            continue
        finding = existing_findings[idx]
        meta = dict(finding.metadata)
        related_scenario = scenario_topics.get(topic, [])
        related_shap = shap_topics.get(topic, [])
        if related_scenario:
            meta["related_scenario_fields"] = ",".join(related_scenario)
        if related_shap:
            meta["related_feature_names"] = ",".join(related_shap)
        meta["topic"] = topic
        existing_findings[idx] = finding.model_copy(update={"metadata": meta})

    # Scenario-driven topics that are not yet evidence-backed
    emitted_topics: set[str] = set()
    for topic in TOPIC_ORDER:
        if topic in evidence_topic_map:
            emitted_topics.add(topic)
            continue
        if topic not in scenario_topics:
            continue
        summary, action, sev = _topic_summary_action(topic)
        related_scenario = scenario_topics.get(topic, [])
        related_shap = shap_topics.get(topic, [])
        fid = f"inspector_{topic}"
        new_findings.append(
            AgentFinding(
                finding_id=fid,
                finding_type=AgentFindingType.TOOL_CHECK,
                severity=sev,
                summary=summary,
                evidence_references=[],
                requires_evidence=False,
                metadata={
                    "topic": topic,
                    "support_status": "scenario_derived",
                    "related_scenario_fields": ",".join(related_scenario),
                    "related_feature_names": ",".join(related_shap),
                },
            )
        )
        action_items.append(
            AgentActionItem(
                action_id=f"action_{topic}",
                summary=action,
                priority=_severity_to_priority(sev),
                evidence_references=[],
                metadata={"related_finding_id": fid, "topic": topic},
            )
        )
        emitted_topics.add(topic)
        if len(new_findings) >= 5:
            break

    # SHAP-only topics not already emitted/evidence-backed
    for topic in TOPIC_ORDER:
        if topic in emitted_topics or topic in evidence_topic_map:
            continue
        if topic not in shap_topics:
            continue
        related_shap = shap_topics.get(topic, [])
        new_findings.append(
            AgentFinding(
                finding_id=f"shap_topic_{topic}",
                finding_type=AgentFindingType.ML_SIGNAL,
                severity=AgentFindingSeverity.LOW,
                summary=f"SHAP attribution suggests the ML model highlighted {topic}-related factors for review; this is not causal proof.",
                evidence_references=[],
                requires_evidence=False,
                metadata={
                    "topic": topic,
                    "support_status": "model_explanation",
                    "related_feature_names": ",".join(related_shap),
                },
            )
        )
        emitted_topics.add(topic)
        if len(new_findings) >= 7:
            break

    # ML uncertainty check
    probs = dict(agent_input.ml_probabilities)
    if len(probs) >= 2:
        ordered = sorted(probs.items(), key=lambda kv: float(kv[1]), reverse=True)
        top_label, top_p = ordered[0]
        second_label, second_p = ordered[1]
        margin = float(top_p) - float(second_p)
        if margin < 0.15:
            new_findings.append(
                AgentFinding(
                    finding_id="ml_probability_uncertainty",
                    finding_type=AgentFindingType.OPERATIONAL_UNCERTAINTY,
                    severity=AgentFindingSeverity.MEDIUM,
                    summary="ML probability distribution suggests uncertainty; avoid overconfident interpretation.",
                    evidence_references=[],
                    requires_evidence=False,
                    metadata={
                        "top_class": str(top_label),
                        "second_class": str(second_label),
                        "top_probability": float(top_p),
                        "second_probability": float(second_p),
                        "probability_margin": margin,
                    },
                )
            )

    # Profile/capability checks (partial availability via metadata only)
    profile_fields = _profile_context_fields(agent_input)
    if not profile_fields:
        limitations.append("Profile capability context is partial in current Stage2 input; inspector used scenario/SHAP/evidence context only.")
    else:
        # payload vs max_payload margin if both available
        payload_val = None
        max_payload_val = None
        for k, v in profile_fields.items():
            lk = k.lower()
            if payload_val is None and ("payload" in lk and "max" not in lk):
                try:
                    payload_val = float(v)
                except Exception:
                    pass
            if max_payload_val is None and ("max_payload" in lk):
                try:
                    max_payload_val = float(v)
                except Exception:
                    pass
        if payload_val is not None and max_payload_val is not None and max_payload_val > 0 and payload_val / max_payload_val >= 0.85:
            fid = "profile_payload_margin"
            new_findings.append(
                AgentFinding(
                    finding_id=fid,
                    finding_type=AgentFindingType.TOOL_CHECK,
                    severity=AgentFindingSeverity.MEDIUM,
                    summary="Mission payload appears close to profile payload capacity; review loading/performance margin.",
                    evidence_references=[],
                    requires_evidence=False,
                    metadata={
                        "topic": "payload",
                        "support_status": "profile_derived",
                        "related_profile_fields": "payload,max_payload",
                    },
                )
            )
            action_items.append(
                AgentActionItem(
                    action_id="action_profile_payload_margin",
                    summary="Review loading/performance margin and confirm expected payload operating envelope.",
                    priority=AgentFindingSeverity.MEDIUM,
                    evidence_references=[],
                    metadata={"related_finding_id": fid, "topic": "payload"},
                )
            )

    return new_findings, action_items, limitations
