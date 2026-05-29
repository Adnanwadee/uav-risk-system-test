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
    profile_context = agent_input.profile_context
    if profile_context is not None:
        return profile_context.model_dump()

    # Backward-compatible fallback for older call paths that only set metadata.
    meta = agent_input.metadata if isinstance(agent_input.metadata, dict) else {}
    return {
        str(k): v
        for k, v in meta.items()
        if any(tok in str(k).lower() for tok in (
            "uav_", "profile", "payload", "mass", "battery", "reserve", "swarm", "runway", "ceiling", "sensor", "gnss", "lidar", "radar", "camera", "detect"
        ))
    }


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"true", "1", "yes", "y", "on"}:
            return True
        if lower in {"false", "0", "no", "n", "off"}:
            return False
    return None


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

    # Profile/capability checks using structured profile_context when available.
    profile_fields = _profile_context_fields(agent_input)
    if not profile_fields:
        limitations.append("Profile capability context is partial in current Stage2 input; inspector used scenario/SHAP/evidence context only.")
    else:
        payload_scenario_keys = [k for k in scenario_keys if any(tok in k.lower() for tok in ("payload", "mass", "weight", "loading"))]
        payload_value = None
        for key in payload_scenario_keys:
            payload_value = _as_float(agent_input.scenario_summary.get(key))
            if payload_value is not None:
                break
        max_payload = _as_float(profile_fields.get("max_payload_kg"))
        if payload_value is not None and max_payload is not None and max_payload > 0 and payload_value / max_payload >= 0.85:
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
                        "related_profile_fields": "max_payload_kg",
                        "related_scenario_fields": ",".join(payload_scenario_keys),
                    },
                )
            )
            action_items.append(
                AgentActionItem(
                    action_id="action_profile_payload_margin",
                    summary="Review payload/performance margin before mission launch.",
                    priority=AgentFindingSeverity.MEDIUM,
                    evidence_references=[],
                    metadata={"related_finding_id": fid, "topic": "payload"},
                )
            )

        swarm_keys = [k for k in scenario_keys if "swarm" in k.lower() or "multi_uas" in k.lower()]
        swarm_requested = any(_as_bool(agent_input.scenario_summary.get(k)) is True for k in swarm_keys)
        swarm_size = None
        for key in swarm_keys:
            if "size" in key.lower():
                swarm_size = _as_float(agent_input.scenario_summary.get(key))
                if swarm_size is not None:
                    break
        profile_swarm_capable = _as_bool(profile_fields.get("swarm_capable"))
        max_swarm_size = _as_float(profile_fields.get("max_swarm_size"))
        if (swarm_requested or (swarm_size is not None and swarm_size > 1)) and profile_swarm_capable is False:
            fid = "profile_swarm_capability"
            new_findings.append(
                AgentFinding(
                    finding_id=fid,
                    finding_type=AgentFindingType.TOOL_CHECK,
                    severity=AgentFindingSeverity.HIGH,
                    summary="Scenario indicates multi-UAS/swarm operation while profile capability is limited; review operational feasibility and coordination controls.",
                    evidence_references=[],
                    requires_evidence=False,
                    metadata={
                        "topic": "swarm",
                        "support_status": "profile_derived",
                        "related_profile_fields": "swarm_capable,max_swarm_size",
                        "related_scenario_fields": ",".join(swarm_keys),
                    },
                )
            )
            action_items.append(
                AgentActionItem(
                    action_id="action_profile_swarm_capability",
                    summary="Verify swarm capability and coordination procedures before launch.",
                    priority=AgentFindingSeverity.HIGH,
                    evidence_references=[],
                    metadata={"related_finding_id": fid, "topic": "swarm"},
                )
            )
        elif swarm_size is not None and max_swarm_size is not None and max_swarm_size > 0 and swarm_size > max_swarm_size:
            fid = "profile_swarm_size_margin"
            new_findings.append(
                AgentFinding(
                    finding_id=fid,
                    finding_type=AgentFindingType.TOOL_CHECK,
                    severity=AgentFindingSeverity.HIGH,
                    summary="Scenario swarm size appears above profile swarm capacity; review operational constraints.",
                    evidence_references=[],
                    requires_evidence=False,
                    metadata={
                        "topic": "swarm",
                        "support_status": "profile_derived",
                        "related_profile_fields": "max_swarm_size",
                        "related_scenario_fields": ",".join(swarm_keys),
                    },
                )
            )

        obstacle_keys = [k for k in scenario_keys if any(tok in k.lower() for tok in ("obstacle", "traffic", "daa", "detect", "avoid"))]
        limited_sensors = []
        if _as_bool(profile_fields.get("detect_and_avoid_available")) is False:
            limited_sensors.append("detect_and_avoid_available")
        for field_name in ("gnss_available", "camera_available", "lidar_available", "radar_available"):
            if _as_bool(profile_fields.get(field_name)) is False:
                limited_sensors.append(field_name)
        if obstacle_keys and limited_sensors:
            fid = "profile_sensor_daa_margin"
            new_findings.append(
                AgentFinding(
                    finding_id=fid,
                    finding_type=AgentFindingType.TOOL_CHECK,
                    severity=AgentFindingSeverity.HIGH,
                    summary="Sensor/DAA capability should be reviewed for obstacle/traffic environment.",
                    evidence_references=[],
                    requires_evidence=False,
                    metadata={
                        "topic": "ground_risk",
                        "support_status": "profile_derived",
                        "related_profile_fields": ",".join(limited_sensors),
                        "related_scenario_fields": ",".join(obstacle_keys),
                    },
                )
            )

        energy_keys = [k for k in scenario_keys if any(tok in k.lower() for tok in ("duration", "flight_time", "energy", "battery", "reserve", "endurance"))]
        reserve_fraction = _as_float(profile_fields.get("reserve_fraction"))
        max_flight_time = _as_float(profile_fields.get("max_flight_time_min"))
        if energy_keys and (reserve_fraction is not None or max_flight_time is not None):
            fid = "profile_energy_margin"
            new_findings.append(
                AgentFinding(
                    finding_id=fid,
                    finding_type=AgentFindingType.TOOL_CHECK,
                    severity=AgentFindingSeverity.MEDIUM,
                    summary="Energy/reserve margin should be reviewed for mission duration and contingencies.",
                    evidence_references=[],
                    requires_evidence=False,
                    metadata={
                        "topic": "energy",
                        "support_status": "profile_derived",
                        "related_profile_fields": "reserve_fraction,max_flight_time_min",
                        "related_scenario_fields": ",".join(energy_keys),
                    },
                )
            )
            action_items.append(
                AgentActionItem(
                    action_id="action_profile_energy_margin",
                    summary="Review energy/reserve margins for mission profile and contingencies.",
                    priority=AgentFindingSeverity.MEDIUM,
                    evidence_references=[],
                    metadata={"related_finding_id": fid, "topic": "energy"},
                )
            )

        altitude_keys = [k for k in scenario_keys if any(tok in k.lower() for tok in ("altitude", "agl", "ceiling"))]
        scenario_altitude = None
        for key in altitude_keys:
            val = _as_float(agent_input.scenario_summary.get(key))
            if val is not None:
                scenario_altitude = val
                break
        max_altitude = _as_float(profile_fields.get("max_altitude_m"))
        hover_ceiling = _as_float(profile_fields.get("hover_ceiling_m"))
        altitude_limit = max(item for item in (max_altitude, hover_ceiling) if item is not None) if any(v is not None for v in (max_altitude, hover_ceiling)) else None
        if scenario_altitude is not None and altitude_limit is not None and altitude_limit > 0 and scenario_altitude / altitude_limit >= 0.85:
            fid = "profile_altitude_margin"
            new_findings.append(
                AgentFinding(
                    finding_id=fid,
                    finding_type=AgentFindingType.TOOL_CHECK,
                    severity=AgentFindingSeverity.MEDIUM,
                    summary="Mission altitude may reduce performance margin relative to profile ceiling/altitude limits.",
                    evidence_references=[],
                    requires_evidence=False,
                    metadata={
                        "topic": "altitude",
                        "support_status": "profile_derived",
                        "related_profile_fields": "max_altitude_m,hover_ceiling_m",
                        "related_scenario_fields": ",".join(altitude_keys),
                    },
                )
            )
            action_items.append(
                AgentActionItem(
                    action_id="action_profile_altitude_margin",
                    summary="Review altitude/performance margins for planned mission profile.",
                    priority=AgentFindingSeverity.MEDIUM,
                    evidence_references=[],
                    metadata={"related_finding_id": fid, "topic": "altitude"},
                )
            )

    return new_findings, action_items, limitations
