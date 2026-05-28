from __future__ import annotations

import re
from typing import Iterable

from uav_risk.stage2.contracts import (
    AgentInput,
    AgentQueryDerivedFrom,
    AgentQuerySourceIntent,
    AgentRAGQueryPlan,
)


def _contains_any(text: str, tokens: Iterable[str]) -> bool:
    return any(token in text for token in tokens)


class AgentQueryPlanner:
    def __init__(self, *, max_queries: int = 5) -> None:
        if max_queries < 1:
            raise ValueError("max_queries must be >= 1")
        self.max_queries = max_queries

    def build_from_agent_input(self, agent_input: AgentInput) -> list[AgentRAGQueryPlan]:
        plans: list[AgentRAGQueryPlan] = []

        self._add_from_shap(agent_input, plans)
        self._add_from_scenario(agent_input, plans)
        self._add_from_operator_notes(agent_input, plans)
        self._add_from_ml_signal(agent_input, plans)

        if not plans:
            plans.append(
                self._plan(
                    query_id="q_default_general_uav",
                    query_text="Part 107 remote pilot small UAS operating rules",
                    query_purpose="Establish baseline UAV operating-rule context.",
                    source_intent=AgentQuerySourceIntent.PART107,
                    expected_source_family="Part 107 / AC 107",
                    derived_from=AgentQueryDerivedFrom.SYSTEM_DEFAULT,
                    related_feature_names=[],
                    priority=5,
                )
            )

        deduped: list[AgentRAGQueryPlan] = []
        seen: set[tuple[str, str]] = set()
        for plan in sorted(plans, key=lambda item: item.priority, reverse=True):
            key = (plan.source_intent.value, plan.query_text.strip().lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(plan)
            if len(deduped) >= self.max_queries:
                break

        return deduped

    def _plan(
        self,
        *,
        query_id: str,
        query_text: str,
        query_purpose: str,
        source_intent: AgentQuerySourceIntent,
        expected_source_family: str | None,
        derived_from: AgentQueryDerivedFrom,
        related_feature_names: list[str],
        priority: int,
    ) -> AgentRAGQueryPlan:
        return AgentRAGQueryPlan(
            query_id=query_id,
            query_text=query_text,
            query_purpose=query_purpose,
            source_intent=source_intent,
            expected_source_family=expected_source_family,
            derived_from=derived_from,
            related_feature_names=related_feature_names,
            evidence_required=True,
            fallback_if_insufficient="Record limitation and keep recommendation cautious/insufficient.",
            priority=priority,
            metadata={},
        )

    def _add_topic_plan(
        self,
        plans: list[AgentRAGQueryPlan],
        *,
        topic: str,
        related: list[str],
        derived_from: AgentQueryDerivedFrom,
        suffix: str,
    ) -> None:
        if topic == "weather":
            plans.append(self._plan(
                query_id=f"q_{suffix}_weather",
                query_text="AC 107-2A preflight weather assessment small UAS wind conditions",
                query_purpose="Retrieve weather/preflight constraints for small UAS operations.",
                source_intent=AgentQuerySourceIntent.AC107,
                expected_source_family="AC_107-2A",
                derived_from=derived_from,
                related_feature_names=related,
                priority=9,
            ))
        elif topic == "airspace":
            plans.append(self._plan(
                query_id=f"q_{suffix}_airspace",
                query_text="AC 107-2A airspace authorization controlled airspace small UAS operation",
                query_purpose="Retrieve controlled/restricted airspace authorization guidance.",
                source_intent=AgentQuerySourceIntent.AC107,
                expected_source_family="AC_107-2A",
                derived_from=derived_from,
                related_feature_names=related,
                priority=10,
            ))
        elif topic == "vlos":
            plans.append(self._plan(
                query_id=f"q_{suffix}_vlos",
                query_text="Part 107 visual line of sight small unmanned aircraft operation",
                query_purpose="Retrieve VLOS operational requirements and constraints.",
                source_intent=AgentQuerySourceIntent.PART107,
                expected_source_family="Part 107 / AC 107",
                derived_from=derived_from,
                related_feature_names=related,
                priority=10,
            ))
        elif topic == "comms":
            plans.append(self._plan(
                query_id=f"q_{suffix}_comms",
                query_text="SORA command and control link reliability operational safety objectives",
                query_purpose="Retrieve C2 link reliability and mitigation guidance.",
                source_intent=AgentQuerySourceIntent.SORA,
                expected_source_family="SORA",
                derived_from=derived_from,
                related_feature_names=related,
                priority=8,
            ))
        elif topic == "energy":
            plans.append(self._plan(
                query_id=f"q_{suffix}_energy",
                query_text="AC 107-2A preflight assessment small UAS aircraft condition and operational planning",
                query_purpose="Retrieve energy/reserve planning guidance.",
                source_intent=AgentQuerySourceIntent.AC107,
                expected_source_family="AC_107-2A",
                derived_from=derived_from,
                related_feature_names=related,
                priority=8,
            ))
        elif topic == "payload":
            plans.append(self._plan(
                query_id=f"q_{suffix}_payload",
                query_text="Part 107 small UAS loading performance preflight operation",
                query_purpose="Retrieve payload/loading and preflight performance guidance.",
                source_intent=AgentQuerySourceIntent.PART107,
                expected_source_family="Part 107 / AC 107",
                derived_from=derived_from,
                related_feature_names=related,
                priority=7,
            ))
        elif topic == "swarm":
            plans.append(self._plan(
                query_id=f"q_{suffix}_swarm",
                query_text="SORA operational safety objectives UAS operational complexity",
                query_purpose="Retrieve multi-UAS operational complexity guidance.",
                source_intent=AgentQuerySourceIntent.SORA,
                expected_source_family="SORA",
                derived_from=derived_from,
                related_feature_names=related,
                priority=7,
            ))
        elif topic == "ground_risk":
            plans.append(self._plan(
                query_id=f"q_{suffix}_ground_risk",
                query_text="SORA ground risk class operational volume adjacent area",
                query_purpose="Retrieve ground-risk and operational-volume guidance.",
                source_intent=AgentQuerySourceIntent.SORA,
                expected_source_family="SORA",
                derived_from=derived_from,
                related_feature_names=related,
                priority=9,
            ))

    def _topics_for_feature_names(self, names: list[str]) -> dict[str, list[str]]:
        buckets: dict[str, list[str]] = {
            "weather": [],
            "airspace": [],
            "vlos": [],
            "comms": [],
            "energy": [],
            "payload": [],
            "swarm": [],
            "ground_risk": [],
        }

        for raw_name in names:
            name = raw_name.lower()
            if _contains_any(name, ("weather", "wind", "gust", "turbulence", "thermal")):
                buckets["weather"].append(raw_name)
            if _contains_any(name, ("airspace", "no_fly", "restricted", "altitude", "agl", "ceiling")):
                buckets["airspace"].append(raw_name)
            if _contains_any(name, ("vlos", "visual_line_of_sight", "los")):
                buckets["vlos"].append(raw_name)
            if _contains_any(name, ("comms", "uplink", "downlink", "c2", "link", "telemetry")):
                buckets["comms"].append(raw_name)
            if _contains_any(name, ("battery", "reserve", "endurance", "fuel", "energy")):
                buckets["energy"].append(raw_name)
            if _contains_any(name, ("payload", "mass", "weight", "loading")):
                buckets["payload"].append(raw_name)
            if _contains_any(name, ("swarm", "multi_uas", "formation")):
                buckets["swarm"].append(raw_name)
            if _contains_any(name, ("ground_risk", "population", "adjacent_area", "operational_volume", "traffic", "obstacle", "landing")):
                buckets["ground_risk"].append(raw_name)

        return {k: v for k, v in buckets.items() if v}

    def _add_from_shap(self, agent_input: AgentInput, plans: list[AgentRAGQueryPlan]) -> None:
        names: list[str] = []
        for item in agent_input.shap_top_features:
            name = str(item.get("feature", "")).strip()
            if name:
                names.append(name)
        for topic, related in self._topics_for_feature_names(names).items():
            self._add_topic_plan(plans, topic=topic, related=related, derived_from=AgentQueryDerivedFrom.SHAP, suffix="shap")

    def _add_from_scenario(self, agent_input: AgentInput, plans: list[AgentRAGQueryPlan]) -> None:
        names = [str(k).strip() for k in agent_input.scenario_summary.keys() if str(k).strip()]
        for topic, related in self._topics_for_feature_names(names).items():
            self._add_topic_plan(plans, topic=topic, related=related, derived_from=AgentQueryDerivedFrom.SCENARIO, suffix="scenario")

    def _add_from_operator_notes(self, agent_input: AgentInput, plans: list[AgentRAGQueryPlan]) -> None:
        notes = (agent_input.operator_notes or "").strip().lower()
        if not notes:
            return

        note_tokens = re.sub(r"[^a-z0-9_\s-]", " ", notes)
        topics: list[str] = []
        if _contains_any(note_tokens, ("wind", "weather", "gust")):
            topics.append("weather")
        if _contains_any(note_tokens, ("airspace", "controlled", "restricted", "airport", "authorization", "no-fly", "nofly")):
            topics.append("airspace")
        if _contains_any(note_tokens, ("vlos", "visual line of sight", "line of sight")):
            topics.append("vlos")
        if _contains_any(note_tokens, ("swarm", "multiple drone", "multi uas")):
            topics.append("swarm")
        if _contains_any(note_tokens, ("export", "ear", "international")):
            plans.append(self._plan(
                query_id="q_notes_ear_export",
                query_text="EAR unmanned aircraft systems export control",
                query_purpose="Retrieve export-control guidance only when explicitly requested by operator notes.",
                source_intent=AgentQuerySourceIntent.EAR_EXPORT,
                expected_source_family="EAR / D0593E",
                derived_from=AgentQueryDerivedFrom.OPERATOR_NOTES,
                related_feature_names=[],
                priority=7,
            ))

        for topic in topics:
            self._add_topic_plan(
                plans,
                topic=topic,
                related=[],
                derived_from=AgentQueryDerivedFrom.OPERATOR_NOTES,
                suffix="notes",
            )

    def _add_from_ml_signal(self, agent_input: AgentInput, plans: list[AgentRAGQueryPlan]) -> None:
        label = (agent_input.ml_prediction or "").strip().lower()
        if "medium" in label or "high" in label:
            plans.append(
                self._plan(
                    query_id="q_ml_special_condition",
                    query_text="special condition UAS medium risk operational limitations",
                    query_purpose="Retrieve medium/high operational limitation context tied to ML risk signal.",
                    source_intent=AgentQuerySourceIntent.SPECIAL_CONDITION,
                    expected_source_family="special_condition",
                    derived_from=AgentQueryDerivedFrom.ML,
                    related_feature_names=[],
                    priority=6,
                )
            )


def build_agent_rag_query_plan(agent_input: AgentInput, *, max_queries: int = 5) -> list[AgentRAGQueryPlan]:
    return AgentQueryPlanner(max_queries=max_queries).build_from_agent_input(agent_input)
