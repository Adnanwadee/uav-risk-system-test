from __future__ import annotations

import inspect

import pytest

from uav_risk.stage2.agent.operational_agent import (
    OperationalAgentV2,
    build_agent_evidence_queries,
    build_agent_query_plan,
)
from uav_risk.stage2.contracts import (
    AgentInput,
    AgentRecommendation,
    AgentResult,
    EvidenceSupportStatus,
    Stage2ProfileContext,
)


def _agent_input(ml_prediction: str = "High Risk") -> AgentInput:
    return AgentInput(
        assessment_id="a1",
        scenario_summary={
            "environment_weather_wind_mps": 8.2,
            "airspace_altitude_agl_max_m": 120.0,
            "comms_uplink_ok": True,
            "environment_gnss_multipath": False,
            "faults_count": 1.0,
        },
        ml_prediction=ml_prediction,
        ml_probabilities={"High Risk": 0.8, "Medium Risk": 0.15, "Low Risk": 0.05}
        if ml_prediction == "High Risk"
        else {"Low Risk": 0.8, "Medium Risk": 0.15, "High Risk": 0.05},
        shap_top_features=[
            {"feature": "environment_weather_wind_mps", "importance": 0.3},
            {"feature": "airspace_altitude_agl_max_m", "importance": 0.2},
            {"feature": "comms_uplink_ok", "importance": 0.15},
        ],
        evidence_bundles=[],
        operator_notes="Check controlled airspace authorization and wind conditions.",
    )


def test_constructor_is_lightweight() -> None:
    agent = OperationalAgentV2()
    assert agent is not None


def test_constructor_rejects_negative_max_queries() -> None:
    with pytest.raises(ValueError):
        OperationalAgentV2(max_queries=-1)


def test_query_plan_maps_wind_to_ac107_weather_query() -> None:
    plan = build_agent_query_plan(_agent_input(), 8)
    texts = [item.query_text for item in plan]
    assert any("AC 107-2A preflight weather assessment small UAS wind conditions" == q for q in texts)


def test_query_plan_maps_airspace_to_ac107_airspace_query() -> None:
    plan = build_agent_query_plan(_agent_input(), 8)
    texts = [item.query_text for item in plan]
    assert any("AC 107-2A airspace authorization controlled airspace small UAS operation" == q for q in texts)


def test_query_plan_maps_comms_to_sora_c2_query() -> None:
    plan = build_agent_query_plan(_agent_input(), 8)
    texts = [item.query_text for item in plan]
    assert any("SORA command and control link reliability operational safety objectives" == q for q in texts)


def test_medium_or_high_ml_does_not_emit_generic_risk_class_query() -> None:
    queries = build_agent_evidence_queries(_agent_input("Medium Risk"), 8)
    joined = " | ".join(queries).lower()
    assert "uav operational guidance for ml risk class" not in joined
    assert "risk class medium risk" not in joined


def test_medium_or_high_ml_can_emit_special_condition_query_derived_from_ml() -> None:
    plan = build_agent_query_plan(_agent_input("High Risk"), 8)
    sc = [item for item in plan if item.source_intent.value == "special_condition"]
    assert sc
    assert any(item.derived_from.value == "ml" for item in sc)


def test_operator_notes_controlled_airspace_maps_to_ac107_airspace_query() -> None:
    ai = _agent_input("Low Risk")
    ai.operator_notes = "Need controlled airspace authorization near airport."
    plan = build_agent_query_plan(ai, 8)
    assert any(item.query_text == "AC 107-2A airspace authorization controlled airspace small UAS operation" for item in plan)


def test_query_builder_deduplicates_topics() -> None:
    ai = _agent_input("Medium Risk")
    ai.shap_top_features.append({"feature": "wind_gust_mps", "importance": 0.1})
    plan = build_agent_query_plan(ai, 8)
    weather = [p for p in plan if "weather assessment" in p.query_text]
    assert len(weather) == 1


def test_query_builder_caps_query_count() -> None:
    plan = build_agent_query_plan(_agent_input(), 2)
    assert len(plan) == 2


@pytest.mark.asyncio
async def test_run_without_rag_adapter_returns_insufficient_evidence() -> None:
    result = await OperationalAgentV2(rag_adapter=None).run(_agent_input())
    assert result.recommendation == AgentRecommendation.INSUFFICIENT_EVIDENCE


@pytest.mark.asyncio
async def test_run_with_fake_supported_adapter_calls_retrieve_evidence() -> None:
    calls: list[str] = []

    class FakeSupportedAdapter:
        async def retrieve_evidence(self, query: str, *, scenario_context=None):
            from uav_risk.stage2.contracts import (
                EvidenceBundle,
                EvidenceCitation,
                EvidenceOrigin,
                EvidenceSourceType,
            )

            calls.append(query)
            citation = EvidenceCitation(
                citation_id=f"c{len(calls)}",
                source_id="s1",
                source_title="AC_107-2A",
                source_type=EvidenceSourceType.INTERNAL_DOC,
                origin=EvidenceOrigin.LOCAL_DOCUMENT,
                quote="retrieved quote about preflight weather assessment and wind constraints",
                metadata={"source_filename": "AC_107-2A.pdf", "page_start": 4},
            )
            return EvidenceBundle(
                bundle_id=f"b{len(calls)}",
                query=query,
                claims=[],
                citations=[citation],
                support_status=EvidenceSupportStatus.SUPPORTED,
                confidence=0.8,
            )

    result = await OperationalAgentV2(rag_adapter=FakeSupportedAdapter(), max_queries=2).run(_agent_input())
    assert calls
    assert all("risk class" not in c.lower() for c in calls)
    assert isinstance(result, AgentResult)
    assert result.tool_trace
    names = {item.tool_name.value for item in result.tool_trace}
    assert "shap_topic_mapper" in names
    assert "rag_retrieval" in names
    assert "scenario_profile_inspector" in names


@pytest.mark.asyncio
async def test_supported_weather_evidence_produces_weather_finding() -> None:
    class FakeAdapter:
        async def retrieve_evidence(self, query: str, *, scenario_context=None):
            from uav_risk.stage2.contracts import EvidenceBundle, EvidenceCitation, EvidenceOrigin, EvidenceSourceType

            citation = EvidenceCitation(
                citation_id="c-weather",
                source_id="src-weather",
                source_title="AC_107-2A",
                source_type=EvidenceSourceType.INTERNAL_DOC,
                origin=EvidenceOrigin.LOCAL_DOCUMENT,
                quote="preflight weather assessment guidance for small UAS",
                metadata={"source_filename": "AC_107-2A.pdf", "page_start": 5},
            )
            return EvidenceBundle(
                bundle_id="b-weather",
                query="AC 107-2A preflight weather assessment small UAS wind conditions",
                claims=[],
                citations=[citation],
                support_status=EvidenceSupportStatus.SUPPORTED,
                confidence=0.7,
                metadata={"source_intent": "ac107", "query_id": "q1", "derived_from": "shap"},
            )

    result = await OperationalAgentV2(rag_adapter=FakeAdapter(), max_queries=1).run(_agent_input("Medium Risk"))
    assert any("Weather and wind conditions require preflight assessment" in f.summary for f in result.findings)


@pytest.mark.asyncio
async def test_supported_airspace_evidence_produces_airspace_finding() -> None:
    class FakeAdapter:
        async def retrieve_evidence(self, query: str, *, scenario_context=None):
            from uav_risk.stage2.contracts import EvidenceBundle, EvidenceCitation, EvidenceOrigin, EvidenceSourceType

            citation = EvidenceCitation(
                citation_id="c-airspace",
                source_id="src-airspace",
                source_title="AC_107-2A",
                source_type=EvidenceSourceType.INTERNAL_DOC,
                origin=EvidenceOrigin.LOCAL_DOCUMENT,
                quote="controlled airspace authorization guidance",
                metadata={"source_filename": "AC_107-2A.pdf", "page_start": 8},
            )
            return EvidenceBundle(
                bundle_id="b-airspace",
                query="AC 107-2A airspace authorization controlled airspace small UAS operation",
                claims=[],
                citations=[citation],
                support_status=EvidenceSupportStatus.SUPPORTED,
                confidence=0.7,
                metadata={"source_intent": "ac107", "query_id": "q2", "derived_from": "scenario"},
            )

    result = await OperationalAgentV2(rag_adapter=FakeAdapter(), max_queries=1).run(_agent_input("Medium Risk"))
    assert any("Controlled/restricted airspace context requires authorization" in f.summary for f in result.findings)


@pytest.mark.asyncio
async def test_supported_sora_c2_evidence_produces_c2_finding() -> None:
    class FakeAdapter:
        async def retrieve_evidence(self, query: str, *, scenario_context=None):
            from uav_risk.stage2.contracts import EvidenceBundle, EvidenceCitation, EvidenceOrigin, EvidenceSourceType

            citation = EvidenceCitation(
                citation_id="c-c2",
                source_id="src-c2",
                source_title="SORA",
                source_type=EvidenceSourceType.INTERNAL_DOC,
                origin=EvidenceOrigin.LOCAL_DOCUMENT,
                quote="command and control link reliability objective",
                metadata={"source_filename": "SORA-v2.5-Main-Body-Release-JAR_doc_25.pdf", "page_start": 12},
            )
            return EvidenceBundle(
                bundle_id="b-c2",
                query="SORA command and control link reliability operational safety objectives",
                claims=[],
                citations=[citation],
                support_status=EvidenceSupportStatus.SUPPORTED,
                confidence=0.8,
                metadata={"source_intent": "sora", "query_id": "q3", "derived_from": "scenario"},
            )

    result = await OperationalAgentV2(rag_adapter=FakeAdapter(), max_queries=1).run(_agent_input("Medium Risk"))
    assert any("Command-and-control link reliability is an operational concern" in f.summary for f in result.findings)


@pytest.mark.asyncio
async def test_shap_related_feature_names_are_included_for_shap_derived_finding() -> None:
    class FakeAdapter:
        async def retrieve_evidence(self, query: str, *, scenario_context=None):
            from uav_risk.stage2.contracts import EvidenceBundle, EvidenceCitation, EvidenceOrigin, EvidenceSourceType

            citation = EvidenceCitation(
                citation_id="c-shap",
                source_id="src-shap",
                source_title="AC_107-2A",
                source_type=EvidenceSourceType.INTERNAL_DOC,
                origin=EvidenceOrigin.LOCAL_DOCUMENT,
                quote="weather preflight quote",
            )
            return EvidenceBundle(
                bundle_id="b-shap",
                query="AC 107-2A preflight weather assessment small UAS wind conditions",
                claims=[],
                citations=[citation],
                support_status=EvidenceSupportStatus.SUPPORTED,
                confidence=0.8,
                metadata={"source_intent": "ac107", "query_id": "q4", "derived_from": "shap"},
            )

    result = await OperationalAgentV2(rag_adapter=FakeAdapter(), max_queries=1).run(_agent_input("Medium Risk"))
    matching = [f for f in result.findings if f.finding_type.value == "evidence_backed"]
    assert matching
    assert any("related_feature_names" in f.metadata for f in matching)


@pytest.mark.asyncio
async def test_insufficient_evidence_creates_uncertainty_and_not_supported_finding() -> None:
    class FakeInsufficientAdapter:
        async def retrieve_evidence(self, query: str, *, scenario_context=None):
            from uav_risk.stage2.contracts import make_insufficient_evidence_bundle

            return make_insufficient_evidence_bundle(query, "No sufficient evidence candidates passed retrieval safety checks.")

    result = await OperationalAgentV2(rag_adapter=FakeInsufficientAdapter(), max_queries=1).run(_agent_input("Medium Risk"))
    assert any(f.finding_type.value in {"operational_uncertainty", "limitation"} for f in result.findings)
    assert not any(f.finding_type.value == "evidence_backed" for f in result.findings)


@pytest.mark.asyncio
async def test_action_items_reference_evidence_when_supported() -> None:
    class FakeAdapter:
        async def retrieve_evidence(self, query: str, *, scenario_context=None):
            from uav_risk.stage2.contracts import EvidenceBundle, EvidenceCitation, EvidenceOrigin, EvidenceSourceType

            citation = EvidenceCitation(
                citation_id="c-action",
                source_id="src-action",
                source_title="AC_107-2A",
                source_type=EvidenceSourceType.INTERNAL_DOC,
                origin=EvidenceOrigin.LOCAL_DOCUMENT,
                quote="airspace authorization quote",
            )
            return EvidenceBundle(
                bundle_id="b-action",
                query="AC 107-2A airspace authorization controlled airspace small UAS operation",
                claims=[],
                citations=[citation],
                support_status=EvidenceSupportStatus.SUPPORTED,
                confidence=0.8,
                metadata={"source_intent": "ac107", "query_id": "q5", "derived_from": "scenario"},
            )

    result = await OperationalAgentV2(rag_adapter=FakeAdapter(), max_queries=1).run(_agent_input("Medium Risk"))
    assert result.action_items
    assert any(item.evidence_references for item in result.action_items)


@pytest.mark.asyncio
async def test_recommendation_caution_for_medium_or_high_with_supported_evidence() -> None:
    class FakeAdapter:
        async def retrieve_evidence(self, query: str, *, scenario_context=None):
            from uav_risk.stage2.contracts import EvidenceBundle, EvidenceCitation, EvidenceOrigin, EvidenceSourceType

            citation = EvidenceCitation(
                citation_id="c-rec",
                source_id="src-rec",
                source_title="Doc",
                source_type=EvidenceSourceType.INTERNAL_DOC,
                origin=EvidenceOrigin.LOCAL_DOCUMENT,
                quote="supported quote",
            )
            return EvidenceBundle(
                bundle_id="b-rec",
                query=query,
                claims=[],
                citations=[citation],
                support_status=EvidenceSupportStatus.SUPPORTED,
                confidence=0.8,
            )

    ai = _agent_input("Medium Risk")
    ai.ml_probabilities = {"Medium Risk": 0.7, "Low Risk": 0.2, "High Risk": 0.1}
    result = await OperationalAgentV2(rag_adapter=FakeAdapter(), max_queries=1).run(ai)
    assert result.recommendation == AgentRecommendation.CAUTION


@pytest.mark.asyncio
async def test_low_risk_does_not_become_go_if_critical_insufficient_evidence_exists() -> None:
    class MixedAdapter:
        def __init__(self) -> None:
            self.calls = 0

        async def retrieve_evidence(self, query: str, *, scenario_context=None):
            from uav_risk.stage2.contracts import EvidenceBundle, EvidenceCitation, EvidenceOrigin, EvidenceSourceType, make_insufficient_evidence_bundle

            self.calls += 1
            if self.calls == 1:
                return make_insufficient_evidence_bundle(query, "No sufficient evidence candidates passed retrieval safety checks.")
            citation = EvidenceCitation(
                citation_id="c-low",
                source_id="src-low",
                source_title="Doc",
                source_type=EvidenceSourceType.INTERNAL_DOC,
                origin=EvidenceOrigin.LOCAL_DOCUMENT,
                quote="supported quote",
            )
            return EvidenceBundle(
                bundle_id=f"b-{self.calls}",
                query=query,
                claims=[],
                citations=[citation],
                support_status=EvidenceSupportStatus.SUPPORTED,
                confidence=0.8,
            )

    result = await OperationalAgentV2(rag_adapter=MixedAdapter(), max_queries=2).run(_agent_input("Low Risk"))
    assert result.recommendation != AgentRecommendation.GO


@pytest.mark.asyncio
async def test_returned_result_is_canonical_agent_result() -> None:
    result = await OperationalAgentV2().run(_agent_input())
    assert isinstance(result, AgentResult)


def test_agent_source_does_not_create_evidence_citation_directly() -> None:
    import uav_risk.stage2.agent.operational_agent as module

    source = inspect.getsource(module)
    assert "EvidenceCitation(" not in source


@pytest.mark.asyncio
async def test_result_does_not_expose_chain_of_thought_fields() -> None:
    result = await OperationalAgentV2().run(_agent_input())
    dumped = result.model_dump_json().lower()
    for token in ("reasoning_chain", "chain_of_thought", "scratchpad", "internal_reasoning", "private_reasoning"):
        assert token not in dumped


def test_agent_source_has_no_groq_llm_usage_or_heavy_runtime_loading() -> None:
    import uav_risk.stage2.agent.operational_agent as module

    source = inspect.getsource(module).lower()
    assert "groq" not in source
    assert "llm" not in source
    assert "faiss" not in source
    assert "vector_db" not in source
    assert "knowledge/models" not in source


@pytest.mark.asyncio
async def test_scenario_payload_field_creates_payload_finding() -> None:
    class FakeInsufficientAdapter:
        async def retrieve_evidence(self, query: str, *, scenario_context=None):
            from uav_risk.stage2.contracts import make_insufficient_evidence_bundle
            return make_insufficient_evidence_bundle(query, "No sufficient evidence candidates passed retrieval safety checks.")

    ai = _agent_input("Medium Risk")
    ai.scenario_summary["uav_payload_mass_kg"] = 3.2
    result = await OperationalAgentV2(rag_adapter=FakeInsufficientAdapter(), max_queries=1).run(ai)
    assert any("Payload/loading should be reviewed" in f.summary for f in result.findings)


@pytest.mark.asyncio
async def test_scenario_swarm_field_creates_swarm_finding() -> None:
    class FakeInsufficientAdapter:
        async def retrieve_evidence(self, query: str, *, scenario_context=None):
            from uav_risk.stage2.contracts import make_insufficient_evidence_bundle
            return make_insufficient_evidence_bundle(query, "No sufficient evidence candidates passed retrieval safety checks.")

    ai = _agent_input("Medium Risk")
    ai.scenario_summary["swarm_count"] = 3
    result = await OperationalAgentV2(rag_adapter=FakeInsufficientAdapter(), max_queries=1).run(ai)
    assert any("Multi-UAS/swarm operation increases operational complexity" in f.summary for f in result.findings)


@pytest.mark.asyncio
async def test_shap_only_topic_creates_cautious_model_explanation_finding() -> None:
    class FakeInsufficientAdapter:
        async def retrieve_evidence(self, query: str, *, scenario_context=None):
            from uav_risk.stage2.contracts import make_insufficient_evidence_bundle
            return make_insufficient_evidence_bundle(query, "No sufficient evidence candidates passed retrieval safety checks.")

    ai = _agent_input("Low Risk")
    ai.scenario_summary = {}
    ai.shap_top_features = [{"feature": "environment_weather_wind_mps", "importance": 0.5}]
    result = await OperationalAgentV2(rag_adapter=FakeInsufficientAdapter(), max_queries=1).run(ai)
    assert any("SHAP attribution suggests the ML model highlighted" in f.summary for f in result.findings)


@pytest.mark.asyncio
async def test_close_ml_probabilities_create_uncertainty_finding() -> None:
    class FakeInsufficientAdapter:
        async def retrieve_evidence(self, query: str, *, scenario_context=None):
            from uav_risk.stage2.contracts import make_insufficient_evidence_bundle
            return make_insufficient_evidence_bundle(query, "No sufficient evidence candidates passed retrieval safety checks.")

    ai = _agent_input("Medium Risk")
    ai.ml_probabilities = {"Medium Risk": 0.41, "High Risk": 0.35, "Low Risk": 0.24}
    result = await OperationalAgentV2(rag_adapter=FakeInsufficientAdapter(), max_queries=1).run(ai)
    assert any("ML probability distribution suggests uncertainty" in f.summary for f in result.findings)


@pytest.mark.asyncio
async def test_evidence_backed_topic_not_duplicated_by_inspector() -> None:
    class FakeAdapter:
        async def retrieve_evidence(self, query: str, *, scenario_context=None):
            from uav_risk.stage2.contracts import EvidenceBundle, EvidenceCitation, EvidenceOrigin, EvidenceSourceType
            citation = EvidenceCitation(
                citation_id="c-airspace-dedup",
                source_id="src-airspace",
                source_title="AC_107-2A",
                source_type=EvidenceSourceType.INTERNAL_DOC,
                origin=EvidenceOrigin.LOCAL_DOCUMENT,
                quote="controlled airspace authorization guidance",
            )
            return EvidenceBundle(
                bundle_id="b-airspace-dedup",
                query="AC 107-2A airspace authorization controlled airspace small UAS operation",
                claims=[],
                citations=[citation],
                support_status=EvidenceSupportStatus.SUPPORTED,
                confidence=0.8,
                metadata={"source_intent": "ac107", "query_id": "q-dedup", "derived_from": "scenario"},
            )

    ai = _agent_input("Medium Risk")
    ai.scenario_summary["airspace_restricted_zone"] = True
    result = await OperationalAgentV2(rag_adapter=FakeAdapter(), max_queries=1).run(ai)
    evidence_airspace = [f for f in result.findings if f.finding_type.value == "evidence_backed" and "airspace" in f.summary.lower()]
    tool_airspace = [f for f in result.findings if f.finding_type.value == "tool_check" and "airspace" in f.summary.lower()]
    assert evidence_airspace
    assert not tool_airspace


@pytest.mark.asyncio
async def test_action_items_link_to_findings_metadata() -> None:
    class FakeInsufficientAdapter:
        async def retrieve_evidence(self, query: str, *, scenario_context=None):
            from uav_risk.stage2.contracts import make_insufficient_evidence_bundle
            return make_insufficient_evidence_bundle(query, "No sufficient evidence candidates passed retrieval safety checks.")

    ai = _agent_input("Medium Risk")
    ai.scenario_summary["comms_uplink_ok"] = False
    result = await OperationalAgentV2(rag_adapter=FakeInsufficientAdapter(), max_queries=1).run(ai)
    assert result.action_items
    assert any("related_finding_id" in item.metadata for item in result.action_items)


@pytest.mark.asyncio
async def test_profile_context_drives_payload_and_swarm_concerns() -> None:
    class FakeAdapter:
        async def retrieve_evidence(self, query: str, *, scenario_context=None):
            from uav_risk.stage2.contracts import EvidenceBundle, EvidenceCitation, EvidenceOrigin, EvidenceSourceType

            citation = EvidenceCitation(
                citation_id="c-prof",
                source_id="src-prof",
                source_title="AC_107-2A",
                source_type=EvidenceSourceType.INTERNAL_DOC,
                origin=EvidenceOrigin.LOCAL_DOCUMENT,
                quote="supported quote",
            )
            return EvidenceBundle(
                bundle_id="b-prof",
                query=query,
                claims=[],
                citations=[citation],
                support_status=EvidenceSupportStatus.SUPPORTED,
                confidence=0.8,
            )

    ai = _agent_input("Medium Risk")
    ai.scenario_summary.update({
        "uav_payload_mass_kg": 9.0,
        "swarm_enabled": True,
        "swarm_size": 4.0,
    })
    ai.profile_context = Stage2ProfileContext(
        profile_id="p1",
        max_payload_kg=10.0,
        swarm_capable=False,
        max_swarm_size=2,
    )

    result = await OperationalAgentV2(rag_adapter=FakeAdapter(), max_queries=1).run(ai)
    summaries = "\n".join(f.summary for f in result.findings)
    assert "payload appears close to profile payload capacity" in summaries.lower()
    assert "multi-uas/swarm operation" in summaries.lower()


@pytest.mark.asyncio
async def test_missing_profile_context_preserves_existing_behavior() -> None:
    class FakeInsufficientAdapter:
        async def retrieve_evidence(self, query: str, *, scenario_context=None):
            from uav_risk.stage2.contracts import make_insufficient_evidence_bundle

            return make_insufficient_evidence_bundle(query, "No sufficient evidence candidates passed retrieval safety checks.")

    ai = _agent_input("Medium Risk")
    ai.profile_context = None
    result = await OperationalAgentV2(rag_adapter=FakeInsufficientAdapter(), max_queries=1).run(ai)
    assert result.findings
    assert result.recommendation in {AgentRecommendation.INSUFFICIENT_EVIDENCE, AgentRecommendation.CAUTION}


@pytest.mark.asyncio
async def test_tool_trace_does_not_expose_private_reasoning_fields() -> None:
    result = await OperationalAgentV2(rag_adapter=None).run(_agent_input())
    serialized = str([item.model_dump() for item in result.tool_trace]).lower()
    assert "chain_of_thought" not in serialized
    assert "scratchpad" not in serialized
    assert "internal_reasoning" not in serialized


@pytest.mark.asyncio
async def test_agent_result_includes_working_memory_signals_and_assessments() -> None:
    class FakeAdapter:
        async def retrieve_evidence(self, query: str, *, scenario_context=None):
            from uav_risk.stage2.contracts import EvidenceBundle, EvidenceCitation, EvidenceOrigin, EvidenceSourceType

            citation = EvidenceCitation(
                citation_id="c-wm",
                source_id="src-wm",
                source_title="AC_107-2A",
                source_type=EvidenceSourceType.INTERNAL_DOC,
                origin=EvidenceOrigin.LOCAL_DOCUMENT,
                quote="weather preflight quote",
            )
            return EvidenceBundle(
                bundle_id="b-wm",
                query=query,
                claims=[],
                citations=[citation],
                support_status=EvidenceSupportStatus.SUPPORTED,
                confidence=0.8,
            )

    ai = _agent_input("Medium Risk")
    ai.profile_context = Stage2ProfileContext(max_payload_kg=2.5, max_altitude_m=120.0, swarm_capable=False)
    result = await OperationalAgentV2(rag_adapter=FakeAdapter(), max_queries=3).run(ai)
    assert result.working_memory is not None
    sources = {s.source.value for s in result.working_memory.input_signals}
    assert {"ml", "shap", "scenario", "profile", "operator_notes"}.issubset(sources)
    assert result.working_memory.feature_assessments


@pytest.mark.asyncio
async def test_high_priority_signals_are_selected_for_rag_queries() -> None:
    class FakeAdapter:
        async def retrieve_evidence(self, query: str, *, scenario_context=None):
            from uav_risk.stage2.contracts import make_insufficient_evidence_bundle

            return make_insufficient_evidence_bundle(query, "not enough")

    result = await OperationalAgentV2(rag_adapter=FakeAdapter(), max_queries=3).run(_agent_input("High Risk"))
    assert result.working_memory is not None
    assert result.working_memory.selected_rag_queries
    assert all("risk class" not in q.lower() for q in result.working_memory.selected_rag_queries)


@pytest.mark.asyncio
async def test_feature_assessments_link_to_evidence_and_findings_when_available() -> None:
    class FakeAdapter:
        async def retrieve_evidence(self, query: str, *, scenario_context=None):
            from uav_risk.stage2.contracts import EvidenceBundle, EvidenceCitation, EvidenceOrigin, EvidenceSourceType

            citation = EvidenceCitation(
                citation_id="c-link",
                source_id="src-link",
                source_title="SORA",
                source_type=EvidenceSourceType.INTERNAL_DOC,
                origin=EvidenceOrigin.LOCAL_DOCUMENT,
                quote="command and control guidance",
            )
            return EvidenceBundle(
                bundle_id="b-link",
                query=query,
                claims=[],
                citations=[citation],
                support_status=EvidenceSupportStatus.SUPPORTED,
                confidence=0.8,
            )

    result = await OperationalAgentV2(rag_adapter=FakeAdapter(), max_queries=2).run(_agent_input("Medium Risk"))
    assert result.working_memory is not None
    assert any(item.evidence_bundle_ids for item in result.working_memory.feature_assessments)
    assert any(item.finding_ids or item.action_item_ids for item in result.working_memory.feature_assessments)


@pytest.mark.asyncio
async def test_action_items_are_deduplicated() -> None:
    class SameAdapter:
        async def retrieve_evidence(self, query: str, *, scenario_context=None):
            from uav_risk.stage2.contracts import EvidenceBundle, EvidenceCitation, EvidenceOrigin, EvidenceSourceType

            citation = EvidenceCitation(
                citation_id="c-dup",
                source_id="src-dup",
                source_title="AC_107-2A",
                source_type=EvidenceSourceType.INTERNAL_DOC,
                origin=EvidenceOrigin.LOCAL_DOCUMENT,
                quote="same quote",
            )
            return EvidenceBundle(
                bundle_id="b-dup",
                query="AC 107-2A airspace authorization controlled airspace small UAS operation",
                claims=[],
                citations=[citation],
                support_status=EvidenceSupportStatus.SUPPORTED,
                confidence=0.8,
                metadata={"source_intent": "ac107", "query_id": "q-dup", "derived_from": "scenario"},
            )

    result = await OperationalAgentV2(rag_adapter=SameAdapter(), max_queries=2).run(_agent_input("Medium Risk"))
    summaries = [item.summary for item in result.action_items]
    assert len(summaries) == len(set(" ".join(s.lower().split()) for s in summaries))


@pytest.mark.asyncio
async def test_tool_trace_includes_feature_risk_assessor() -> None:
    result = await OperationalAgentV2(rag_adapter=None).run(_agent_input("Medium Risk"))
    names = {item.tool_name.value for item in result.tool_trace}
    assert "feature_risk_assessor" in names



@pytest.mark.asyncio
async def test_agent_requested_retrieval_is_bounded_and_tagged() -> None:
    calls: list[str] = []

    class FakeAdapter:
        async def retrieve_evidence(self, query: str, *, scenario_context=None, retrieval_origin=None):
            from uav_risk.stage2.contracts import EvidenceBundle, EvidenceCitation, EvidenceOrigin, EvidenceSourceType

            calls.append(query)
            citation = EvidenceCitation(
                citation_id=f"c-{len(calls)}",
                source_id="src",
                source_title="AC_107-2A",
                source_type=EvidenceSourceType.INTERNAL_DOC,
                origin=EvidenceOrigin.LOCAL_DOCUMENT,
                quote="bounded retrieval quote with provenance and enough operational detail.",
                metadata={"source_filename": "AC_107-2A.pdf", "page_start": 2},
            )
            return EvidenceBundle(
                bundle_id=f"b-{len(calls)}",
                query=query,
                claims=[],
                citations=[citation],
                support_status=EvidenceSupportStatus.SUPPORTED,
                confidence=0.7,
                metadata={"retrieval_origin": retrieval_origin or "agent_requested"},
            )

    result = await OperationalAgentV2(rag_adapter=FakeAdapter(), max_queries=5, max_agent_queries=1).run(_agent_input("Medium Risk"))
    assert len(calls) <= 1
    tagged = [b for b in result.evidence_bundles if isinstance(b.metadata, dict) and b.metadata.get("retrieval_origin") == "agent_requested"]
    assert tagged
