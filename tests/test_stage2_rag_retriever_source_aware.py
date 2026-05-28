from __future__ import annotations

from uav_risk.stage2.rag.hybrid_retriever import HybridRetriever, RetrievedDocument


def test_source_intent_detector_detects_part107() -> None:
    intent = HybridRetriever.detect_source_intent("Part 107 remote pilot rules")
    assert intent["explicit_source_intent"] is True
    assert intent["intent_name"] == "part107"


def test_source_intent_detector_detects_sora() -> None:
    intent = HybridRetriever.detect_source_intent("SORA ground risk operational volume")
    assert intent["explicit_source_intent"] is True
    assert "sora" in intent["intent_name"]


def test_source_intent_detector_detects_sora_annexes() -> None:
    assert HybridRetriever.detect_source_intent("SORA Annex A requirements")["intent_name"] == "sora_annex_a"
    assert HybridRetriever.detect_source_intent("SORA Annex B requirements")["intent_name"] == "sora_annex_b"
    assert HybridRetriever.detect_source_intent("SORA Annex E requirements")["intent_name"] == "sora_annex_e"
    assert HybridRetriever.detect_source_intent("SORA Annex F requirements")["intent_name"] == "sora_annex_f"


def test_source_intent_detector_detects_special_condition() -> None:
    intent = HybridRetriever.detect_source_intent("special condition uas medium risk")
    assert intent["intent_name"] == "special_condition"


def test_source_intent_detector_detects_ear_export() -> None:
    intent = HybridRetriever.detect_source_intent("EAR export control unmanned aircraft systems")
    assert intent["intent_name"] == "ear_export"


def test_source_intent_detector_marks_unrelated_domain_false() -> None:
    intent = HybridRetriever.detect_source_intent("bake sourdough bread")
    assert intent["domain_match"] is False


def _doc(source_id: str, score: float, page: int) -> RetrievedDocument:
    return RetrievedDocument(
        doc_id=f"{source_id}_{page}",
        text="A sufficiently long text quote for retrieval validation purposes only.",
        source=source_id,
        source_id=source_id,
        source_filename=f"{source_id}.pdf",
        source_title=source_id,
        page_start=page,
        page_end=page,
        chunk_id=f"chunk_{source_id}_{page}",
        final_score=score,
        source_match_score=0.0,
        provenance_complete=True,
    )


def test_anti_collapse_limits_same_source_duplicates_for_generic_queries() -> None:
    r = HybridRetriever()
    docs = [_doc("s1", 0.9, 1), _doc("s1", 0.8, 2), _doc("s1", 0.7, 3), _doc("s2", 0.6, 1)]
    out = r._apply_diversity(docs, {"explicit_source_intent": False}, top_k=4)
    assert len([d for d in out if d.source_id == "s1"]) <= 2


def test_explicit_source_intent_not_harmed_by_diversity() -> None:
    r = HybridRetriever()
    docs = [_doc("part107", 0.9, 1), _doc("part107", 0.8, 2), _doc("part107", 0.7, 3), _doc("other", 0.6, 1)]
    out = r._apply_diversity(docs, {"explicit_source_intent": True}, top_k=3)
    assert len(out) == 3
    assert all(d.source_id == "part107" for d in out[:2])


def test_source_intent_detector_detects_scenario_airspace_guidance() -> None:
    intent = HybridRetriever.detect_source_intent("UAS operation controlled airspace restricted area no-fly zone guidance")
    assert intent["intent_name"] == "scenario_airspace"
    assert intent["explicit_source_intent"] is True


def test_source_intent_detector_detects_scenario_weather_guidance() -> None:
    intent = HybridRetriever.detect_source_intent("UAS operation wind conditions flight risk guidance")
    assert intent["intent_name"] == "scenario_weather"
    assert intent["explicit_source_intent"] is True


def test_source_intent_detector_detects_scenario_medium_risk_guidance() -> None:
    intent = HybridRetriever.detect_source_intent("uav operational guidance for risk class Medium Risk")
    assert intent["intent_name"] in {"scenario_medium_risk", "special_condition"}
    assert intent["explicit_source_intent"] is True
