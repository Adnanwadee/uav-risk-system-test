from __future__ import annotations

import inspect

import pytest

from uav_risk.stage2.contracts import EvidenceSupportStatus
from uav_risk.stage2.rag.quality import (
    build_default_rag_quality_queries,
    evaluate_rag_adapter_quality,
)


class _FakeAdapter:
    async def retrieve_evidence(self, query: str):
        from uav_risk.stage2.contracts import (
            EvidenceBundle,
            EvidenceCitation,
            EvidenceOrigin,
            EvidenceSourceType,
            make_insufficient_evidence_bundle,
        )

        q = query.lower()
        if any(x in q for x in ["sourdough", "stock market", "chest pain", "brake pads"]):
            return make_insufficient_evidence_bundle(query, "outside domain")

        if "part 107" in q:
            title = "14 CFR Part 107 (up to date)"
            sid = "14_cfr_part_107"
        elif "ac 107" in q or "advisory circular" in q:
            title = "AC_107-2A"
            sid = "ac_107_2a"
        elif "annex a" in q:
            title = "SORA-v2.5-Annex-A"
            sid = "sora_annex_a"
        elif "annex b" in q:
            title = "SORA-v2.5-Annex-B"
            sid = "sora_annex_b"
        elif "annex e" in q:
            title = "SORA-v2.5-Annex-E"
            sid = "sora_annex_e"
        elif "sora" in q:
            title = "SORA-v2.5-Main-Body"
            sid = "sora_main"
        elif "special condition" in q:
            title = "special_condition_sc_light-uas_medium_risk_01"
            sid = "special_condition"
        elif "ear" in q or "export" in q:
            title = "D0593E_2024-07-10_06.26.37_EAR-for-Unmanned-Aircraft-Systems"
            sid = "d0593e_ear"
        else:
            return make_insufficient_evidence_bundle(query, "unknown")

        c = EvidenceCitation(
            citation_id=f"c_{sid}",
            source_id=sid,
            source_title=title,
            source_type=EvidenceSourceType.INTERNAL_DOC,
            origin=EvidenceOrigin.RETRIEVAL_SYSTEM,
            page=2,
            chunk_id=f"chunk_{sid}",
            quote="Retrieved UAV regulatory evidence quote with sufficient detail.",
            metadata={"source_filename": f"{title}.pdf", "page_start": 2, "page_end": 2},
        )
        return EvidenceBundle(
            bundle_id=f"b_{sid}",
            query=query,
            claims=[],
            citations=[c],
            support_status=EvidenceSupportStatus.SUPPORTED,
            confidence=0.8,
        )


def test_default_queries_non_empty() -> None:
    queries = build_default_rag_quality_queries()
    assert queries


def test_default_queries_include_supported_and_unsupported() -> None:
    queries = build_default_rag_quality_queries()
    assert any(q.should_have_evidence for q in queries)
    assert any(not q.should_have_evidence for q in queries)


@pytest.mark.asyncio
async def test_quality_harness_requires_unsupported_cases_insufficient() -> None:
    report = await evaluate_rag_adapter_quality(_FakeAdapter(), queries=build_default_rag_quality_queries())  # type: ignore[arg-type]
    assert report.metadata["unsupported_correct_cases"] == report.metadata["unsupported_total_cases"]


@pytest.mark.asyncio
async def test_quality_harness_requires_source_match_for_critical_supported_cases() -> None:
    report = await evaluate_rag_adapter_quality(_FakeAdapter(), queries=build_default_rag_quality_queries())  # type: ignore[arg-type]
    assert report.metadata["matched_expected_source_cases"] == report.metadata["expected_evidence_cases"]


@pytest.mark.asyncio
async def test_quality_harness_tracks_citation_provenance_complete_cases() -> None:
    report = await evaluate_rag_adapter_quality(_FakeAdapter(), queries=build_default_rag_quality_queries())  # type: ignore[arg-type]
    assert report.metadata["citation_provenance_complete_cases"] == report.metadata["expected_evidence_cases"]


@pytest.mark.asyncio
async def test_quality_metadata_fields_present() -> None:
    report = await evaluate_rag_adapter_quality(_FakeAdapter(), queries=build_default_rag_quality_queries())  # type: ignore[arg-type]
    for key in [
        "retrieval_usable",
        "quality_is_proven",
        "supported_expected_cases",
        "supported_total_cases",
        "unsupported_correct_cases",
        "unsupported_total_cases",
        "matched_expected_source_cases",
        "citation_provenance_complete_cases",
    ]:
        assert key in report.metadata


@pytest.mark.asyncio
async def test_quality_is_proven_true_when_all_cases_meet_expectations() -> None:
    report = await evaluate_rag_adapter_quality(_FakeAdapter(), queries=build_default_rag_quality_queries())  # type: ignore[arg-type]
    assert report.metadata["quality_is_proven"] is True


def test_quality_module_no_groq_or_llm_imports() -> None:
    import uav_risk.stage2.rag.quality as module

    source = inspect.getsource(module)
    assert "groq" not in source.lower()
    assert "report_writer" not in source.lower()
