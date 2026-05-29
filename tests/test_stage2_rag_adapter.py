from __future__ import annotations

from dataclasses import dataclass

import pytest

from uav_risk.stage2.contracts import EvidenceBundle, EvidenceOrigin, EvidenceSupportStatus
from uav_risk.stage2.rag.adapter import Stage2RAGAdapter
from uav_risk.stage2.rag.schemas import SearchResult


class FakeRAGRaises:
    async def search_scenario(self, **kwargs):
        raise RuntimeError("boom stacktrace internal error details")


class FakeRAGDict:
    async def search_scenario(self, **kwargs):
        return {
            "documents": [
                {
                    "citation_id": "c1",
                    "source_id": "doc-1",
                    "source_title": "14 CFR Part 107",
                    "source_filename": "14 CFR Part 107.pdf",
                    "chunk_id": "chunk_1",
                    "page_start": 2,
                    "text": "Retrieved passage text that is long enough for support gating in adapter.",
                    "final_score": 0.5,
                    "source_match_score": 0.8,
                    "origin": "retrieval_system",
                }
            ]
        }


class ObjResult:
    def __init__(self) -> None:
        self.documents = [
            {
                "doc_id": "doc-2",
                "chunk_id": "chunk_2",
                "source": "SORA-v2.5-Main-Body",
                "source_id": "sora_main",
                "source_filename": "SORA-v2.5-Main-Body-Release-JAR_doc_25.pdf",
                "page_start": 4,
                "excerpt": "Chunk excerpt from local corpus with enough words for quality validation.",
                "score": 0.7,
                "final_score": 0.7,
                "source_match_score": 0.8,
                "origin": "local_document",
            }
        ]


class FakeRAGObject:
    async def search_scenario(self, **kwargs):
        return ObjResult()


@dataclass
class ResultWithSearchResult:
    documents: list[SearchResult]


class FakeRAGSearchResultClass:
    async def search_scenario(self, **kwargs):
        return ResultWithSearchResult(
            documents=[
                SearchResult(
                    doc_id="doc-3",
                    text="SearchResult chunk text with enough detail for adapter support decision.",
                    source="AC_107-2A.pdf",
                    final_score=0.66,
                )
            ]
        )


@pytest.mark.asyncio
async def test_empty_query_raises_value_error() -> None:
    adapter = Stage2RAGAdapter(FakeRAGDict())
    with pytest.raises(ValueError):
        await adapter.retrieve_evidence("   ")


@pytest.mark.asyncio
async def test_max_claims_less_than_one_raises_value_error() -> None:
    adapter = Stage2RAGAdapter(FakeRAGDict())
    with pytest.raises(ValueError):
        await adapter.retrieve_evidence("query", max_claims=0)


@pytest.mark.asyncio
async def test_missing_rag_core_returns_insufficient_bundle() -> None:
    adapter = Stage2RAGAdapter(None)
    bundle = await adapter.retrieve_evidence("uav part 107 query")
    assert bundle.support_status == EvidenceSupportStatus.INSUFFICIENT_EVIDENCE
    assert bundle.no_evidence_reason == "RAG core is not configured."


@pytest.mark.asyncio
async def test_rag_core_exception_returns_insufficient_bundle() -> None:
    adapter = Stage2RAGAdapter(FakeRAGRaises())
    bundle = await adapter.retrieve_evidence("uav part 107 query")
    assert bundle.support_status == EvidenceSupportStatus.INSUFFICIENT_EVIDENCE
    assert bundle.claims == []
    assert bundle.citations == []


@pytest.mark.asyncio
async def test_rag_core_exception_reason_does_not_leak_stack_text() -> None:
    adapter = Stage2RAGAdapter(FakeRAGRaises())
    bundle = await adapter.retrieve_evidence("uav part 107 query")
    assert bundle.no_evidence_reason == "Evidence retrieval failed."
    assert "stacktrace" not in (bundle.no_evidence_reason or "").lower()


@pytest.mark.asyncio
async def test_empty_dict_like_result_returns_insufficient_bundle() -> None:
    class EmptyRAG:
        async def search_scenario(self, **kwargs):
            return {}

    adapter = Stage2RAGAdapter(EmptyRAG())
    bundle = await adapter.retrieve_evidence("uav query")
    assert bundle.support_status == EvidenceSupportStatus.INSUFFICIENT_EVIDENCE


@pytest.mark.asyncio
async def test_dict_like_result_returns_supported_bundle() -> None:
    adapter = Stage2RAGAdapter(FakeRAGDict())
    bundle = await adapter.retrieve_evidence("Part 107 remote pilot")
    assert bundle.support_status == EvidenceSupportStatus.SUPPORTED
    assert len(bundle.citations) == 1


@pytest.mark.asyncio
async def test_object_like_result_returns_supported_bundle() -> None:
    adapter = Stage2RAGAdapter(FakeRAGObject())
    bundle = await adapter.retrieve_evidence("SORA risk assessment")
    assert bundle.support_status == EvidenceSupportStatus.SUPPORTED


@pytest.mark.asyncio
async def test_real_schema_search_result_without_provenance_is_insufficient() -> None:
    adapter = Stage2RAGAdapter(FakeRAGSearchResultClass())
    bundle = await adapter.retrieve_evidence("AC 107 guidance")
    assert bundle.support_status == EvidenceSupportStatus.INSUFFICIENT_EVIDENCE


@pytest.mark.asyncio
async def test_citation_origin_is_only_retrieval_or_local_document() -> None:
    adapter = Stage2RAGAdapter(FakeRAGObject())
    bundle = await adapter.retrieve_evidence("SORA operation")
    origins = {citation.origin for citation in bundle.citations}
    assert origins.issubset({EvidenceOrigin.RETRIEVAL_SYSTEM, EvidenceOrigin.LOCAL_DOCUMENT})


@pytest.mark.asyncio
async def test_origin_llm_synthesis_candidate_is_skipped() -> None:
    class UnsafeRAG:
        async def search_scenario(self, **kwargs):
            return {
                "documents": [
                    {
                        "source_id": "doc",
                        "source_title": "Doc",
                        "source_filename": "Doc.pdf",
                        "chunk_id": "c",
                        "page_start": 1,
                        "text": "LLM-style quote with enough length but unsafe origin.",
                        "origin": "llm_synthesis",
                        "final_score": 0.6,
                    }
                ]
            }

    adapter = Stage2RAGAdapter(UnsafeRAG())
    bundle = await adapter.retrieve_evidence("uav query")
    assert bundle.support_status == EvidenceSupportStatus.INSUFFICIENT_EVIDENCE


@pytest.mark.asyncio
async def test_origin_hyde_generated_candidate_is_skipped() -> None:
    class UnsafeRAG:
        async def search_scenario(self, **kwargs):
            return {
                "documents": [
                    {
                        "source_id": "doc",
                        "source_title": "Doc",
                        "source_filename": "Doc.pdf",
                        "chunk_id": "c",
                        "page_start": 1,
                        "text": "HyDE text with enough length but unsafe origin for citation.",
                        "origin": "hyde_generated",
                        "final_score": 0.6,
                    }
                ]
            }

    adapter = Stage2RAGAdapter(UnsafeRAG())
    bundle = await adapter.retrieve_evidence("uav query")
    assert bundle.support_status == EvidenceSupportStatus.INSUFFICIENT_EVIDENCE


@pytest.mark.asyncio
async def test_candidate_without_quote_or_text_is_skipped() -> None:
    class NoQuoteRAG:
        async def search_scenario(self, **kwargs):
            return {
                "documents": [
                    {
                        "source_id": "doc-1",
                        "source_title": "Doc 1",
                        "source_filename": "doc1.pdf",
                        "chunk_id": "chunk_1",
                        "page_start": 1,
                        "origin": "retrieval_system",
                        "final_score": 0.6,
                    }
                ]
            }

    adapter = Stage2RAGAdapter(NoQuoteRAG())
    bundle = await adapter.retrieve_evidence("uav query")
    assert bundle.support_status == EvidenceSupportStatus.INSUFFICIENT_EVIDENCE


@pytest.mark.asyncio
async def test_supported_claim_always_has_at_least_one_citation() -> None:
    adapter = Stage2RAGAdapter(FakeRAGDict())
    bundle = await adapter.retrieve_evidence("Part 107 remote pilot")
    assert bundle.support_status == EvidenceSupportStatus.SUPPORTED
    assert all(len(claim.citations) >= 1 for claim in bundle.claims)


@pytest.mark.asyncio
async def test_duplicate_citation_ids_are_deduplicated() -> None:
    class DupRAG:
        async def search_scenario(self, **kwargs):
            return {
                "documents": [
                    {
                        "citation_id": "dup",
                        "source_id": "doc-1",
                        "source_title": "14 CFR Part 107",
                        "source_filename": "14 CFR Part 107.pdf",
                        "chunk_id": "c1",
                        "page_start": 1,
                        "text": "Retrieved evidence quote long enough for inclusion.",
                        "final_score": 0.6,
                        "source_match_score": 0.8,
                        "origin": "retrieval_system",
                    },
                    {
                        "citation_id": "dup",
                        "source_id": "doc-1",
                        "source_title": "14 CFR Part 107",
                        "source_filename": "14 CFR Part 107.pdf",
                        "chunk_id": "c1",
                        "page_start": 1,
                        "text": "Retrieved evidence quote long enough for inclusion.",
                        "final_score": 0.6,
                        "source_match_score": 0.8,
                        "origin": "retrieval_system",
                    },
                ]
            }

    adapter = Stage2RAGAdapter(DupRAG())
    bundle = await adapter.retrieve_evidence("Part 107 remote pilot")
    assert isinstance(bundle, EvidenceBundle)
    assert len(bundle.citations) == 1


@pytest.mark.asyncio
async def test_adapter_returns_insufficient_for_out_of_domain_query() -> None:
    adapter = Stage2RAGAdapter(FakeRAGDict())
    bundle = await adapter.retrieve_evidence("how to bake sourdough bread")
    assert bundle.support_status == EvidenceSupportStatus.INSUFFICIENT_EVIDENCE


@pytest.mark.asyncio
async def test_adapter_refuses_missing_provenance() -> None:
    class WeakRAG:
        async def search_scenario(self, **kwargs):
            return {
                "documents": [
                    {
                        "source_id": "",
                        "source_title": "x",
                        "text": "Some text with enough length to pass length gate alone.",
                        "final_score": 0.8,
                        "origin": "retrieval_system",
                    }
                ]
            }

    adapter = Stage2RAGAdapter(WeakRAG())
    bundle = await adapter.retrieve_evidence("Part 107 query")
    assert bundle.support_status == EvidenceSupportStatus.INSUFFICIENT_EVIDENCE


@pytest.mark.asyncio
async def test_citations_are_ranked_and_confidence_labeled_by_final_score() -> None:
    class RankedRAG:
        async def search_scenario(self, **kwargs):
            return {
                "documents": [
                    {
                        "citation_id": "c_low",
                        "source_id": "doc-low",
                        "source_title": "AC_107-2A",
                        "source_filename": "AC_107-2A.pdf",
                        "chunk_id": "chunk_low",
                        "page_start": 3,
                        "text": "Long enough low-score citation text for ranking validation in adapter output.",
                        "final_score": 0.32,
                        "dense_score": 0.31,
                        "sparse_score": 0.22,
                        "source_match_score": 0.2,
                        "origin": "retrieval_system",
                    },
                    {
                        "citation_id": "c_high",
                        "source_id": "doc-high",
                        "source_title": "14 CFR Part 107",
                        "source_filename": "14 CFR Part 107.pdf",
                        "chunk_id": "chunk_high",
                        "page_start": 1,
                        "text": "Long enough high-score citation text for ranking validation in adapter output.",
                        "final_score": 0.82,
                        "dense_score": 0.79,
                        "sparse_score": 0.61,
                        "source_match_score": 0.9,
                        "origin": "retrieval_system",
                    },
                ]
            }

    adapter = Stage2RAGAdapter(RankedRAG())
    bundle = await adapter.retrieve_evidence("Part 107 remote pilot")
    assert bundle.support_status == EvidenceSupportStatus.SUPPORTED
    assert len(bundle.citations) == 2
    first = bundle.citations[0]
    second = bundle.citations[1]

    assert first.citation_id == "c_high"
    assert first.metadata.get("rank") == 1
    assert second.metadata.get("rank") == 2

    assert first.metadata.get("confidence_label") in {"HIGH", "MEDIUM", "LOW", "VERY LOW"}
    assert second.metadata.get("confidence_label") in {"HIGH", "MEDIUM", "LOW", "VERY LOW"}
    assert first.metadata.get("retrieval_score") is not None
    assert first.metadata.get("top_score") is not None
    assert first.retrieval_score is not None


@pytest.mark.asyncio
async def test_bundle_metadata_includes_retrieval_origin_when_specified() -> None:
    adapter = Stage2RAGAdapter(FakeRAGDict())
    bundle = await adapter.retrieve_evidence("Part 107 remote pilot", retrieval_origin="agent_requested")
    assert bundle.support_status == EvidenceSupportStatus.SUPPORTED
    assert bundle.metadata.get("retrieval_origin") == "agent_requested"
    assert bundle.metadata.get("evidence_status") == "grounded"


@pytest.mark.asyncio
async def test_synthetic_only_candidates_mark_bundle_as_synthetic_only() -> None:
    class SyntheticOnlyRAG:
        async def search_scenario(self, **kwargs):
            return {
                "documents": [
                    {
                        "source_id": "doc",
                        "source_title": "Doc",
                        "source_filename": "Doc.pdf",
                        "chunk_id": "c",
                        "page_start": 1,
                        "text": "Synthetic style text with enough length to be considered but should never become grounded evidence.",
                        "origin": "hyde_generated",
                        "final_score": 0.92,
                    }
                ]
            }

    adapter = Stage2RAGAdapter(SyntheticOnlyRAG())
    bundle = await adapter.retrieve_evidence("uav query")
    assert bundle.support_status == EvidenceSupportStatus.INSUFFICIENT_EVIDENCE
    assert bundle.metadata.get("evidence_status") == "synthetic_only"
    assert bundle.metadata.get("synthetic") is True
