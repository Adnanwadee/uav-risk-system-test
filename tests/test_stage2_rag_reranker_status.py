from __future__ import annotations

from types import SimpleNamespace

import pytest

from uav_risk.stage2.rag.hybrid_retriever import HybridRetriever, RetrievedDocument


class FakeReranker:
    def __init__(self) -> None:
        self.called = False

    def predict(self, pairs):
        self.called = True
        class _Out(list):
            def tolist(self):
                return list(self)
        return _Out([0.9 for _ in pairs])


@pytest.mark.asyncio
async def test_reranker_is_invoked_when_available_and_enabled() -> None:
    retriever = HybridRetriever(
        reranker=FakeReranker(),
        config=SimpleNamespace(USE_RERANKER=True, DENSE_WEIGHT=0.6, SPARSE_WEIGHT=0.4),
    )
    docs = [
        RetrievedDocument(doc_id="d1", text="a" * 80, source="s", source_id="sid", source_filename="f", source_title="t", chunk_id="c1", final_score=0.2, rrf_score=0.2),
        RetrievedDocument(doc_id="d2", text="b" * 80, source="s", source_id="sid", source_filename="f", source_title="t", chunk_id="c2", final_score=0.3, rrf_score=0.3),
    ]

    reranked = await retriever.rerank("uav query", docs)
    status = retriever.get_reranker_status()

    assert reranked
    assert retriever.reranker.called is True
    assert status["reranker_configured"] is True
    assert status["reranker_available"] is True
    assert status["reranker_used"] is True
    assert status["reranker_reason"] == "reranker_invoked"


@pytest.mark.asyncio
async def test_reranker_status_reports_unavailable_when_missing() -> None:
    retriever = HybridRetriever(
        reranker=None,
        config=SimpleNamespace(USE_RERANKER=True, DENSE_WEIGHT=0.6, SPARSE_WEIGHT=0.4),
    )
    docs = [
        RetrievedDocument(doc_id="d1", text="a" * 80, source="s", source_id="sid", source_filename="f", source_title="t", chunk_id="c1", final_score=0.2, rrf_score=0.2),
    ]

    _ = await retriever.rerank("uav query", docs)
    status = retriever.get_reranker_status()

    assert status["reranker_configured"] is True
    assert status["reranker_available"] is False
    assert status["reranker_used"] is False
    assert status["reranker_reason"] == "reranker_not_available"
