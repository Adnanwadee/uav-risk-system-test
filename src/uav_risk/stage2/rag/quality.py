from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from uav_risk.stage2.contracts import EvidenceSupportStatus, Stage2Error, Stage2Status
from uav_risk.stage2.rag.adapter import Stage2RAGAdapter

JsonScalar = str | int | float | bool | None


class _AsyncEmbedderFromLangChain:
    def __init__(self, model):
        self.model = model

    async def embed(self, text: str) -> list[float]:
        import asyncio

        loop = asyncio.get_event_loop()
        vec = await loop.run_in_executor(None, self.model.embed_query, text)
        return list(vec)





def _collect_corpus_coverage_status(index_dir: Path) -> dict[str, JsonScalar]:
    metadata_path = index_dir / "metadata.json"
    dense_mapping_path = index_dir / "dense_mapping.json"

    expected_ids: list[str] = []
    expected_titles: list[str] = []
    expected_source_count: int | None = None

    try:
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(metadata, dict):
                docs = metadata.get("source_documents")
                if isinstance(docs, list):
                    for item in docs:
                        if not isinstance(item, dict):
                            continue
                        sid = str(item.get("source_id") or "").strip()
                        if sid and sid not in expected_ids:
                            expected_ids.append(sid)
                        title = str(item.get("filename") or item.get("source_title") or sid).strip()
                        if title and title not in expected_titles:
                            expected_titles.append(title)
                source_count = metadata.get("source_count")
                if isinstance(source_count, (int, float)):
                    expected_source_count = int(source_count)
    except Exception:
        pass

    indexed_ids: list[str] = []
    indexed_titles: list[str] = []
    try:
        if dense_mapping_path.exists():
            mapping = json.loads(dense_mapping_path.read_text(encoding="utf-8"))
            if isinstance(mapping, dict):
                chunks = mapping.get("chunks")
                if isinstance(chunks, list):
                    for item in chunks:
                        if not isinstance(item, dict):
                            continue
                        sid = str(item.get("source_id") or "").strip()
                        if sid and sid not in indexed_ids:
                            indexed_ids.append(sid)
                        title = str(item.get("source_title") or item.get("source_filename") or sid).strip()
                        if title and title not in indexed_titles:
                            indexed_titles.append(title)
    except Exception:
        pass

    indexed_source_count = len(indexed_ids) if indexed_ids else None
    missing_sources: list[str] = []
    if expected_ids and indexed_ids:
        missing_sources = sorted(set(expected_ids) - set(indexed_ids))

    coverage_status = "unknown"
    if expected_source_count is not None:
        idx_count = indexed_source_count or 0
        if expected_source_count > 0 and idx_count >= expected_source_count and not missing_sources:
            coverage_status = "complete"
        elif expected_source_count > 0:
            coverage_status = "partial"

    return {
        "corpus_coverage_status": coverage_status,
        "expected_source_count": expected_source_count,
        "indexed_source_count": indexed_source_count,
        "missing_sources": ",".join(missing_sources[:64]),
        "source_ids": ",".join((indexed_ids or expected_ids)[:64]),
        "source_titles": ",".join((indexed_titles or expected_titles)[:64]),
    }


def _build_local_reranker_if_available(config_module) -> tuple[object | None, dict[str, JsonScalar]]:
    configured = bool(getattr(config_module, "USE_RERANKER", True))
    model_path_raw = getattr(config_module, "RERANKER_PATH", None)
    model_path = Path(str(model_path_raw)).expanduser().resolve() if model_path_raw else None

    status: dict[str, JsonScalar] = {
        "reranker_configured": configured,
        "reranker_available": False,
        "reranker_used": False,
        "reranker_reason": "disabled_by_configuration" if not configured else "reranker_not_configured",
    }

    if not configured:
        return None, status

    if model_path is None or not model_path.exists():
        status["reranker_reason"] = "reranker_model_missing"
        return None, status

    try:
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder(str(model_path), device="cpu")
        status["reranker_available"] = True
        status["reranker_reason"] = "reranker_model_loaded"
        return reranker, status
    except Exception:
        status["reranker_reason"] = "reranker_init_failed"
        return None, status


class RAGQualityQuery(BaseModel):
    query_id: str
    query: str
    expected_source_keywords: list[str] = Field(default_factory=list)
    minimum_citations: int = 1
    should_have_evidence: bool = True
    critical: bool = True


class RAGQualityCaseResult(BaseModel):
    query_id: str
    support_status: EvidenceSupportStatus
    citation_count: int
    matched_expected_source: bool
    citation_ids: list[str] = Field(default_factory=list)
    source_titles: list[str] = Field(default_factory=list)
    citation_provenance_complete: bool = False
    warnings: list[str] = Field(default_factory=list)


class RAGQualityReport(BaseModel):
    status: Stage2Status
    cases: list[RAGQualityCaseResult] = Field(default_factory=list)
    errors: list[Stage2Error] = Field(default_factory=list)
    metadata: dict[str, JsonScalar] = Field(default_factory=dict)


def build_default_rag_quality_queries() -> list[RAGQualityQuery]:
    return [
        RAGQualityQuery(
            query_id="part107_remote_pilot",
            query="Part 107 remote pilot small UAS operating rules",
            expected_source_keywords=["14 cfr part 107", "ac_107", "ac 107"],
            should_have_evidence=True,
            critical=True,
        ),
        RAGQualityQuery(
            query_id="part107_vlos",
            query="Part 107 visual line of sight small unmanned aircraft operation",
            expected_source_keywords=["14 cfr part 107", "ac_107", "ac 107"],
            should_have_evidence=True,
            critical=True,
        ),
        RAGQualityQuery(
            query_id="ac107_advisory",
            query="AC 107-2A advisory circular small UAS guidance",
            expected_source_keywords=["ac_107-2a", "ac 107"],
            should_have_evidence=True,
            critical=True,
        ),
        RAGQualityQuery(
            query_id="sora_ground_risk",
            query="SORA ground risk class operational volume adjacent area",
            expected_source_keywords=["sora"],
            should_have_evidence=True,
            critical=True,
        ),
        RAGQualityQuery(
            query_id="sora_annex_a",
            query="SORA Annex A operational safety objectives",
            expected_source_keywords=["annex a", "annex-a", "sora"],
            should_have_evidence=True,
            critical=True,
        ),
        RAGQualityQuery(
            query_id="sora_annex_b",
            query="SORA Annex B integrity and assurance levels",
            expected_source_keywords=["annex b", "annex-b", "sora"],
            should_have_evidence=True,
            critical=True,
        ),
        RAGQualityQuery(
            query_id="sora_annex_e",
            query="SORA Annex E containment requirements",
            expected_source_keywords=["annex e", "annex-e", "sora"],
            should_have_evidence=True,
            critical=True,
        ),
        RAGQualityQuery(
            query_id="special_condition_medium_risk",
            query="special condition UAS medium risk operational limitations",
            expected_source_keywords=["special_condition", "special condition"],
            should_have_evidence=True,
            critical=True,
        ),
        RAGQualityQuery(
            query_id="ear_export_control",
            query="EAR unmanned aircraft systems export control",
            expected_source_keywords=["ear", "d0593e"],
            should_have_evidence=True,
            critical=True,
        ),
        RAGQualityQuery(
            query_id="cooking_recipe",
            query="how to bake sourdough bread at home",
            expected_source_keywords=[],
            minimum_citations=0,
            should_have_evidence=False,
            critical=False,
        ),
        RAGQualityQuery(
            query_id="stock_market",
            query="best stock market investment strategy for next week",
            expected_source_keywords=[],
            minimum_citations=0,
            should_have_evidence=False,
            critical=False,
        ),
        RAGQualityQuery(
            query_id="medical",
            query="diagnose chest pain and prescribe medication",
            expected_source_keywords=[],
            minimum_citations=0,
            should_have_evidence=False,
            critical=False,
        ),
        RAGQualityQuery(
            query_id="unrelated_vehicle",
            query="how to replace brake pads on a passenger car",
            expected_source_keywords=[],
            minimum_citations=0,
            should_have_evidence=False,
            critical=False,
        ),
    ]


def _contains_expected_keyword(
    titles: list[str],
    source_ids: list[str],
    metadata_strings: list[str],
    keywords: list[str],
) -> bool:
    if not keywords:
        return True
    haystack = " ".join(titles + source_ids + metadata_strings).lower()
    return any(keyword.lower() in haystack for keyword in keywords)


def _citation_provenance_complete(bundle) -> bool:
    if not bundle.citations:
        return False
    for c in bundle.citations:
        if not (c.source_id and c.source_title and c.quote and c.chunk_id):
            return False
        sf = c.metadata.get("source_filename") if isinstance(c.metadata, dict) else None
        if not isinstance(sf, str) or not sf.strip():
            return False
        page_start = c.metadata.get("page_start") if isinstance(c.metadata, dict) else None
        if page_start is not None:
            try:
                if int(page_start) < 1:
                    return False
            except Exception:
                return False
    return True


async def evaluate_rag_adapter_quality(
    rag_adapter: Stage2RAGAdapter,
    queries: list[RAGQualityQuery] | None = None,
    *,
    provenance_status: str | None = None,
    extra_metadata: dict[str, JsonScalar] | None = None,
) -> RAGQualityReport:
    test_queries = queries or build_default_rag_quality_queries()
    cases: list[RAGQualityCaseResult] = []
    errors: list[Stage2Error] = []
    warning_count = 0

    for item in test_queries:
        warnings: list[str] = []
        try:
            bundle = await rag_adapter.retrieve_evidence(item.query)
            citation_ids = [citation.citation_id for citation in bundle.citations]
            source_titles = [citation.source_title for citation in bundle.citations]
            source_ids = [citation.source_id for citation in bundle.citations]
            metadata_strings: list[str] = []
            for citation in bundle.citations:
                for value in citation.metadata.values():
                    if isinstance(value, (str, int, float, bool)):
                        metadata_strings.append(str(value))

            matched = _contains_expected_keyword(
                source_titles,
                source_ids,
                metadata_strings,
                item.expected_source_keywords,
            )
            prov_ok = _citation_provenance_complete(bundle) if bundle.support_status == EvidenceSupportStatus.SUPPORTED else True

            if item.should_have_evidence:
                if bundle.support_status == EvidenceSupportStatus.INSUFFICIENT_EVIDENCE:
                    warnings.append("Expected evidence but received insufficient evidence.")
                if len(bundle.citations) < item.minimum_citations:
                    warnings.append("Citation count is below minimum expectation.")
                if item.expected_source_keywords and not matched:
                    warnings.append("Expected source keyword was not matched.")
                if not prov_ok:
                    warnings.append("Citation provenance incomplete.")
            else:
                if bundle.support_status != EvidenceSupportStatus.INSUFFICIENT_EVIDENCE:
                    warnings.append("Unsupported query unexpectedly returned supported evidence.")

            warning_count += len(warnings)
            cases.append(
                RAGQualityCaseResult(
                    query_id=item.query_id,
                    support_status=bundle.support_status,
                    citation_count=len(bundle.citations),
                    matched_expected_source=matched,
                    citation_ids=citation_ids,
                    source_titles=source_titles,
                    citation_provenance_complete=prov_ok,
                    warnings=warnings,
                )
            )
        except Exception:
            errors.append(
                Stage2Error(
                    code="rag_quality_case_failed",
                    message=f"RAG quality evaluation failed for query_id={item.query_id}.",
                    details={"query_id": item.query_id},
                )
            )
            cases.append(
                RAGQualityCaseResult(
                    query_id=item.query_id,
                    support_status=EvidenceSupportStatus.INSUFFICIENT_EVIDENCE,
                    citation_count=0,
                    matched_expected_source=False,
                    citation_ids=[],
                    source_titles=[],
                    citation_provenance_complete=False,
                    warnings=["Adapter call failed for this query."],
                )
            )
            warning_count += 1

    expected_cases = [q for q in test_queries if q.should_have_evidence]
    unsupported_cases = [q for q in test_queries if not q.should_have_evidence]
    case_map = {c.query_id: c for c in cases}

    supported_expected_cases = 0
    matched_expected_source_cases = 0
    citation_provenance_complete_cases = 0
    critical_supported_ok = True

    for q in expected_cases:
        case = case_map.get(q.query_id)
        if not case:
            continue
        if case.support_status == EvidenceSupportStatus.SUPPORTED and case.citation_count >= q.minimum_citations:
            supported_expected_cases += 1
            if case.matched_expected_source:
                matched_expected_source_cases += 1
            if case.citation_provenance_complete:
                citation_provenance_complete_cases += 1
        if q.critical:
            if not (
                case.support_status == EvidenceSupportStatus.SUPPORTED
                and case.matched_expected_source
                and case.citation_provenance_complete
            ):
                critical_supported_ok = False

    unsupported_correct_cases = 0
    for q in unsupported_cases:
        case = case_map.get(q.query_id)
        if case and case.support_status == EvidenceSupportStatus.INSUFFICIENT_EVIDENCE:
            unsupported_correct_cases += 1

    supported_total_cases = sum(1 for c in cases if c.support_status == EvidenceSupportStatus.SUPPORTED)
    insufficient_cases = sum(1 for c in cases if c.support_status == EvidenceSupportStatus.INSUFFICIENT_EVIDENCE)

    retrieval_usable = supported_expected_cases > 0 and critical_supported_ok

    quality_is_proven = bool(
        not errors
        and expected_cases
        and supported_expected_cases == len(expected_cases)
        and matched_expected_source_cases == len(expected_cases)
        and citation_provenance_complete_cases == len(expected_cases)
        and unsupported_correct_cases == len(unsupported_cases)
        and warning_count == 0
    )

    status = Stage2Status.COMPLETED if quality_is_proven else Stage2Status.DEGRADED

    metadata: dict[str, JsonScalar] = {
        "query_count": len(test_queries),
        "total_cases": len(cases),
        "supported_total_cases": supported_total_cases,
        "insufficient_cases": insufficient_cases,
        "expected_evidence_cases": len(expected_cases),
        "supported_expected_cases": supported_expected_cases,
        "unsupported_total_cases": len(unsupported_cases),
        "unsupported_correct_cases": unsupported_correct_cases,
        "matched_expected_source_cases": matched_expected_source_cases,
        "citation_provenance_complete_cases": citation_provenance_complete_cases,
        "warning_count": warning_count,
        "critical_supported_ok": critical_supported_ok,
        "retrieval_usable": retrieval_usable,
        "rag_quality_is_proven": quality_is_proven,
        "quality_is_proven": quality_is_proven,
        "quality_is_provisional": provenance_status not in {None, "current"},
    }

    if provenance_status is not None:
        metadata["provenance_status"] = provenance_status
    if extra_metadata:
        metadata.update(extra_metadata)

    return RAGQualityReport(status=status, cases=cases, errors=errors, metadata=metadata)


def build_runtime_rag_adapter_if_available() -> Stage2RAGAdapter | None:
    docs_dir = Path("src/uav_risk/stage2/docs")
    models_dir = Path("src/uav_risk/stage2/knowledge/models")
    if not docs_dir.exists() or not models_dir.exists():
        return None

    try:
        import pickle

        import uav_risk.stage2.rag.config_v3 as config_v3
        from langchain_huggingface import HuggingFaceEmbeddings

        from uav_risk.stage2.rag.config_v3 import EMBEDDING_PATH, get_index_dir, get_sparse_index_path
        from uav_risk.stage2.rag.rag_core_v3 import AsyncRAGCoreV3

        reranker, reranker_status = _build_local_reranker_if_available(config_v3)
        index_dir = get_index_dir()
        sparse_path = get_sparse_index_path()

        sparse_index = None
        if sparse_path.exists():
            with sparse_path.open("rb") as f:
                sparse_index = pickle.load(f)

        embedding_model_dir = Path(EMBEDDING_PATH).expanduser().resolve()
        if not embedding_model_dir.exists():
            return None

        embedder = _AsyncEmbedderFromLangChain(
            HuggingFaceEmbeddings(
                model_name=str(embedding_model_dir),
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        )

        rag_core = AsyncRAGCoreV3(
            config_module=config_v3,
            embedder=embedder,
            reranker=reranker,
            sparse_index_builder=None,
            index_dir=str(index_dir),
        )

        if sparse_index is not None:
            rag_core.sparse_builder = None
            rag_core._preloaded_sparse_index = sparse_index  # type: ignore[attr-defined]

        rag_core._runtime_status = {  # type: ignore[attr-defined]
            **reranker_status,
            **_collect_corpus_coverage_status(index_dir),
            "index_dir": str(index_dir),
            "sparse_index_loaded": bool(sparse_index is not None),
        }

        adapter = Stage2RAGAdapter(rag_core=rag_core)
        adapter.runtime_status = dict(rag_core._runtime_status)  # type: ignore[attr-defined]
        return adapter
    except Exception:
        return None
