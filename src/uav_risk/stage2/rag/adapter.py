from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from uav_risk.stage2.contracts import (
    EvidenceBundle,
    EvidenceCitation,
    EvidenceClaim,
    EvidenceOrigin,
    EvidenceSourceType,
    EvidenceSupportStatus,
    EvidenceUse,
    collect_unique_citations,
    make_insufficient_evidence_bundle,
)
from uav_risk.stage2.rag.hybrid_retriever import HybridRetriever

JsonScalar = str | int | float | bool | None


class Stage2RAGAdapter:
    """Strict adapter that normalizes heterogeneous RAG outputs into EvidenceBundle."""

    def __init__(self, rag_core: Any | None = None) -> None:
        self._rag_core = rag_core

    async def retrieve_evidence(
        self,
        query: str,
        *,
        scenario_context: dict[str, JsonScalar] | None = None,
        max_claims: int = 3,
        retrieval_origin: str = "scenario_driven",
    ) -> EvidenceBundle:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must be non-empty")
        if max_claims < 1:
            raise ValueError("max_claims must be >= 1")

        intent = HybridRetriever.detect_source_intent(normalized_query)
        safe_origin = self._normalize_retrieval_origin(retrieval_origin)
        if not intent.get("domain_match", False):
            return self._insufficient_bundle_with_metadata(
                normalized_query,
                "Query appears outside UAV/regulatory corpus scope.",
                retrieval_origin=safe_origin,
                runtime_metadata={},
                evidence_status="unavailable",
            )

        if self._rag_core is None:
            return self._insufficient_bundle_with_metadata(
                normalized_query,
                "RAG core is not configured.",
                retrieval_origin=safe_origin,
                runtime_metadata={},
                evidence_status="unavailable",
            )

        try:
            raw_result = await self._call_rag_core(
                normalized_query, scenario_context=scenario_context
            )
        except Exception:
            return self._insufficient_bundle_with_metadata(
                normalized_query,
                "Evidence retrieval failed.",
                retrieval_origin=safe_origin,
                runtime_metadata={},
                evidence_status="unavailable",
            )

        runtime_metadata = self._extract_runtime_metadata(raw_result)
        candidates = self._iter_candidate_items(raw_result)
        citations: list[EvidenceCitation] = []
        supported_rows = 0
        saw_synthetic_candidate = False

        for idx, candidate in enumerate(candidates):
            mapping = self._to_mapping(candidate)
            origin = self._normalize_origin(self._safe_get(mapping, ("origin",), default=None))
            if origin in {EvidenceOrigin.LLM_SYNTHESIS, EvidenceOrigin.HYDE_GENERATED}:
                saw_synthetic_candidate = True
            if not self._candidate_passes_sufficiency(mapping, intent):
                continue
            citation = self._candidate_to_citation(mapping, idx)
            if citation is None:
                continue
            citations.append(citation)
            supported_rows += 1

        citations = self._sort_and_annotate_citations(citations)

        if not citations or supported_rows < 1:
            synthetic_only = bool(saw_synthetic_candidate and not citations)
            return self._insufficient_bundle_with_metadata(
                normalized_query,
                "No sufficient evidence candidates passed retrieval safety checks.",
                retrieval_origin=safe_origin,
                runtime_metadata=runtime_metadata,
                evidence_status="synthetic_only" if synthetic_only else "insufficient",
                synthetic=synthetic_only,
            )

        return self._make_supported_bundle(
            normalized_query,
            citations=citations,
            max_claims=max_claims,
            intent=intent,
            retrieval_origin=safe_origin,
            runtime_metadata=runtime_metadata,
        )

    async def _call_rag_core(
        self,
        query: str,
        *,
        scenario_context: dict[str, JsonScalar] | None,
    ) -> Any:
        context = scenario_context or {}
        if not context:
            context = {
                "operation_type": query,
                "regulatory_framework": query,
                "flight_mission": query,
            }

        search_method = getattr(self._rag_core, "search_scenario", None)
        if not callable(search_method):
            raise RuntimeError("RAG core does not expose search_scenario")

        return await search_method(
            core_features=context,
            optional_features=None,
            shap_features=None,
            free_text=query,
            ml_risk_score=None,
        )

    def _normalize_retrieval_origin(self, value: str | None) -> str:
        normalized = str(value or "").strip().lower().replace("-", "_")
        if normalized in {"scenario_driven", "agent_requested", "fallback"}:
            return normalized
        return "scenario_driven"

    def _extract_runtime_metadata(self, raw_result: Any) -> dict[str, JsonScalar]:
        mapping = self._to_mapping(raw_result)
        analysis = mapping.get("analysis") if isinstance(mapping, dict) else None
        if not isinstance(analysis, dict):
            return {}

        reranker_status = analysis.get("reranker_status")
        runtime_status = analysis.get("runtime_status")

        meta: dict[str, JsonScalar] = {}
        if isinstance(reranker_status, dict):
            for key in ("reranker_configured", "reranker_available", "reranker_used", "reranker_reason"):
                value = reranker_status.get(key)
                if isinstance(value, (str, int, float, bool)) or value is None:
                    meta[key] = value
        if isinstance(runtime_status, dict):
            for key in (
                "reranker_configured",
                "reranker_available",
                "reranker_used",
                "reranker_reason",
                "corpus_coverage_status",
                "expected_source_count",
                "indexed_source_count",
                "missing_sources",
                "source_ids",
                "source_titles",
            ):
                if key in meta:
                    continue
                value = runtime_status.get(key)
                if isinstance(value, (str, int, float, bool)) or value is None:
                    meta[key] = value
        return meta

    def _insufficient_bundle_with_metadata(
        self,
        query: str,
        reason: str,
        *,
        retrieval_origin: str,
        runtime_metadata: dict[str, JsonScalar],
        evidence_status: str,
        synthetic: bool = False,
    ) -> EvidenceBundle:
        bundle = make_insufficient_evidence_bundle(query, reason)
        merged_meta: dict[str, JsonScalar] = {
            "retrieval_origin": retrieval_origin,
            "evidence_status": evidence_status,
            "synthetic": bool(synthetic),
        }
        merged_meta.update(runtime_metadata)
        return bundle.model_copy(update={"metadata": merged_meta})

    def _safe_get(self, obj: Any, names: tuple[str, ...], default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            for name in names:
                if name in obj:
                    return obj[name]
            return default
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
        return default

    def _to_mapping(self, obj: Any) -> dict[str, Any]:
        if isinstance(obj, dict):
            return obj
        if is_dataclass(obj):
            return asdict(obj)
        dump = getattr(obj, "model_dump", None)
        if callable(dump):
            value = dump()
            if isinstance(value, dict):
                return value
        if hasattr(obj, "__dict__"):
            return {key: value for key, value in vars(obj).items() if not key.startswith("_")}
        return {}

    def _iter_candidate_items(self, result: Any) -> list[Any]:
        if result is None:
            return []
        if isinstance(result, list):
            return result
        mapping = self._to_mapping(result)
        if not mapping:
            return []

        container = self._safe_get(
            mapping,
            ("results", "documents", "docs", "chunks", "evidence", "citations", "retrieved_docs"),
            default=[],
        )
        if isinstance(container, list):
            return container
        if container is None:
            return []
        return [container]

    def _normalize_origin(self, value: Any) -> EvidenceOrigin:
        if isinstance(value, EvidenceOrigin):
            return value
        if value is None:
            return EvidenceOrigin.RETRIEVAL_SYSTEM
        normalized = str(value).strip().lower()
        for member in EvidenceOrigin:
            if normalized == member.value:
                return member
        return EvidenceOrigin.RETRIEVAL_SYSTEM

    def _normalize_score(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(1.0, score))

    def _derive_confidence_label(self, *, final_score: float | None, dense_score: float | None, sparse_score: float | None, source_match_score: float | None) -> str:
        fs = final_score or 0.0
        ds = dense_score or 0.0
        ss = sparse_score or 0.0
        sm = source_match_score or 0.0
        blended = 0.6 * fs + 0.2 * ds + 0.1 * ss + 0.1 * sm
        if blended >= 0.65:
            return "HIGH"
        if blended >= 0.45:
            return "MEDIUM"
        if blended >= 0.25:
            return "LOW"
        return "VERY LOW"

    def _sort_and_annotate_citations(self, citations: list[EvidenceCitation]) -> list[EvidenceCitation]:
        def score_of(c: EvidenceCitation) -> float:
            meta = c.metadata if isinstance(c.metadata, dict) else {}
            for key in ("final_score", "retrieval_score", "dense_score", "sparse_score"):
                v = self._normalize_score(meta.get(key))
                if v is not None:
                    return v
            return self._normalize_score(c.retrieval_score) or 0.0

        ordered = sorted(citations, key=score_of, reverse=True)
        out: list[EvidenceCitation] = []
        for idx, c in enumerate(ordered, start=1):
            meta = dict(c.metadata or {})
            final_score = self._normalize_score(meta.get("final_score"))
            dense_score = self._normalize_score(meta.get("dense_score"))
            sparse_score = self._normalize_score(meta.get("sparse_score"))
            source_match_score = self._normalize_score(meta.get("source_match_score"))
            retrieval_score = self._normalize_score(meta.get("retrieval_score"))
            if retrieval_score is None:
                retrieval_score = final_score or dense_score or sparse_score or 0.0

            meta["rank"] = idx
            meta["retrieval_score"] = retrieval_score
            meta["top_score"] = retrieval_score
            meta.setdefault("support_status", "grounded")
            meta.setdefault("synthetic", False)
            meta["confidence_label"] = self._derive_confidence_label(
                final_score=final_score,
                dense_score=dense_score,
                sparse_score=sparse_score,
                source_match_score=source_match_score,
            )

            out.append(
                c.model_copy(
                    update={
                        "retrieval_score": retrieval_score,
                        "metadata": meta,
                    }
                )
            )
        return out

    def _candidate_passes_sufficiency(self, candidate: dict[str, Any], intent: dict[str, Any]) -> bool:
        origin = self._normalize_origin(self._safe_get(candidate, ("origin",), default=None))
        if origin in {EvidenceOrigin.LLM_SYNTHESIS, EvidenceOrigin.HYDE_GENERATED}:
            return False

        quote_value = self._safe_get(candidate, ("quote", "text", "chunk_text", "excerpt", "passage", "content"), default=None)
        if not isinstance(quote_value, str) or len(quote_value.strip()) < 40:
            return False

        source_id = self._safe_get(candidate, ("source_id",), default=None)
        source_filename = self._safe_get(candidate, ("source_filename", "source_title", "source", "doc_id"), default=None)
        page_start = self._safe_get(candidate, ("page_start", "page", "page_number"), default=None)
        chunk_id = self._safe_get(candidate, ("chunk_id",), default=None)

        if not (isinstance(source_id, str) and source_id.strip()):
            return False
        if not (isinstance(source_filename, str) and source_filename.strip()):
            return False
        if not (isinstance(chunk_id, str) and chunk_id.strip()):
            return False
        if page_start is not None:
            try:
                if int(page_start) < 1:
                    return False
            except Exception:
                return False

        final_score = self._normalize_score(
            self._safe_get(candidate, ("final_score", "score", "retrieval_score"), default=None)
        )
        source_match = self._normalize_score(self._safe_get(candidate, ("source_match_score",), default=None))

        if final_score is None or final_score < 0.08:
            return False

        if intent.get("explicit_source_intent") and (source_match is None or source_match <= 0):
            return False

        return True

    def _candidate_to_citation(self, mapping: dict[str, Any], index: int) -> EvidenceCitation | None:
        origin = self._normalize_origin(self._safe_get(mapping, ("origin",), default=None))
        if origin in {EvidenceOrigin.LLM_SYNTHESIS, EvidenceOrigin.HYDE_GENERATED}:
            return None

        quote_value = self._safe_get(
            mapping,
            ("quote", "text", "chunk_text", "excerpt", "passage", "content"),
            default=None,
        )
        if not isinstance(quote_value, str) or not quote_value.strip():
            return None

        source_id = self._safe_get(mapping, ("source_id",), default=None)
        source_title = self._safe_get(mapping, ("source_title", "source_filename", "source", "doc_id"), default=None)
        if source_id is None or source_title is None:
            return None

        page_value = self._safe_get(mapping, ("page_start", "page", "page_number"), default=None)
        try:
            page = int(page_value) if page_value is not None else None
        except (TypeError, ValueError):
            page = None

        citation_id = self._safe_get(mapping, ("citation_id",), default=None)
        if not isinstance(citation_id, str) or not citation_id.strip():
            cid_source = self._safe_get(mapping, ("chunk_id", "doc_id", "source_id"), default=None)
            if isinstance(cid_source, str) and cid_source.strip():
                citation_id = f"cit_{cid_source.strip()}"
            else:
                citation_id = f"cit_{index}"

        metadata: dict[str, JsonScalar] = {}
        for key in (
            "source_filename",
            "source_id",
            "source_title",
            "chunk_id",
            "page_start",
            "page_end",
            "section_title",
            "text_sha256",
            "dense_score",
            "sparse_score",
            "rerank_score",
            "final_score",
            "retrieval_score",
            "retrieval_method",
            "source_match_score",
            "retrieval_origin",
            "synthetic",
            "support_status",
        ):
            value = mapping.get(key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                metadata[key] = value

        try:
            return EvidenceCitation(
                citation_id=str(citation_id).strip(),
                source_id=str(source_id).strip(),
                source_title=str(source_title).strip(),
                source_type=EvidenceSourceType.INTERNAL_DOC,
                origin=origin if origin in {EvidenceOrigin.LOCAL_DOCUMENT, EvidenceOrigin.RETRIEVAL_SYSTEM} else EvidenceOrigin.RETRIEVAL_SYSTEM,
                section=self._safe_get(mapping, ("section_title", "section"), default=None),
                page=page,
                chunk_id=self._safe_get(mapping, ("chunk_id",), default=None),
                quote=quote_value.strip(),
                retrieval_score=self._normalize_score(
                    self._safe_get(mapping, ("retrieval_score", "score", "final_score", "dense_score", "sparse_score"), default=None)
                ),
                rerank_score=self._normalize_score(self._safe_get(mapping, ("rerank_score",), default=None)),
                metadata=metadata,
            )
        except Exception:
            return None

    def _make_supported_bundle(
        self,
        query: str,
        *,
        citations: list[EvidenceCitation],
        max_claims: int,
        intent: dict[str, Any],
        retrieval_origin: str,
        runtime_metadata: dict[str, JsonScalar],
    ) -> EvidenceBundle:
        claim_count = min(max_claims, len(citations))
        claims: list[EvidenceClaim] = []
        top_retrieval_score = 0.0
        for idx in range(claim_count):
            citation = citations[idx]
            c_meta = citation.metadata if isinstance(citation.metadata, dict) else {}
            c_score = self._normalize_score(c_meta.get("retrieval_score"))
            if c_score is None:
                c_score = self._normalize_score(citation.retrieval_score) or 0.0
            top_retrieval_score = max(top_retrieval_score, c_score)
            claims.append(
                EvidenceClaim(
                    claim_id=f"claim_{idx + 1}",
                    claim=f"Retrieved evidence is available for the query: {query}",
                    support_status=EvidenceSupportStatus.SUPPORTED,
                    evidence_use=EvidenceUse.RETRIEVAL_CONTEXT,
                    citations=[citation],
                    confidence=c_score,
                    limitations=[],
                    conflicts=[],
                    metadata={
                        "intent_name": intent.get("intent_name"),
                        "retrieval_origin": retrieval_origin,
                        "evidence_status": "grounded",
                        "synthetic": False,
                    },
                )
            )

        unique_citations = collect_unique_citations(claims)
        bundle_meta: dict[str, JsonScalar] = {
            "intent_name": intent.get("intent_name"),
            "citation_count": len(unique_citations),
            "retrieval_origin": retrieval_origin,
            "evidence_status": "grounded",
            "synthetic": False,
        }
        bundle_meta.update(runtime_metadata)
        return EvidenceBundle(
            bundle_id=f"bundle_{abs(hash(query)) % 10_000_000}",
            query=query,
            claims=claims,
            citations=unique_citations,
            support_status=EvidenceSupportStatus.SUPPORTED,
            confidence=top_retrieval_score,
            no_evidence_reason=None,
            metadata=bundle_meta,
        )
