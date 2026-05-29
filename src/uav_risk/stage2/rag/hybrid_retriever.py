"""
Hybrid Retriever - Dense + Sparse + RRF + optional reranker with source-aware routing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .query_intelligence import AdaptiveRRF

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    doc_id: str
    text: str
    source: str
    source_id: str = ""
    source_filename: str = ""
    source_title: str = ""
    page_start: int | None = None
    page_end: int | None = None
    section_title: str | None = None
    chunk_id: str | None = None
    vector_id: int | None = None
    text_sha256: str | None = None
    retrieval_method: str = "hybrid"
    domain_match: bool = True
    source_match_score: float = 0.0
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    is_duplicate: bool = False
    chunk_hash: Optional[str] = None
    provenance_complete: bool = False


class SimHash:
    def __init__(self, hashbits: int = 64):
        self.hashbits = hashbits

    def _hash_func(self, token: str) -> int:
        return hash(token) & ((1 << self.hashbits) - 1)

    def compute(self, text: str) -> int:
        tokens = text.lower().split()
        v = [0] * self.hashbits
        for token in tokens:
            h = self._hash_func(token)
            for i in range(self.hashbits):
                bit = (h >> i) & 1
                v[i] += 1 if bit else -1
        fingerprint = 0
        for i in range(self.hashbits):
            if v[i] > 0:
                fingerprint |= 1 << i
        return fingerprint

    def hamming_distance(self, hash1: int, hash2: int) -> int:
        x = hash1 ^ hash2
        distance = 0
        while x:
            distance += 1
            x &= x - 1
        return distance

    def is_duplicate(self, text1: str, text2: str, threshold: int = 3) -> bool:
        h1 = self.compute(text1)
        h2 = self.compute(text2)
        return self.hamming_distance(h1, h2) <= threshold


class HybridRetriever:
    def __init__(
        self,
        dense_index=None,
        sparse_index=None,
        embedder=None,
        reranker=None,
        config=None,
        index_dir: Optional[str] = None,
    ):
        self.dense_index = dense_index
        self.sparse_index = sparse_index
        self.embedder = embedder
        self.reranker = reranker
        self.config = config
        self.index_dir = Path(index_dir).expanduser().resolve() if index_dir else None
        if self.index_dir is None and config is not None and getattr(config, "INDEX_DIR", None):
            self.index_dir = Path(config.INDEX_DIR).expanduser().resolve()

        self.simhash = SimHash(hashbits=64)
        self.dense_doc_ids: dict[int, str] = {}
        self.chunk_by_id: dict[str, dict[str, Any]] = {}
        self.sparse_lookup: dict[str, dict[str, Any]] = {}
        self._load_dense_mapping()
        self._load_sparse_lookup()

        self.dense_weight = getattr(config, "DENSE_WEIGHT", 0.6) if config else 0.6
        self.sparse_weight = getattr(config, "SPARSE_WEIGHT", 0.4) if config else 0.4
        self.use_reranker = getattr(config, "USE_RERANKER", True) if config else True
        self._reranker_status: dict[str, Any] = {
            "reranker_configured": bool(self.use_reranker),
            "reranker_available": bool(self.reranker is not None),
            "reranker_used": False,
            "reranker_reason": "configured_and_available" if self.use_reranker and self.reranker is not None else (
                "disabled_by_configuration" if not self.use_reranker else "reranker_not_available"
            ),
        }

        self.min_final_score = 0.08
        self.min_quote_chars = 40

    def _set_reranker_status(
        self,
        *,
        configured: bool,
        available: bool,
        used: bool,
        reason: str,
    ) -> None:
        self._reranker_status = {
            "reranker_configured": configured,
            "reranker_available": available,
            "reranker_used": used,
            "reranker_reason": reason,
        }

    def get_reranker_status(self) -> dict[str, Any]:
        return dict(self._reranker_status)

    @staticmethod
    def detect_source_intent(query: str) -> dict[str, Any]:
        q = query.lower().strip()
        if not q:
            return {
                "intent_name": "unknown",
                "preferred_source_patterns": [],
                "negative_source_patterns": [],
                "explicit_source_intent": False,
                "domain_match": False,
                "confidence": 0.0,
            }

        aviation_terms = [
            "uas",
            "uav",
            "drone",
            "part 107",
            "sora",
            "airspace",
            "vlos",
            "remote pilot",
            "no-fly",
            "export",
            "ear",
            "special condition",
        ]
        domain_match = any(t in q for t in aviation_terms)

        rules = [
            (
                "sora_annex_a",
                ["annex a"],
                ["annex-a", "annex a", "sora-v2.5-annex-a", "sora"],
            ),
            (
                "sora_annex_b",
                ["annex b"],
                ["annex-b", "annex b", "sora-v2.5-annex-b", "sora"],
            ),
            (
                "sora_annex_e",
                ["annex e"],
                ["annex-e", "annex e", "sora-v2.5-annex-e", "sora"],
            ),
            (
                "sora_annex_f",
                ["annex f"],
                ["annex-f", "annex f", "sora-v2.5-annex-f", "sora"],
            ),
            (
                "part107",
                ["part 107", "14 cfr part 107", "small uas", "remote pilot", "visual line of sight", "vlos"],
                ["14 cfr part 107", "part 107", "ac_107", "ac 107"],
            ),
            (
                "ac107",
                ["ac 107", "advisory circular", "107-2a"],
                ["ac_107-2a", "ac 107", "advisory circular"],
            ),
            (
                "sora",
                ["sora", "ground risk", "air risk", "operational volume", "adjacent area", "containment"],
                ["sora"],
            ),
            (
                "special_condition",
                ["special condition", "medium risk", "uas medium risk"],
                ["special_condition", "special condition"],
            ),
            (
                "ear_export",
                ["ear", "export", "export control", "unmanned aircraft systems export"],
                ["ear", "d0593e"],
            ),
            (
                "scenario_airspace",
                ["controlled airspace", "restricted area", "no-fly", "airspace authorization"],
                ["14 cfr part 107", "part 107", "ac_107", "ac 107"],
            ),
            (
                "scenario_vlos",
                ["visual line of sight", "vlos", "remote pilot"],
                ["14 cfr part 107", "part 107", "ac_107", "ac 107"],
            ),
            (
                "scenario_weather",
                ["wind conditions", "weather assessment", "wind speed", "preflight weather"],
                ["ac_107", "ac 107", "14 cfr part 107", "part 107"],
            ),
            (
                "scenario_medium_risk",
                ["risk class medium", "medium risk"],
                ["special_condition", "special condition", "sora"],
            ),
        ]

        for name, triggers, preferred in rules:
            if any(t in q for t in triggers):
                return {
                    "intent_name": name,
                    "preferred_source_patterns": preferred,
                    "negative_source_patterns": [],
                    "explicit_source_intent": True,
                    "domain_match": True,
                    "confidence": 0.9,
                }

        return {
            "intent_name": "general_uav" if domain_match else "unsupported_domain",
            "preferred_source_patterns": [],
            "negative_source_patterns": [],
            "explicit_source_intent": False,
            "domain_match": domain_match,
            "confidence": 0.6 if domain_match else 0.1,
        }

    @staticmethod
    def _source_pattern_score(text: str, patterns: list[str]) -> float:
        if not patterns:
            return 0.0
        t = text.lower()
        matches = sum(1 for p in patterns if p.lower() in t)
        return min(1.0, matches / max(1, len(patterns)))

    def _load_dense_mapping(self) -> None:
        if not self.index_dir:
            return
        mapping_path = self.index_dir / "dense_mapping.json"
        if not mapping_path.exists():
            return

        try:
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
            chunks = mapping.get("chunks") if isinstance(mapping, dict) else None
            if isinstance(chunks, list) and chunks:
                for i, chunk in enumerate(chunks):
                    if not isinstance(chunk, dict):
                        continue
                    chunk_id = str(chunk.get("chunk_id") or f"chunk_{i}")
                    vector_id = int(chunk.get("vector_id", i))
                    self.dense_doc_ids[vector_id] = chunk_id
                    self.chunk_by_id[chunk_id] = {
                        "doc_id": chunk_id,
                        "text": str(chunk.get("text") or ""),
                        "source": str(chunk.get("source_filename") or ""),
                        "source_id": str(chunk.get("source_id") or ""),
                        "source_filename": str(chunk.get("source_filename") or ""),
                        "source_title": str(chunk.get("source_title") or ""),
                        "page_start": chunk.get("page_start"),
                        "page_end": chunk.get("page_end"),
                        "section_title": chunk.get("section_title"),
                        "chunk_id": chunk_id,
                        "vector_id": vector_id,
                        "text_sha256": chunk.get("text_sha256"),
                    }
            else:
                doc_ids = mapping.get("doc_ids", []) if isinstance(mapping, dict) else []
                texts = mapping.get("texts", []) if isinstance(mapping, dict) else []
                sources = mapping.get("sources", []) if isinstance(mapping, dict) else []
                pages = mapping.get("pages", []) if isinstance(mapping, dict) else []
                for idx, doc_id in enumerate(doc_ids):
                    did = str(doc_id)
                    self.dense_doc_ids[idx] = did
                    self.chunk_by_id[did] = {
                        "doc_id": did,
                        "text": texts[idx] if idx < len(texts) else "",
                        "source": sources[idx] if idx < len(sources) else "",
                        "source_id": sources[idx] if idx < len(sources) else "",
                        "source_filename": sources[idx] if idx < len(sources) else "",
                        "source_title": sources[idx] if idx < len(sources) else "",
                        "page_start": pages[idx] if idx < len(pages) else None,
                        "page_end": pages[idx] if idx < len(pages) else None,
                        "section_title": None,
                        "chunk_id": did,
                        "vector_id": idx,
                        "text_sha256": None,
                    }
            logger.info("Loaded dense mapping with %d chunks", len(self.chunk_by_id))
        except Exception as exc:
            logger.warning("Failed to load dense mapping: %s", exc)

    def _load_sparse_lookup(self) -> None:
        if not self.sparse_index:
            return
        doc_ids = self.sparse_index.get("doc_ids", [])
        texts = self.sparse_index.get("doc_texts", [])
        sources = self.sparse_index.get("doc_sources", [])
        for i, doc_id in enumerate(doc_ids):
            did = str(doc_id)
            self.sparse_lookup[did] = {
                "text": texts[i] if i < len(texts) else "",
                "source": sources[i] if i < len(sources) else "",
            }

    async def search_dense(self, query: str, query_embedding: Optional[list[float]] = None, top_k: int = 20) -> list[tuple[str, float]]:
        if not self.dense_index or not self.embedder:
            return []
        try:
            if query_embedding is None:
                query_embedding = await self.embedder.embed(query)
            qv = np.array(query_embedding).astype("float32").reshape(1, -1)
            faiss = __import__("faiss")
            faiss.normalize_L2(qv)
            distances, indices = self.dense_index.search(qv, top_k)
            out: list[tuple[str, float]] = []
            for idx, dist in zip(indices[0], distances[0]):
                if int(idx) < 0:
                    continue
                doc_id = self.dense_doc_ids.get(int(idx), f"dense_{idx}")
                out.append((doc_id, max(0.0, min(1.0, float(dist)))))
            return out
        except Exception as exc:
            logger.error("Dense search failed: %s", exc)
            return []

    async def search_sparse(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        if not self.sparse_index:
            return []
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._search_sparse_sync, query, top_k)

    def _search_sparse_sync(self, query: str, top_k: int) -> list[tuple[str, float]]:
        try:
            index = self.sparse_index
            tokens = [t for t in query.lower().split() if t]
            N = int(index.get("N", 0))
            if N <= 0:
                return []
            doc_scores: defaultdict[int, float] = defaultdict(float)
            k1 = index.get("k1", 1.5)
            b = index.get("b", 0.75)
            avgdl = max(1e-9, index.get("avg_doc_length", 1.0))
            for token in tokens:
                if token not in index.get("term_doc_freq", {}):
                    continue
                idf = index.get("idf", {}).get(token, 0.0)
                doc_freqs = index["term_doc_freq"][token]
                for doc_idx, freq in doc_freqs.items():
                    doc_len = index["doc_lengths"][doc_idx]
                    denom = freq + k1 * (1 - b + b * (doc_len / avgdl))
                    score = idf * (freq * (k1 + 1)) / max(1e-9, denom)
                    doc_scores[doc_idx] += score
            top = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            doc_ids = index.get("doc_ids", [])
            out: list[tuple[str, float]] = []
            for doc_idx, score in top:
                if doc_idx < len(doc_ids):
                    out.append((str(doc_ids[doc_idx]), float(score)))
            if out:
                max_score = max(s for _, s in out)
                if max_score > 0:
                    out = [(d, s / max_score) for d, s in out]
            return out
        except Exception as exc:
            logger.error("Sparse search failed: %s", exc)
            return []

    async def rerank(self, query: str, documents: list[RetrievedDocument]) -> list[RetrievedDocument]:
        if not self.reranker or not self.use_reranker or not documents:
            reason = "reranker_not_available"
            if not self.use_reranker:
                reason = "disabled_by_configuration"
            elif not documents:
                reason = "no_documents_to_rerank"
            self._set_reranker_status(
                configured=bool(self.use_reranker),
                available=bool(self.reranker is not None),
                used=False,
                reason=reason,
            )
            return documents
        try:
            loop = asyncio.get_event_loop()
            pairs = [(query, d.text) for d in documents]
            scores = await loop.run_in_executor(None, self._rerank_sync, pairs)
            for d, s in zip(documents, scores):
                d.rerank_score = float(s)
                d.final_score = 0.45 * d.final_score + 0.35 * d.rerank_score + 0.2 * d.rrf_score
            documents.sort(key=lambda x: x.final_score, reverse=True)
            self._set_reranker_status(
                configured=True,
                available=True,
                used=True,
                reason="reranker_invoked",
            )
        except Exception as exc:
            logger.warning("Reranking failed: %s", exc)
            self._set_reranker_status(
                configured=bool(self.use_reranker),
                available=bool(self.reranker is not None),
                used=False,
                reason="reranker_failed",
            )
        return documents

    def _rerank_sync(self, pairs: list[tuple[str, str]]) -> list[float]:
        try:
            if hasattr(self.reranker, "predict"):
                return self.reranker.predict(pairs).tolist()
            if hasattr(self.reranker, "compute_score"):
                return [self.reranker.compute_score(p) for p in pairs]
            return [0.5] * len(pairs)
        except Exception:
            return [0.5] * len(pairs)

    def _is_provenance_complete(self, info: dict[str, Any]) -> bool:
        text = str(info.get("text") or "").strip()
        return bool(
            str(info.get("source_id") or "").strip()
            and str(info.get("source_filename") or "").strip()
            and text
            and str(info.get("chunk_id") or "").strip()
        )

    def _candidate_passes_gate(self, doc: RetrievedDocument, intent: dict[str, Any]) -> bool:
        if not intent.get("domain_match", False):
            return False
        if not doc.provenance_complete:
            return False
        if len(doc.text.strip()) < self.min_quote_chars:
            return False
        if doc.final_score < self.min_final_score:
            return False
        if intent.get("explicit_source_intent") and doc.source_match_score <= 0.0 and doc.final_score < 0.25:
            return False
        return True

    def _apply_diversity(self, docs: list[RetrievedDocument], intent: dict[str, Any], top_k: int) -> list[RetrievedDocument]:
        explicit = bool(intent.get("explicit_source_intent"))
        max_per_source = 3 if explicit else 2
        out: list[RetrievedDocument] = []
        per_source: defaultdict[str, int] = defaultdict(int)
        seen_pairs: set[tuple[str, int | None]] = set()

        for d in docs:
            source_key = d.source_id or d.source_filename or d.source or "unknown"
            page_key = (d.source_filename, d.page_start)
            if page_key in seen_pairs:
                continue
            if per_source[source_key] >= max_per_source:
                continue
            out.append(d)
            per_source[source_key] += 1
            seen_pairs.add(page_key)
            if len(out) >= top_k:
                break

        if not explicit and len({(d.source_id or d.source_filename) for d in out}) < 2:
            remaining = [d for d in docs if d not in out]
            for d in remaining:
                sk = d.source_id or d.source_filename or d.source
                if sk not in {x.source_id or x.source_filename or x.source for x in out}:
                    out.append(d)
                    if len(out) >= top_k:
                        break

        return out[:top_k]

    async def search(
        self,
        query: str,
        query_embedding: Optional[list[float]] = None,
        top_k: int = 20,
        rrf_k: int = 60,
        use_hyde: bool = False,
    ) -> list[RetrievedDocument]:
        self._set_reranker_status(
            configured=bool(self.use_reranker),
            available=bool(self.reranker is not None),
            used=False,
            reason="pending",
        )
        intent = self.detect_source_intent(query)
        if not intent.get("domain_match", False):
            self._set_reranker_status(
                configured=bool(self.use_reranker),
                available=bool(self.reranker is not None),
                used=False,
                reason="domain_not_supported",
            )
            return []

        dense_task = self.search_dense(query, query_embedding, top_k * 3)
        sparse_task = self.search_sparse(query, top_k * 3)
        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)

        rrf = AdaptiveRRF()
        fused = rrf.fuse(dense_results, sparse_results, rrf_k, self.dense_weight, self.sparse_weight)

        dense_map = {d: s for d, s in dense_results}
        sparse_map = {d: s for d, s in sparse_results}

        candidates: list[RetrievedDocument] = []
        seen_texts: list[str] = []
        for doc_id, rrf_score in fused[: top_k * 6]:
            info = self.chunk_by_id.get(doc_id)
            if info is None:
                sparse_info = self.sparse_lookup.get(doc_id, {})
                info = {
                    "doc_id": doc_id,
                    "text": sparse_info.get("text", ""),
                    "source": sparse_info.get("source", ""),
                    "source_id": sparse_info.get("source", ""),
                    "source_filename": sparse_info.get("source", ""),
                    "source_title": sparse_info.get("source", ""),
                    "page_start": None,
                    "page_end": None,
                    "section_title": None,
                    "chunk_id": doc_id,
                    "vector_id": None,
                    "text_sha256": None,
                }

            text = str(info.get("text") or "")
            if not text.strip():
                continue

            if any(self.simhash.is_duplicate(text, old, threshold=3) for old in seen_texts):
                continue
            seen_texts.append(text)

            source_text = " ".join(
                [
                    str(info.get("source_id") or ""),
                    str(info.get("source_filename") or ""),
                    str(info.get("source_title") or ""),
                ]
            ).lower()
            source_match = self._source_pattern_score(source_text, intent.get("preferred_source_patterns", []))
            source_boost = 0.18 * source_match if intent.get("explicit_source_intent") else 0.05 * source_match
            source_penalty = 0.08 if intent.get("explicit_source_intent") and source_match <= 0 else 0.0

            dense_score = float(dense_map.get(doc_id, 0.0))
            sparse_score = float(sparse_map.get(doc_id, 0.0))
            base = 0.5 * float(rrf_score) + 0.3 * dense_score + 0.2 * sparse_score
            final_score = base + source_boost - source_penalty

            d = RetrievedDocument(
                doc_id=str(info.get("doc_id") or doc_id),
                text=text,
                source=str(info.get("source") or info.get("source_filename") or ""),
                source_id=str(info.get("source_id") or ""),
                source_filename=str(info.get("source_filename") or ""),
                source_title=str(info.get("source_title") or ""),
                page_start=info.get("page_start"),
                page_end=info.get("page_end"),
                section_title=info.get("section_title"),
                chunk_id=str(info.get("chunk_id") or doc_id),
                vector_id=info.get("vector_id"),
                text_sha256=info.get("text_sha256"),
                retrieval_method="dense_sparse_rrf",
                domain_match=bool(intent.get("domain_match", False)),
                source_match_score=source_match,
                dense_score=dense_score,
                sparse_score=sparse_score,
                rrf_score=float(rrf_score),
                final_score=float(final_score),
                chunk_hash=str(self.simhash.compute(text)),
                provenance_complete=self._is_provenance_complete(info),
            )
            candidates.append(d)

        if self.use_reranker and self.reranker:
            candidates = await self.rerank(query, candidates)
        else:
            candidates.sort(key=lambda x: x.final_score, reverse=True)

        gated = [d for d in candidates if self._candidate_passes_gate(d, intent)]

        if intent.get("explicit_source_intent"):
            matched = [d for d in gated if d.source_match_score > 0]
            if matched:
                gated = matched
            else:
                return []

        if not gated:
            return []

        gated.sort(key=lambda x: x.final_score, reverse=True)
        return self._apply_diversity(gated, intent, top_k)

    async def batch_search(
        self,
        queries: list[str],
        query_embeddings: Optional[list[list[float]]] = None,
        top_k: int = 20,
        max_concurrent: int = 5,
    ) -> list[list[RetrievedDocument]]:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _search_single(idx: int):
            async with semaphore:
                emb = query_embeddings[idx] if query_embeddings else None
                return await self.search(queries[idx], emb, top_k)

        tasks = [_search_single(i) for i in range(len(queries))]
        return await asyncio.gather(*tasks)
