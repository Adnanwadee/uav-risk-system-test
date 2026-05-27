"""
Hybrid Retriever - TRUE Dense + Sparse + RRF + Reranker + SimHash Deduplication
V3.1 FIXES:
- Fixed imports with package prefix
- Fixed dense doc_id mapping via dense_mapping.json
- Added cross-encoder reranker support
- Fixed score normalization for IndexFlatIP
"""
import logging
import asyncio
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
import json

from .query_intelligence import AdaptiveRRF
from .schemas import RerankerProtocol

logger = logging.getLogger(__name__)

@dataclass
class RetrievedDocument:
    doc_id: str
    text: str
    source: str
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    is_duplicate: bool = False
    chunk_hash: Optional[str] = None

class SimHash:
    """
    SimHash for fast near-duplicate detection.
    Replaces O(N²) SequenceMatcher.
    """

    def __init__(self, hashbits: int = 64):
        self.hashbits = hashbits

    def _hash_func(self, token: str) -> int:
        """Simple hash function"""
        return hash(token) & ((1 << self.hashbits) - 1)

    def compute(self, text: str) -> int:
        """Compute SimHash for text"""
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
                fingerprint |= (1 << i)

        return fingerprint

    def hamming_distance(self, hash1: int, hash2: int) -> int:
        """Calculate Hamming distance between two hashes"""
        x = hash1 ^ hash2
        distance = 0
        while x:
            distance += 1
            x &= x - 1
        return distance

    def is_duplicate(self, text1: str, text2: str, threshold: int = 3) -> bool:
        """Check if two texts are near-duplicates"""
        h1 = self.compute(text1)
        h2 = self.compute(text2)
        return self.hamming_distance(h1, h2) <= threshold


class HybridRetriever:
    """
    TRUE Hybrid Retriever combining Dense + Sparse + RRF + Reranker.
    Features:
    - SimHash deduplication (O(1) per check vs O(N²))
    - Async sparse search (no Event Loop blocking)
    - Cross-encoder reranker for precision boost
    - Weighted confidence fusion
    """

    def __init__(self, 
                 dense_index=None,
                 sparse_index=None,
                 embedder=None,
                 reranker=None,
                 config=None,
                 index_dir: Optional[str] = None):
        self.dense_index = dense_index
        self.sparse_index = sparse_index
        self.embedder = embedder
        self.reranker = reranker
        self.config = config
        self.index_dir = Path(index_dir) if index_dir else None

        self.simhash = SimHash(hashbits=64)
        self.seen_hashes: set = set()

        # Load dense mapping if available
        self.dense_doc_ids: Dict[int, str] = {}
        self.dense_texts: Dict[int, str] = {}
        self._load_dense_mapping()

        # Weights for final fusion
        self.dense_weight = getattr(config, "DENSE_WEIGHT", 0.6) if config else 0.6
        self.sparse_weight = getattr(config, "SPARSE_WEIGHT", 0.4) if config else 0.4
        self.use_reranker = getattr(config, "USE_RERANKER", True) if config else True

    def _load_dense_mapping(self):
        """Load dense_mapping.json to map FAISS indices to real doc_ids"""
        if not self.index_dir:
            return

        mapping_path = self.index_dir / "dense_mapping.json"
        if mapping_path.exists():
            try:
                with open(mapping_path, "r", encoding="utf-8") as f:
                    mapping = json.load(f)

                doc_ids = mapping.get("doc_ids", [])
                texts = mapping.get("texts", [])

                for idx, doc_id in enumerate(doc_ids):
                    self.dense_doc_ids[idx] = doc_id
                    if idx < len(texts):
                        self.dense_texts[idx] = texts[idx]

                logger.info(f"Loaded dense mapping: {len(self.dense_doc_ids)} documents")
            except Exception as e:
                logger.warning(f"Failed to load dense mapping: {e}")

    async def search_dense(self, query: str, 
                          query_embedding: Optional[List[float]] = None,
                          top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Dense retrieval using vector similarity.
        Uses dense_mapping.json for proper doc_id resolution.
        """
        if not self.dense_index or not self.embedder:
            return []

        try:
            # Get query embedding
            if query_embedding is None:
                query_embedding = await self.embedder.embed(query)

            query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)

            # Normalize query embedding (must match index normalization)
            faiss = __import__("faiss")
            faiss.normalize_L2(query_embedding)

            # Search FAISS index
            distances, indices = self.dense_index.search(query_embedding, top_k)

            # For IndexFlatIP with normalized vectors: distance = cosine similarity
            # Range: [-1, 1], typically [0, 1] for positive similarity
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx >= 0:
                    # Map to real doc_id if mapping exists
                    doc_id = self.dense_doc_ids.get(int(idx), f"dense_{idx}")
                    # Score is already cosine similarity [-1, 1], clip to [0, 1]
                    score = max(0.0, min(1.0, float(dist)))
                    results.append((doc_id, score))

            return results

        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []

    async def search_sparse(self, query: str, 
                           top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Sparse retrieval using BM25.
        Runs in thread pool to avoid blocking Event Loop.
        """
        if not self.sparse_index:
            return []

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._search_sparse_sync, query, top_k)

    def _search_sparse_sync(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Synchronous BM25 search"""
        try:
            index = self.sparse_index
            tokens = query.lower().split()

            N = index.get("N", 0)
            if N == 0:
                return []

            doc_scores = defaultdict(float)

            k1 = index.get("k1", 1.5)
            b = index.get("b", 0.75)
            avgdl = index.get("avg_doc_length", 1.0)

            for token in tokens:
                if token not in index.get("term_doc_freq", {}):
                    continue

                idf = index.get("idf", {}).get(token, 0)
                doc_freqs = index["term_doc_freq"][token]

                for doc_idx, freq in doc_freqs.items():
                    doc_len = index["doc_lengths"][doc_idx]

                    # BM25 scoring
                    denom = freq + k1 * (1 - b + b * (doc_len / avgdl))
                    score = idf * (freq * (k1 + 1)) / denom
                    doc_scores[doc_idx] += score

            # Get top-k
            top_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

            # Map to doc_ids
            doc_ids = index.get("doc_ids", [])
            results = []
            for doc_idx, score in top_results:
                if doc_idx < len(doc_ids):
                    results.append((doc_ids[doc_idx], score))

            # Normalize scores to 0-1
            if results:
                max_score = max(s for _, s in results)
                if max_score > 0:
                    results = [(doc_id, score / max_score) for doc_id, score in results]

            return results

        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            return []

    async def rerank(self, query: str, 
                    documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """
        Rerank documents using cross-encoder.
        Significantly improves precision for top-k results.
        """
        if not self.reranker or not self.use_reranker or not documents:
            return documents

        try:
            loop = asyncio.get_event_loop()

            # Prepare query-document pairs
            pairs = [(query, doc.text) for doc in documents]

            # Run reranker in thread pool (CPU-intensive)
            scores = await loop.run_in_executor(
                None, 
                self._rerank_sync, 
                pairs
            )

            # Update rerank scores
            for doc, score in zip(documents, scores):
                doc.rerank_score = float(score)
                # Blend rerank score into final score
                doc.final_score = (
                    0.5 * doc.rrf_score +
                    0.3 * doc.rerank_score +
                    0.1 * doc.dense_score +
                    0.1 * doc.sparse_score
                )

            # Re-sort by blended score
            documents.sort(key=lambda d: d.final_score, reverse=True)
            logger.info(f"Reranked {len(documents)} documents")

        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Using original scores.")

        return documents

    def _rerank_sync(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Synchronous reranker inference"""
        try:
            # HuggingFace cross-encoder interface
            if hasattr(self.reranker, 'predict'):
                return self.reranker.predict(pairs).tolist()
            elif hasattr(self.reranker, 'compute_score'):
                return [self.reranker.compute_score(p) for p in pairs]
            else:
                logger.warning("Reranker has no predict or compute_score method")
                return [0.5] * len(pairs)
        except Exception as e:
            logger.error(f"Reranker inference failed: {e}")
            return [0.5] * len(pairs)

    async def search(self, query: str,
                    query_embedding: Optional[List[float]] = None,
                    top_k: int = 20,
                    rrf_k: int = 60,
                    use_hyde: bool = False) -> List[RetrievedDocument]:
        """
        Full hybrid search: Dense + Sparse + RRF + Reranker + Deduplication.
        """
        # Run dense and sparse in parallel
        dense_task = self.search_dense(query, query_embedding, top_k * 2)
        sparse_task = self.search_sparse(query, top_k * 2)

        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)

        logger.info(
            f"Dense: {len(dense_results)} results, "
            f"Sparse: {len(sparse_results)} results"
        )

        # RRF Fusion
        rrf = AdaptiveRRF()
        fused = rrf.fuse(
            dense_results, sparse_results, rrf_k, 
            self.dense_weight, self.sparse_weight
        )

        # Build documents
        documents = []
        doc_lookup = {}

        # Create lookup from sparse index
        if self.sparse_index:
            sparse_docs = self.sparse_index.get("doc_ids", [])
            sparse_texts = self.sparse_index.get("doc_texts", [])
            sparse_sources = self.sparse_index.get("doc_sources", [])

            for i, doc_id in enumerate(sparse_docs):
                doc_lookup[doc_id] = {
                    "text": sparse_texts[i] if i < len(sparse_texts) else "",
                    "source": sparse_sources[i] if i < len(sparse_sources) else ""
                }

        # Also add dense mapping texts
        for idx, text in self.dense_texts.items():
            doc_id = self.dense_doc_ids.get(idx, f"dense_{idx}")
            if doc_id not in doc_lookup:
                doc_lookup[doc_id] = {"text": text, "source": "dense_index"}

        # Build result documents with deduplication
        seen_texts = []
        for doc_id, rrf_score in fused[:top_k * 3]:  # Get extra for dedup
            doc_info = doc_lookup.get(doc_id, {"text": "", "source": ""})
            text = doc_info["text"]

            if not text:
                continue

            # SimHash deduplication
            is_dup = False
            for seen_text in seen_texts:
                if self.simhash.is_duplicate(text, seen_text, threshold=3):
                    is_dup = True
                    break

            if is_dup:
                continue

            seen_texts.append(text)

            # Get individual scores
            dense_score = next((s for d, s in dense_results if d == doc_id), 0.0)
            sparse_score = next((s for d, s in sparse_results if d == doc_id), 0.0)

            # Compute chunk hash
            chunk_hash = str(self.simhash.compute(text))

            doc = RetrievedDocument(
                doc_id=doc_id,
                text=text,
                source=doc_info["source"],
                dense_score=dense_score,
                sparse_score=sparse_score,
                rrf_score=rrf_score,
                chunk_hash=chunk_hash
            )

            documents.append(doc)

            if len(documents) >= top_k * 2:  # Get extra for reranker
                break

        # Calculate initial confidence scores
        for doc in documents:
            doc.final_score = (
                0.5 * doc.rrf_score +
                0.3 * doc.dense_score +
                0.2 * doc.sparse_score
            )

        # Rerank if enabled
        if self.use_reranker and self.reranker:
            documents = await self.rerank(query, documents)

        # Sort by final score and return top_k
        documents.sort(key=lambda d: d.final_score, reverse=True)
        return documents[:top_k]

    async def batch_search(self, queries: List[str],
                          query_embeddings: Optional[List[List[float]]] = None,
                          top_k: int = 20,
                          max_concurrent: int = 5) -> List[List[RetrievedDocument]]:
        """
        Batch search with concurrency control.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _search_single(idx):
            async with semaphore:
                emb = query_embeddings[idx] if query_embeddings else None
                return await self.search(queries[idx], emb, top_k)

        tasks = [_search_single(i) for i in range(len(queries))]
        return await asyncio.gather(*tasks)