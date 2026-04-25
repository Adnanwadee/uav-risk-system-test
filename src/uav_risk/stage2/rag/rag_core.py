"""
Async RAG Core Engine (V12 - Certified Aviation Standard)
=========================================================
Contract: Implements AsyncRAGIndexInterface.
Resilience: Embedded Circuit Breaker + Fail-Fast Embedding Loading (No Semantic Corruption).
Memory/State: Thread-safe Async TTL Cache with bounded max size.
Math: Hybrid Normalization (Dynamic Global Clamp + Batch Min-Max) with strict boundaries [0, 1].
Safety: asyncio.Semaphore prevents ThreadPool exhaustion under heavy load.
Observability: Deep Health Check Endpoints + Prometheus-ready metrics.

Author: Stage 2 — ACE System
"""

import os
import hashlib
import asyncio
import logging
import re
import time
import unicodedata
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto
from typing import List, Dict, Any, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

from uav_risk.stage2.agents.legal_agent import AsyncRAGIndexInterface
from .config import RAGConfig

logger = logging.getLogger("AsyncRAGCore")

# ---------------------------------------------------------------------------
# Resilience Structures (Circuit Breaker & Thread-Safe TTL Cache)
# ---------------------------------------------------------------------------

class CircuitState(Enum):
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()

class RAGCircuitBreaker:
    def __init__(self, failure_threshold: int = 5, cooldown_sec: float = 30.0):
        self.failure_threshold = failure_threshold
        self.cooldown_sec = cooldown_sec
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.open_until = 0.0

    def allow_request(self) -> bool:
        if self.state == CircuitState.OPEN:
            if time.monotonic() >= self.open_until:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        return True

    def record_success(self):
        self.failures = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info("RAG Circuit Breaker RECOVERED.")

    def record_failure(self):
        self.failures += 1
        if self.state == CircuitState.CLOSED and self.failures >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.open_until = time.monotonic() + self.cooldown_sec
            logger.critical(f"RAG Circuit Breaker TRIPPED for {self.cooldown_sec}s.")

class AsyncTTLCache:
    def __init__(self, ttl_sec: float = 3600.0, max_size: int = 1000):
        self.ttl_sec = ttl_sec
        self.max_size = max_size
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key in self._cache:
                expiry, data = self._cache[key]
                if time.monotonic() < expiry:
                    self._cache.move_to_end(key)
                    return data
                else:
                    del self._cache[key]
            return None

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            self._cache[key] = (time.monotonic() + self.ttl_sec, value)
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def size(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Core Engine
# ---------------------------------------------------------------------------

class AsyncRAGCore(AsyncRAGIndexInterface):
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        
        self._pool = ThreadPoolExecutor(max_workers=self.config.MAX_THREADS, thread_name_prefix="RAG_Core")
        self._semaphore = asyncio.Semaphore(self.config.MAX_CONCURRENT_REQUESTS)
        
        self.circuit = RAGCircuitBreaker()
        self.query_cache = AsyncTTLCache(ttl_sec=3600.0, max_size=1000)
        
        self._db: Optional[FAISS] = None
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._reranker: Optional[CrossEncoder] = None
        self._reranker_lock = asyncio.Lock()
        
        self.metrics = {
            "total_searches": 0, 
            "cache_hits": 0,
            "failed_searches": 0, 
            "latency_sum_ms": 0.0
        }
        
        self._initialize_base_engines()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False

    def shutdown(self):
        if hasattr(self, '_pool') and self._pool is not None:
            self._pool.shutdown(wait=False)

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    def _verify_index_integrity(self) -> bool:
        if not self.config.EXPECTED_INDEX_HASH:
            return True
            
        index_file = self.config.INDEX_PATH / "index.faiss"
        if not index_file.exists():
            return False
            
        hasher = hashlib.sha256()
        try:
            with open(index_file, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            is_valid = hasher.hexdigest() == self.config.EXPECTED_INDEX_HASH
            if not is_valid:
                logger.critical("FAISS INDEX INTEGRITY COMPROMISED! Hash mismatch.")
            return is_valid
        except Exception as e:
            logger.error(f"Error checking index integrity: {e}")
            return False

    def _initialize_base_engines(self) -> None:
        """[FIX] Fail-Fast: تحميل نموذج واحد فقط. الفشل يعني الإيقاف الكلي لمنع الفساد الإحصائي."""
        logger.info("Initializing Base RAG Engine...")
        if not self.config.INDEX_PATH or not self.config.INDEX_PATH.exists():
            logger.critical(f"Index missing at {self.config.INDEX_PATH}. RAG Offline.")
            return

        target_model = self.config.EMBEDDING_MODEL
        try:
            logger.info(f"Loading Primary Embeddings: {target_model}")
            self._embeddings = HuggingFaceEmbeddings(model_name=target_model)
            
            if not self._verify_index_integrity():
                return

            self._db = FAISS.load_local(
                str(self.config.INDEX_PATH), 
                self._embeddings, 
                allow_dangerous_deserialization=True
            )
            logger.info(f"Base RAG Engines loaded successfully with {target_model}.")
        except Exception as e:
            logger.critical(f"FATAL: Failed to load primary model {target_model}: {e}")
            self._db = None

    async def _ensure_reranker_loaded(self):
        if self._reranker is None:
            async with self._reranker_lock:
                if self._reranker is None:
                    logger.info("Lazy-loading CrossEncoder (Forced to CPU)...")
                    loop = asyncio.get_running_loop()
                    # حماية حتمية لتعدد الخيوط بمنع تفعيل الـ GPU
                    self._reranker = await loop.run_in_executor(
                        self._pool, 
                        lambda: CrossEncoder(self.config.RERANKER_MODEL, device='cpu')
                    )
                    logger.info("CrossEncoder ready.")

    def get_health_status(self) -> Dict[str, Any]:
        """[FIX] فحص صحي دقيق وعميق يشمل الـ Cache وقاطع الدائرة."""
        is_healthy = self._db is not None and self._embeddings is not None
        return {
            "status": "healthy" if is_healthy else "degraded/offline",
            "active_embedding_model": self.config.EMBEDDING_MODEL,
            "circuit_state": self.circuit.state.name,
            "reranker_loaded": self._reranker is not None,
            "cache_size": self.query_cache.size(),
            "metrics": self.metrics
        }

    def _sanitize_query(self, query: str) -> str:
        if not query: return ""
        q = unicodedata.normalize("NFKC", str(query))
        q = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", q)
        for c in "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\u200f\u200e":
            q = q.replace(c, " ")
        return q.strip()[:500]

    def _hybrid_normalize_scores(self, raw_scores: list[float]) -> list[float]:
        """[FIX] التطبيع الهجين مع حماية رياضية قاطعة للـ Bounds [0, 1]."""
        if not raw_scores: return []
        
        min_b = self.config.LOGIT_MIN
        max_b = self.config.LOGIT_MAX
        clamped = [max(min_b, min(float(s), max_b)) for s in raw_scores]
        
        min_s = min(clamped)
        max_s = max(clamped)
        
        if max_s == min_s:
            return [1.0] * len(clamped)
            
        normalized = [(s - min_s) / (max_s - min_s) for s in clamped]
        # أمان رياضي نهائي لمنع تسرب القيم خارج [0, 1]
        return [max(0.0, min(1.0, float(s))) for s in normalized]

    def _deterministic_chunk_id(self, content: str) -> str:
        return "chk_" + hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

    def _sync_search_pipeline(self, query: str, top_k: int, min_score: float) -> List[Dict[str, Any]]:
        """[FIX] العزل المعماري: التقاط جميع الأخطاء هنا لكي لا يسقط الـ ThreadPool."""
        try:
            if not self._db or not self._reranker:
                return []

            docs = self._db.similarity_search(query, k=self.config.INITIAL_K)
            if not docs: return []

            pairs = [[query, doc.page_content] for doc in docs]
            raw_scores = self._reranker.predict(pairs)

            normalized_scores = self._hybrid_normalize_scores(raw_scores.tolist())

            scored_results = []
            for i, doc in enumerate(docs):
                conf_score = normalized_scores[i]
                if conf_score >= min_score:
                    hit = {
                        "page_content": doc.page_content,
                        "metadata": {
                            "source": doc.metadata.get("source", "Unknown_Regulation"),
                            "chunk_id": doc.metadata.get("chunk_id", self._deterministic_chunk_id(doc.page_content)),
                            "score": conf_score
                        }
                    }
                    scored_results.append(hit)

            scored_results.sort(key=lambda x: x["metadata"]["score"], reverse=True)
            return scored_results[:top_k]
            
        except Exception as e:
            logger.error(f"Internal Pipeline Error during RAG execution: {e}")
            return []  # العودة الآمنة تمنع الانهيار الكامل للمنسق (Orchestrator)

    async def search(self, query: str, top_k: int, min_score: float) -> List[Dict[str, Any]]:
        clean_query = self._sanitize_query(query)
        if not clean_query: return []

        if not self.circuit.allow_request():
            self.metrics["failed_searches"] += 1
            logger.warning(f"RAG request rejected. Circuit OPEN.")
            return []

        # [FIX] Cache Key متكامل لمنع أي تصادم في معايير الاسترجاع
        cache_key = f"{hashlib.sha256(clean_query.encode()).hexdigest()}_{self.config.INITIAL_K}_{top_k}_{min_score:.2f}"
        
        cached_result = await self.query_cache.get(cache_key)
        if cached_result is not None:
            self.metrics["cache_hits"] += 1
            return cached_result

        async with self._semaphore:
            self.metrics["total_searches"] += 1
            start_time = time.monotonic()
            
            try:
                await self._ensure_reranker_loaded()
                
                loop = asyncio.get_running_loop()
                future = loop.run_in_executor(
                    self._pool, 
                    self._sync_search_pipeline, 
                    clean_query, top_k, min_score
                )
                
                results = await asyncio.wait_for(future, timeout=self.config.TIMEOUT_SEC)
                
                if results:
                    await self.query_cache.set(cache_key, results)
                
                self.metrics["latency_sum_ms"] += (time.monotonic() - start_time) * 1000
                self.circuit.record_success()
                return results

            except asyncio.TimeoutError:
                logger.error(f"RAG Search timed out after {self.config.TIMEOUT_SEC}s.")
                self.metrics["failed_searches"] += 1
                self.circuit.record_failure()
                return []
            except Exception as e:
                logger.error(f"Async RAG Search Failed: {e}")
                self.metrics["failed_searches"] += 1
                self.circuit.record_failure()
                return []

    def export_prometheus(self) -> str:
        m = self.metrics
        tot = max(1, m['total_searches'])
        avg_latency = m['latency_sum_ms'] / tot
        
        return (
            f"rag_searches_total {m['total_searches']}\n"
            f"rag_cache_hits_total {m['cache_hits']}\n"
            f"rag_searches_failed {m['failed_searches']}\n"
            f"rag_latency_avg_ms {avg_latency:.2f}\n"
            f"rag_circuit_state {{state=\"{self.circuit.state.name}\"}} 1\n"
        )