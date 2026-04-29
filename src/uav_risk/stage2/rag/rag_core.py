"""
Async RAG Core Engine (V15.1 - Pydantic Validation Fix)
=========================================================
التحديثات:
- إصلاح خطأ الـ Extra inputs: نقل local_files_only إلى model_kwargs.
- الربط المباشر مع مجلد knowledge/models المحلي.
- تفعيل local_files_only لضمان عدم التعليق نهائياً.
- تحسين استقرار الـ Reranker باستخدام مسارات Pathlib المطلقة.
"""

import os
import hashlib
import asyncio
import logging
import time
from pathlib import Path
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

# ───────────────────────────────────────────────────────────────────────────
# 1. إعداد المسارات المحلية الدقيقة (المجلد الذي أنشأته)
# ───────────────────────────────────────────────────────────────────────────
# المسار النسبي: يرجع من rag إلى stage2 ثم يدخل knowledge/models
CURRENT_DIR = Path(__file__).resolve().parent
MODELS_DIR = CURRENT_DIR.parent / "knowledge" / "models"

EMBEDDING_PATH = MODELS_DIR / "embedding"
RERANKER_PATH = MODELS_DIR / "reranker"

# ───────────────────────────────────────────────────────────────────────────
# 2. الهياكل المرنة (Circuit Breaker & Cache)
# ───────────────────────────────────────────────────────────────────────────

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

# ───────────────────────────────────────────────────────────────────────────
# 3. المحرك الأساسي (AsyncRAGCore)
# ───────────────────────────────────────────────────────────────────────────

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
        
        self.metrics = {"total_searches": 0, "cache_hits": 0, "failed_searches": 0, "latency_sum_ms": 0.0}
        
        self._initialize_base_engines()

    def _initialize_base_engines(self) -> None:
        """تحميل محرك التضمين وقاعدة البيانات أوفلاين."""
        if not self.config.INDEX_PATH or not self.config.INDEX_PATH.exists():
            logger.critical(f"Index missing at {self.config.INDEX_PATH}. RAG Offline.")
            return

        try:
            logger.info(f"📡 Accessing local embeddings at: {EMBEDDING_PATH}")
            # [تعديل حاسم]: تم نقل local_files_only لداخل model_kwargs لحل خطأ Pydantic
            self._embeddings = HuggingFaceEmbeddings(
                model_name=str(EMBEDDING_PATH),
                model_kwargs={
                    'device': 'cpu',
                    'local_files_only': True # إجبار الأوفلاين داخل القاموس
                }
            )
            
            self._db = FAISS.load_local(
                str(self.config.INDEX_PATH), 
                self._embeddings, 
                allow_dangerous_deserialization=True
            )
            logger.info("✅ FAISS & Embeddings loaded successfully from local models.")
        except Exception as e:
            logger.critical(f"FATAL: Failed to load RAG engines: {e}")
            self._db = None

    async def _ensure_reranker_loaded(self):
        """تحميل الـ Reranker من المجلد المحلي عند الطلب."""
        if self._reranker is None:
            async with self._reranker_lock:
                if self._reranker is None:
                    logger.info(f"📡 Initializing local Reranker from: {RERANKER_PATH}")
                    loop = asyncio.get_running_loop()
                    self._reranker = await loop.run_in_executor(
                        self._pool, 
                        lambda: CrossEncoder(str(RERANKER_PATH), device='cpu', local_files_only=True)
                    )

    def _hybrid_normalize_scores(self, raw_scores: list[float]) -> list[float]:
        if not raw_scores: return []
        min_b, max_b = self.config.LOGIT_MIN, self.config.LOGIT_MAX
        clamped = [max(min_b, min(float(s), max_b)) for s in raw_scores]
        min_s, max_s = min(clamped), max(clamped)
        if max_s == min_s: return [1.0] * len(clamped)
        return [max(0.0, min(1.0, (s - min_s) / (max_s - min_s))) for s in clamped]

    def _sync_search_pipeline(self, query: str, top_k: int, min_score: float) -> List[Dict[str, Any]]:
        """خط إنتاج البحث لاستخراج الأدلة الجنائية."""
        try:
            if not self._db or not self._reranker: return []

            docs = self._db.similarity_search(query, k=self.config.INITIAL_K)
            if not docs: return []

            pairs = [[query, doc.page_content] for doc in docs]
            raw_scores = self._reranker.predict(pairs)
            normalized_scores = self._hybrid_normalize_scores(raw_scores.tolist())

            scored_results = []
            for i, doc in enumerate(docs):
                conf_score = normalized_scores[i]
                if conf_score >= min_score:
                    source = doc.metadata.get("source", "Aviation Regulation")
                    article_id = doc.metadata.get("article_id", "N/A")
                    
                    hit = {
                        "page_content": doc.page_content,
                        "metadata": {
                            "source": source,
                            "article_id": article_id,
                            "chunk_id": hashlib.sha256(doc.page_content.encode()).hexdigest()[:16],
                            "score": conf_score,
                            "citation": f"[{source} | Article: {article_id}]"
                        }
                    }
                    scored_results.append(hit)

            scored_results.sort(key=lambda x: x["metadata"]["score"], reverse=True)
            return scored_results[:top_k]
        except Exception as e:
            logger.error(f"Internal RAG Pipeline Error: {e}")
            return []

    async def search(self, query: str, top_k: int, min_score: float) -> List[Dict[str, Any]]:
        """دالة البحث الأساسية المحمية."""
        if not self.circuit.allow_request():
            self.metrics["failed_searches"] += 1
            return []

        cache_key = f"{hashlib.sha256(query.encode()).hexdigest()}_{top_k}_{min_score}"
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
                results = await asyncio.wait_for(
                    loop.run_in_executor(self._pool, self._sync_search_pipeline, query, top_k, min_score),
                    timeout=self.config.TIMEOUT_SEC
                )
                if results: await self.query_cache.set(cache_key, results)
                self.metrics["latency_sum_ms"] += (time.monotonic() - start_time) * 1000
                self.circuit.record_success()
                return results
            except Exception as e:
                logger.error(f"Async RAG Search Failed: {e}")
                self.metrics["failed_searches"] += 1
                self.circuit.record_failure()
                return []

    def shutdown(self):
        if hasattr(self, '_pool'): self._pool.shutdown(wait=False)

    def __del__(self):
        self.shutdown()