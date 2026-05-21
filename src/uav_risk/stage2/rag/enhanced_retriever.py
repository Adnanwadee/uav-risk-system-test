"""
Module: src/uav_risk/stage2/rag/enhanced_retriever.py
Author: Elite Technical Partner
Description: Advanced, aviation-aware semantic retriever featuring ontological expansion,
             hybrid reranking, HyDE search, semantic deduplication, and caching.
"""

import asyncio
import hashlib
from typing import List, Optional, Dict, Any, Tuple
import difflib
import structlog

# استيراد العقود البياناتية الصارمة من الطبقة الحاضنة
from uav_risk.stage2.rag.schemas import RetrievedChunk

logger = structlog.get_logger()

# أنطولوجيا مصطلحات فيزياء وطيران الدرون لربط الميزات الجافة باللوائح التشريعية
AVIATION_ONTOLOGY: Dict[str, List[str]] = {
    "aerodynamic": ["wing loading", "aspect ratio", "rotorcraft disk area", "aerodynamic stall", "mass category", "MTOW"],
    "environmental": ["wind gusts", "visibility minimums", "precipitation", "air density", "weather severity", "thermal"],
    "battery": ["powerplant failure", "energy reserve", "battery capacity mah", "voltage drop", "emergency landing contingency"],
    "mission": ["BVLOS operations", "altitude ceiling", "vlos distance limits", "populated area overflight", "restricted airspace"],
    "gps": ["gnss jamming", "satellite fix quality", "hdop dilution", "navigation degradation", "lost link profile"],
    "comms": ["c2 link failure", "radio frequency interference", "rssi attenuation", "signal strength minimums", "containment"],
    "operator": ["remote pilot certificate", "license privileges", "experience hours", "atc clearance", "faa part 107 compliance"]
}

class EnhancedLegalRetriever:
    """High-performance legal retriever optimized for unmanned aviation risk assessment compliance."""
    
    def __init__(
        self,
        vector_store,  # كائن لـ FAISS
        embeddings,    # كائن لـ HuggingFaceEmbeddings
        reranker: Optional[Any] = None,  # كائن لـ CrossEncoder المحلي
        llm: Optional[Any] = None,       # كائن لـ GroqLLM
        cache_size: int = 128
    ):
        self._vector_store = vector_store
        self._embeddings = embeddings
        self._reranker = reranker
        self._llm = llm
        self._cache_size = cache_size
        self._cache: Dict[int, List[RetrievedChunk]] = {}
        
        # إحصائيات الأداء التشغيلي
        self._hit_count = 0
        self._miss_count = 0
        
        logger.info("enhanced_retriever_initialized", 
                    has_reranker=self._reranker is not None, 
                    has_llm=self._llm is not None, 
                    cache_size=cache_size)

    def _expand_query_with_ontology(self, query: str) -> str:
        """Expands the raw technical text or feature name with aerospace industry vocabulary keywords."""
        lowered_query = query.lower()
        expanded_keywords = []
        
        for category, keywords in AVIATION_ONTOLOGY.items():
            if category in lowered_query or any(kw in lowered_query for kw in keywords):
                expanded_keywords.extend(keywords)
        
        if expanded_keywords:
            unique_keywords = list(dict.fromkeys(expanded_keywords))
            expanded_query = f"{query} ({', '.join(unique_keywords[:4])})"
            logger.debug("query_ontological_expansion_applied", original=query, expanded=expanded_query)
            return expanded_query
            
        return query

    def _normalize_l2_score(self, faiss_score: float) -> float:
        """Transforms raw FAISS L2 distance into a bounded 0.0 - 1.0 similarity score (Higher is better)."""
        # في مسافات L2: الصفر يمثل تطابقاً مطلقاً. نقوم بقلب المعادلة رياضياً بشكل آمن.
        return 1.0 / (1.0 + float(faiss_score))

    def _diverse_results(self, chunks: List[RetrievedChunk], similarity_threshold: float = 0.85) -> List[RetrievedChunk]:
        """Filters out high-density textual overlaps to preserve agent context memory window from redundancy."""
        unique_chunks: List[RetrievedChunk] = []
        
        for chunk in chunks:
            is_duplicate = False
            for existing in unique_chunks:
                # استخدام SequenceMatcher المدمج والآمن لتقييم مدى التشابه النصي
                ratio = difflib.SequenceMatcher(None, chunk.content, existing.content).ratio()
                if ratio > similarity_threshold:
                    is_duplicate = True
                    logger.debug("redundant_regulatory_chunk_filtered", 
                                 source=chunk.source_file, page=chunk.page_number, ratio=f"{ratio:.2f}")
                    break
            if not is_duplicate:
                unique_chunks.append(chunk)
                
        return unique_chunks

    async def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        initial_k: int = 20,
        min_score: float = 0.3
    ) -> List[RetrievedChunk]:
        """Performs localized vector search combined with a deep learning re-ranking protocol."""
        # 1. التدقيق والبحث في الـ Cache لضمان السرعة التشغيلية الفائقة للوكيل
        cache_key = hash(f"{query}|{top_k}|{min_score}")
        if cache_key in self._cache:
            self._hit_count += 1
            logger.debug("retriever_cache_hit", query=query)
            return self._cache[cache_key]
            
        self._miss_count += 1
        
        # 2. حقن التوسيع الأنطولوجي لربط الـ Feature بالبند القانوني المقابل له
        expanded_query = self._expand_query_with_ontology(query)
        
        # 3. الاسترجاع الأولي الكثيف من قاعدة البيانات المتجهية المحلية
        # جلب أفق أوسع لتمكين الـ Reranker من العمل بمرونة هندسية
        docs_with_scores = await asyncio.to_thread(
            self._vector_store.similarity_search_with_score, expanded_query, k=initial_k
        )
        
        if not docs_with_scores:
            return []
            
        raw_chunks: List[RetrievedChunk] = []
        for doc, raw_score in docs_with_scores:
            norm_score = self._normalize_l2_score(raw_score)
            chunk = RetrievedChunk(
                content=doc.page_content,
                source_file=doc.metadata.get("source_file", "unknown_policy.pdf"),
                page_number=int(doc.metadata.get("page_number", 0)),
                relevance_score=norm_score,
                reranker_score=0.0
            )
            raw_chunks.append(chunk)

        # 4. تفعيل أداة إعادة الترتيب العميقة (Cross-Encoder Reranking Shield)
        if self._reranker and raw_chunks:
            try:
                pairs = [[query, chunk.content] for chunk in raw_chunks]
                # استدعاء نموذج الـ CrossEncoder المحلي أوفلاين بالكامل
                reranker_scores = await asyncio.to_thread(self._reranker.predict, pairs)
                
                for idx, score in enumerate(reranker_scores):
                    # تحديث أوزان الكتل بناء على مخرجات مفسر الترتيب
                    # نقوم بدمج الـ Score بعد تطبيعه داخل الكائن المعدل
                    object.__setattr__(raw_chunks[idx], 'reranker_score', float(score))
                    # دمج الأوزان للحصول على تقييم نهائي عيار 0-1
                    final_score = (raw_chunks[idx].relevance_score * 0.4) + (float(score) * 0.6)
                    object.__setattr__(raw_chunks[idx], 'relevance_score', max(0.0, min(1.0, final_score)))
                    
                # إعادة رص المصفوفة خطياً تنازلياً حسب دقة المطابقة الأعلى
                raw_chunks.sort(key=lambda x: x.relevance_score, reverse=True)
                logger.debug("cross_encoder_reranking_complete", top_score=raw_chunks[0].relevance_score)
            except Exception as e:
                logger.warning("reranker_failed_falling_back_to_vector_scores", error=str(e))

        # 5. الفلترة والتنظيف (تصفية المخرجات الضعيفة ومنع التشابه الكثيف)
        filtered_chunks = [c for c in raw_chunks if c.relevance_score >= min_score]
        diverse_chunks = self._diverse_results(filtered_chunks, similarity_threshold=0.85)
        final_results = diverse_chunks[:top_k]
        
        # إدارة حجم مخزن الـ Cache لمنع تضخم الذاكرة في السيرفر الجوي
        if len(self._cache) >= self._cache_size:
            first_key = next(iter(self._cache))
            self._cache.pop(first_key)
            
        self._cache[cache_key] = final_results
        return final_results

    async def hyde_search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """Executes a Hypothetical Document Embeddings (HyDE) cycle to capture regulatory intents."""
        if not self._llm:
            logger.warning("hyde_search_requested_but_llm_unavailable")
            return []
            
        try:
            # صياغة موجه طيران صارم لمنع الهلوسة واستخراج مسودة تشريعية افتراضية دقيقة
            hyde_prompt = (
                f"You are an aviation regulatory compliance expert. Write a brief, highly technical hypothetical "
                f"excerpt from the FAA Part 107 or EASA SORA regulations that precisely answers this query: '{query}'. "
                f"Use official legal language, terms like compliance, mitigation, or certification. Do not add intro or outro."
            )
            
            # توليد الوثيقة الافتراضية حياً عبر محرك Groq السريع
            hypothetical_answer = await self._llm.generate(hyde_prompt)
            logger.debug("hyde_hypothetical_document_generated", doc_len=len(hypothetical_answer))
            
            # البحث باستخدام المسودة الافتراضية للحصول على نصوص حقيقية مطابقة دلالياً للهيكل التنظيمي
            hyde_results = await self.hybrid_search(query=hypothetical_answer, top_k=top_k, min_score=0.25)
            return hyde_results
        except Exception as exc:
            logger.error("hyde_search_pipeline_failed", error=str(exc))
            return []

    async def adaptive_search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """Dynamically shifts search strategies based on information density and baseline scores."""
        # البدء فوراً بالبحث الهجين القياسي والمحمي بالـ Reranker
        hybrid_results = await self.hybrid_search(query=query, top_k=top_k, min_score=0.3)
        
        # حساب متوسط كفاءة وجودة البيانات المسترجعة حياً لتقييم الموقف
        avg_score = sum(c.relevance_score for c in hybrid_results) / len(hybrid_results) if hybrid_results else 0.0
        
        logger.info("adaptive_search_checkpoint", 
                    query=query, results_found=len(hybrid_results), avg_score=f"{avg_score:.2f}")

        # درع حماية الطيران الذكي: إذا كانت النتائج شحيحة أو جودتها متدنية وخارج نطاق الثقة، نطلق الـ HyDE فوراً
        if len(hybrid_results) < 3 or avg_score < 0.45:
            logger.info("launching_hyde_recovery_protocol", reason="Low density or precision drop")
            hyde_results = await self.hyde_search(query=query, top_k=top_k)
            
            # دمج النتائج بالكامل وتمريرها لـ الفلتر لمنع تكرار الرموز الفرعية ذات المعرف المتطابق
            combined_pool = hybrid_results + hyde_results
            deduplicated_pool: List[RetrievedChunk] = []
            seen_ids = set()
            
            for chunk in combined_pool:
                if chunk.chunk_id not in seen_ids:
                    seen_ids.add(chunk.chunk_id)
                    deduplicated_pool.append(chunk)
            
            # إعادة فرز وتصفية القائمة المدمجة لضمان تقديم الزبدة التشريعية للـ Agent
            deduplicated_pool.sort(key=lambda x: x.relevance_score, reverse=True)
            return deduplicated_pool[:top_k]
            
        return hybrid_results

    def clear_cache(self) -> None:
        """Purges the operational retriever cache completely."""
        self._cache.clear()
        logger.info("retriever_cache_purged_successfully")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Returns diagnostic metrics regarding cache efficiency and hit-rates."""
        total_requests = self._hit_count + self._miss_count
        hit_rate = (self._hit_count / total_requests) if total_requests > 0 else 0.0
        return {
            "cache_current_size": len(self._cache),
            "total_calls_received": total_requests,
            "hit_rate_percentage": f"{hit_rate * 100:.1f}%"
        }

# =====================================================================
# Stage 2 Architectural Dependency Comment Block:
# This component orchestrates advanced query routing and dense retrieval optimization.
# Dependencies: src/uav_risk/stage2/rag/schemas.py -> RetrievedChunk
# Dependent Files: Coupled directly into src/uav_risk/stage2/rag/rag_core.py
# =====================================================================