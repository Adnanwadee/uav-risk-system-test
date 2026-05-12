"""
Enhanced Legal Retriever for UAV RAG System (V3.0 - Improved)
============================================================
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from .groq_llm import GroqLLM
from .prompts import HYDE_PROMPT

logger = logging.getLogger("EnhancedRetriever")


class EnhancedLegalRetriever:
    """محسن الاسترجاع مع تحسينات كبيرة في الجودة"""
    
    def __init__(self, vector_store: FAISS, embeddings: HuggingFaceEmbeddings, 
                 reranker: CrossEncoder, llm: Optional[GroqLLM] = None):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.reranker = reranker
        self.llm = llm
    
    async def hybrid_search(self, query: str, top_k: int = 10, initial_k: int = 60) -> List[Dict[str, Any]]:
        """البحث الهجين المحسن مع جلب نتائج أكثر"""
        
        docs = self.vector_store.similarity_search_with_score(query, k=initial_k)
        
        if not docs:
            return []
        
        pairs = [[query, doc[0].page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        
        results = []
        for i, (doc, score) in enumerate(zip([d[0] for d in docs], scores)):
            normalized_score = (float(score) + 10) / 20
            normalized_score = max(0.0, min(1.0, normalized_score))
            
            results.append({
                "page_content": doc.page_content,
                "metadata": {
                    "source": doc.metadata.get("source", "Unknown"),
                    "article_id": doc.metadata.get("article_id", "N/A"),
                    "score": normalized_score,
                },
                "relevance": normalized_score
            })
        
        results.sort(key=lambda x: x["relevance"], reverse=True)
        
        # إزالة التكرارات
        seen_content = set()
        unique_results = []
        for r in results:
            content_hash = hash(r["page_content"][:200])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(r)
        
        unique_results.sort(key=lambda x: x["relevance"], reverse=True)
        
        logger.info(f"   Retrieved {len(docs)} docs → {len(unique_results)} unique → {min(top_k, len(unique_results))} final")
        
        return unique_results[:top_k]
    
    async def hyde_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """HyDE مع تحسين الوثيقة الافتراضية"""
        
        if not self.llm:
            return await self.hybrid_search(query, top_k)
        
        prompt = HYDE_PROMPT.format(query=query)
        hypothetical = await self.llm.generate(prompt, include_system=False)
        
        enhanced_query = f"Original query: {query}\n\nRegulatory context: {hypothetical[:800]}"
        
        docs = self.vector_store.similarity_search_with_score(enhanced_query, k=top_k * 3)
        
        results = []
        for doc, score in docs:
            normalized_score = max(0.0, min(1.0, 1.0 / (1.0 + score)))
            results.append({
                "page_content": doc.page_content,
                "metadata": {
                    "source": doc.metadata.get("source", "Unknown"),
                    "score": normalized_score,
                    "hyde_used": True
                },
                "relevance": normalized_score
            })
        
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:top_k]
    
    def _diverse_results(self, results: List[Dict]) -> List[Dict]:
        """تحسين التنوع مع الحفاظ على الجودة"""
        
        if len(results) <= 3:
            return results
        
        faa_results = [r for r in results if "FAA" in r["metadata"].get("source", "")]
        easa_results = [r for r in results if "EASA" in r["metadata"].get("source", "")]
        
        diverse = []
        
        for source_results in [faa_results, easa_results]:
            for i, r in enumerate(source_results[:3]):
                if r not in diverse:
                    diverse.append(r)
        
        if len(diverse) < 5:
            remaining = [r for r in results if r not in diverse]
            diverse.extend(remaining[:5 - len(diverse)])
        
        seen = set()
        final = []
        for r in diverse:
            key = r["page_content"][:150]
            if key not in seen:
                seen.add(key)
                final.append(r)
        
        return final[:8]
    
    async def adaptive_search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """بحث تكيفي مع تحسين الجودة"""
        
        query_lower = query.lower()
        
        comparison_keywords = ["compare", "difference", "between", "vs", "versus"]
        is_comparison = any(kw in query_lower for kw in comparison_keywords)
        
        specific_keywords = ["section", "paragraph", "article", "§", "part"]
        is_specific = any(kw in query_lower for kw in specific_keywords)
        
        if is_comparison and self.llm:
            results = await self.hyde_search(query, top_k)
        elif is_specific:
            results = await self.hybrid_search(query, top_k, initial_k=80)
        else:
            results = await self.hybrid_search(query, top_k)
        
        results = self._diverse_results(results)
        
        faa_count = sum(1 for r in results if "FAA" in r["metadata"].get("source", ""))
        easa_count = sum(1 for r in results if "EASA" in r["metadata"].get("source", ""))
        
        logger.info(f"   Final: FAA={faa_count}, EASA={easa_count}, Total={len(results)}")
        
        return results