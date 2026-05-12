"""
Async RAG Core Engine (V17.0 - Fixed with Debug)
=================================================
"""

import os
import hashlib
import asyncio
import logging
import time
from pathlib import Path
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

from .config import RAGConfig, GroqSettings
from .groq_llm import GroqLLM, GroqLLMConfig
from .enhanced_legal_agent import EnhancedLegalAgent, LegalAnswer
from .enhanced_retriever import EnhancedLegalRetriever

logger = logging.getLogger("AsyncRAGCore")

MODELS_DIR = Path(__file__).resolve().parent.parent / "knowledge" / "models"
EMBEDDING_PATH = MODELS_DIR / "embedding"
RERANKER_PATH = MODELS_DIR / "reranker"


class AsyncRAGCore:
    """المحرك الأساسي مع دعم التصحيح"""
    
    def __init__(self, config: Optional[RAGConfig] = None, groq_api_key: Optional[str] = None):
        self.config = config or RAGConfig()
        self.debug_mode = self.config.DEBUG_MODE
        
        self._pool = ThreadPoolExecutor(max_workers=self.config.MAX_THREADS)
        self._db: Optional[FAISS] = None
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._reranker: Optional[CrossEncoder] = None
        self._retriever: Optional[EnhancedLegalRetriever] = None
        self._llm: Optional[GroqLLM] = None
        self._legal_agent: Optional[EnhancedLegalAgent] = None
        
        self.metrics = {"searches": 0, "cache_hits": 0, "errors": 0}
        
        self._initialize_embeddings_and_db()
        self._initialize_reranker()
        self._initialize_llm_and_agent(groq_api_key)
        self._initialize_retriever()
        
        if self.debug_mode:
            self._print_status()
    
    def _initialize_embeddings_and_db(self):
        if self.debug_mode:
            logger.info("[1/4] Loading embeddings and FAISS...")
        
        try:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=str(EMBEDDING_PATH),
                model_kwargs={'device': 'cpu', 'local_files_only': True}
            )
            
            self._db = FAISS.load_local(
                str(self.config.INDEX_PATH),
                self._embeddings,
                allow_dangerous_deserialization=True
            )
            
            if self.debug_mode:
                logger.info(f"   FAISS loaded from {self.config.INDEX_PATH}")
        except Exception as e:
            logger.error(f"Failed to load FAISS: {e}")
            self._db = None
    
    def _initialize_reranker(self):
        if self.debug_mode:
            logger.info("[2/4] Loading reranker...")
        
        try:
            self._reranker = CrossEncoder(str(RERANKER_PATH), device='cpu')
            if self.debug_mode:
                logger.info(f"   Reranker loaded from {RERANKER_PATH}")
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
            self._reranker = None
    
    def _initialize_llm_and_agent(self, groq_api_key: Optional[str] = None):
        if self.debug_mode:
            logger.info("[3/4] Loading Groq LLM and Legal Agent...")
        
        api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        
        if not api_key:
            logger.warning("No Groq API key. LLM disabled.")
            return
        
        try:
            llm_config = GroqLLMConfig(api_key=api_key)
            self._llm = GroqLLM(llm_config)
            self._legal_agent = EnhancedLegalAgent(self._llm, self.config)
            
            if self.debug_mode:
                logger.info(f"   Groq LLM ready (model: {llm_config.model})")
        except Exception as e:
            logger.error(f"Failed to load Groq: {e}")
    
    def _initialize_retriever(self):
        if self.debug_mode:
            logger.info("[4/4] Initializing retriever...")
        
        try:
            self._retriever = EnhancedLegalRetriever(
                vector_store=self._db,
                embeddings=self._embeddings,
                reranker=self._reranker,
                llm=self._llm
            )
            if self.debug_mode:
                logger.info("   Retriever ready")
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
    
    def _print_status(self):
        print("\n" + "=" * 60)
        print("RAG SYSTEM STATUS")
        print("=" * 60)
        print(f"FAISS Index: {'✅' if self._db else '❌'}")
        print(f"Embeddings: {'✅' if self._embeddings else '❌'}")
        print(f"Reranker: {'✅' if self._reranker else '❌'}")
        print(f"Groq LLM: {'✅' if self._llm else '❌'}")
        print(f"Legal Agent: {'✅' if self._legal_agent else '❌'}")
        print(f"Retriever: {'✅' if self._retriever else '❌'}")
        print(f"Debug Mode: {'✅' if self.debug_mode else '❌'}")
        print("=" * 60 + "\n")
    
    async def search(self, query: str, top_k: int = 8, min_score: float = 0.3) -> List[Dict]:
        """البحث مع طباعة تفصيلية"""
        
        if self.debug_mode:
            logger.info(f"\nSEARCH: '{query[:60]}...'")
            start_time = time.time()
        
        if not self._retriever:
            logger.error("Retriever not available")
            return []
        
        try:
            results = await self._retriever.adaptive_search(query, top_k)
            filtered = [r for r in results if r.get("relevance", 0) >= min_score]
            
            if self.debug_mode:
                elapsed = (time.time() - start_time) * 1000
                logger.info(f"   Found {len(results)} docs, filtered to {len(filtered)} (min_score={min_score})")
                logger.info(f"   Search time: {elapsed:.0f}ms")
                
                for i, r in enumerate(filtered[:4]):
                    score = r.get("relevance", 0)
                    source = r["metadata"].get("source", "Unknown")
                    logger.info(f"      [{i+1}] score={score:.3f} | source={source}")
            
            return filtered
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            self.metrics["errors"] += 1
            return []
    
    async def ask_legal_question(self, query: str, top_k: int = 8, min_score: float = 0.3) -> LegalAnswer:
        """سؤال قانوني مع طباعة تفصيلية"""
        
        if self.debug_mode:
            logger.info(f"\nLEGAL QUESTION: '{query[:80]}...'")
        
        results = await self.search(query, top_k, min_score)
        
        if not results:
            return LegalAnswer(
                answer="No relevant information found.",
                citations=[],
                confidence_score=0.0,
                debug_info={"error": "no_results"}
            )
        
        if not self._legal_agent:
            return LegalAnswer(
                answer="Legal Agent not available.",
                citations=[],
                confidence_score=0.0
            )
        
        answer = await self._legal_agent.build_final_answer(query, results)
        
        if self.debug_mode:
            logger.info(f"   Answer ready. Confidence: {answer.confidence_score:.3f}")
            logger.info(f"   Citations: {len(answer.citations)}")
        
        return answer
    
    def shutdown(self):
        if hasattr(self, '_pool'):
            self._pool.shutdown(wait=False)