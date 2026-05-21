"""
Module: src/uav_risk/stage2/rag/rag_core.py
Author: Elite Technical Partner
Description: Central lifecycle orchestrator for the Legislative RAG subsystem,
             supporting structural degraded modes and strict semantic routing.
"""

import os
import time
from typing import List, Optional, Dict, Any
import structlog
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# الاستيراد المطلق الصارم للعقود والمكونات المطهّرة
from uav_risk.stage2.rag.schemas import RetrievedChunk, LegalCitation, LegalAnswer
from uav_risk.stage2.rag.config import RAGConfig, GroqLLMConfig
from uav_risk.stage2.rag.groq_llm import GroqLLM
from uav_risk.stage2.rag.enhanced_retriever import EnhancedLegalRetriever
from uav_risk.stage2.rag.enhanced_legal_agent import EnhancedLegalAgent

logger = structlog.get_logger()


class AsyncRAGCore:
    """The exclusive entry point for aviation regulatory indexing, retrieval, and synthesis."""

    def __init__(self, config: Optional[RAGConfig] = None, groq_api_key: Optional[str] = None):
        self.config = config or RAGConfig()
        self._groq_api_key = groq_api_key
        
        # حواضن الكائنات والميكرو-مكونات الداخلية
        self._db: Optional[FAISS] = None
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._reranker: Optional[Any] = None
        self._llm: Optional[GroqLLM] = None
        self._retriever: Optional[EnhancedLegalRetriever] = None
        self._agent: Optional[EnhancedLegalAgent] = None
        
        self._initialized = False
        self._init_status: Dict[str, str] = {
            "db": "uninitialized",
            "reranker": "uninitialized",
            "llm": "uninitialized",
            "retriever": "uninitialized"
        }

    async def initialize(self) -> bool:
        """Sequential initialization of components with robust degraded mode isolation fallback."""
        start_time = time.perf_counter()
        logger.info("rag_core_lifecycle_initialization_started")

        await self._initialize_embeddings_and_db()
        await self._initialize_reranker()
        await self._initialize_llm_and_agent()
        await self._initialize_retriever()

        self._initialized = True
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info("rag_core_lifecycle_initialization_concluded", 
                    ready=self.is_ready(), elapsed_ms=f"{elapsed_ms:.2f}ms", status=self._init_status)
        return self.is_ready()

    async def _initialize_embeddings_and_db(self) -> None:
        """Loads embedding matrices and vector files entirely offline from verified config paths."""
        try:
            emb_path = str(self.config.EMBEDDING_PATH)
            idx_path = str(self.config.INDEX_PATH)

            if not os.path.exists(emb_path) or not os.path.exists(idx_path):
                raise FileNotFoundError(f"Missing core assets paths: {emb_path} or {idx_path}")

            self._embeddings = HuggingFaceEmbeddings(
                model_name=emb_path,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            self._db = FAISS.load_local(
                folder_path=idx_path,
                embeddings=self._embeddings,
                allow_dangerous_deserialization=True
            )
            self._init_status["db"] = "ready"
        except Exception as exc:
            self._init_status["db"] = f"failed: {str(exc)}"
            logger.error("vector_db_loading_failed_fallback_active", error=str(exc))

    async def _initialize_reranker(self) -> None:
        """Loads local cross-encoder weights if populated inside current knowledge pool."""
        try:
            rank_path = str(self.config.RERANKER_PATH)
            if os.path.exists(rank_path) and os.listdir(rank_path):
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(rank_path)
                self._init_status["reranker"] = "ready"
            else:
                self._reranker = None
                self._init_status["reranker"] = "unavailable"
        except Exception as exc:
            self._reranker = None
            self._init_status["reranker"] = f"failed: {str(exc)}"
            logger.warning("reranker_load_skipped_continuing_safely", error=str(exc))

    async def _initialize_llm_and_agent(self) -> None:
        """Establishes authenticated async gateway instances to the Groq processing engine."""
        try:
            api_key = self._groq_api_key or os.getenv("GROQ_API_KEY")
            if not api_key:
                self._init_status["llm"] = "no_api_key"
                logger.warning("llm_initialization_skipped_empty_credentials")
                return

            llm_config = GroqLLMConfig(api_key=api_key)
            self._llm = GroqLLM(config=llm_config)
            self._agent = EnhancedLegalAgent(llm=self._llm, config=self.config)
            self._init_status["llm"] = "ready"
        except Exception as exc:
            self._llm = None
            self._agent = None
            self._init_status["llm"] = f"failed: {str(exc)}"
            logger.error("llm_gateway_setup_failed", error=str(exc))

    async def _initialize_retriever(self) -> None:
        """Binds the active storage engines inside the adaptive retriever core pipeline."""
        if self._db is None:
            self._init_status["retriever"] = "no_db"
            return
        try:
            self._retriever = EnhancedLegalRetriever(
                vector_store=self._db,
                embeddings=self._embeddings,
                reranker=self._reranker,
                llm=self._llm,
                cache_size=128
            )
            self._init_status["retriever"] = "ready"
        except Exception as exc:
            self._retriever = None
            self._init_status["retriever"] = f"failed: {str(exc)}"
            logger.error("retriever_attachment_aborted", error=str(exc))

    async def search(self, query: str, top_k: int = 5, min_score: float = 0.3) -> List[RetrievedChunk]:
        """Exposes clean, deduplicated vector retrieval search matching user specifications."""
        if not self._retriever:
            logger.error("search_rejected_retriever_offline")
            return []
        try:
            # استخدام البحث المتكيف الذكي المانع لحشو السياق النصي
            return await self._retriever.adaptive_search(query=query, top_k=top_k)
        except Exception as exc:
            logger.error("search_execution_failed_empty_pool_returned", error=str(exc))
            return []

    async def ask_legal_question(self, query: str, top_k: int = 5, min_score: float = 0.3) -> LegalAnswer:
        """Principal compliance portal queried by the ReAct Agent to return verified answers."""
        if not self._retriever or self._db is None:
            logger.warning("rag_operating_in_hardcoded_degraded_mode_due_to_missing_assets")
            return LegalAnswer(
                query=query,
                answer="Legal RAG unavailable. Manual aviation regulatory review required.",
                citations=[],
                confidence_score=0.0,
                rag_available=False
            )

        try:
            chunks = await self.search(query=query, top_k=top_k, min_score=min_score)
            
            # نمط الطوارئ المحلي الصارم (Local Degraded Template Mode) عند سقوط الـ LLM السحابي
            if not self._agent or not self._llm:
                logger.info("llm_absent_assembling_raw_regulatory_structural_bundle")
                citations = [
                    LegalCitation(source_file=c.source_file, page_number=c.page_number, full_text=c.content)
                    for c in chunks
                ]
                raw_bundle = "### [LOCAL DEGRADED MODE - RAW REGULATORY EXCERPTS]:\n\n" + "\n\n".join([c.to_citation_text() for c in chunks])
                return LegalAnswer(
                    query=query,
                    answer=raw_bundle,
                    citations=citations,
                    confidence_score=0.4,
                    rag_available=True
                )

            # النمط الكامل المتكامل عبر عميل الاستنتاج والوكيل القانوني الذكي
            return await self._agent.build_final_answer(query=query, chunks=chunks)
        except Exception as exc:
            logger.critical("fatal_failure_inside_legal_question_pipeline", error=str(exc))
            return LegalAnswer(
                query=query,
                answer=f"Pipeline error encountered during regulatory extraction: {str(exc)}",
                citations=[],
                confidence_score=0.0,
                rag_available=False
            )

    def is_ready(self) -> bool:
        """True if the vector database layer is successfully attached and online."""
        return self._db is not None

    def get_status(self) -> Dict[str, Any]:
        """Provides operational metrics mapping the diagnostic condition of microcomponents."""
        return {
            **self._init_status,
            "overall_ready": self.is_ready(),
            "cache_metrics": self._retriever.get_cache_stats() if self._retriever else None
        }

    async def shutdown(self):
        """Safely purges in-memory caching pools to completely free system footprint blocks."""
        logger.info("rag_core_shutdown_sequence_triggered")
        if self._retriever:
            self._retriever.clear_cache()
        self._db = None
        self._embeddings = None
        self._reranker = None
        self._llm = None
        self._agent = None
        logger.info("rag_core_shutdown_sequence_finalized")


# =====================================================================
# Stage 2 Architectural Dependency Comment Block:
# Main orchestrator governing lifecycle, cache metrics and state switches of RAG components.
# Dependencies: src/uav_risk/stage2/rag/schemas.py -> Contracts Models
#               src/uav_risk/stage2/rag/config.py -> Paths Profiles
#               src/uav_risk/stage2/rag/enhanced_retriever.py -> Search Nodes
#               src/uav_risk/stage2/rag/enhanced_legal_agent.py -> Answer Shields
# Dependent Files: Evaluated globally inside web engine via src/uav_risk/api/main.py
# =====================================================================