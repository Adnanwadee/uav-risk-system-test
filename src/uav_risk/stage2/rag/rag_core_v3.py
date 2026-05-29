"""
RAG Core V3.1 - Intelligent Orchestrator
Production Fixes:
- Fixed imports with package prefix
- Proper dense doc_id mapping support
- Reranker integration
- Enhanced error handling
"""
import os
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RAGResult:
    """Final RAG result"""
    documents: List[Dict]
    analysis: Dict[str, Any]
    scenario_type: str
    confidence: float
    latency_ms: float
    evidence_log_id: Optional[str] = None

class AsyncRAGCoreV3:
    """
    Intelligent RAG Orchestrator V3.1
    """

    def __init__(self, 
                 config_module=None,
                 llm_client=None,
                 embedder=None,
                 reranker=None,
                 dense_index=None,
                 sparse_index_builder=None,
                 index_dir: Optional[str] = None):
        self.config = config_module
        self.llm = llm_client
        self.embedder = embedder
        self.reranker = reranker
        self.dense_index = dense_index
        self.sparse_builder = sparse_index_builder
        self.index_dir = Path(index_dir).expanduser().resolve() if index_dir else None

        self.intelligence = None
        self.mapper = None
        self.retriever = None
        self.hyde = None
        self.evidence_log = None

        self._initialized = False

    def _resolve_index_dir(self) -> Path:
        if self.index_dir is not None:
            return self.index_dir

        config_index_dir = getattr(self.config, "INDEX_DIR", None) if self.config else None
        if config_index_dir:
            return Path(config_index_dir).expanduser().resolve()

        from .config_v3 import get_index_dir

        return get_index_dir()

    async def initialize(self):
        """Initialize all components with safe loading"""
        if self._initialized:
            return

        logger.info("Initializing RAG Core V3.1...")

        # Import components with package prefix
        from .query_intelligence import QueryIntelligence
        from .feature_query_mapper import FeatureQueryMapper
        from .hybrid_retriever import HybridRetriever
        from .hyde_pipeline import TargetedHyDE
        from .evidence_logger import BoundedEvidenceLog
        from .faiss_security import verify_and_safely_load_faiss
        from .schemas import SyncEmbedderWrapper

        # Wrap sync embedder if needed
        if self.embedder and not asyncio.iscoroutinefunction(getattr(self.embedder, 'embed', None)):
            logger.info("Wrapping sync embedder with SyncEmbedderWrapper")
            self.embedder = SyncEmbedderWrapper(self.embedder)

        # Load dense index if not provided
        if self.dense_index is None:
            index_dir = self._resolve_index_dir()
            faiss_path = index_dir / 'dense_index.faiss'

            if faiss_path.exists():
                try:
                    secret = os.getenv('UAV_FAISS_SECRET')
                    self.dense_index, meta = verify_and_safely_load_faiss(
                        str(faiss_path),
                        secret_key=secret,
                        allow_unsigned=(secret is None)
                    )
                    logger.info(f"Loaded FAISS index: {faiss_path} (ntotal={self.dense_index.ntotal})")
                except Exception as e:
                    logger.error(f"Failed to load FAISS index: {e}")
                    self.dense_index = None
            else:
                logger.warning(f"FAISS index not found at {faiss_path}")

        # Initialize intelligence
        history_path = Path("query_history.json")
        if self.config and hasattr(self.config, "CACHE_DIR"):
            history_path = self.config.CACHE_DIR / "query_history.json"

        self.intelligence = QueryIntelligence(
            history_path=history_path,
            config_module=self.config
        )

        # Initialize mapper
        self.mapper = FeatureQueryMapper(config_module=self.config)

        # Build/load sparse index
        sparse_index = getattr(self, "_preloaded_sparse_index", None)
        if sparse_index is None and self.sparse_builder:
            try:
                sparse_index = await self.sparse_builder.build_or_load()
            except Exception as e:
                logger.error(f"Failed to build/load sparse index: {e}")

        # Initialize retriever with reranker
        index_dir_for_retriever = self._resolve_index_dir()
        self.retriever = HybridRetriever(
            dense_index=self.dense_index,
            sparse_index=sparse_index,
            embedder=self.embedder,
            reranker=self.reranker,
            config=self.config,
            index_dir=str(index_dir_for_retriever) if index_dir_for_retriever else None
        )

        # Initialize HyDE
        self.hyde = TargetedHyDE(llm_client=self.llm, embedding_model=self.embedder)

        # Initialize evidence log
        log_dir = Path("evidence_logs")
        max_entries = 1000
        if self.config:
            log_dir = getattr(self.config, "LOG_DIR", log_dir)
            max_entries = getattr(self.config, "MAX_EVIDENCE_ENTRIES", 1000)

        self.evidence_log = BoundedEvidenceLog(
            max_entries=max_entries,
            log_dir=log_dir
        )

        self._initialized = True
        logger.info("RAG Core V3.1 initialized successfully")

    async def search_scenario(self,
                             core_features: Dict[str, Any],
                             optional_features: Optional[Dict[str, Any]] = None,
                             shap_features: Optional[List[Tuple[str, float]]] = None,
                             free_text: Optional[str] = None,
                             ml_risk_score: Optional[float] = None) -> RAGResult:
        """
        Main entry: Search for a complete scenario.
        """
        import time
        start_time = time.time()

        if not self._initialized:
            await self.initialize()

        # Step 1: Analyze scenario
        all_features = dict(core_features)
        if optional_features:
            all_features.update(optional_features)

        scenario_analysis = self.intelligence.analyze_scenario(
            all_features, free_text
        )
        scenario_type = scenario_analysis["scenario_type"]

        logger.info(f"Scenario classified as: {scenario_type} "
                   f"(complexity: {scenario_analysis['complexity']:.2f})")

        # Step 2: Map features to queries
        feature_queries = self.mapper.map_scenario(
            core_features, optional_features, shap_features
        )

        # Step 3: Decide on HyDE
        use_hyde = scenario_analysis["recommended_hyde"]

        # Step 4: Execute batch search with try/except fallback
        try:
            search_results = await self._execute_batch_search(
                feature_queries, all_features, scenario_type, use_hyde, free_text
            )
        except Exception as e:
            logger.error(f"Batch search failed: {e}. Falling back to sparse-only search.")
            # Fallback: try sparse-only search
            try:
                search_results = await self._fallback_search(
                    feature_queries, top_k=10
                )
            except Exception as e2:
                logger.error(f"Fallback search also failed: {e2}")
                search_results = [[] for _ in feature_queries]

        # Step 5: Deduplicate and rank
        final_docs = self._merge_and_rank(search_results)

        # Step 6: Calculate confidence
        confidence = self._calculate_confidence(final_docs, scenario_analysis)

        # Step 7: Log evidence
        latency = (time.time() - start_time) * 1000
        await self._log_evidence(
            feature_queries, final_docs, scenario_type, 
            confidence, latency, use_hyde
        )

        # Step 8: Build analysis
        reranker_status = {}
        if self.retriever is not None and hasattr(self.retriever, "get_reranker_status"):
            try:
                reranker_status = dict(self.retriever.get_reranker_status())
            except Exception:
                reranker_status = {}

        analysis = {
            "scenario_type": scenario_type,
            "complexity": scenario_analysis["complexity"],
            "priority_features": scenario_analysis["priority_features"],
            "risk_indicators": scenario_analysis["risk_indicators"],
            "hyde_used": use_hyde,
            "query_count": len(feature_queries),
            "retrieval_stats": {
                "total_docs": len(final_docs),
                "avg_score": sum(float(getattr(d, "final_score", 0.0)) for d in final_docs) / len(final_docs) if final_docs else 0,
                "top_score": float(getattr(final_docs[0], "final_score", 0.0)) if final_docs else 0
            },
            "reranker_status": reranker_status,
            "runtime_status": dict(getattr(self, "_runtime_status", {}) or {}),
        }

        # Step 9: Record for learning
        top_score = float(getattr(final_docs[0], "final_score", 0.0)) if final_docs else 0
        self.intelligence.record_query(
            query=feature_queries[0].query_text if feature_queries else "",
            scenario_type=scenario_type,
            used_hyde=use_hyde,
            top_score=top_score,
            result_count=len(final_docs),
            latency_ms=latency
        )

        return RAGResult(
            documents=[{
                "doc_id": d.doc_id,
                "chunk_id": d.chunk_id,
                "vector_id": d.vector_id,
                "text": d.text,
                "source": d.source,
                "source_id": d.source_id,
                "source_filename": d.source_filename,
                "source_title": d.source_title,
                "page_start": d.page_start,
                "page_end": d.page_end,
                "section_title": d.section_title,
                "text_sha256": d.text_sha256,
                "score": d.final_score,
                "final_score": d.final_score,
                "dense_score": d.dense_score,
                "sparse_score": d.sparse_score,
                "rerank_score": d.rerank_score,
                "source_match_score": d.source_match_score,
                "retrieval_method": d.retrieval_method,
                "domain_match": d.domain_match,
            } for d in final_docs],
            analysis=analysis,
            scenario_type=scenario_type,
            confidence=confidence,
            latency_ms=latency
        )

    async def _execute_batch_search(self,
                                   feature_queries: List,
                                   all_features: Dict,
                                   scenario_type: str,
                                   use_hyde: bool,
                                   free_text: Optional[str] = None) -> List[List]:
        """Execute batch search for all feature queries"""

        corpus_size = 0
        if self.retriever.sparse_index:
            corpus_size = self.retriever.sparse_index.get("N", 0)

        rrf_k = self.intelligence.get_optimal_rrf_k(corpus_size, scenario_type)

        queries = [fq.query_text for fq in feature_queries]
        if free_text and free_text.strip():
            q = free_text.strip()
            if q not in queries:
                queries.insert(0, q)

        # Generate HyDE if needed
        hyde_embeddings = None
        if use_hyde and self.hyde:
            logger.info("Generating HyDE embeddings...")
            hyde_results = await self.hyde.generate_batch(
                queries=queries,
                features_list=[all_features] * len(queries),
                scenario_types=[scenario_type] * len(queries),
                max_concurrent=3
            )

            hyde_queries = [h.hypothetical_doc for h in hyde_results]

            if self.embedder:
                hyde_embeddings = await asyncio.gather(*[
                    self.embedder.embed(q) for q in hyde_queries
                ])

        logger.info(f"Executing batch search for {len(queries)} queries...")
        search_results = await self.retriever.batch_search(
            queries=queries,
            query_embeddings=hyde_embeddings,
            top_k=10,
            max_concurrent=5
        )

        return search_results

    async def _fallback_search(self, feature_queries: List, top_k: int = 10) -> List[List]:
        """Fallback sparse-only search when hybrid fails"""
        logger.warning("Using fallback sparse-only search")
        results = []
        for fq in feature_queries:
            try:
                docs = await self.retriever.search_sparse(fq.query_text, top_k)
                results.append(docs)
            except Exception as e:
                logger.error(f"Fallback search failed for query: {e}")
                results.append([])
        return results

    def _merge_and_rank(self, search_results: List[List]) -> List:
        """Merge results from multiple queries and deduplicate"""
        from .hybrid_retriever import SimHash

        simhash = SimHash()
        all_docs = {}

        for results in search_results:
            for doc in results:
                doc_id = doc.doc_id

                if doc_id in all_docs:
                    if doc.final_score > all_docs[doc_id].final_score:
                        all_docs[doc_id] = doc
                else:
                    all_docs[doc_id] = doc

        # Deduplicate by content
        unique_docs = []
        seen_hashes = set()

        for doc in sorted(all_docs.values(), key=lambda d: d.final_score, reverse=True):
            doc_hash = doc.chunk_hash or str(simhash.compute(doc.text))

            if doc_hash not in seen_hashes:
                seen_hashes.add(doc_hash)
                unique_docs.append(doc)

        return unique_docs[:20]

    def _calculate_confidence(self, docs: List, 
                             scenario_analysis: Dict) -> float:
        """Calculate overall confidence score"""
        if not docs:
            return 0.0

        top_scores = [d.final_score for d in docs[:5]]
        avg_top = sum(top_scores) / len(top_scores) if top_scores else 0

        coverage = min(1.0, len(docs) / 10)
        complexity_penalty = scenario_analysis.get("complexity", 0) * 0.1

        confidence = (0.6 * avg_top + 0.3 * coverage - complexity_penalty)
        return max(0.0, min(1.0, confidence))

    async def _log_evidence(self,
                           feature_queries: List,
                           docs: List,
                           scenario_type: str,
                           confidence: float,
                           latency: float,
                           use_hyde: bool):
        """Log evidence for audit trail"""
        if not self.evidence_log:
            return

        # Log all queries, not just the first one
        all_queries = " | ".join([fq.query_text for fq in feature_queries[:5]])

        results = [{
            "doc_id": d.doc_id,
            "source": d.source,
            "score": d.final_score,
            "rerank_score": d.rerank_score
        } for d in docs]

        await self.evidence_log.log(
            query=all_queries,
            scenario_type=scenario_type,
            results=results,
            confidence_scores=[d.final_score for d in docs],
            retrieval_method="hyde" if use_hyde else "hybrid",
            latency_ms=latency,
            feature_count=len(feature_queries)
        )

    async def synthesize(self, 
                        rag_result: RAGResult,
                        validated_features: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize RAG results into final answer for Agent"""
        from .prompts_v3 import build_synthesis_prompt

        if not self.llm:
            logger.warning("No LLM available for synthesis")
            return {
                "status": "raw_results",
                "documents": rag_result.documents,
                "analysis": rag_result.analysis
            }

        prompt = build_synthesis_prompt(
            retrieval_results=rag_result.documents,
            scenario_analysis=rag_result.analysis,
            ml_risk_score=rag_result.analysis.get("ml_risk_score")
        )

        try:
            response = await self.llm.generate(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.2
            )

            import json
            try:
                parsed = json.loads(response)
                parsed["rag_metadata"] = {
                    "confidence": rag_result.confidence,
                    "latency_ms": rag_result.latency_ms,
                    "scenario_type": rag_result.scenario_type
                }
                return parsed
            except json.JSONDecodeError:
                return {
                    "status": "parse_error",
                    "raw_response": response,
                    "documents": rag_result.documents
                }

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "documents": rag_result.documents
            }

    async def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = {
            "initialized": self._initialized,
            "intelligence": self.intelligence.get_stats() if self.intelligence else {},
            "evidence": await self.evidence_log.get_stats() if self.evidence_log else {}
        }

        if self.sparse_builder:
            stats["sparse_index"] = self.sparse_builder.get_stats()

        return stats