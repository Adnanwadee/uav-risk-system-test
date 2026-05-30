"""
HyDE Pipeline - Targeted Hypothetical Document Embedding
Generates contextual hypothetical documents based on query features.
"""
# STAGE6_CLEANUP_REVIEW:
# Classification: OPTIONAL_RAG_COMPONENT_KEEP_NOW
# Plan lineage: PLAN3_OPTIONAL_RAG_SUPPORT
# Runtime status: Imported by rag_core_v3.py as optional HyDE/query-expansion support.
# Legacy signal: Not an evidence source by itself; keep while rag_core_v3 imports TargetedHyDE.
# Replacement: None currently. Review only if rag_core_v3 no longer imports or uses TargetedHyDE.
# Action rule: Do not delete now. Any cleanup must first update rag_core_v3 and RAG quality/runtime diagnostics.
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class HyDEResult:
    """Result from HyDE generation"""
    hypothetical_doc: str
    query_embedding: Optional[List[float]] = None
    confidence: float = 0.0
    generation_time_ms: float = 0.0

class TargetedHyDE:
    """
    Targeted HyDE that generates hypothetical documents
    based on specific feature values, not generic text.
    """

    def __init__(self, llm_client=None, embedding_model=None):
        self.llm = llm_client
        self.embedder = embedding_model

    def _build_prompt(self, query: str, features: Dict[str, Any], 
                     scenario_type: str) -> str:
        """Build targeted prompt for HyDE generation"""

        # Extract key features for context
        feature_context = ""
        critical_features = [
            "flight_altitude_m", "wind_speed_kt", "temperature_c",
            "visibility_km", "obstacle_proximity_m", "battery_capacity_mah",
            "communication_range_km", "population_density", "airspace_class"
        ]

        for feat in critical_features:
            if feat in features:
                feature_context += f"- {feat}: {features[feat]}\n"

        prompt = f"""You are an aviation regulatory expert. Given the following UAV operational scenario, generate a detailed regulatory document excerpt that would be relevant to this situation.

SCENARIO TYPE: {scenario_type}
QUERY: {query}

OPERATIONAL PARAMETERS:
{feature_context}

Generate a 200-word regulatory document excerpt that:
1. Addresses the specific operational parameters above
2. Cites relevant aviation regulations (EASA, FAA, or ICAO)
3. Provides clear operational limits or requirements
4. Uses formal regulatory language

The document should be factual, specific, and directly relevant to the scenario. Do not include disclaimers or notes about being AI-generated."""

        return prompt

    async def generate(self, query: str, 
                      features: Dict[str, Any],
                      scenario_type: str = "general",
                      max_tokens: int = 300) -> HyDEResult:
        """
        Generate targeted hypothetical document.

        Args:
            query: Original search query
            features: Operational features for targeting
            scenario_type: Classified scenario type
            max_tokens: Max generation length

        Returns:
            HyDEResult with hypothetical document
        """
        import time
        start_time = time.time()

        if not self.llm:
            logger.warning("No LLM client available for HyDE")
            return HyDEResult(
                hypothetical_doc=query,
                confidence=0.0,
                generation_time_ms=0.0
            )

        try:
            # Build targeted prompt
            prompt = self._build_prompt(query, features, scenario_type)

            # Generate hypothetical document
            # Assuming llm_client has async generate method
            hypothetical_doc = await self.llm.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.3  # Low temp for factual consistency
            )

            # Generate embedding if embedder available
            query_embedding = None
            if self.embedder:
                query_embedding = await self.embedder.embed(hypothetical_doc)

            # Calculate confidence based on length and specificity
            confidence = self._calculate_confidence(hypothetical_doc, features)

            elapsed = (time.time() - start_time) * 1000

            return HyDEResult(
                hypothetical_doc=hypothetical_doc,
                query_embedding=query_embedding,
                confidence=confidence,
                generation_time_ms=elapsed
            )

        except Exception as e:
            logger.error(f"HyDE generation failed: {e}")
            return HyDEResult(
                hypothetical_doc=query,
                confidence=0.0,
                generation_time_ms=(time.time() - start_time) * 1000
            )

    def _calculate_confidence(self, doc: str, features: Dict[str, Any]) -> float:
        """Calculate confidence score for generated document"""
        score = 0.5

        # Length check
        word_count = len(doc.split())
        if 100 <= word_count <= 400:
            score += 0.2

        # Feature mention check
        feature_mentions = sum(1 for f in features.keys() if f.replace("_", " ") in doc.lower())
        score += min(0.2, feature_mentions / 10)

        # Regulatory citation check
        regulatory_terms = ["regulation", "EASA", "FAA", "ICAO", "compliance", 
                           "requirement", "limit", "authorization", "certificate"]
        citation_count = sum(1 for term in regulatory_terms if term.lower() in doc.lower())
        score += min(0.1, citation_count / 5)

        return min(1.0, score)

    async def generate_batch(self, queries: List[str],
                          features_list: List[Dict[str, Any]],
                          scenario_types: List[str],
                          max_concurrent: int = 3) -> List[HyDEResult]:
        """
        Generate HyDE for multiple queries with concurrency control.

        Args:
            queries: List of queries
            features_list: List of feature dicts
            scenario_types: List of scenario types
            max_concurrent: Max parallel generations

        Returns:
            List of HyDEResults
        """
        import asyncio
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _generate_single(q, f, s):
            async with semaphore:
                return await self.generate(q, f, s)

        tasks = [
            _generate_single(q, f, s)
            for q, f, s in zip(queries, features_list, scenario_types)
        ]

        return await asyncio.gather(*tasks)