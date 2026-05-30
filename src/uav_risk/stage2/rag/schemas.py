"""
Schemas - Data Models + Protocols for RAG System V3.1
"""
# STAGE6_CLEANUP_REVIEW:
# Classification: COMPATIBILITY_BRIDGE_KEEP_NOW
# Plan lineage: PLAN3_COMPATIBILITY_BRIDGE
# Runtime status: RAG adapter/tests still import SearchResult and canonical evidence aliases from this module.
# Legacy signal: Keeps old RAG schema import paths stable while re-exporting canonical stage2 contracts.
# Replacement: Canonical evidence models live in src/uav_risk/stage2/contracts.py.
# Action rule: Do not delete now. Review only after all RAG imports are migrated or compatibility tests are updated.
from typing import Optional, List, Dict, Any, Protocol, runtime_checkable
from dataclasses import dataclass, field
from datetime import datetime

from uav_risk.stage2.contracts import (
    EvidenceBundle,
    EvidenceCitation,
    EvidenceClaim,
    EvidenceOrigin,
    EvidenceSourceType,
    EvidenceSupportStatus,
    EvidenceUse,
    LegalCitation,
)

# ═══════════════════════════════════════════════════════════
# Protocols - توثيق الواجهات المطلوبة
# ═══════════════════════════════════════════════════════════

@runtime_checkable
class EmbedderProtocol(Protocol):
    """Protocol for embedding models"""

    async def embed(self, text: str) -> List[float]:
        """
        Embed text into vector.
        Must be async. If using sync embedder (HuggingFace),
        wrap with asyncio.to_thread().
        """
        ...


@runtime_checkable
class LLMProtocol(Protocol):
    """Protocol for LLM clients"""

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.2,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate text from prompt"""
        ...


@runtime_checkable
class RerankerProtocol(Protocol):
    """Protocol for cross-encoder rerankers"""

    def predict(self, pairs: List[tuple]) -> List[float]:
        """
        Rerank query-document pairs.
        Returns list of scores (higher = more relevant).
        """
        ...

    def compute_score(self, pair: tuple) -> float:
        """
        Compute score for single query-document pair.
        """
        ...


class SyncEmbedderWrapper:
    """
    Wrapper to convert sync embedder to async.
    Usage:
        sync_model = SentenceTransformer("model")
        embedder = SyncEmbedderWrapper(sync_model)
        embedding = await embedder.embed("text")
    """

    def __init__(self, sync_model):
        self.model = sync_model

    async def embed(self, text: str) -> List[float]:
        import asyncio

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.model.encode, text)
        return result.tolist()


# ═══════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════

@dataclass
class DocumentChunk:
    """A chunk of a document with deduplication support"""

    chunk_id: str
    text: str
    source: str
    chunk_hash: str  # SimHash for fast dedup
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "chunk_hash": self.chunk_hash,
            "metadata": self.metadata,
        }


@dataclass
class SearchResult:
    """Result from hybrid search"""

    doc_id: str
    text: str
    source: str
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    chunk_hash: Optional[str] = None
    is_duplicate: bool = False


@dataclass
class ScenarioFeatures:
    """Complete scenario feature set"""

    core_features: Dict[str, Any] = field(default_factory=dict)
    optional_features: Dict[str, Any] = field(default_factory=dict)
    shap_features: List[tuple] = field(default_factory=list)
    free_text: Optional[str] = None
    ml_risk_score: Optional[float] = None

    def all_features(self) -> Dict[str, Any]:
        """Merge all features"""
        merged = dict(self.core_features)
        merged.update(self.optional_features)
        return merged


@dataclass
class RAGResponse:
    """Final RAG response"""

    documents: List[SearchResult]
    analysis: Dict[str, Any]
    scenario_type: str
    confidence: float
    latency_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


__all__ = [
    "DocumentChunk",
    "SearchResult",
    "ScenarioFeatures",
    "RAGResponse",
    "EmbedderProtocol",
    "LLMProtocol",
    "RerankerProtocol",
    "SyncEmbedderWrapper",
    "EvidenceCitation",
    "LegalCitation",
    "EvidenceClaim",
    "EvidenceBundle",
    "EvidenceSourceType",
    "EvidenceSupportStatus",
    "EvidenceUse",
    "EvidenceOrigin",
]
