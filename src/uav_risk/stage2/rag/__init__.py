"""Lightweight RAG package exports for Stage2 integration.

This module intentionally avoids importing heavy retrieval/index/LLM components at
import-time to keep compatibility checks and API startup lightweight.
"""

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

from .adapter import Stage2RAGAdapter
from .quality import (
    RAGQualityCaseResult,
    RAGQualityQuery,
    RAGQualityReport,
    build_default_rag_quality_queries,
    build_runtime_rag_adapter_if_available,
    get_cached_runtime_rag_adapter,
    clear_runtime_rag_adapter_cache_for_tests,
    evaluate_rag_adapter_quality,
)
from .runtime_diagnostics import (
    RAGIndexProvenanceStatus,
    RAGRuntimeDiagnosticResult,
    RAGRuntimeResourceStatus,
    inspect_rag_index_provenance,
    inspect_rag_runtime_resources,
    run_rag_runtime_diagnostic,
)
from .schemas import (
    DocumentChunk,
    EmbedderProtocol,
    LLMProtocol,
    RAGResponse,
    RerankerProtocol,
    ScenarioFeatures,
    SearchResult,
    SyncEmbedderWrapper,
)

__all__ = [
    "Stage2RAGAdapter",
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
    "RAGQualityQuery",
    "RAGQualityCaseResult",
    "RAGQualityReport",
    "build_default_rag_quality_queries",
    "evaluate_rag_adapter_quality",
    "build_runtime_rag_adapter_if_available",
    "get_cached_runtime_rag_adapter",
    "clear_runtime_rag_adapter_cache_for_tests",
    "RAGRuntimeResourceStatus",
    "RAGIndexProvenanceStatus",
    "RAGRuntimeDiagnosticResult",
    "inspect_rag_runtime_resources",
    "inspect_rag_index_provenance",
    "run_rag_runtime_diagnostic",
]
