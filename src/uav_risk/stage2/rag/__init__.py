"""
RAG V3.1 Package - Intelligent Aviation Regulatory Retrieval
Production-ready with local offline models and full reranker support.
"""

# Core Configuration
from .config_v3 import (
    RAGConfig,
    GroqLLMConfig,
    DynamicThresholdManager,
    THRESHOLD_MANAGER,
    BASE_DIR,
    INDEX_DIR,
    CACHE_DIR,
    LOG_DIR,
    DOCS_PATH,
    EMBEDDING_PATH,
    RERANKER_PATH,
    CORE_FEATURE_QUERIES,
    OPTIONAL_FEATURE_QUERIES,
    ALL_FEATURE_QUERIES,
    RISK_THRESHOLDS,
    load_risk_thresholds,
    save_risk_thresholds
)

# Security
from .faiss_security import (
    FAISSSecurityError,
    FAISSIndexVerifier,
    verify_and_safely_load_faiss
)

# Index Building
from .sparse_index_builder import (
    CorpusState,
    SparseIndexBuilder
)

# Intelligence
from .query_intelligence import (
    QueryIntelligence,
    AdaptiveRRF,
    ScenarioProfile,
    QueryHistory
)

# Feature Mapping
from .feature_query_mapper import (
    FeatureQueryMapper,
    FeatureQuery,
    RiskLevel
)

# HyDE
from .hyde_pipeline import (
    TargetedHyDE,
    HyDEResult
)

# Retrieval
from .hybrid_retriever import (
    HybridRetriever,
    RetrievedDocument,
    SimHash
)

# Evidence
from .evidence_logger import (
    BoundedEvidenceLog,
    EvidenceEntry
)

# Orchestrator
from .rag_core_v3 import (
    AsyncRAGCoreV3,
    RAGResult
)

# Schemas (with Protocols)
from .schemas import (
    DocumentChunk,
    SearchResult,
    ScenarioFeatures,
    RAGResponse,
    EmbedderProtocol,
    LLMProtocol,
    SyncEmbedderWrapper
)

# Prompts
from . import prompts_v3

# Build Index
from .build_index import (
    build_rag_index,
    _convert_to_native_faiss,
    _build_sparse_index,
    _sanity_check
)

# LLM
from .groq_llm import (
    GroqLLM
)

# Download
from .force_download import (
    download_with_retry
)

__version__ = "3.1.1"
__all__ = [
    # Config
    "RAGConfig",
    "GroqLLMConfig",
    "DynamicThresholdManager",
    "THRESHOLD_MANAGER",
    "BASE_DIR",
    "INDEX_DIR",
    "CACHE_DIR",
    "LOG_DIR",
    "DOCS_PATH",
    "EMBEDDING_PATH",
    "RERANKER_PATH",
    "CORE_FEATURE_QUERIES",
    "OPTIONAL_FEATURE_QUERIES",
    "ALL_FEATURE_QUERIES",
    "RISK_THRESHOLDS",
    "load_risk_thresholds",
    "save_risk_thresholds",

    # Security
    "FAISSSecurityError",
    "FAISSIndexVerifier",
    "verify_and_safely_load_faiss",

    # Index
    "CorpusState",
    "SparseIndexBuilder",

    # Intelligence
    "QueryIntelligence",
    "AdaptiveRRF",
    "ScenarioProfile",
    "QueryHistory",

    # Features
    "FeatureQueryMapper",
    "FeatureQuery",
    "RiskLevel",

    # HyDE
    "TargetedHyDE",
    "HyDEResult",

    # Retrieval
    "HybridRetriever",
    "RetrievedDocument",
    "SimHash",

    # Evidence
    "BoundedEvidenceLog",
    "EvidenceEntry",

    # Core
    "AsyncRAGCoreV3",
    "RAGResult",

    # Schemas
    "DocumentChunk",
    "SearchResult",
    "ScenarioFeatures",
    "RAGResponse",
    "EmbedderProtocol",
    "LLMProtocol",
    "SyncEmbedderWrapper",

    # Build
    "build_rag_index",
    "_convert_to_native_faiss",
    "_build_sparse_index",
    "_sanity_check",

    # LLM
    "GroqLLM",

    # Download
    "download_with_retry",

    # Prompts module
    "prompts_v3",
]