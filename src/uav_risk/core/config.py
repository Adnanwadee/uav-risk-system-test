from __future__ import annotations
from functools import lru_cache
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SystemSettings(BaseSettings):
    """Centralized system settings read from environment or a .env file.
    All settings have sensible defaults and can be overridden via environment
    variables or a `.env` file located at the project root.
    """
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_file_encoding="utf-8")

    # Paths
    UAV_ARTIFACTS_DIR: str = Field("./artifacts", description="Directory for artifacts")
    UAV_KNOWLEDGE_DIR: str = Field("./knowledge", description="Directory for knowledge store")
    UAV_REPORTS_DIR: str = Field("./data/reports", description="Directory for generated reports")

    # Decision weights
    DECISION_WEIGHT_ML: float = Field(0.15, description="Weight applied to ML decision")
    DECISION_WEIGHT_AGENT: float = Field(0.55, description="Weight applied to agent decision")
    DECISION_WEIGHT_COMPLIANCE: float = Field(0.30, description="Weight applied to compliance checks")

    # RAG configuration
    RAG_MIN_CONFIDENCE: float = Field(0.55, description="Minimum confidence threshold for RAG results")
    RAG_TOP_K: int = Field(5, description="Number of top retrievals to return")
    RAG_USE_RERANKER: bool = Field(True, description="Whether to apply reranker to retrieved candidates")

    # Timeouts (seconds)
    ML_INFERENCE_TIMEOUT_SEC: float = Field(2.0, description="Timeout for ML inference in seconds")
    AGENT_REASONING_TIMEOUT_SEC: float = Field(10.0, description="Timeout for agent reasoning in seconds")

    # Security signatures (optional)
    ML_BUNDLE_SHA256: Optional[str] = Field(None, description="Optional SHA256 for ML bundle integrity")
    FAISS_INDEX_SHA256: Optional[str] = Field(None, description="Optional SHA256 for FAISS index integrity")


@lru_cache()
def get_settings() -> SystemSettings:
    """Return a cached SystemSettings instance (singleton pattern)."""
    return SystemSettings()