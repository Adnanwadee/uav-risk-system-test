"""
Centralized Configuration Module
Manages environment variables, paths, and strict thresholds for the ACE System.
"""

from __future__ import annotations
from functools import lru_cache
from typing import Optional
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class SystemSettings(BaseSettings):
    """
    Centralized system settings read from environment or a .env file.
    """
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_file_encoding="utf-8")

    # ==========================================
    # System & Environment
    # ==========================================
    ENVIRONMENT: str = Field("development", description="development or production")
    LOG_LEVEL: str = Field("INFO", description="Standard logging level")

    # ==========================================
    # Paths (Aligned with actual project structure)
    # ==========================================
    UAV_ARTIFACTS_DIR: str = Field("artifacts", description="Directory for ML stage 1 artifacts")
    UAV_KNOWLEDGE_DIR: str = Field("src/uav_risk/stage2/knowledge", description="Directory for RAG documents and vector DB")
    UAV_REPORTS_DIR: str = Field("data/reports", description="Directory for generated final markdown reports")

    # ==========================================
    # ReAct Agent & LLM Config (Groq)
    # ==========================================
    GROQ_API_KEY: Optional[SecretStr] = Field(None, description="API Key for Groq Cloud")
    AGENT_MODEL_NAME: str = Field("llama3-70b-8192", description="Primary model used by the ReAct Agent")
    AGENT_REASONING_TIMEOUT_SEC: float = Field(30.0, description="Absolute timeout for agent to deliberate")
    REPORT_MODEL_NAME: str = Field("mixtral-8x7b-32768", description="Model used for fast report generation")

    # ==========================================
    # RAG Engine Configuration
    # ==========================================
    RAG_MIN_CONFIDENCE: float = Field(0.55, description="Minimum confidence threshold for FAISS retrievals")
    RAG_TOP_K: int = Field(5, description="Number of top document chunks to retrieve")
    RAG_USE_RERANKER: bool = Field(True, description="Whether to apply cross-encoder reranking")

    # ==========================================
    # ML Inference Config
    # ==========================================
    ML_INFERENCE_TIMEOUT_SEC: float = Field(2.0, description="Timeout for LightGBM model inference")
    ML_BUNDLE_FILENAME: str = Field("stage1_production_bundle.pkl", description="Exact name of the ML bundle")


@lru_cache()
def get_settings() -> SystemSettings:
    """
    Cached function to return system settings. 
    Prevents reading the .env file multiple times.
    """
    return SystemSettings()