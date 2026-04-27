"""
RAG Configuration (V6.1 - Hardened Path Resolution)
==================================================
Fixes: 
- Corrected INDEX_PATH resolution to include the 'uav_risk' package directory.
- Enhanced BASE_DIR logic to be more resilient across different environments.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("RAGConfig")

def _safe_int(env_key: str, default: int, min_val: int, max_val: int) -> int:
    try:
        val = int(os.getenv(env_key, str(default)))
        return max(min_val, min(val, max_val))
    except ValueError:
        logger.warning(f"Invalid integer for {env_key}. Using default: {default}")
        return default

def _safe_float(env_key: str, default: float, min_val: float, max_val: float) -> float:
    try:
        val = float(os.getenv(env_key, str(default)))
        return max(min_val, min(val, max_val))
    except ValueError:
        logger.warning(f"Invalid float for {env_key}. Using default: {default}")
        return default

@dataclass
class RAGConfig:
    # [FIX] جعل المسار المرجعي هو مجلد 'uav_risk' (parents[2]) بدلاً من 'src'
    BASE_DIR: Path = Path(os.getenv("ACE_KNOWLEDGE_BASE", Path(__file__).resolve().parents[2]))
    INDEX_PATH: Optional[Path] = field(default=None, init=False)
    
    EMBEDDING_MODEL: str = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    RERANKER_MODEL: str = os.getenv("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    EXPECTED_INDEX_HASH: str = os.getenv("RAG_INDEX_HASH", "")
    
    INITIAL_K: int = _safe_int("RAG_INITIAL_K", 15, min_val=5, max_val=100)
    MAX_THREADS: int = _safe_int("RAG_MAX_THREADS", 4, min_val=1, max_val=32)
    TIMEOUT_SEC: float = _safe_float("RAG_TIMEOUT_SEC", 8.0, min_val=3.0, max_val=60.0)
    MAX_CONCURRENT_REQUESTS: int = _safe_int("RAG_MAX_CONCURRENT", 10, min_val=1, max_val=100)
    
    LOGIT_MIN: float = _safe_float("RAG_LOGIT_MIN", -10.0, min_val=-100.0, max_val=0.0)
    LOGIT_MAX: float = _safe_float("RAG_LOGIT_MAX", 10.0, min_val=0.0, max_val=100.0)

    FAIL_ON_MISSING_INDEX: bool = os.getenv("RAG_FAIL_ON_MISSING_INDEX", "True").lower() == "true"

    def __post_init__(self):
        """تحقق صارم وآمن من المسارات والبيانات."""
        # [FIX] التأكد من أن المسار يشير إلى uav_risk/stage2/knowledge/vector_db
        self.INDEX_PATH = self.BASE_DIR / "stage2" / "knowledge" / "vector_db"
        
        logger.info(f"RAG Core attempting to load index from: {self.INDEX_PATH}")

        if self.FAIL_ON_MISSING_INDEX:
            if not self.INDEX_PATH.exists():
                # محاولة ثانية ذكية (Smart Fallback) إذا كان الهيكل مختلفاً قليلاً
                alt_path = Path(__file__).resolve().parents[1] / "knowledge" / "vector_db"
                if alt_path.exists():
                    self.INDEX_PATH = alt_path
                    logger.info(f"Using alternate path fallback: {self.INDEX_PATH}")
                else:
                    raise ValueError(f"CRITICAL ERROR: FAISS Vector DB missing at {self.INDEX_PATH}")
        
        if not self.EXPECTED_INDEX_HASH:
            logger.warning("RAG_INDEX_HASH is empty. Index integrity will NOT be verified!")