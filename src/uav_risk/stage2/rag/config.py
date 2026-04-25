"""
RAG Configuration (V6 - Enterprise Aviation Grade)
==================================================
Dynamic path resolution, strict Fail-Fast logic, safe environment parsing, 
and dynamic logit bounds for hybrid normalization.
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
    # مسار آمن وصارم مع دعم المتغيرات البيئية للبيئات المختلفة
    BASE_DIR: Path = Path(os.getenv("ACE_KNOWLEDGE_BASE", Path(__file__).resolve().parents[3]))
    INDEX_PATH: Optional[Path] = field(default=None, init=False)
    
    # النماذج (Fail-Fast: لا يوجد Fallback لمنع تناقض الأبعاد مع FAISS)
    EMBEDDING_MODEL: str = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    RERANKER_MODEL: str = os.getenv("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    EXPECTED_INDEX_HASH: str = os.getenv("RAG_INDEX_HASH", "")
    
    # إعدادات البحث والأداء
    INITIAL_K: int = _safe_int("RAG_INITIAL_K", 15, min_val=5, max_val=100)
    MAX_THREADS: int = _safe_int("RAG_MAX_THREADS", 4, min_val=1, max_val=32)
    TIMEOUT_SEC: float = _safe_float("RAG_TIMEOUT_SEC", 8.0, min_val=3.0, max_val=60.0)
    MAX_CONCURRENT_REQUESTS: int = _safe_int("RAG_MAX_CONCURRENT", 10, min_val=1, max_val=100)
    
    # حدود التطبيع الرياضي للـ CrossEncoder
    LOGIT_MIN: float = _safe_float("RAG_LOGIT_MIN", -10.0, min_val=-100.0, max_val=0.0)
    LOGIT_MAX: float = _safe_float("RAG_LOGIT_MAX", 10.0, min_val=0.0, max_val=100.0)

    FAIL_ON_MISSING_INDEX: bool = os.getenv("RAG_FAIL_ON_MISSING_INDEX", "True").lower() == "true"

    def __post_init__(self):
        """تحقق صارم وآمن من المسارات والبيانات."""
        try:
            if not self.BASE_DIR.exists():
                logger.error(f"BASE_DIR not found at {self.BASE_DIR}")
        except OSError as e:
            logger.error(f"OS Error while accessing BASE_DIR: {e}")

        self.INDEX_PATH = self.BASE_DIR / "stage2" / "knowledge" / "vector_db"
        
        if self.FAIL_ON_MISSING_INDEX:
            try:
                if not self.INDEX_PATH.exists():
                    raise ValueError(f"CRITICAL ERROR: FAISS Vector DB missing at {self.INDEX_PATH}")
            except OSError as e:
                logger.error(f"OS Error while verifying FAISS index: {e}")
        
        if not self.EXPECTED_INDEX_HASH:
            logger.warning("RAG_INDEX_HASH is empty. Index integrity will NOT be verified!")