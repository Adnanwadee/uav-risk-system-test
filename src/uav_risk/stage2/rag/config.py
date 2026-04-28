"""
RAG Configuration (V13.1 - Path Fix for Codespaces)
==================================================
تم إصلاح مشكلة تكرار المسار (src/src) لضمان تحميل قاعدة البيانات القانونية.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("RAGConfig")

@dataclass
class RAGConfig:
    # [تعديل المسار]: نرجع 4 مستويات للوصول للمجلد الرئيسي للمشروع (Root)
    BASE_DIR: Path = Path(__file__).resolve().parents[4] 
    INDEX_PATH: Optional[Path] = field(default=None, init=False)
    
    EMBEDDING_MODEL: str = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    RERANKER_MODEL: str = os.getenv("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    MIN_RELEVANCE_SCORE: float = 0.65
    TOP_K: int = 5
    INITIAL_K: int = 15
    MAX_THREADS: int = 4
    TIMEOUT_SEC: float = 10.0
    MAX_CONCURRENT_REQUESTS: int = 10
    
    LOGIT_MIN: float = -10.0
    LOGIT_MAX: float = 10.0

    def __post_init__(self):
        # [إصلاح حاسم]: بناء المسار من الجذر مباشرة لضمان الدقة
        self.INDEX_PATH = self.BASE_DIR / "src" / "uav_risk" / "stage2" / "knowledge" / "vector_db"
        
        if not self.INDEX_PATH.exists():
            # محاولة البحث في المسار البديل إذا كان المشروع في مجلد فرعي
            alt_path = Path(__file__).resolve().parents[1] / "knowledge" / "vector_db"
            if alt_path.exists():
                self.INDEX_PATH = alt_path
            else:
                logger.error(f"CRITICAL: RAG Index NOT found at {self.INDEX_PATH}")