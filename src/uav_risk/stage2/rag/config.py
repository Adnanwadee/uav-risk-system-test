"""
RAG Configuration (V18.0 - Production Ready)
===========================================
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("RAGConfig")


@dataclass
class RAGConfig:
    """التهيئة الأساسية لنظام RAG - نسخة محسنة"""
    
    # المسارات
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    
    @property
    def INDEX_PATH(self) -> Path:
        return self.BASE_DIR / "knowledge" / "vector_db"
    
    @property
    def MODELS_DIR(self) -> Path:
        return self.BASE_DIR / "knowledge" / "models"
    
    @property
    def EMBEDDING_PATH(self) -> Path:
        return self.MODELS_DIR / "embedding"
    
    @property
    def RERANKER_PATH(self) -> Path:
        return self.MODELS_DIR / "reranker"
    
    # معاملات الأداء - محسنة لرفع الثقة
    MIN_RELEVANCE_SCORE: float = 0.30      # أقل للسماح بمصادر أكثر
    TOP_K: int = 10                        # نتائج نهائية أكثر
    INITIAL_K: int = 60                    # بحث أولي أكبر
    MAX_THREADS: int = 4
    TIMEOUT_SEC: float = 20.0
    MAX_CONCURRENT_REQUESTS: int = 10
    
    # معاملات الـ Reranker
    LOGIT_MIN: float = -10.0
    LOGIT_MAX: float = 10.0
    
    # وضع التصحيح - يعرض كل خطوة
    DEBUG_MODE: bool = True
    
    def __post_init__(self):
        """التحقق من وجود الموارد"""
        if self.DEBUG_MODE:
            logger.info(f"BASE_DIR: {self.BASE_DIR}")
            logger.info(f"INDEX_PATH: {self.INDEX_PATH}")
            logger.info(f"EMBEDDING_PATH: {self.EMBEDDING_PATH}")
            logger.info(f"RERANKER_PATH: {self.RERANKER_PATH}")
        
        if not self.INDEX_PATH.exists():
            logger.warning(f"Index not found at {self.INDEX_PATH}")
        if not self.EMBEDDING_PATH.exists():
            logger.warning(f"Embedding model not found at {self.EMBEDDING_PATH}")
        if not self.RERANKER_PATH.exists():
            logger.warning(f"Reranker model not found at {self.RERANKER_PATH}")


@dataclass
class GroqSettings:
    """إعدادات Groq API - محسنة"""
    api_key: str = field(default_factory=lambda: os.environ.get("GROQ_API_KEY", ""))
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.1
    max_tokens: int = 8192          # زيادة كبيرة للإجابات الأطول
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


# تحميل API key من ملف .env
def load_api_key():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.startswith("GROQ_API_KEY"):
                    key = line.split("=", 1)[1].strip()
                    os.environ["GROQ_API_KEY"] = key
                    return key
    return None


load_api_key()

if not os.environ.get("GROQ_API_KEY"):
    logger.warning("GROQ_API_KEY not set. LLM features disabled.")