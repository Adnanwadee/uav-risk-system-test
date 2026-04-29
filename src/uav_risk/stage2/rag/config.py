"""
RAG Configuration (V15.0 - Strict Offline & Path Optimized)
==========================================================
التعديلات الهندسية:
1. الموديلات المحلية: تم ربط EMBEDDING_MODEL و RERANKER_MODEL بالمسارات الفيزيائية داخل مجلد models.
2. استدامة المسارات: استخدام Pathlib لضمان العثور على الموارد في بيئة Codespaces دون تعليق.
3. التوافق مع الرادار: ضبط المعاملات لضمان دقة استجابة الوكيل القانوني بنسبة عالية.
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
    
    # تعريف المسارات الأساسية للموارد المحلية
    INDEX_PATH: Optional[Path] = field(default=None, init=False)
    MODELS_DIR: Optional[Path] = field(default=None, init=False)
    
    # [تعديل حاسم]: تحويل الموديلات للإشارة للمجلدات المحلية بدلاً من Hugging Face
    EMBEDDING_MODEL: str = field(default=None, init=False)
    RERANKER_MODEL: str = field(default=None, init=False)
    
    # معاملات الأداء والدقة
    MIN_RELEVANCE_SCORE: float = 0.65
    TOP_K: int = 5
    INITIAL_K: int = 15
    MAX_THREADS: int = 4
    TIMEOUT_SEC: float = 10.0
    MAX_CONCURRENT_REQUESTS: int = 10
    
    # معاملات الـ Reranker لتطبيع النتائج
    LOGIT_MIN: float = -10.0
    LOGIT_MAX: float = 10.0

    def __post_init__(self):
        """بناء المسارات من الجذر مباشرة لضمان الدقة وتجنب خطأ 404 أو 429."""
        
        # 1. تحديد موقع مجلد الـ Stage2
        STAGE2_PATH = self.BASE_DIR / "src" / "uav_risk" / "stage2"
        
        # 2. تحديد مسار قاعدة البيانات الـ Vector DB
        self.INDEX_PATH = STAGE2_PATH / "knowledge" / "vector_db"
        
        # 3. تحديد مسار الموديلات المحلية التي قمت بنقلها يدوياً
        self.MODELS_DIR = STAGE2_PATH / "knowledge" / "models"
        
        # 4. ربط الموديلات بالمسارات الفعلية لضمان عمل local_files_only=True
        self.EMBEDDING_MODEL = str(self.MODELS_DIR / "embedding")
        self.RERANKER_MODEL = str(self.MODELS_DIR / "reranker")
        
        # 5. التحقق الوقائي من وجود الموارد قبل إقلاع السيرفر
        if not self.INDEX_PATH.exists():
            logger.error(f"CRITICAL: RAG Index NOT found at {self.INDEX_PATH}")
            
        if not Path(self.EMBEDDING_MODEL).exists():
            logger.warning(f"Offline Embedding model NOT found at {self.EMBEDDING_MODEL}. System may hang.")
            
        if not Path(self.RERANKER_MODEL).exists():
            logger.warning(f"Offline Reranker model NOT found at {self.RERANKER_MODEL}. RAG reranking will fail.")

        logger.info("✅ RAG Configuration finalized for STRICT OFFLINE mode.")