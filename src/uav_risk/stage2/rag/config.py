"""
Module: src/uav_risk/stage2/rag/config.py
Author: Elite Technical Partner
Description: Centralized production configuration module for the Legislative RAG system.
             Unifies path resolutions, performance constraints, and credential auditing.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any , Optional

logger = logging.getLogger("RAGConfig")


@dataclass
class RAGConfig:
    """Central configuration for paths and execution constants of the UAV RAG pipeline."""
    
    # احتساب المسار الأساسي لـ stage2 بشكل ديناميكي صارم ومنع الفجوات النسبية
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    
    # معاملات الأداء التشغيلي - محاذة ومحسنة لرفع ثقة قرارات الطيران
    MIN_RELEVANCE_SCORE: float = 0.30      # عتبة الاسترجاع الدنيا المعتمدة
    TOP_K: int = 10                        # عدد النتائج النهائية الكثيفة المسلمة للوكيل
    INITIAL_K: int = 60                    # أفق البحث الأولي المعزز قبل الترتيب
    MAX_THREADS: int = 4
    TIMEOUT_SEC: float = 20.0
    MAX_CONCURRENT_REQUESTS: int = 10
    
    # نطاقات تصفية معاملات الـ Reranker المحلي
    LOGIT_MIN: float = -10.0
    LOGIT_MAX: float = 10.0
    
    DEBUG_MODE: bool = True

    @property
    def INDEX_PATH(self) -> Path:
        """مسار قاعدة البيانات المتجهية الموحد داخل مجلد المعرفة الاستراتيجي أوفلاين"""
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
    
    @property
    def DOCS_PATH(self) -> Path:
        """مجلد وثائق القوانين الجوية الحية الحاضن لملفات الـ PDF"""
        return self.BASE_DIR / "docs"

    def __post_init__(self):
        """تدقيق فوري للبيئة المحلية عند بدء تشغيل المحطة الأرضية"""
        if self.DEBUG_MODE:
            logger.info(f"[RAG_INIT] BASE_DIR set dynamically to: {self.BASE_DIR}")
            logger.info(f"[RAG_INIT] INDEX_PATH mapped to: {self.INDEX_PATH}")
            logger.info(f"[RAG_INIT] EMBEDDING_PATH mapped to: {self.EMBEDDING_PATH}")
            logger.info(f"[RAG_INIT] RERANKER_PATH mapped to: {self.RERANKER_PATH}")
            logger.info(f"[RAG_INIT] DOCS_PATH mapped to: {self.DOCS_PATH}")
        
        # إطلاق تحذيرات جنائية فورية صامتة Danger في حال غياب الموارد الحتمية
        if not self.INDEX_PATH.exists():
            logger.warning(f"FAISS Vector Index artifact path not found at: {self.INDEX_PATH}")
        if not self.EMBEDDING_PATH.exists():
            logger.warning(f"Offline Embedding Model weights folder not found at: {self.EMBEDDING_PATH}")
        if not self.RERANKER_PATH.exists():
            logger.warning(f"Offline Cross-Encoder Reranker folder not found at: {self.RERANKER_PATH}")

    def verify_system_paths(self) -> Dict[str, Dict[str, Any]]:
        """Integrated path verification health-check replacing the redundant paths.py file."""
        paths_map = {
            "vector_db": self.INDEX_PATH,
            "embedding": self.EMBEDDING_PATH,
            "reranker": self.RERANKER_PATH,
            "docs": self.DOCS_PATH,
            "env": self.BASE_DIR / ".env"
        }
        
        audit_results = {}
        for name, path in paths_map.items():
            exists = path.exists()
            audit_results[name] = {"path": str(path), "exists": exists}
            if not exists:
                logger.warning(f"System audit warning: {name} path is missing on active target environment.")
                
        return audit_results


@dataclass(frozen=True)
class GroqLLMConfig:
    """Strict configuration contract for the Groq inference engine client."""
    api_key: str = field(default_factory=lambda: os.environ.get("GROQ_API_KEY", ""))
    model: str = "llama-3.3-70b-versatile"  # النموذج المعتمد فائق السرعة
    temperature: float = 0.1                # درجة حرارة منخفضة لضمان الدقة التشريعية المطلقة ومنع الـ Creativity
    max_tokens: int = 8192                  # طاقة استيعابية تضمن إنتاج تقارير كاملة بدون بتر نصوص الإجابات
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


def load_and_inject_env_key() -> Optional[str]:
    """Scans and securely binds the localized .env credentials to operating environment variables."""
    # البحث المباشر في المجلد الأب لـ stage2 لتحديد ملف التحقق
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("GROQ_API_KEY"):
                    try:
                        key = line.split("=", 1)[1].strip()
                        os.environ["GROQ_API_KEY"] = key
                        return key
                    except IndexError:
                        logger.error("Failed to parse GROQ_API_KEY string format in .env file.")
    return os.environ.get("GROQ_API_KEY")


# تأمين تشغيل الحقن التلقائي قبل أي استدعاء خارجي للمكونات السحابية
ACTIVE_API_KEY = load_and_inject_env_key()

if not os.environ.get("GROQ_API_KEY"):
    logger.warning("CRITICAL CRADENTIAL ALERT: GROQ_API_KEY environment variable is not configured. LLM pipelines will be disabled.")

# =====================================================================
# Stage 2 Architectural Dependency Comment Block:
# Central state config file mapping system paths and constants.
# Dependencies: None (Root Submodule Asset)
# Dependent Files: src/uav_risk/stage2/rag/groq_llm.py, rag_core.py, enhanced_legal_agent.py
# =====================================================================