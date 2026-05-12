"""
Unified Paths for RAG System - المركز الوحيد للمسارات
"""

import os
from pathlib import Path
from typing import Optional

class RAGPaths:
    """مركز موحد لجميع مسارات النظام"""
    
    @staticmethod
    def get_stage2_path() -> Path:
        """الحصول على مسار stage2 بشكل موثوق"""
        current = Path(__file__).resolve().parent.parent
        return current
    
    @staticmethod
    def get_knowledge_path() -> Path:
        return RAGPaths.get_stage2_path() / "knowledge"
    
    @staticmethod
    def get_vector_db_path() -> Path:
        return RAGPaths.get_knowledge_path() / "vector_db"
    
    @staticmethod
    def get_models_path() -> Path:
        return RAGPaths.get_knowledge_path() / "models"
    
    @staticmethod
    def get_embedding_path() -> Path:
        return RAGPaths.get_models_path() / "embedding"
    
    @staticmethod
    def get_reranker_path() -> Path:
        return RAGPaths.get_models_path() / "reranker"
    
    @staticmethod
    def get_docs_path() -> Path:
        return RAGPaths.get_stage2_path() / "docs"
    
    @staticmethod
    def get_env_path() -> Path:
        return RAGPaths.get_stage2_path() / ".env"
    
    @staticmethod
    def verify_paths() -> dict:
        """التحقق من وجود جميع المسارات"""
        paths = {
            "vector_db": RAGPaths.get_vector_db_path(),
            "embedding": RAGPaths.get_embedding_path(),
            "reranker": RAGPaths.get_reranker_path(),
            "docs": RAGPaths.get_docs_path(),
            "env": RAGPaths.get_env_path()
        }
        
        results = {}
        for name, path in paths.items():
            exists = path.exists()
            results[name] = {"path": str(path), "exists": exists}
            if not exists:
                print(f"⚠️ {name} not found at {path}")
        
        return results

# للاستخدام السريع
PATHS = RAGPaths()
