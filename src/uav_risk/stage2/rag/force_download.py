"""
Module: src/uav_risk/stage2/rag/force_download.py
Author: Elite Technical Partner
Description: Binds with RAGConfig paths to pull embedding and reranker weights offline.
"""

import os
import structlog
from huggingface_hub import snapshot_download
from uav_risk.stage2.rag.config import RAGConfig

logger = structlog.get_logger()
config = RAGConfig()

def download_with_retry(repo_id: str, local_dir: str):
    logger.info("attempting_model_snapshot_download", repo_id=repo_id, destination=local_dir)
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            endpoint="https://hf-mirror.com",  # لتخطي حظر الـ IPs
            local_files_only=False
        )
        logger.info("model_download_success", repo_id=repo_id)
    except Exception as e:
        logger.error("model_download_failed_critical", repo_id=repo_id, error=str(e))
        raise e

if __name__ == "__main__":
    # إنشاء المجلدات الرسمية المعتمدة في Config حياً
    os.makedirs(config.EMBEDDING_PATH, exist_ok=True)
    os.makedirs(config.RERANKER_PATH, exist_ok=True)
    
    # 1. جلب نموذج التضمين المحلي
    download_with_retry("sentence-transformers/all-MiniLM-L6-v2", config.EMBEDDING_PATH)
    
    # 2. جلب مفسر إعادة الترتيب المحلي
    download_with_retry("cross-encoder/ms-marco-MiniLM-L-6-v2", config.RERANKER_PATH)

# =====================================================================
# Stage 2 Architectural Dependency Comment Block:
# Deployment bootstrap utility to seed local intelligence weights.
# Dependencies: src/uav_risk/stage2/rag/config.py -> RAGConfig Paths
# Dependent Files: None (Standalone Setup Script)
# =====================================================================