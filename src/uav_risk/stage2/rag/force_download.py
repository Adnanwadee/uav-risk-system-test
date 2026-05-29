"""
Legacy/demo/manual-only model bootstrap helper.

This module is not canonical runtime and should not be used as final readiness evidence.
Canonical readiness commands are listed in README.md.

Module: src/uav_risk/stage2/rag/force_download.py
Author: Elite Technical Partner + V3.1 Fix
Description: Binds with RAGConfig paths to pull embedding and reranker weights offline.
             Updated for config_v3 with local model paths.
"""



import os
import logging
from huggingface_hub import snapshot_download
from .config_v3 import RAGConfig, EMBEDDING_PATH, RERANKER_PATH

logger = logging.getLogger(__name__)
config = RAGConfig()


def download_with_retry(repo_id: str, local_dir: str, max_retries: int = 3):
    """Download model with retry logic."""
    logger.info("attempting_model_snapshot_download", repo_id=repo_id, destination=local_dir)

    last_error = None
    for attempt in range(max_retries):
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                local_files_only=False,
                resume_download=True
            )
            logger.info("model_download_success", repo_id=repo_id)
            return
        except Exception as e:
            last_error = e
            logger.warning(
                f"model_download_attempt_{attempt+1}_failed", 
                repo_id=repo_id, 
                error=str(e)
            )

    logger.error("model_download_failed_critical", repo_id=repo_id, error=str(last_error))
    raise last_error


if __name__ == "__main__":
    # إنشاء المجلدات الرسمية المعتمدة في Config
    os.makedirs(EMBEDDING_PATH, exist_ok=True)
    os.makedirs(RERANKER_PATH, exist_ok=True)

    # 1. جلب نموذج التضمين المحلي (MiniLM - الخفيف والسريع)
    logger.info("downloading_embedding_model", model="sentence-transformers/all-MiniLM-L6-v2")
    download_with_retry("sentence-transformers/all-MiniLM-L6-v2", EMBEDDING_PATH)

    # 2. جلب مفسر إعادة الترتيب المحلي
    logger.info("downloading_reranker_model", model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    download_with_retry("cross-encoder/ms-marco-MiniLM-L-6-v2", RERANKER_PATH)

    logger.info("all_models_downloaded_successfully")

# =====================================================================
# Legacy/demo/manual-only bootstrap utility.
# Not canonical runtime and not final readiness evidence.
# Canonical readiness commands are listed in README.md.
# =====================================================================