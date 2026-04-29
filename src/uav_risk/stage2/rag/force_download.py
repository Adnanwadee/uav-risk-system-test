"""
ACE Model Force Downloader (V1.0 - Mirror & Structure Optimized)
================================================================
يقوم هذا السكربت بتحميل الموديلات من المرآة العالمية وحفظها محلياً 
داخل مجلد الـ knowledge لضمان تشغيل النظام أوفلاين للأبد.
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download

# بناء المسارات بناءً على الهيكلية التي أرفقتها
BASE_DIR = Path(__file__).resolve().parents[3] # يرجع من rag إلى stage2
MODELS_DIR = BASE_DIR / "src" / "uav_risk" / "stage2" / "knowledge" / "models"

EMBEDDING_DIR = MODELS_DIR / "embedding"
RERANKER_DIR = MODELS_DIR / "reranker"

# إنشاء المجلدات إذا لم تكن موجودة
os.makedirs(EMBEDDING_DIR, exist_ok=True)
os.makedirs(RERANKER_DIR, exist_ok=True)

def download_with_retry(repo_id, local_dir):
    print(f"🚀 Attempting to download {repo_id} to {local_dir}...")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            endpoint="https://hf-mirror.com", # استخدام المرآة لتجاوز حظر الـ IP
            local_files_only=False,
            # استبدل YOUR_TOKEN بالتوكن الجديد الذي أنشأته (اختياري مع المرآة)
            # token="YOUR_NEW_TOKEN" 
        )
        print(f"✅ Success: {repo_id} is now offline-ready.")
    except Exception as e:
        print(f"❌ Failed to download {repo_id}: {e}")

if __name__ == "__main__":
    # 1. تحميل موديل الـ Embedding
    download_with_retry("sentence-transformers/all-MiniLM-L6-v2", EMBEDDING_DIR)
    
    # 2. تحميل موديل الـ Reranker (المسبب الرئيسي للمشكلة)
    download_with_retry("cross-encoder/ms-marco-MiniLM-L-6-v2", RERANKER_DIR)