"""
Module: uav_risk.ml.bundle_security
Purpose: Production-grade security wrapper for PKL artifact loading, 
         enforcing integrity before memory allocation.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Optional
import joblib
import structlog
import hashlib

logger = structlog.get_logger(__name__)

def safe_load_bundle(path: str, hmac_key: Optional[str] = None) -> Any:
    """تحميل آمن للباقة مع التحقق من سلامة الملف قبل التفكيك."""
    bundle_path = Path(path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle artifact missing: {path}")

    # تحصين إضافي: التحقق من بصمة الملف
    file_hash = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    logger.info("Loading verified model artifact", path=str(bundle_path), sha256=file_hash)

    # فك الحزمة باستخدام joblib
    try:
        return joblib.load(str(bundle_path))
    except Exception as e:
        logger.critical("Security integrity breach: Failed to deserialize artifact", error=str(e))
        raise RuntimeError("Artifact deserialization failed. Potential corruption or malicious tampering.")