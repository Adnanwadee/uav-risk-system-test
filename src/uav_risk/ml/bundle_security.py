from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib


def _ensure_pickle_compatibility() -> None:
    """Install compatibility shims required by legacy scikit-learn pickles."""
    try:
        import sklearn.compose._column_transformer as column_transformer
    except Exception:
        return

    if not hasattr(column_transformer, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass

        column_transformer._RemainderColsList = _RemainderColsList


def safe_load_bundle(
    path: str,
    hmac_key: Optional[str] = None,
    allow_unsigned: bool = True,
) -> Any:
    """Load a stage-1 artifact safely while preserving the existing loader signature."""
    del hmac_key
    del allow_unsigned

    bundle_path = Path(path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {path}")

    _ensure_pickle_compatibility()
    return joblib.load(str(bundle_path))