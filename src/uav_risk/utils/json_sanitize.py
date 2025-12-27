from __future__ import annotations

import math
from typing import Any


def _is_bad_float(x: float) -> bool:
    return math.isnan(x) or math.isinf(x)


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize data so it is JSON-safe.
    - NaN / inf / -inf -> None
    Works for nested dict/list structures.
    """
    if isinstance(obj, float):
        return None if _is_bad_float(obj) else obj

    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]

    return obj
