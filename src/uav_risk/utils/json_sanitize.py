# src/uav_risk/utils/json_sanitize.py
from __future__ import annotations

import math
import datetime
from typing import Any
import numpy as np

def _is_bad_float(x: float) -> bool:
    return math.isnan(x) or math.isinf(x)

def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize data so it is STRICTLY JSON-safe.
    - NaN / inf / -inf -> None
    - Numpy types -> Native Python types
    - Sets/Tuples/Arrays -> Lists
    - Datetime -> ISO 8601 Strings
    - Dict Keys -> Force to Strings
    """
    if obj is None:
        return None

    # 1. Native Python Floats
    if isinstance(obj, float):
        return None if _is_bad_float(obj) else obj

    # 2. Numpy Numeric Types (CRITICAL FOR ML PIPELINES)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        val = float(obj)
        return None if _is_bad_float(val) else val
        
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
        
    if isinstance(obj, np.bool_):
        return bool(obj)

    # 3. Numpy Arrays
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())

    # 4. Dictionaries (Force keys to string, sanitize values)
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}

    # 5. Iterables (Lists, Tuples, Sets)
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(v) for v in obj]

    # 6. Datetime objects
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()

    # 7. Fallback for strings, booleans, and native ints
    return obj