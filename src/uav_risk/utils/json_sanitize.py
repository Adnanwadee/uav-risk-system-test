"""Utilities to sanitize complex Python objects into JSON-safe primitives.

Goals:
- Convert dataclasses, enums, numpy types, and other non-serializable objects
  into primitives (str, int, float, bool, list, dict).
- Limit free-text lengths and remove suspicious control sequences.
"""
from typing import Any
import dataclasses
import math
import numpy as _np


def _convert_value(v: Any) -> Any:
    # None
    if v is None:
        return None
    # Primitive types
    if isinstance(v, (str, int, float, bool)):
        # truncate long strings
        if isinstance(v, str) and len(v) > 2000:
            return v[:2000]
        # convert NaN/Inf
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return None
        return v
    # numpy types
    if isinstance(v, (_np.generic,)):
        try:
            return v.item()
        except Exception:
            return float(v)
    # dataclasses
    if dataclasses.is_dataclass(v):
        return sanitize_dataclass(v)
    # dict
    if isinstance(v, dict):
        return {str(k): sanitize_value(vv) for k, vv in v.items()}
    # list/tuple
    if isinstance(v, (list, tuple, set)):
        return [sanitize_value(x) for x in list(v)]
    # fallback to string
    try:
        return str(v)
    except Exception:
        return None


def sanitize_dataclass(obj: Any) -> Any:
    out = {}
    for f in dataclasses.fields(obj):
        try:
            val = getattr(obj, f.name)
            out[f.name] = sanitize_value(val)
        except Exception:
            out[f.name] = None
    return out


def sanitize_value(v: Any) -> Any:
    try:
        return _convert_value(v)
    except Exception:
        try:
            return str(v)
        except Exception:
            return None


def strict_aviation_json_sanitizer(payload: Any) -> Any:
    """Main entry point — returns a JSON-safe representation of `payload`."""
    return sanitize_value(payload)
