"""Utilities to sanitize complex Python objects into JSON-safe primitives.

Goals:
- Convert dataclasses, enums, numpy types, and other non-serializable objects
  into primitives (str, int, float, bool, list, dict).
- Limit free-text lengths and remove suspicious control sequences.
"""
# STAGE6_CLEANUP_REVIEW:
# Classification: MIXED_ACTIVE_LEGACY_SANITIZER
# Plan lineage: PLAN3_ACTIVE public-safe API trace sanitizers plus PLAN1/PLAN2 legacy strict JSON sanitizer.
# Runtime status: sanitize_system_work_trace_public(), sanitize_tool_trace_public(), and sanitize_working_memory_public() are active API safety paths.
# Legacy signal: strict_aviation_json_sanitizer() remains for legacy ACE/report_writer callers.
# Replacement: New public API exposure should use the explicit public-safe sanitizers below.
# Action rule: Do not delete this file. Review strict_aviation_json_sanitizer only after ACE/report_writer legacy callers are removed.
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

FORBIDDEN_PUBLIC_KEYS = {
    "reasoning_steps",
    "chain_of_thought",
    "reasoning_chain",
    "thoughts",
    "thought",
    "raw_prompt",
    "prompt",
    "raw_completion",
    "completion",
    "raw_llm_response",
    "tool_history",
    "internal_memory",
    "scratchpad",
    "hidden",
    "api_key",
    "secret",
    "token",
    "authorization",
    "internal_reasoning",
    "private_reasoning",
}

_SECRET_KEY_TOKENS = {
    "api_key",
    "apikey",
    "secret",
    "token",
    "authorization",
    "password",
    "bearer",
}

_MAX_PUBLIC_STRING_LENGTH = 500
_MAX_PUBLIC_LIST_ITEMS = 64
_MAX_PUBLIC_DEPTH = 8


def _is_forbidden_public_key(key: str) -> bool:
    normalized = key.strip().lower()
    if normalized in FORBIDDEN_PUBLIC_KEYS:
        return True
    return any(token in normalized for token in _SECRET_KEY_TOKENS)


def _truncate_public_text(value: str, max_len: int = _MAX_PUBLIC_STRING_LENGTH) -> str:
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def _to_plain_object(value: Any) -> Any:
    if hasattr(value, "model_dump") and callable(getattr(value, "model_dump")):
        try:
            return value.model_dump()
        except Exception:
            return sanitize_value(value)
    return value


def _sanitize_public_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= _MAX_PUBLIC_DEPTH:
        return "[truncated-depth]"

    value = _to_plain_object(value)

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if isinstance(value, str):
        return _truncate_public_text(value)

    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, child in value.items():
            normalized_key = str(key)
            if _is_forbidden_public_key(normalized_key):
                continue
            sanitized[normalized_key] = _sanitize_public_value(child, depth=depth + 1)
        return sanitized

    if isinstance(value, (list, tuple, set)):
        sanitized_items = [_sanitize_public_value(item, depth=depth + 1) for item in list(value)[:_MAX_PUBLIC_LIST_ITEMS]]
        return sanitized_items

    return sanitize_value(value)


def _sanitize_public_dict_with_allowlist(value: Any, allowed_keys: set[str]) -> dict[str, Any]:
    plain = _to_plain_object(value)
    if not isinstance(plain, dict):
        return {}

    sanitized: dict[str, Any] = {}
    for key, child in plain.items():
        normalized_key = str(key)
        if normalized_key not in allowed_keys:
            continue
        if _is_forbidden_public_key(normalized_key):
            continue
        sanitized[normalized_key] = _sanitize_public_value(child, depth=1)
    return sanitized


def sanitize_working_memory_public(value: Any) -> dict[str, Any]:
    allowed = {
        "coverage_summary",
        "reasoning_summary",
        "limitations",
        "selected_rag_queries",
        "skipped_rag_queries",
        "top_input_signals",
        "top_feature_assessments",
        "signal_count",
        "feature_assessment_count",
    }
    return _sanitize_public_dict_with_allowlist(value, allowed)


def sanitize_tool_trace_public(value: Any) -> list[dict[str, Any]]:
    allowed = {
        "step_id",
        "stage",
        "tool_name",
        "purpose",
        "status",
        "input_summary",
        "output_summary",
        "related_query_ids",
        "related_evidence_ids",
        "related_finding_ids",
        "evidence_ids",
        "warnings",
        "duration_ms",
        "started_at",
        "completed_at",
        "public_safe",
    }

    plain = _to_plain_object(value)
    if not isinstance(plain, list):
        return []

    sanitized_entries: list[dict[str, Any]] = []
    for item in plain[:_MAX_PUBLIC_LIST_ITEMS]:
        if isinstance(item, dict):
            sanitized_entries.append(_sanitize_public_dict_with_allowlist(item, allowed))
        else:
            normalized = _sanitize_public_value(item, depth=1)
            if isinstance(normalized, dict):
                sanitized_entries.append(_sanitize_public_dict_with_allowlist(normalized, allowed))
    return sanitized_entries


def sanitize_system_work_trace_public(value: Any) -> dict[str, Any]:
    allowed = {"entries", "summary", "public_safe"}
    trace = _sanitize_public_dict_with_allowlist(value, allowed)

    entries = trace.get("entries")
    trace["entries"] = sanitize_tool_trace_public(entries if isinstance(entries, list) else [])

    summary = trace.get("summary")
    if isinstance(summary, str):
        trace["summary"] = _truncate_public_text(summary)
    elif summary is None:
        trace["summary"] = None
    else:
        trace["summary"] = _truncate_public_text(str(summary))

    trace["public_safe"] = True
    return trace
