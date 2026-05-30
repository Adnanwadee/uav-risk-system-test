# STAGE6_CLEANUP_REVIEW:
# Classification: LEGACY_CATALOG_HELPER_USED_BY_OLD_PIPELINE
# Plan lineage: PLAN1_OR_PLAN2_RELIC
# Runtime status: Not part of current raw-profile API assessment path.
# Legacy signal: Used by old stage2/pipeline.py to resolve UAV limits/catalog specs.
# Replacement: Current API profile contract stores explicit DroneProfileRaw capability fields.
# Action rule: Do not call from new code. Remove only after legacy pipeline.py is resolved.
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Canonical packaged location for reference catalog.
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema" / "uav_catalog.json"


def _load_catalog() -> Dict[str, Any]:
    try:
        if SCHEMA_PATH.exists():
            with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return {}
    return {}


_CATALOG = _load_catalog()


def get_uav_limits(model_id: Optional[str], operator_spec: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Return limits for a given model_id or operator_spec.

    Priority:
      1. If operator_spec provided, return it (marking source='operator')
      2. Else, lookup model_id in packaged catalog
      3. Else, return None

    The caller is responsible for auditing/trusting operator_spec; this helper
    will annotate the returned dict with a `__source` key for traceability.
    """
    if operator_spec and isinstance(operator_spec, dict):
        spec = dict(operator_spec)
        spec.setdefault("source", "operator")
        spec["__source"] = "operator"
        return spec

    if not model_id:
        return None

    entry = _CATALOG.get(model_id)
    if not entry:
        return None
    out = dict(entry)
    out["__source"] = out.get("source", "catalog")
    return out
