from __future__ import annotations
from typing import Any, Dict, List, Tuple
import math

from ..schemas import DataQuality


REQUIRED_CORE_KEYS = [
    "uav.mass_kg",
    "uav.max_speed_mps",
    "environment.weather.wind_mps",
    "environment.gnss_jam_dbm",
]


def compute_data_quality(snapshot: Dict[str, Any]) -> DataQuality:
    # Count present vs missing for keys in snapshot
    total = len(snapshot)
    missing_keys: List[str] = []
    present = 0

    def is_missing(v: Any) -> bool:
        if v is None:
            return True
        try:
            if isinstance(v, float) and math.isnan(v):
                return True
        except Exception:
            pass
        return False

    for k, v in snapshot.items():
        if is_missing(v):
            missing_keys.append(k)
        else:
            present += 1

    ratio = (present / total) if total else 0.0

    required_inputs: List[str] = []
    for k in REQUIRED_CORE_KEYS:
        if k in snapshot and is_missing(snapshot.get(k)):
            required_inputs.append(k)

    # Quality level heuristic (deterministic)
    if ratio >= 0.85 and not required_inputs:
        level = "HIGH"
        modifier = 0.0
    elif ratio >= 0.65:
        level = "MEDIUM"
        modifier = -0.10
    else:
        level = "LOW"
        modifier = -0.25

    return DataQuality(
        present_count=present,
        total_count=total,
        completeness_ratio=float(ratio),
        missing_keys=missing_keys,
        quality_level=level,  # type: ignore[arg-type]
        confidence_modifier=float(modifier),
        required_inputs=required_inputs,
    )
