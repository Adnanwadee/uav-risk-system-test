from __future__ import annotations

import math
import logging
from typing import Any, Dict, List, Tuple

from .contracts import FlightInput

logger = logging.getLogger(__name__)

# Deterministic mapping from raw API fields to 198-feature vector indices.
# Aligns with stage1_feature_mapping.json used during ML training.
_RAW_TO_VECTOR_INDEX: Dict[str, int] = {
    "speed": 0,
    "altitude": 1,
    "distance": 2,
    "wind_speed": 3,
    "visibility": 4,
    "alt_dist_ratio": 5,
    "tier_numeric": 6,
}

# Indices 7..197 are reserved for OneHot, Missing Indicators, Statistical & Derived features.
_VECTOR_SIZE = 198


class FeatureRouter:
    """Deterministic router: FlightInput → 198-dim ML vector + Context Pool."""

    def __init__(self) -> None:
        self._nan = math.nan

    def _safe_float(self, value: Any) -> float:
        """Convert to float; explicitly reject bools and return NaN on failure."""
        if value is None or isinstance(value, bool):
            return self._nan
        try:
            return float(value)
        except (TypeError, ValueError):
            return self._nan

    def _extract_wind_speed(self, wind: Any) -> float:
        """Extract scalar wind speed from float or dict representations."""
        if wind is None:
            return self._nan
        if isinstance(wind, (int, float)):
            return self._safe_float(wind)
        if isinstance(wind, dict):
            for key in ("speed", "wind_speed", "magnitude"):
                if key in wind:
                    return self._safe_float(wind[key])
            if "u" in wind and "v" in wind:
                u, v = self._safe_float(wind["u"]), self._safe_float(wind["v"])
                if not math.isnan(u) and not math.isnan(v):
                    return math.hypot(u, v)
        return self._nan

    def route_payload(self, payload: FlightInput) -> Tuple[List[float], Dict[str, Any]]:
        """Route validated payload into ML vector and agent context pool."""
        # 1. Extract raw values safely
        speed = self._safe_float(payload.speed)
        altitude = self._safe_float(payload.altitude)
        distance = self._safe_float(payload.distance)
        visibility = self._safe_float(payload.visibility)
        wind_speed = self._extract_wind_speed(payload.wind)

        alt_dist_ratio = (
            altitude / distance
            if not math.isnan(altitude) and not math.isnan(distance) and distance != 0
            else self._nan
        )

        tier_numeric = float(payload.tier_level)

        # 2. Build 198-dim vector (NaN-initialized)
        ml_vector: List[float] = [self._nan] * _VECTOR_SIZE
        raw_map = {
            "speed": speed,
            "altitude": altitude,
            "distance": distance,
            "wind_speed": wind_speed,
            "visibility": visibility,
            "alt_dist_ratio": alt_dist_ratio,
            "tier_numeric": tier_numeric,
        }
        for feat_name, val in raw_map.items():
            idx = _RAW_TO_VECTOR_INDEX.get(feat_name)
            if idx is not None:
                ml_vector[idx] = val

        # 3. Context Pool for Agents / RAG / Reporting
        context_pool: Dict[str, Any] = {
            "flight_id": payload.flight_id,
            "tier": payload.tier,
            "tier_level": payload.tier_level,
            "bounds_warnings": payload.validate_bounds(),
            "missing_fields": [k for k, v in raw_map.items() if math.isnan(v)],
            "original_wind": payload.wind,
        }

        return ml_vector, context_pool