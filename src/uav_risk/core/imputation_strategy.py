from __future__ import annotations

import math
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class ImputationStrategy:
    """3-Layer deterministic imputation for 198-dim ML vectors.
    Layer 1: Physics/Legal safe baselines
    Layer 2: Statistical population averages
    Layer 3: Derived computations (alt/dist, ratios)
    All mutations are logged to context_pool["imputation_log"].
    """

    def __init__(self) -> None:
        self._nan = math.nan

    @staticmethod
    def _is_nan(value: Any) -> bool:
        try:
            return isinstance(value, float) and math.isnan(value)
        except Exception:
            return False

    def apply_imputation(
        self, ml_vector: List[float], context_pool: Dict[str, Any]
    ) -> Tuple[List[float], Dict[str, Any]]:
        if not isinstance(ml_vector, list):
            raise TypeError("ml_vector must be a list of floats")
        if not isinstance(context_pool, dict):
            raise TypeError("context_pool must be a dict")

        log: List[Dict[str, Any]] = []
        existing_log = context_pool.get("imputation_log", [])
        if isinstance(existing_log, list):
            log.extend(existing_log)

        # Layer 1: Physics/Legal Safe Defaults
        physics_safe = {
            1: 120.0,   # altitude (m AGL baseline)
            3: 5000.0,  # visibility (safe baseline assumption)
        }

        # Layer 2: Statistical Baselines
        stat_baseline = {
            0: 12.0,    # speed
            2: 1000.0,  # distance
            4: 10.0,    # wind speed
            6: 3.0,     # tier numeric default
        }

        for idx in range(len(ml_vector)):
            if not self._is_nan(ml_vector[idx]):
                continue

            imputed_val: float
            layer: str
            reason: str

            if idx in physics_safe:
                imputed_val = physics_safe[idx]
                layer = "physics_safe_baseline"
                reason = f"Missing index {idx} assumed physics/legal safe default"
            elif idx in stat_baseline:
                imputed_val = stat_baseline[idx]
                layer = "statistical_baseline"
                reason = f"Missing index {idx} imputed with population baseline"
            else:
                imputed_val = 0.0
                layer = "fallback_zero"
                reason = f"Missing index {idx} defaulted to zero (non-critical)"

            ml_vector[idx] = imputed_val
            log.append({
                "feature_index": idx,
                "original": "NaN",
                "imputed": imputed_val,
                "layer": layer,
                "reason": reason,
            })

        # Layer 3: Derivations
        self._derive_features(ml_vector, log)

        context_pool["imputation_log"] = log
        return ml_vector, context_pool

    def _derive_features(self, vec: List[float], log: List[Dict[str, Any]]) -> None:
        """Compute derived features when primary components are available."""
        if len(vec) > 5 and self._is_nan(vec[5]):
            alt, dist = vec[1], vec[2]
            if not self._is_nan(alt) and not self._is_nan(dist) and dist != 0:
                vec[5] = alt / dist
                log.append({
                    "feature_index": 5,
                    "original": "NaN",
                    "imputed": vec[5],
                    "layer": "derivation",
                    "reason": "Derived alt/dist ratio from imputed altitude & distance",
                })
            else:
                vec[5] = 0.0
                log.append({
                    "feature_index": 5,
                    "original": "NaN",
                    "imputed": 0.0,
                    "layer": "derivation_fallback",
                    "reason": "Could not derive ratio; applied zero fallback",
                })