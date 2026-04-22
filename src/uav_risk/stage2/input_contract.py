from __future__ import annotations

from typing import Any, Dict, List


# ============================
# Input Contract (Stage-2)
# ============================
# These are the features your models were trained with (core validity).
MANDATORY_FOR_MODEL: List[str] = [
    "uav.mass_kg",
    "uav.max_speed_mps",
    "uav.battery_model.hover_power_W",
    "environment.weather.wind_mps",
    "environment.weather.visibility",
]

# Safety-critical for issuing hard NO_GO based on model alone
MANDATORY_FOR_SAFETY: List[str] = [
    "environment.weather.gust_mps",
    "environment.gnss_jam_dbm",
    "environment.gnss_multipath",
    "environment.em_interference",
    "daa.sep_threshold_m",
    "daa.ttc_threshold_s",
    "airspace.altitude_agl_max_m",
]


def validate_input_contract(inputs_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate inputs without imputing values.
    We only check presence (None after sanitize == missing).
    """
    missing_model: List[str] = []
    missing_safety: List[str] = []

    for k in MANDATORY_FOR_MODEL:
        if inputs_snapshot.get(k) is None:
            missing_model.append(k)

    for k in MANDATORY_FOR_SAFETY:
        if inputs_snapshot.get(k) is None:
            missing_safety.append(k)

    return {
        "model_ready": len(missing_model) == 0,
        "safety_ready": len(missing_safety) == 0,
        "missing_model_keys": missing_model,
        "missing_safety_keys": missing_safety,
        "mandatory_for_model": list(MANDATORY_FOR_MODEL),
        "mandatory_for_safety": list(MANDATORY_FOR_SAFETY),
    }
