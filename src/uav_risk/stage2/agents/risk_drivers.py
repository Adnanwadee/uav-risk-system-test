from __future__ import annotations
from typing import Any, Dict, List

from ..schemas import RiskDriver


def derive_risk_drivers(stage1_facts: Dict[str, Any], snapshot: Dict[str, Any]) -> List[RiskDriver]:
    drivers: List[RiskDriver] = []

    # Deterministic drivers from known keys (no SHAP yet)
    jam = snapshot.get("environment.gnss_jam_dbm")
    if jam is not None:
        drivers.append(
            RiskDriver(
                driver="GNSS interference (jamming)",
                value=jam,
                note="Elevated jamming can degrade navigation and trigger failsafe or drift."
            )
        )

    wind = snapshot.get("environment.weather.wind_mps")
    gust = snapshot.get("environment.weather.gust_mps")
    if wind is not None:
        note = "Wind impacts stability and control margin."
        if gust is not None:
            note += " Gusts increase transient load and control risk."
        drivers.append(RiskDriver(driver="Weather wind/gust", value={"wind_mps": wind, "gust_mps": gust}, note=note))

    pred = stage1_facts.get("predicted_class")
    conf = stage1_facts.get("confidence")
    if pred is not None:
        drivers.append(
            RiskDriver(
                driver="Model risk classification",
                value={"predicted_class": pred, "confidence": conf},
                note="Classifier aggregates learned multi-factor risk patterns."
            )
        )

    return drivers
