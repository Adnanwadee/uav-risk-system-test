from __future__ import annotations

from typing import List

from uav_risk.stage2.agents.risk_context import RiskDriver


def decide_from_risk_context(
    risk_drivers: List[RiskDriver],
    safety_ready: bool,
) -> str:
    """
    Lightweight decision helper (legacy).
    NOTE: The authoritative policy is implemented in stage2/pipeline.py (_stage2_decision_policy).
    This function remains for backward compatibility and simple calls.
    """

    # If safety isn't ready, never return GO
    if not safety_ready:
        return "CAUTION"

    hard_hits = 0
    caution_hits = 0

    for d in (risk_drivers or []):
        sev = str(getattr(d, "severity", "UNKNOWN")).upper().strip()
        did = str(getattr(d, "driver_id", "")).upper().strip()

        # HARD: clearly unsafe domains with HIGH severity
        if did in {"WEATHER_GUST", "GNSS_JAMMING"} and sev == "HIGH":
            hard_hits += 1

        # CAUTION: medium severity drivers
        if sev == "MEDIUM":
            caution_hits += 1

    if hard_hits > 0:
        return "NO_GO"
    if caution_hits > 0:
        return "CAUTION"
    return "GO"
