from __future__ import annotations

from typing import List

from uav_risk.stage2.agents.risk_context import RiskDriver


def decide_from_risk_context(
    risk_drivers: List[RiskDriver],
    safety_ready: bool,
) -> str:
    """
    Product-grade Stage-2 decision policy.
    - Model output is informational (NOT a veto).
    - Decision is based on operational drivers + safety readiness.
    """

    # Hard veto conditions derived from operational/safety drivers
    hard_hits = []
    caution_hits = []

    for d in risk_drivers:
        # HARD: clearly unsafe domains with HIGH severity
        if d.driver_id in {
            "WEATHER_GUST",
            "GNSS_JAMMING",
            "GNSS_MULTIPATH",
            "EM_INTERFERENCE",
        } and d.severity == "HIGH":
            hard_hits.append(d.driver_id)

        # CAUTION: medium severity drivers
        if d.severity == "MEDIUM":
            caution_hits.append(d.driver_id)

    # If safety isn't ready, never return GO
    if not safety_ready:
        return "CAUTION" if not hard_hits else "NO_GO"

    if hard_hits:
        return "NO_GO"

    if caution_hits:
        return "CAUTION"

    return "GO"
