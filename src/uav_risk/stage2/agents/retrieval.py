from __future__ import annotations
from typing import List
from ..schemas import RiskDriver


def build_retrieval_queries(drivers: List[RiskDriver]) -> List[str]:
    qs: List[str] = []
    for d in drivers:
        name = d.driver.lower()
        if "gnss" in name:
            qs.append("GNSS interference mitigation steps for UAV operations")
            qs.append("GNSS jamming operational limits and safety checklist")
        if "weather" in name or "wind" in name:
            qs.append("recommended wind and gust limits for UAV flight operations")
            qs.append("visibility limits and weather go/no-go checklist for UAV")
        if "model risk classification" in name:
            qs.append("risk scoring interpretation and mitigation planning for UAV flights")
    # Deduplicate while preserving order
    seen = set()
    out = []
    for q in qs:
        if q not in seen:
            out.append(q)
            seen.add(q)
    return out[:6]
