from __future__ import annotations
from typing import List
from ..schemas import RiskDriver


from typing import List
from uav_risk.stage2.schemas import EvidenceSnippet
from uav_risk.stage2.rag.loader import DocChunk


def chunks_to_evidence(chunks: List[DocChunk], max_chars: int = 900) -> List[EvidenceSnippet]:
    out: List[EvidenceSnippet] = []
    for c in chunks:
        text = (c.text or "").strip()
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."
        out.append(
            EvidenceSnippet(
                source=c.source,
                citation=c.citation,
                content=text,
            )
        )
    return out

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
