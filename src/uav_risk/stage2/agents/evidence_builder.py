from __future__ import annotations

from typing import Any, Dict, List

from ..rag.index import RAGIndex
from .risk_context import RiskDriver


def _domain_to_query_suffix(domain: str) -> str:
    return {
        "weather": "wind gust visibility operational limits vlos bvlos",
        "safety": "interference mitigation risk navigation reliability contingency procedures",
        "regulations": "operational requirements compliance limitations approvals restrictions",
        "company_sop": "standard operating procedure policy mitigation checklists go no-go",
        "uav_manual": "manufacturer guidance limitations operating conditions environmental limits",
    }.get(domain, "")


def _driver_to_query_prefix(d: RiskDriver) -> str:
    """
    Give BM25 stronger lexical overlap using driver_id + severity.
    """
    sev = (d.severity or "UNKNOWN").upper()
    parts = [d.driver_id, sev, d.title]
    if d.value is not None:
        parts.append(str(d.value))
    return " ".join(parts).strip()


def build_evidence_snippets(
    index: RAGIndex,
    drivers: List[RiskDriver],
    top_k_per_driver: int = 2,
) -> List[Dict[str, Any]]:
    """
    Returns EvidenceSnippet list:
      { evidence_id, source, title, content, citation, score, linked_driver }

    Stable citation:
      "<title> — <source>::<chunkNNN>"
    """
    evidence: List[Dict[str, Any]] = []
    seen_ids = set()

    if not getattr(index, "chunks", None):
        return []

    for d in (drivers or []):
        domains = (d.domains or [])
        if not domains:
            continue

        base_q = _driver_to_query_prefix(d)

        for domain in domains:
            q = f"{base_q} {_domain_to_query_suffix(domain)}".strip()
            hits = index.search(q, top_k=top_k_per_driver, min_score=0.0)

            for chunk, score in hits:
                if chunk.evidence_id in seen_ids:
                    continue
                seen_ids.add(chunk.evidence_id)

                citation = f"{chunk.title} — {chunk.source}::{chunk.evidence_id.split('::')[-1]}"

                evidence.append({
                    "evidence_id": chunk.evidence_id,
                    "source": chunk.source,
                    "title": chunk.title,
                    "content": chunk.text,
                    "citation": citation,
                    "score": float(score),
                    "linked_driver": d.driver_id,
                })

    evidence.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return evidence[:12]
