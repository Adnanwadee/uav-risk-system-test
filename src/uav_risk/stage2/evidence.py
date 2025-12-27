from __future__ import annotations
from typing import Any, Dict, List

from pydantic import BaseModel

from .schemas import EvidenceSnippet, DataQualitySummary


class EvidencePack(BaseModel):
    inputs_snapshot: Dict[str, Any]
    stage1_facts: Dict[str, Any]
    rules: Dict[str, Any]
    risk_drivers: List[Dict[str, Any]]
    data_quality: DataQualitySummary
    evidence_snippets: List[EvidenceSnippet]


def build_evidence_pack(
    *,
    inputs_snapshot: Dict[str, Any],
    stage1_facts: Dict[str, Any],
    rules: Any,
) -> EvidencePack:
    """
    Build a structured evidence pack that Stage-2 and LLMs can rely on.
    This function MUST NOT hallucinate.
    """

    # -------------------------
    # Data Quality
    # -------------------------
    missing_keys = [
        k for k, v in inputs_snapshot.items()
        if v is None or (isinstance(v, float) and str(v) == "nan")
    ]

    present_count = len(inputs_snapshot) - len(missing_keys)
    total_count = len(inputs_snapshot)

    dq = DataQualitySummary(
        present_count=present_count,
        total_count=total_count,
        completeness_ratio=present_count / max(total_count, 1),
        missing_keys=missing_keys,
    )

    # -------------------------
    # Risk Drivers (baseline)
    # -------------------------
    risk_drivers: List[Dict[str, Any]] = []

    if "environment.gnss_jam_dbm" in inputs_snapshot:
        val = inputs_snapshot.get("environment.gnss_jam_dbm")
        if val is not None:
            risk_drivers.append(
                {
                    "driver": "GNSS: jamming",
                    "value": val,
                    "note": "Stronger interference may degrade navigation reliability.",
                }
            )

    # -------------------------
    # Evidence snippets (RAG later)
    # -------------------------
    evidence_snippets: List[EvidenceSnippet] = []

    return EvidencePack(
        inputs_snapshot=inputs_snapshot,
        stage1_facts=stage1_facts,
        rules=rules.model_dump() if hasattr(rules, "model_dump") else rules,
        risk_drivers=risk_drivers,
        data_quality=dq,
        evidence_snippets=evidence_snippets,
    )
