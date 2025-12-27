from __future__ import annotations

import math
from typing import Any, Dict, List

from pydantic import BaseModel

from .schemas import EvidenceSnippet, DataQualitySummary
from uav_risk.utils.json_sanitize import sanitize_for_json


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return True
    return False


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
    Build a structured evidence pack that Stage-2 and (later) LLMs can rely on.
    This function MUST NOT hallucinate.
    """

    # Force JSON-safe early
    inputs_snapshot = sanitize_for_json(inputs_snapshot)
    stage1_facts = sanitize_for_json(stage1_facts)

    # Normalize rules into dict
    if isinstance(rules, dict):
        rules_dict = rules
    elif rules is not None and hasattr(rules, "model_dump"):
        rules_dict = rules.model_dump()
    else:
        rules_dict = {}

    rules_dict = sanitize_for_json(rules_dict)

    # -------------------------
    # Data Quality
    # -------------------------
    missing_keys = [k for k, v in inputs_snapshot.items() if _is_missing(v)]

    total_count = len(inputs_snapshot)
    present_count = total_count - len(missing_keys)

    dq = DataQualitySummary(
        present_count=present_count,
        total_count=total_count,
        completeness_ratio=present_count / max(total_count, 1),
        missing_keys=missing_keys,
    )

    # -------------------------
    # Risk Drivers (deterministic)
    # -------------------------
    risk_drivers: List[Dict[str, Any]] = []

    # Only add a driver if value is real (not None/NaN/inf)
    val = inputs_snapshot.get("environment.gnss_jam_dbm")
    if not _is_missing(val):
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
        rules=rules_dict,
        risk_drivers=risk_drivers,
        data_quality=dq,
        evidence_snippets=evidence_snippets,
    )
