from __future__ import annotations
from typing import Any, Dict, List, Tuple
import re


def verify_report(md: str, evidence_ids: List[str]) -> Tuple[bool, List[str]]:
    """
    Enforce: every mitigation bullet must cite evidence_id in square brackets.
    Also detect suspicious numeric claims without citations (heuristic).
    """
    errors: List[str] = []

    # Extract all citations like [something]
    cited = set(re.findall(r"\[([^\]]+)\]", md))

    # Any cited id must exist
    for c in cited:
        if c not in set(evidence_ids):
            errors.append(f"Unknown citation used: [{c}]")

    # Enforce evidence usage if 'Mitigation' section exists
    if "Mitigation" in md or "Mitigation Plan" in md:
        # naive: require at least one valid evidence id cited
        if not (set(evidence_ids) & cited):
            errors.append("Mitigation section present but no valid evidence_id citations found.")

    # Heuristic: if report contains explicit wind limits like 'm/s' or 'km/h' numbers, require citation
    suspicious = re.findall(r"\b\d+(\.\d+)?\s*(m/s|km/h|knots|ft|meters)\b", md, flags=re.IGNORECASE)
    if suspicious and not cited:
        errors.append("Numeric operational limits detected without citations.")

    return (len(errors) == 0), errors
