from __future__ import annotations

from typing import Any, Dict, List


def _fmt(x: Any, nd: int = 2) -> str:
    if x is None:
        return "N/A"
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}"
    return str(x)


def render_report_md(ep: Dict[str, Any]) -> str:
    # --------------------------------------------------
    # Canonical inputs (defensive reads)
    # --------------------------------------------------
    decision = ep.get("decision", "INSUFFICIENT_DATA")

    s1 = ep.get("stage1_facts", {}) or {}
    dq = ep.get("data_quality", {}) or {}
    drivers = ep.get("risk_drivers", []) or []
    rules = ep.get("rules", {}) or {}
    ev = ep.get("evidence_snippets", []) or []
    contract = ep.get("input_contract", {}) or {}

    present_count = dq.get("present_count")
    total_count = dq.get("total_count")
    completeness = dq.get("completeness_ratio")

    # --------------------------------------------------
    # Report body
    # --------------------------------------------------
    lines: List[str] = []

    # ===============================
    # Header
    # ===============================
    lines.append("# UAV Flight Risk Assessment Report")
    lines.append("")

    # ===============================
    # 1) Executive Summary
    # ===============================
    lines.append("## 1) Executive Summary")
    lines.append(f"- **Final Operational Decision (Stage-2):** {decision}")
    lines.append(
        f"- **Model Assessment (Stage-1):** "
        f"{s1.get('predicted_class', 'N/A')} "
        f"(model decision: {s1.get('decision', 'N/A')})"
    )
    lines.append(f"- **Risk Score (Regression):** {_fmt(s1.get('risk_score'), 4)}")
    lines.append(f"- **Model Confidence:** {_fmt(s1.get('confidence'), 4)}")
    lines.append(
        f"- **Input Completeness:** {_fmt(completeness, 2)} "
        f"({present_count if present_count is not None else 'N/A'} / "
        f"{total_count if total_count is not None else 'N/A'})"
    )
    lines.append("")

    # ===============================
    # 2) Input Contract & Readiness
    # ===============================
    if contract:
        lines.append("## 2) Input Contract & Readiness")
        lines.append(f"- **Model-Ready:** {contract.get('model_ready', 'N/A')}")
        lines.append(f"- **Safety-Ready:** {contract.get('safety_ready', 'N/A')}")

        mmk = contract.get("missing_model_keys", []) or []
        msk = contract.get("missing_safety_keys", []) or []

        if mmk:
            lines.append("- **Missing mandatory model inputs:**")
            for k in mmk:
                lines.append(f"  - `{k}`")

        if msk:
            lines.append("- **Missing safety-critical inputs:**")
            for k in msk:
                lines.append(f"  - `{k}`")

        lines.append("")

    # ===============================
    # 3) Key Risk Drivers
    # ===============================
    lines.append("## 3) Key Risk Drivers")
    if drivers:
        for d in drivers[:10]:
            lines.append(
                f"- **{d.get('driver', 'N/A')}**: "
                f"{d.get('value', 'N/A')} — {d.get('note', '')}"
            )
    else:
        lines.append("- No explicit risk drivers identified from available data.")
    lines.append("")

    # ===============================
    # 4) Rules & Compliance Analysis
    # ===============================
    lines.append("## 4) Rules & Compliance Analysis")

    hard = rules.get("hard_violations", []) or []
    adv = rules.get("advisories", []) or []

    if hard:
        lines.append("### HARD Violations (Operational NO-GO)")
        for r in hard:
            lines.append(
                f"- [{r.get('rule_id')}] {r.get('message')} "
                f"| evidence={r.get('evidence')}"
            )
    else:
        lines.append("- No hard safety violations detected.")

    lines.append("")

    if adv:
        lines.append("### Advisories & Mitigation Flags")
        for r in adv:
            lines.append(
                f"- [{r.get('rule_id')}] {r.get('message')} "
                f"| evidence={r.get('evidence')}"
            )
    else:
        lines.append("- No advisory conditions raised.")

    lines.append("")

    # ===============================
    # 5) Evidence & References
    lines.append("## 5) Evidence & Citations")
    if ev:
        for i, e in enumerate(ev[:12], start=1):
            cit = e.get("citation") or e.get("evidence_id") or "N/A"
            lines.append(f"- **E{i}**: {cit}")
    else:
        lines.append("- No external evidence retrieved for this assessment.")
    lines.append("")


    lines.append("")

    # ===============================
    # 6) Data Quality & LimitationsJ
    # ===============================
    lines.append("## 6) Data Quality & Limitations")

    missing = dq.get("missing_keys", []) or []
    if missing:
        lines.append(
            f"- {len(missing)} input fields are missing or undefined. "
            "This may limit the reliability of the assessment."
        )
        for k in missing[:14]:
            lines.append(f"  - `{k}`")
        if len(missing) > 14:
            lines.append(f"  - ... and {len(missing) - 14} additional fields")
    else:
        lines.append("- All required inputs were provided.")

    lines.append("")
    return "\n".join(lines)
