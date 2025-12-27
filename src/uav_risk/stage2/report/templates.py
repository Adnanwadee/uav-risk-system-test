from __future__ import annotations
from typing import Any, Dict, List


def render_report_md(ep: Dict[str, Any]) -> str:
    decision = ep.get("decision", "INSUFFICIENT_DATA")
    s1 = ep.get("stage1_facts", {})
    dq = ep.get("data_quality", {})
    drivers = ep.get("risk_drivers", [])
    rules = ep.get("rules", {})
    ev = ep.get("evidence_snippets", [])

    lines: List[str] = []
    lines.append("# UAV Flight Risk Report (Stage-2)")
    lines.append("")
    lines.append("## 1) Executive Summary")
    lines.append(f"- **Decision:** {decision}")
    lines.append(f"- **Predicted Class:** {s1.get('predicted_class')}")
    lines.append(f"- **Risk Score:** {s1.get('risk_score')}")
    lines.append(f"- **Confidence:** {s1.get('confidence')}")
    lines.append(f"- **Data Quality:** {dq.get('quality_level')} | Completeness: {dq.get('completeness_ratio'):.2f} ({dq.get('present_count')}/{dq.get('total_count')})")
    lines.append("")

    lines.append("## 2) Key Risk Drivers")
    if drivers:
        for d in drivers[:8]:
            lines.append(f"- **{d.get('driver')}**: {d.get('value')} — {d.get('note')}")
    else:
        lines.append("- No explicit drivers available.")
    lines.append("")

    lines.append("## 3) Rules & Compliance Findings")
    hard = rules.get("hard_violations", []) or []
    adv = rules.get("advisories", []) or []
    if hard:
        lines.append("### HARD Violations (NO-GO)")
        for r in hard:
            lines.append(f"- [{r.get('rule_id')}] {r.get('message')} | evidence={r.get('evidence')}")
    else:
        lines.append("- No hard violations detected.")
    lines.append("")
    if adv:
        lines.append("### Advisories")
        for r in adv:
            lines.append(f"- [{r.get('rule_id')}] {r.get('message')} | evidence={r.get('evidence')}")
    else:
        lines.append("- No advisories detected.")
    lines.append("")

    lines.append("## 4) Evidence Retrieved (for Mitigations)")
    if ev:
        for e in ev:
            lines.append(f"- **{e.get('evidence_id')}** ({e.get('source')}) score={e.get('score'):.2f}")
    else:
        lines.append("- No evidence snippets available.")
    lines.append("")

    lines.append("## 5) Data Quality & Limitations")
    missing = dq.get("missing_keys", []) or []
    if missing:
        lines.append(f"- Missing/unknown inputs detected: {len(missing)} fields are null/NaN, which may reduce trustworthiness.")
        for k in missing[:12]:
            lines.append(f"  - `{k}`")
        if len(missing) > 12:
            lines.append(f"  - ... and {len(missing) - 12} more")
    else:
        lines.append("- No missing inputs detected in the provided snapshot.")
    lines.append("")
    return "\n".join(lines)
