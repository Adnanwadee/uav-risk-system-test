from __future__ import annotations

from typing import Any, Dict, List


def _fmt(x: Any, nd: int = 2) -> str:
    if x is None:
        return "N/A"
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}"
    return str(x)


def render_report_md(ep: Dict[str, Any]) -> str:
    decision = ep.get("decision", "INSUFFICIENT_DATA")
    s1 = ep.get("stage1_facts", {}) or {}
    dq = ep.get("data_quality", {}) or {}
    drivers = ep.get("risk_drivers", []) or []
    rules = ep.get("rules", {}) or {}
    ev = ep.get("evidence_snippets", []) or []

    present_count = dq.get("present_count")
    total_count = dq.get("total_count")
    completeness = dq.get("completeness_ratio")

    lines: List[str] = []
    lines.append("# UAV Flight Risk Report (Stage-2)")
    lines.append("")
    lines.append("## 1) Executive Summary")
    lines.append(f"- **Decision:** {decision}")
    lines.append(f"- **Predicted Class:** {s1.get('predicted_class', 'N/A')}")
    lines.append(f"- **Risk Score:** {_fmt(s1.get('risk_score'), 4)}")
    lines.append(f"- **Confidence:** {_fmt(s1.get('confidence'), 4)}")
    lines.append(
        f"- **Completeness:** {_fmt(completeness, 2)} "
        f"({present_count if present_count is not None else 'N/A'}/"
        f"{total_count if total_count is not None else 'N/A'})"
    )
    lines.append("")

    lines.append("## 2) Key Risk Drivers")
    if drivers:
        for d in drivers[:8]:
            lines.append(f"- **{d.get('driver', 'N/A')}**: {d.get('value', 'N/A')} — {d.get('note', '')}")
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
    # EvidenceSnippet schema: {source, content, citation}
    if ev:
        for i, e in enumerate(ev[:10], start=1):
            src = e.get("source", "unknown")
            cit = e.get("citation")
            content = (e.get("content") or "").strip()
            content_short = content[:220] + ("..." if len(content) > 220 else "")
            if cit:
                lines.append(f"- **E{i}** ({src}) — {cit}")
            else:
                lines.append(f"- **E{i}** ({src})")
            if content_short:
                lines.append(f"  - {content_short}")
    else:
        lines.append("- No evidence snippets available.")
    lines.append("")

    lines.append("## 5) Data Quality & Limitations")
    missing = dq.get("missing_keys", []) or []
    if missing:
        lines.append(
            f"- Missing/unknown inputs detected: {len(missing)} fields are null/NaN, "
            "which may reduce trustworthiness."
        )
        for k in missing[:12]:
            lines.append(f"  - `{k}`")
        if len(missing) > 12:
            lines.append(f"  - ... and {len(missing) - 12} more")
    else:
        lines.append("- No missing inputs detected in the provided snapshot.")
    lines.append("")
    return "\n".join(lines)
