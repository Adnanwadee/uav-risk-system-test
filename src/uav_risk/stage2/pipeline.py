from __future__ import annotations

from typing import Any, Dict

from uav_risk.stage1.infer import run_stage1_inference
from uav_risk.stage1.canonicalize import canonicalize_scenario

from uav_risk.stage2.rules import run_rules
from uav_risk.stage2.evidence import build_evidence_pack
from uav_risk.stage2.report.templates import render_report_md
from uav_risk.stage2.schemas import Stage2Response
from uav_risk.utils.json_sanitize import sanitize_for_json


def run_stage2_report(
    scenario: Dict[str, Any],
    artifacts_dir: str = "artifacts",
) -> Stage2Response:
    # --------------------------------------------------
    # 1) Stage-1 inference (FACTS ONLY)
    # --------------------------------------------------
    s1_raw = run_stage1_inference(scenario, artifacts_dir=artifacts_dir)
    s1 = sanitize_for_json(s1_raw)

    # --------------------------------------------------
    # 2) Canonicalize inputs → snapshot (and sanitize)
    # --------------------------------------------------
    df = canonicalize_scenario(scenario)
    inputs_snapshot_raw = df.iloc[0].to_dict()
    inputs_snapshot = sanitize_for_json(inputs_snapshot_raw)

    # --------------------------------------------------
    # 3) Rules engine (MUST return RulesResult)
    # --------------------------------------------------
    rules = run_rules(stage1_facts=s1, inputs_snapshot=inputs_snapshot)
    if rules is None:
        raise RuntimeError("run_rules returned None – BUG: must return RulesResult")

    rules_dict = rules.model_dump()

    # --------------------------------------------------
    # 4) Evidence Pack (single source of truth)
    # --------------------------------------------------
    ep = build_evidence_pack(
        inputs_snapshot=inputs_snapshot,
        stage1_facts=s1,
        rules=rules_dict,  # pass dict
    )

    # --------------------------------------------------
    # 5) Build sections (sanitize-safe)
    # --------------------------------------------------
    sections: Dict[str, Any] = {
        "decision": s1.get("decision", "INSUFFICIENT_DATA"),
        "stage1_facts": s1,
        "rules": rules_dict,
        "inputs_snapshot": inputs_snapshot,
        "data_quality": ep.data_quality.model_dump(),
        "risk_drivers": ep.risk_drivers,
        "evidence_snippets": [e.model_dump() for e in ep.evidence_snippets],
    }
    sections = sanitize_for_json(sections)

    # --------------------------------------------------
    # 6) Render report (no logic)
    # --------------------------------------------------
    report_md = render_report_md(sections)

    # --------------------------------------------------
    # 7) Final decision (policy layer)
    # --------------------------------------------------
    decision = s1.get("decision", "INSUFFICIENT_DATA")

    if rules.hard_violations:
        decision = "NO_GO"
    elif rules.advisories and decision == "GO":
        decision = "CAUTION"
    elif ep.data_quality.completeness_ratio < 0.65:
        decision = "INSUFFICIENT_DATA"

    # --------------------------------------------------
    # 8) Response (return Stage2Response, not dict)
    # --------------------------------------------------
    resp = Stage2Response(
        status="OK",
        decision=decision,  # type: ignore[arg-type]
        facts={
            "stage1": s1,
            "rules": rules_dict,
        },
        report_md=report_md,
        report_json={"sections": sections},
        evidence=ep.evidence_snippets,  # typed models
        quality={"data_quality": ep.data_quality.model_dump()},
    )

    # Ensure absolutely JSON-safe, but keep Stage2Response type
    payload = sanitize_for_json(resp.model_dump())
    return Stage2Response.model_validate(payload)
