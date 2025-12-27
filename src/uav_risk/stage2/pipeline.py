from __future__ import annotations
from typing import Any, Dict

from uav_risk.stage1.infer import run_stage1_inference
from uav_risk.stage1.canonicalize import canonicalize_scenario

from .rules import run_rules
from .evidence import build_evidence_pack
from .report.templates import render_report_md
from .schemas import Stage2Response


def run_stage2_report(
    scenario: Dict[str, Any],
    artifacts_dir: str = "artifacts",
) -> Stage2Response:
    # --------------------------------------------------
    # 1) Stage-1 inference (FACTS ONLY)
    # --------------------------------------------------
    s1 = run_stage1_inference(
        scenario,
        artifacts_dir=artifacts_dir,
    )

    # --------------------------------------------------
    # 2) Canonicalize inputs → snapshot
    # --------------------------------------------------
    df = canonicalize_scenario(scenario)
    inputs_snapshot = df.iloc[0].to_dict()

    # --------------------------------------------------
    # 3) Rules engine
    # --------------------------------------------------
    rules = run_rules(
        stage1_facts=s1,
        inputs_snapshot=inputs_snapshot,
    )

    # --------------------------------------------------
    # 4) Evidence Pack (SINGLE SOURCE OF TRUTH)
    # --------------------------------------------------
    ep = build_evidence_pack(
        inputs_snapshot=inputs_snapshot,
        stage1_facts=s1,
        rules=rules,
    )

    sections = {
        "decision": s1.get("decision", "INSUFFICIENT_DATA"),
        "stage1_facts": s1,
        "rules": rules.model_dump(),
        "inputs_snapshot": inputs_snapshot,
        "data_quality": ep.data_quality.model_dump(),
        "risk_drivers": ep.risk_drivers,
    }

    # --------------------------------------------------
    # 5) Render report (NO LOGIC)
    # --------------------------------------------------
    report_md = render_report_md(sections)

    # --------------------------------------------------
    # 6) Final decision (policy layer)
    # --------------------------------------------------
    decision = s1.get("decision", "INSUFFICIENT_DATA")

    if rules.hard_violations:
        decision = "NO_GO"
    elif rules.advisories and decision == "GO":
        decision = "CAUTION"
    elif ep.data_quality.completeness_ratio < 0.65:
        decision = "INSUFFICIENT_DATA"

    # --------------------------------------------------
    # 7) API Response
    # --------------------------------------------------
    return Stage2Response(
        status="OK",
        decision=decision,  # type: ignore[arg-type]
        facts={
            "stage1": s1,
            "rules": rules.model_dump(),
        },
        report_md=report_md,
        report_json={"sections": sections},
        evidence=ep.evidence_snippets,
        quality={"data_quality": ep.data_quality.model_dump()},
    )
