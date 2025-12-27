from __future__ import annotations
from typing import Any, Dict

from uav_risk.stage1.infer import run_stage1_inference
from uav_risk.stage1.canonicalize import canonicalize_scenario

from uav_risk.stage2.rules import run_rules
from uav_risk.stage2.evidence import build_evidence_pack
from uav_risk.stage2.report.templates import  render_report_md
from uav_risk.stage2.schemas import Stage2Response


def run_stage2_report(
    scenario: Dict[str, Any],
    artifacts_dir: str = "artifacts",
) -> Stage2Response:
    """
    Stage-2 Orchestrator
    - Calls Stage-1
    - Applies rules
    - Builds evidence pack
    - Renders report
    - Returns STRICT typed response
    """

    # --------------------------------------------------
    # 1) Stage-1 facts (numerical truth)
    # --------------------------------------------------
    s1 = run_stage1_inference(scenario, artifacts_dir=artifacts_dir)

    # --------------------------------------------------
    # 2) Canonical snapshot
    # --------------------------------------------------
    df = canonicalize_scenario(scenario)
    snapshot = df.iloc[0].to_dict()

    # --------------------------------------------------
    # 3) Rules engine
    # --------------------------------------------------
    rules = run_rules(
        stage1_facts=s1,
        inputs_snapshot=snapshot,
    )

    # --------------------------------------------------
    # 4) Evidence pack (typed models)
    # --------------------------------------------------
    ep = build_evidence_pack(
        inputs_snapshot=snapshot,
        stage1_facts=s1,
        rules=rules,
    )

    ep_dict = ep.model_dump()

    # --------------------------------------------------
    # 5) Report rendering (pure function)
    # --------------------------------------------------
    md_final = render_report_md(ep_dict)

    # --------------------------------------------------
    # 6) Final decision resolution
    # --------------------------------------------------
    decision = s1.get("decision", "INSUFFICIENT_DATA")

    if rules.hard_violations:
        decision = "NO_GO"
    elif rules.advisories and decision == "GO":
        decision = "CAUTION"

    # --------------------------------------------------
    # 7) RETURN — ✅ schema-correct
    # --------------------------------------------------
    return Stage2Response(
        status="OK",
        decision=decision,  # type: ignore[arg-type]
        facts={
            "stage1": s1,
            "rules": rules.model_dump(),
        },
        report_md=md_final,
        report_json={
            "sections": ep_dict,
        },
        evidence=ep.evidence_snippets,   # ✅ models, NOT dicts
        quality={
            "data_quality": ep.data_quality,  # ✅ model, NOT dict
        },
    )
