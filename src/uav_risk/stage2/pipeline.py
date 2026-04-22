from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Tuple

from uav_risk.stage1.infer import run_stage1_inference
from uav_risk.stage1.canonicalize import canonicalize_scenario

from uav_risk.stage2.rules import run_rules
from uav_risk.stage2.evidence import build_evidence_pack
from uav_risk.stage2.report.templates import render_report_md
from uav_risk.stage2.schemas import Stage2Response
from uav_risk.utils.json_sanitize import sanitize_for_json
from uav_risk.stage2.input_contract import validate_input_contract

# NEW (AI / RAG / LLM)
from uav_risk.stage2.agents.risk_context import build_risk_context, RiskDriver
from uav_risk.stage2.agents.evidence_builder import build_evidence_snippets

# RAG
from uav_risk.stage2.rag.index import RAGIndex
from uav_risk.stage2.rag.loader import load_knowledge_chunks

# LLM prompt block builder (deterministic)
from uav_risk.stage2.llm.report_writer import write_llm_report


# ============================
# Stage-2 Decision Policy v4 (Product-grade, Non-fragile)
# ============================
# Principles:
# - Stage-2 is the authority.
# - Stage-1 model is a signal (NOT a veto).
# - HARD violations come only from explicit constraints (rules engine).
# - If safety inputs are missing => never return GO.
# - Use aggregated risk scoring to support GO / CAUTION / NO_GO robustly.


CRITICAL_KEYS = [
    "uav.mass_kg",
    "uav.max_speed_mps",
    "uav.battery_model.hover_power_W",
    "environment.weather.wind_mps",
    "environment.weather.gust_mps",
    "environment.weather.visibility",
    "environment.gnss_jam_dbm",
    "environment.gnss_multipath",
    "environment.em_interference",
]

CRITICAL_WEIGHTS = {
    "uav.mass_kg": 1.0,
    "uav.max_speed_mps": 1.0,
    "uav.battery_model.hover_power_W": 0.8,
    "environment.weather.wind_mps": 1.0,
    "environment.weather.gust_mps": 0.7,
    "environment.weather.visibility": 0.6,
    "environment.gnss_jam_dbm": 0.9,
    "environment.gnss_multipath": 0.6,
    "environment.em_interference": 0.6,
}


def _missing_critical(inputs_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    missing = []
    present_weight = 0.0
    total_weight = 0.0

    for k in CRITICAL_KEYS:
        w = float(CRITICAL_WEIGHTS.get(k, 1.0))
        total_weight += w
        v = inputs_snapshot.get(k, None)
        is_missing = (v is None)
        if is_missing:
            missing.append(k)
        else:
            present_weight += w

    critical_completeness = (present_weight / total_weight) if total_weight > 0 else 0.0
    return {
        "missing_keys": missing,
        "critical_completeness": critical_completeness,
        "missing_count": len(missing),
        "total_count": len(CRITICAL_KEYS),
    }


def _serialize_risk_drivers(drivers: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for d in (drivers or []):
        if is_dataclass(d):
            out.append(asdict(d))
        elif isinstance(d, dict):
            out.append(d)
        else:
            out.append({"value": str(d)})
    return out


def _build_rag_index(knowledge_dir: str) -> RAGIndex:
    chunks = load_knowledge_chunks(knowledge_dir, allow_prefixes=None)
    return RAGIndex.build(chunks)


def _risk_driver_stats(drivers: List[RiskDriver]) -> Dict[str, Any]:
    """
    Compute severity stats for operational drivers.

    IMPORTANT:
    - Excludes meta drivers that must not trigger decisions directly:
      MODEL_PREDICTION, RULES_TRIGGERED, MISSING_SAFETY_INPUTS
    - Correctly counts LOW / MEDIUM / HIGH / UNKNOWN.
    """
    exclude = {"MODEL_PREDICTION", "RULES_TRIGGERED", "MISSING_SAFETY_INPUTS"}

    hi = 0
    med = 0
    low = 0
    unk = 0
    used: List[RiskDriver] = []

    for d in (drivers or []):
        if getattr(d, "driver_id", "") in exclude:
            continue

        sev = str(getattr(d, "severity", "UNKNOWN")).upper().strip()
        used.append(d)

        if sev == "HIGH":
            hi += 1
        elif sev == "MEDIUM":
            med += 1
        elif sev == "LOW":
            low += 1
        else:
            unk += 1

    return {"high": hi, "medium": med, "low": low, "unknown": unk, "drivers_used": used}


def _model_points(predicted_class: str, confidence: float) -> int:
    pred = str(predicted_class or "").lower().strip()

    if pred.startswith("high"):
        if confidence >= 0.98:
            return 5
        if confidence >= 0.90:
            return 4
        return 3

    if pred.startswith("medium"):
        return 3 if confidence >= 0.90 else 2

    if pred.startswith("low"):
        return 0 if confidence >= 0.90 else 1

    # Unknown / missing
    return 2


def _risk_score_points(risk_score: float | None) -> int:
    if risk_score is None:
        return 1
    try:
        rs = float(risk_score)
    except Exception:
        return 1

    if rs >= 2.7:
        return 2
    if rs >= 2.2:
        return 1
    return 0


def _stage2_decision_policy(
    *,
    rules_dict: Dict[str, Any],
    contract: Dict[str, Any],
    inputs_snapshot: Dict[str, Any],
    stage1_facts: Dict[str, Any],
    risk_drivers: List[RiskDriver],
    data_quality: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (final_decision, policy_debug).

    Product-grade policy (stable, non-fragile):
    1) HARD violations => NO_GO (explicit rules only).
    2) safety_ready gate => never GO (=> CAUTION).
    3) Operational HIGH drivers => NO_GO (they represent quantified/validated hazards).
    4) Aggregate points decide CAUTION/GO.
       - Model contributes points but never alone forces NO_GO.
       - Model high-conf High Risk => at least CAUTION (signal).
    """

    policy_debug: Dict[str, Any] = {"notes": []}

    hard = rules_dict.get("hard_violations") or []
    adv = rules_dict.get("advisories") or []
    safety_ready = bool(contract.get("safety_ready", True))

    # 1) HARD constraints => NO_GO
    if isinstance(hard, list) and len(hard) > 0:
        policy_debug["notes"].append("HARD violations present => NO_GO.")
        policy_debug["risk_points"] = 999
        return "NO_GO", policy_debug

    # 2) Safety gate: never GO if safety inputs missing
    if not safety_ready:
        policy_debug["notes"].append("Safety not ready (missing safety-critical inputs) => CAUTION.")
        return "CAUTION", policy_debug

    # Stats from drivers (excludes meta drivers)
    stats = _risk_driver_stats(risk_drivers)
    policy_debug["driver_stats"] = {k: stats[k] for k in ("high", "medium", "low", "unknown")}

    # 3) Operational HIGH drivers are authoritative hazards
    if int(stats["high"]) >= 1:
        policy_debug["notes"].append(">=1 HIGH operational driver => NO_GO (operational hazard).")
        policy_debug["risk_points"] = 900
        return "NO_GO", policy_debug

    # Model signal
    predicted_class = stage1_facts.get("predicted_class", "UNKNOWN")
    confidence = float(stage1_facts.get("confidence", 0.0) or 0.0)
    risk_score = stage1_facts.get("risk_score", None)

    policy_debug["model_predicted_class"] = predicted_class
    policy_debug["model_confidence"] = confidence
    policy_debug["model_risk_score"] = risk_score

    model_high_conf = str(predicted_class).lower().startswith("high") and confidence >= 0.95
    policy_debug["model_high_conf"] = model_high_conf

    # Data quality
    completeness = float(data_quality.get("completeness_ratio", 0.0) or 0.0)
    missing_keys = data_quality.get("missing_keys") or []
    policy_debug["completeness_ratio"] = completeness
    policy_debug["missing_key_count"] = len(missing_keys)

    # dq flags
    dq_comms_present = int(inputs_snapshot.get("dq_comms_present", 1) or 0)
    dq_sensors_present_pct = float(inputs_snapshot.get("dq_sensors_present_pct", 1.0) or 0.0)

    # 4) Aggregate risk points (NO direct NO_GO from model)
    points = 0

    # (A) Model contributes points (signal only)
    points += _model_points(str(predicted_class), confidence)
    points += _risk_score_points(risk_score)

    # (B) Operational non-HIGH drivers
    points += 1 * int(stats["medium"])
    points += min(2, int(stats["unknown"]))  # cap unknown penalty

    # (C) Data quality penalties
    if completeness < 0.85:
        points += 3
    elif completeness < 0.95:
        points += 1

    if dq_comms_present == 0:
        points += 1
    if dq_sensors_present_pct < 0.5:
        points += 1

    # (D) Advisories (soft)
    if isinstance(adv, list) and len(adv) > 0:
        points += min(2, len(adv))

    policy_debug["risk_points"] = points

    # 5) Decision thresholds
    # - Model high-conf High Risk => at least CAUTION (not NO_GO)
    if model_high_conf:
        policy_debug["notes"].append("Model high-confidence High Risk => minimum CAUTION (signal only).")
        return "CAUTION", policy_debug

    # - Otherwise use points
    if points >= 6:
        policy_debug["notes"].append("Aggregate risk points >= 6 => CAUTION.")
        return "CAUTION", policy_debug

    if points >= 3:
        return "CAUTION", policy_debug

    return "GO", policy_debug


def run_stage2_report(
    scenario: Dict[str, Any],
    artifacts_dir: str = "artifacts",
) -> Stage2Response:
    # --------------------------------------------------
    # 1) Canonicalize inputs → snapshot (sanitize)
    # --------------------------------------------------
    df = canonicalize_scenario(scenario)
    inputs_snapshot_raw = df.iloc[0].to_dict()
    inputs_snapshot = sanitize_for_json(inputs_snapshot_raw)

    # --------------------------------------------------
    # 2) Input Contract Gate (NO imputation)
    # --------------------------------------------------
    contract = validate_input_contract(inputs_snapshot)
    contract = sanitize_for_json(contract)

    # If model inputs are not ready, skip Stage-1 and return INSUFFICIENT_DATA.
    if not contract.get("model_ready", False):
        sections: Dict[str, Any] = {
            "decision": "INSUFFICIENT_DATA",
            "stage1_facts": {
                "status": "SKIPPED",
                "decision": "INSUFFICIENT_DATA",
                "predicted_class": "UNKNOWN",
                "risk_score": None,
                "confidence": 0.0,
                "probabilities": {},
                "reason": "Missing mandatory inputs required by model training.",
                "missing_model_keys": contract.get("missing_model_keys", []),
            },
            "rules": {"hard_violations": [], "advisories": [], "computed": {"rule_count": 0}},
            "inputs_snapshot": inputs_snapshot,
            "data_quality": {
                "present_count": int(df.notna().sum(axis=1).iloc[0]),
                "total_count": int(df.shape[1]),
                "completeness_ratio": float(df.notna().mean(axis=1).iloc[0]),
                "missing_keys": contract.get("missing_model_keys", []),
            },
            "risk_drivers": [],
            "risk_context": {"risk_drivers": [], "input_contract": contract},
            "evidence_snippets": [],
            "llm_section": None,
            "input_contract": contract,
            "policy_debug": {"notes": ["Model not ready => INSUFFICIENT_DATA."]},
        }
        sections = sanitize_for_json(sections)
        report_md = render_report_md(sections)

        resp = Stage2Response(
            status="OK",
            decision="INSUFFICIENT_DATA",
            facts={"stage1": sections["stage1_facts"], "rules": sections["rules"], "input_contract": contract},
            report_md=report_md,
            report_json={"sections": sections},
            evidence=[],
            quality={"input_contract": contract, "data_quality": sections["data_quality"]},
        )
        payload = sanitize_for_json(resp.model_dump())
        return Stage2Response.model_validate(payload)

    # --------------------------------------------------
    # 3) Stage-1 inference (FACTS)
    # --------------------------------------------------
    s1_raw = run_stage1_inference(scenario, artifacts_dir=artifacts_dir)
    s1 = sanitize_for_json(s1_raw)

    # --------------------------------------------------
    # 4) Rules engine (explicit constraints only)
    # --------------------------------------------------
    rules = run_rules(stage1_facts=s1, inputs_snapshot=inputs_snapshot)
    if rules is None:
        raise RuntimeError("run_rules returned None – must return RulesResult")
    rules_dict = sanitize_for_json(rules.model_dump())

    # --------------------------------------------------
    # 5) Evidence Pack (legacy deterministic pack)
    # --------------------------------------------------
    ep = build_evidence_pack(inputs_snapshot=inputs_snapshot, stage1_facts=s1, rules=rules_dict)

    # --------------------------------------------------
    # 6) Build Risk Context (structured, comprehensive)
    # --------------------------------------------------
    risk_context_raw = build_risk_context(
        inputs_snapshot=inputs_snapshot,
        stage1_facts=s1,
        rules=rules_dict,
        input_contract=contract,
    )

    rc_drivers: List[RiskDriver] = (risk_context_raw.get("risk_drivers") or [])
    rc_contract: Dict[str, Any] = (risk_context_raw.get("input_contract") or contract)

    risk_context_json = {
        "risk_drivers": _serialize_risk_drivers(rc_drivers),
        "input_contract": sanitize_for_json(rc_contract),
    }
    risk_context_json = sanitize_for_json(risk_context_json)

    # --------------------------------------------------
    # 7) Stage-2 Final Decision (AVIATION-GRADE authority)
    # --------------------------------------------------
    # IMPORTANT: We DO NOT take Stage-1 "decision" as base. Stage-1 is signal-only.
    dq_summary = ep.data_quality.model_dump()
    dq_summary = sanitize_for_json(dq_summary)

    decision, policy_debug = _stage2_decision_policy(
        rules_dict=rules_dict,
        contract=contract,
        inputs_snapshot=inputs_snapshot,
        stage1_facts=s1,
        risk_drivers=rc_drivers,
        data_quality=dq_summary,
    )

    policy_debug = sanitize_for_json(policy_debug)

    # --------------------------------------------------
    # 8) RAG Evidence Snippets (BM25 over knowledge files)
    # --------------------------------------------------
    knowledge_dir = "src/uav_risk/stage2/knowledge"
    rag_index = _build_rag_index(knowledge_dir)

    rag_evidence = build_evidence_snippets(
        index=rag_index,
        drivers=rc_drivers,
        top_k_per_driver=3,
    )
    rag_evidence = sanitize_for_json(rag_evidence)

    # --------------------------------------------------
    # 9) LLM Report Prompt Block (deterministic)
    # --------------------------------------------------
    llm_section = write_llm_report(
        decision=decision,
        stage1_facts=s1,
        rules=rules_dict,
        risk_context=risk_context_json,
        evidence_snippets=rag_evidence,
    )
    llm_section = sanitize_for_json(llm_section)

    # --------------------------------------------------
    # 10) Build sections (template consumes this)
    # --------------------------------------------------
    sections: Dict[str, Any] = {
        "decision": decision,
        "stage1_facts": s1,
        "rules": rules_dict,
        "inputs_snapshot": inputs_snapshot,
        "data_quality": dq_summary,
        "risk_drivers": ep.risk_drivers,           # legacy (keep)
        "risk_context": risk_context_json,         # comprehensive context
        "evidence_snippets": rag_evidence,         # RAG evidence
        "llm_section": llm_section,                # prompt block
        "input_contract": contract,
        "policy_debug": policy_debug,              # auditable behavior
    }
    sections = sanitize_for_json(sections)

    # --------------------------------------------------
    # 11) Render report
    # --------------------------------------------------
    report_md = render_report_md(sections)

    # --------------------------------------------------
    # 12) Response
    # Keep evidence=ep.evidence_snippets to avoid schema mismatches if Stage2Response expects the old shape.
    # RAG evidence is in report_json.sections["evidence_snippets"].
    # --------------------------------------------------
    resp = Stage2Response(
        status="OK",
        decision=decision,  # type: ignore[arg-type]
        facts={"stage1": s1, "rules": rules_dict, "input_contract": contract},
        report_md=report_md,
        report_json={"sections": sections},
        evidence=ep.evidence_snippets,
        quality={"data_quality": dq_summary, "input_contract": contract},
    )

    payload = sanitize_for_json(resp.model_dump())
    return Stage2Response.model_validate(payload)
