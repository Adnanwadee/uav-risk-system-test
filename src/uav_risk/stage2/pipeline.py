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
# Stage-2 Decision Policy v3 (Product-grade)
# ============================
# Principles:
# - Stage-2 is the authority.
# - Stage-1 model is a signal (evidence), not a veto by itself.
# - HARD violations come only from explicit constraints (rules engine), not the model alone.
# - If safety inputs are missing => INSUFFICIENT_DATA (regulator-grade posture).


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
    - Excludes meta drivers that must not trigger NO_GO:
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

def _stage2_decision_policy(
    *,
    base_decision: str,
    rules_dict: Dict[str, Any],
    contract: Dict[str, Any],
    inputs_snapshot: Dict[str, Any],
    stage1_facts: Dict[str, Any],
    risk_drivers: List[RiskDriver],
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (final_decision, policy_debug).

    Product-grade policy:
    1) Any HARD violation => NO_GO (explicit constraints only).
    2) If safety_ready is False => never GO.
    3) Operational drivers drive the decision; the model is a strong signal but not a veto.
    4) >=2 HIGH operational drivers => at least CAUTION.
       Escalate to NO_GO only if model_high_conf ALSO present (or if hard violations exist).
    5) Model high-conf High Risk:
       - Alone => CAUTION
       - With >=1 HIGH operational driver => NO_GO (strong combined evidence)
    6) Any advisories => at least CAUTION (unless already NO_GO / INSUFFICIENT_DATA)
    """

    policy_debug: Dict[str, Any] = {"notes": []}

    hard = rules_dict.get("hard_violations") or []
    adv = rules_dict.get("advisories") or []
    safety_ready = bool(contract.get("safety_ready", True))

    # Stats from drivers (excludes MODEL_PREDICTION / RULES_TRIGGERED / MISSING_SAFETY_INPUTS)
    stats = _risk_driver_stats(risk_drivers)
    policy_debug["driver_stats"] = {k: stats[k] for k in ("high", "medium", "unknown")}

    # Model signal
    predicted = str(stage1_facts.get("predicted_class", "")).lower()
    confidence = float(stage1_facts.get("confidence", 0.0) or 0.0)
    model_high_conf = predicted.startswith("high") and confidence >= 0.95
    policy_debug["model_high_conf"] = model_high_conf
    policy_debug["model_confidence"] = confidence
    policy_debug["model_predicted_class"] = stage1_facts.get("predicted_class")

    # 1) HARD constraints => NO_GO
    if isinstance(hard, list) and len(hard) > 0:
        policy_debug["notes"].append("HARD violations present => NO_GO.")
        return "NO_GO", policy_debug

    # 2) Safety gate: if not safety_ready => never GO
    if not safety_ready:
        policy_debug["notes"].append("Safety not ready => decision cannot be GO.")
        if base_decision == "INSUFFICIENT_DATA":
            return "INSUFFICIENT_DATA", policy_debug
        return "CAUTION", policy_debug

    # Start from base decision
    decision = base_decision

    # Normalize: if base is NO_GO but only model caused it upstream, we still let policy decide.
    # (We do not trust base_decision as an authority—Stage-2 is authority.)
    if decision not in {"GO", "CAUTION", "NO_GO", "INSUFFICIENT_DATA"}:
        decision = "INSUFFICIENT_DATA"

    # 3) Operational escalation (product-grade)
    # >=2 HIGH operational drivers => at least CAUTION (not automatic NO_GO)
    if stats["high"] >= 2 and decision != "INSUFFICIENT_DATA":
        policy_debug["notes"].append(">=2 HIGH operational drivers => at least CAUTION.")
        decision = "CAUTION" if decision == "GO" else decision

        # Escalate to NO_GO only if model_high_conf supports it
        if model_high_conf:
            policy_debug["notes"].append(">=2 HIGH operational drivers + model high-conf => NO_GO.")
            decision = "NO_GO"

    # 4) Model high-confidence High Risk handling
    if model_high_conf:
        if decision == "GO":
            policy_debug["notes"].append("Model High-Risk high-confidence => elevate GO to CAUTION (not veto).")
            decision = "CAUTION"

        # If at least one HIGH operational driver exists, we accept combined evidence => NO_GO
        if stats["high"] >= 1:
            policy_debug["notes"].append("Model high-conf + >=1 HIGH operational driver => NO_GO.")
            decision = "NO_GO"

    # 5) Advisories ensure at least CAUTION (unless already NO_GO / INSUFFICIENT_DATA)
    if isinstance(adv, list) and len(adv) > 0 and decision == "GO":
        policy_debug["notes"].append("Advisories present and decision GO => CAUTION.")
        decision = "CAUTION"

    # 6) Extra guard: missing safety keys => cannot be GO
    missing_safety = contract.get("missing_safety_keys") or []
    if missing_safety and decision == "GO":
        policy_debug["notes"].append("Missing safety keys => cannot be GO, forcing CAUTION.")
        decision = "CAUTION"

    return decision, policy_debug


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
    # Base decision is informational only (Stage-2 is the authority)
    base_decision = str(s1.get("decision", "INSUFFICIENT_DATA"))

    decision, policy_debug = _stage2_decision_policy(
        base_decision=base_decision,
        rules_dict=rules_dict,
        contract=contract,
        inputs_snapshot=inputs_snapshot,
        stage1_facts=s1,
        risk_drivers=rc_drivers,
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
        "data_quality": ep.data_quality.model_dump(),
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
        quality={"data_quality": ep.data_quality.model_dump(), "input_contract": contract},
    )

    payload = sanitize_for_json(resp.model_dump())
    return Stage2Response.model_validate(payload)
