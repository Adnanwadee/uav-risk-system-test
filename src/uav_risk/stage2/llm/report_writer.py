from __future__ import annotations

from typing import Any, Dict, List, Optional

# ============================
# System Prompt (LLM Behavior)
# ============================

SYSTEM_PROMPT = """You are an aviation safety analyst and technical report writer.
Write in professional, conservative aviation language suitable for regulators and enterprise UAV operators.

Hard rules:
- Do NOT invent facts.
- Every safety-relevant claim must cite at least one evidence item (E1, E2, ...).
- If evidence is insufficient, explicitly state "Evidence is insufficient" and recommend data collection.
- Never change numeric values provided in the scenario.
- NEVER change the final decision provided by Stage-2.
"""


# ============================
# Prompt Builder
# ============================

def build_user_prompt(ep: Dict[str, Any]) -> str:
    decision = ep.get("decision")
    s1 = ep.get("stage1_facts") or {}
    contract = ep.get("input_contract") or {}
    drivers = ep.get("risk_drivers") or []
    evidence = ep.get("evidence_snippets") or []

    # Evidence listing (short, indexed)
    ev_lines: List[str] = []
    for i, e in enumerate(evidence, start=1):
        ev_lines.append(
            f"E{i}: {e.get('citation')}\n{(e.get('content') or '').strip()}"
        )

    return f"""
Generate a UAV Flight Risk Assessment Report in English.

Inputs:
- Final Stage-2 decision: {decision}
- Stage-1 model output:
  predicted_class={s1.get('predicted_class')}
  decision={s1.get('decision')}
  risk_score={s1.get('risk_score')}
  confidence={s1.get('confidence')}

- Input contract:
  model_ready={contract.get('model_ready')}
  safety_ready={contract.get('safety_ready')}
  missing_model_keys={contract.get('missing_model_keys')}
  missing_safety_keys={contract.get('missing_safety_keys')}

Risk drivers (structured, factual):
{drivers}

Evidence (you must cite using E1..En):
{chr(10).join(ev_lines)}

Report requirements:
1. Executive Summary (final decision + rationale)
2. Operational Risk Interpretation (map drivers to operational consequences; cite)
3. Compliance & Constraints (hard vs advisory; cite)
4. Mitigation Recommendations (actionable, prioritized; cite)
5. Data Quality & Gaps (missing inputs and impact on confidence)
6. References (list E1..En with citation strings)

Formatting: Markdown.
"""


# ============================
# LLM Report Writer (ENTRY POINT)
# ============================

def write_llm_report(
    decision: str,
    stage1_facts: Dict[str, Any],
    rules: Dict[str, Any],
    risk_context: Dict[str, Any],
    evidence_snippets: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generates a professional, cited LLM-written report section.
    This function does NOT make decisions. It only explains them.

    Returns a structured block to be embedded into Stage-2 report.
    """

    # Assemble EP object expected by prompt builder
    ep = {
        "decision": decision,
        "stage1_facts": stage1_facts,
        "rules": rules,
        "risk_drivers": risk_context.get("risk_drivers", []),
        "input_contract": risk_context.get("input_contract", {}),
        "evidence_snippets": evidence_snippets,
    }

    user_prompt = build_user_prompt(ep)

    # NOTE:
    # We intentionally do NOT call Groq API here directly
    # to keep pipeline deterministic and testable.
    #
    # The caller (or future infra layer) can send:
    #   SYSTEM_PROMPT + user_prompt
    # to Groq and store the response.

    return {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "expected_style": "aviation_regulatory",
        "citations_required": True,
        "llm_provider": "groq",
    }
