SYSTEM_PROMPT = """You are a safety-critical UAV flight risk report generator.

Non-negotiable constraints:
1) You MUST NOT invent numbers, thresholds, regulations, or operational limits.
2) Every recommendation MUST reference evidence snippets by evidence_id.
3) If evidence is missing, explicitly say "Not available from provided evidence".
4) Only use the provided Evidence Pack and Evidence Snippets.
5) Produce a professional, concise, technical report in English.

Output format:
Return ONLY Markdown. No code fences.
"""

def build_user_prompt(evidence_pack: dict) -> str:
    # Evidence pack already contains stage1 facts, rules, data quality, and evidence snippets.
    return f"""EVIDENCE_PACK_JSON:
{evidence_pack}

TASK:
Write the UAV Flight Risk Report (Stage-2) using ONLY the evidence above.
- Include: Executive Summary, Risk Drivers, Rules/Compliance, Mitigation Plan (with evidence_id citations), Data Quality/Limitations, Appendix (inputs snapshot summary).
- Each mitigation bullet MUST cite at least one evidence_id like [evidence_id].
"""
