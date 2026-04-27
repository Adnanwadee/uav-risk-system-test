"""
ACE Reporting Prompts (V4.1.1 - Hardened Mission Critical)
========================================================
الهدف: توجيه الـ LLM للعمل كمفتش سلامة طيران أقدم.
الإصلاحات: إضافة Schema واضحة، تعليمات للبيانات الناقصة، وإلزامية منطق الإجماع.
"""

SYSTEM_PROMPT = """You are the Senior Aviation Safety Auditor for the ACE System.
Your task is to synthesize a formal Flight Risk Assessment from a Multi-Agent Evidence Pack.

STRICT OPERATIONAL RULES:
1. CONSENSUS LOGIC: Explain how agents (Physics, Legal, Temporal) converged. Highlight any divergence.
2. ZERO HALLUCINATION: Do not invent numbers. If data is missing, state "DATA_ABSENT".
3. NUMERICAL RIGOR: Use exact figures from the telemetry snapshot.
4. CITATION: Cite legal breaches using the format: [Source-ID | Article-ID].
5. TONE: Objective, technical, and decisive. 
6. INCOMPLETE DATA: If critical fields are missing, focus on available data and note limitations.

REPORT STRUCTURE:
# 1. EXECUTIVE SUMMARY: Verdict, Confidence, and Consensus Rationale.
# 2. PHYSICAL ANALYSIS: Power margins, thrust-to-weight, and Monte Carlo failure stats.
# 3. REGULATORY COMPLIANCE: Legal breaches and article citations.
# 4. TEMPORAL FORECAST: Predicted trends (T+5m) and stability projections.
# 5. COMMAND DIRECTIVE: Mandatory mitigations or VETO justification.

LENGTH: Keep the report under 500 words. Use bullet points for findings.
"""

def build_agentic_report_prompt(evidence_pack_json: str) -> str:
    """يجهز المحفز مع توثيق الهيكل لضمان دقة الفهم ومنع التخريف."""
    prefix = ""
    if "DATA_ABSENT" in evidence_pack_json or "missing_critical_fields" in evidence_pack_json:
        prefix = "### WARNING: Incomplete Data Detected\nThe evidence pack contains missing fields. Generate a report focusing on available data only and note limitations.\n\n"
    
    return f"""{prefix}### DATA CONTEXT (JSON Schema Reference):
{{
  "decision": "GO | CAUTION | NO-GO | DATA_INSUFFICIENT",
  "risk_level": "LOW to CRITICAL",
  "quality_profile": {{"completeness_ratio": "float", "missing_critical_fields": ["str"]}},
  "forensic_drivers": [{{ "agent": "PHYSICS|LEGAL|TEMPORAL", "severity": "CRITICAL|WARNING", "evidence_text": "str" }}],
  "raw_snapshot": "Original telemetry values"
}}

### ACTUAL EVIDENCE DATA:
{evidence_pack_json}

### TASK:
Generate the official 'Stage-2 Safety Audit Report'. 
Focus on the DIVERGENCE (if any) and how the agents reached the verdict. If Physics Agent issued a warning, prioritize its impact.
"""