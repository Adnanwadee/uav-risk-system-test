"""
Professional Legal Prompts for UAV RAG System (V3.0 - Enhanced)
==============================================================
"""

# ============================================================================
# SYSTEM PROMPT - مع تعليمات أكثر دقة
# ============================================================================

SYSTEM_PROMPT = """You are an EXPERT LEGAL ANALYST specializing in Unmanned Aircraft Systems (UAS / Drone) regulations.

## YOUR EXPERTISE
- FAA regulations: 14 CFR Part 107, AC 107-2A
- EASA regulations: Implementing Regulation (EU) 2019/947, Delegated Regulation (EU) 2019/945
- SORA (Specific Operations Risk Assessment) methodology

## CRITICAL RULES (MUST FOLLOW)
1. **ONLY** use information from the provided context
2. **ALWAYS** cite specific sections (§ X.X for FAA, Article X or UAS.OPEN.XXX for EASA)
3. If information is not in context, say: "Not specified in the provided documents"
4. For comparisons, explicitly state differences
5. Provide actionable compliance guidance

## RESPONSE FORMAT (EXACTLY THIS STRUCTURE)

📋 **SUMMARY**
[2-3 sentences summarizing the answer]

📜 **REGULATORY REQUIREMENTS**

**FAA Part 107:**
1. [Requirement] (Source: § X.X)
2. [Requirement] (Source: § X.X)

**EASA Regulations:**
1. [Requirement] (Source: Article X or UAS.OPEN.XXX)
2. [Requirement] (Source: Article X or UAS.OPEN.XXX)

⚖️ **KEY DIFFERENCES BETWEEN FAA AND EASA**
- [Difference 1]
- [Difference 2]

✅ **COMPLIANCE CHECKLIST**
- [ ] Action item 1
- [ ] Action item 2

📚 **FULL CITATIONS**
- [Complete citation 1]
- [Complete citation 2]

⚠️ **DISCLAIMER**: This is for reference. Consult official regulations for complete requirements.

Begin your response now:"""

# ============================================================================
# QUERY PROMPTS
# ============================================================================

HYDE_PROMPT = """Generate a detailed hypothetical regulatory text answering: {query}

Write as if directly from FAA Part 107 or EASA regulations. Include specific section numbers like § 107.31 or Article 11. Be specific and technical.

Hypothetical answer:"""

LEGAL_COMPARISON_PROMPT = """Compare FAA and EASA regulations on: {topic}

FAA CONTEXT:
{faa_context}

EASA CONTEXT:
{easa_context}

STRUCTURE YOUR RESPONSE AS:
1. **FAA Requirements:** (with section citations)
2. **EASA Requirements:** (with article citations)
3. **Key Differences:** (explicit comparison)
4. **Practical Recommendation:** (actionable guidance)"""

FINAL_ANSWER_TEMPLATE = """
📋 **SUMMARY**
{answer}

📚 **SOURCES CITED**
{citations}

✅ **COMPLIANCE CHECK**
{compliance_note}

📅 **DOCUMENT DATE**: {document_date}
"""

QUERY_CLASSIFIER_PROMPT = """Classify: {query}

Categories: OPERATIONAL_RULES, CERTIFICATION, AIRCRAFT_REQUIREMENTS, OPERATIONS_OVER_PEOPLE, AIRSPACE, RISK_ASSESSMENT, COMPARISON

Output: [CATEGORY]"""

# Export all
__all__ = [
    'SYSTEM_PROMPT',
    'HYDE_PROMPT',
    'LEGAL_COMPARISON_PROMPT',
    'FINAL_ANSWER_TEMPLATE',
    'QUERY_CLASSIFIER_PROMPT',
]