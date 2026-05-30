"""
Prompts V3 - Contextual, Structured, Feature-aware
Production prompts with helper functions for LLM integration.
"""
# STAGE6_CLEANUP_REVIEW:
# Classification: OPTIONAL_RAG_PROMPTS_KEEP_NOW
# Plan lineage: PLAN3_OPTIONAL_RAG_SUPPORT
# Runtime status: Used by rag_core_v3.py synthesis path and legacy GroqLLM/HyDE prompt helpers.
# Legacy signal: Some prompt helpers support old groq_llm.py, but build_synthesis_prompt is still referenced by rag_core_v3.py.
# Replacement: None currently. Consider splitting later only if needed.
# Action rule: Do not delete now. Review function-by-function after RAG synthesis/HyDE usage is clarified.
from typing import Dict, List, Any, Optional

# ═══════════════════════════════════════════════════════════
# System Prompts for Groq LLM
# ═══════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an aviation regulatory analysis system specialized in UAV (drone) operations compliance. 
Your knowledge spans FAA (US), EASA (EU), ICAO (international), and CAAC (China) regulations.
Provide factual, well-cited answers based strictly on provided context.
Never invent regulations or citations."""

QUERY_CLASSIFIER_PROMPT = """Analyze this UAV operational query and classify it into exactly one category:
- EMERGENCY_RESPONSE: Emergency procedures, flight termination, lost link
- ADVERSE_WEATHER: Wind, precipitation, visibility, icing conditions
- REGULATORY_COMPLIANCE: Airspace, authorization, certification, NOTAM
- TECHNICAL_DEGRADATION: Battery, motor, GPS, communication issues
- SURVEY_MISSION: Aerial survey, mapping, inspection
- DELIVERY_MISSION: Cargo delivery, transport
- INSPECTION_MISSION: Infrastructure inspection, monitoring
- GENERAL_OPERATION: Standard flight operations

Query: {query}

Return ONLY JSON: {{"category": "CATEGORY_NAME", "confidence": 0.95}}"""

HYDE_PROMPT = """Generate a regulatory document excerpt relevant to this UAV scenario:

Scenario Type: {scenario_type}
Query: {query}

Key Parameters:
{feature_context}

Generate a 200-word excerpt that:
1. References specific operational limits
2. Cites regulatory sources if known (FAA, EASA, ICAO)
3. Uses formal aviation language
4. Is directly relevant to the parameters above

Excerpt:"""

# ═══════════════════════════════════════════════════════════
# Builder Functions
# ═══════════════════════════════════════════════════════════

def build_retrieval_prompt(query: str, 
                          features: Dict[str, Any],
                          scenario_type: str,
                          top_results: List[Dict]) -> str:
    """
    Build contextual prompt for retrieval analysis.
    Does NOT hardcode FAA/EASA - extracts from retrieved context only.
    """

    # Build feature summary
    feature_lines = []
    for feat, val in features.items():
        feature_lines.append(f"- {feat}: {val}")

    feature_summary = "\n".join(feature_lines[:20])  # Limit to top 20

    # Build context from results
    context_blocks = []
    for i, result in enumerate(top_results[:5], 1):
        text = result.get("text", "")[:500]  # Truncate long texts
        source = result.get("source", "Unknown")
        score = result.get("final_score", 0)
        context_blocks.append(f"""[Document {i}] Score: {score:.3f} | Source: {source}
{text}
""")

    context = "\n\n".join(context_blocks)

    prompt = f"""You are an aviation regulatory analysis system. Analyze the following UAV operational scenario against the retrieved regulatory documents.

## SCENARIO TYPE: {scenario_type}

## OPERATIONAL PARAMETERS:
{feature_summary}

## RETRIEVED REGULATORY CONTEXT:
{context}

## ANALYSIS INSTRUCTIONS:
1. Identify which regulations from the retrieved context apply to this scenario
2. Determine compliance status for each parameter
3. Highlight any violations or risks
4. Provide actionable recommendations
5. Cite specific sources from the retrieved documents

## OUTPUT FORMAT:
Return a JSON object with this structure:
{{
    "applicable_regulations": [
        {{
            "regulation_id": "string",
            "source": "string",
            "relevance_score": float,
            "applicability": "string"
        }}
    ],
    "compliance_analysis": [
        {{
            "parameter": "string",
            "value": "string",
            "status": "compliant|non_compliant|warning",
            "regulation_reference": "string",
            "notes": "string"
        }}
    ],
    "risk_assessment": {{
        "overall_risk": "low|medium|high|critical",
        "primary_risks": ["string"],
        "mitigation_required": bool
    }},
    "recommendations": [
        {{
            "priority": 1-10,
            "action": "string",
            "justification": "string"
        }}
    ],
    "confidence": float
}}

Base your analysis ONLY on the retrieved context above. Do not invent regulations."""

    return prompt


def build_hyde_prompt(query: str,
                     features: Dict[str, Any],
                     scenario_type: str) -> str:
    """Build prompt for targeted HyDE generation"""

    feature_context = "\n".join([
        f"- {k}: {v}" 
        for k, v in list(features.items())[:15]
    ])

    return f"""Generate a regulatory document excerpt relevant to this UAV scenario:

Scenario Type: {scenario_type}
Query: {query}

Key Parameters:
{feature_context}

Generate a 200-word excerpt that:
1. References specific operational limits
2. Cites regulatory sources if known
3. Uses formal aviation language
4. Is directly relevant to the parameters above"""


def build_synthesis_prompt(retrieval_results: List[Dict],
                          scenario_analysis: Dict,
                          ml_risk_score: Optional[float] = None) -> str:
    """
    Build final synthesis prompt for Agent consumption.
    """

    # Summarize retrieval results
    result_summary = []
    for r in retrieval_results[:5]:
        result_summary.append(
            f"- [{r.get('source', 'Unknown')}] Score: {r.get('final_score', 0):.3f} | "
            f"Text: {r.get('text', '')[:300]}..."
        )

    risk_context = f"ML Risk Score: {ml_risk_score:.2f}\n" if ml_risk_score else ""

    prompt = f"""## RETRIEVAL RESULTS SUMMARY

{risk_context}
Scenario Type: {scenario_analysis.get('scenario_type', 'unknown')}
Complexity: {scenario_analysis.get('complexity', 0):.2f}
Priority Features: {', '.join(scenario_analysis.get('priority_features', []))}
Risk Indicators: {', '.join(scenario_analysis.get('risk_indicators', []))}

## TOP RETRIEVED DOCUMENTS
{chr(10).join(result_summary)}

## SYNTHESIS INSTRUCTIONS
Synthesize the above into a comprehensive legal/regulatory assessment:

1. **Regulatory Framework**: Which regulations apply and how
2. **Compliance Status**: Specific compliance determination
3. **Risk Matrix**: Visual risk assessment
4. **Required Actions**: Step-by-step compliance actions
5. **Legal Basis**: Specific legal citations from retrieved docs

## OUTPUT FORMAT (JSON)
{{
    "legal_assessment": {{
        "applicable_frameworks": ["string"],
        "jurisdiction": "string",
        "operator_responsibilities": ["string"]
    }},
    "compliance_determination": {{
        "status": "compliant|conditional|non_compliant",
        "violations": ["string"],
        "waivers_required": ["string"]
    }},
    "risk_matrix": {{
        "severity": 1-5,
        "probability": 1-5,
        "risk_level": "low|medium|high|extreme",
        "mitigation_priority": 1-10
    }},
    "action_items": [
        {{
            "step": 1,
            "action": "string",
            "deadline": "string",
            "responsible_party": "string"
        }}
    ],
    "citations": [
        {{
            "source": "string",
            "relevant_text": "string",
            "confidence": float
        }}
    ],
    "overall_confidence": float
}}

Ensure all citations reference the retrieved documents above."""

    return prompt


def build_scenario_planning_prompt(features: Dict[str, Any],
                                  free_text: Optional[str] = None) -> str:
    """Build prompt for scenario analysis and query planning"""

    feature_summary = "\n".join([
        f"- {k}: {v}" 
        for k, v in list(features.items())[:25]
    ])

    free_text_section = f"\nAdditional Context: {free_text}" if free_text else ""

    return f"""Analyze this UAV operational scenario and determine retrieval strategy:

Operational Parameters:
{feature_summary}
{free_text_section}

Determine:
1. Scenario classification (emergency, weather, regulatory, technical, etc.)
2. Query complexity (0-1)
3. Priority features requiring immediate lookup
4. Recommended retrieval approach
5. Whether HyDE would improve results

Output JSON:
{{
    "scenario_type": "string",
    "complexity": float,
    "priority_features": ["string"],
    "retrieval_strategy": "direct|hyde|hybrid",
    "recommended_top_k": int,
    "risk_indicators": ["string"]
}}"""