"""
ACE Report Verifier (V5.0.0 - Grounding Guard)
"""

import re
from typing import List, Tuple, Any, NamedTuple

class VerificationError(NamedTuple):
    error_type: str
    field: str
    message: str

class ReportVerifier:
    @staticmethod
    def verify_grounding(md_content: str, evidence: Any) -> Tuple[bool, List[VerificationError]]:
        errors = []
        md_upper = md_content.upper()

        # 1. فحص الهيكل (الأقسام السبعة الجديدة)
        required_sections = [
            "EXECUTIVE SUMMARY", "PHYSICAL", "REGULATORY COMPLIANCE", 
            "TEMPORAL", "ML CONSULTANT", "DATA AUDIT"
        ]
        for section in required_sections:
            if section not in md_upper:
                errors.append(VerificationError("MISSING_SECTION", section, f"Section {section} is missing."))

        # 2. فحص استشهادات RAG (يجب أن تحتوي على [ | ])
        if evidence.get("legal_rag_citations") and len(evidence["legal_rag_citations"]) > 0:
            if not re.search(r"\[.*\|.*\]", md_content):
                errors.append(VerificationError("CITATION_MISSING", "legal", "RAG evidence found but not cited in text."))

        # 3. فحص تطابق القرار
        expected_decision = str(evidence["decision"]).upper()
        if expected_decision not in md_upper:
             errors.append(VerificationError("DECISION_MISMATCH", "final", "LLM changed the official verdict."))

        return len(errors) == 0, errors