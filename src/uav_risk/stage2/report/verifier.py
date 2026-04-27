"""
ACE Report Verifier (V4.1.1 - Deep Grounding Guard)
=================================================
الدور: التدقيق البعدي للتقرير المنتج لضمان صحة الأرقام والاستشهادات.
الإصلاحات: استخدام Decimal للمطابقة الدقيقة، ونظام VerificationError المهيكل.
"""

import re
from decimal import Decimal, getcontext
from typing import List, Tuple, Any, NamedTuple

getcontext().prec = 10

class VerificationError(NamedTuple):
    error_type: str
    field: str
    message: str

class ReportVerifier:
    # أرقام عامة يتم استثناؤها من فحص الهلوسة (مثل أرقام الإصدار أو العتبات الافتراضية)
    GENERAL_NUMBERS = {
        Decimal("0.00"), Decimal("1.00"), Decimal("5.00"), Decimal("4.10"), 
        Decimal("10.00"), Decimal("100.00"), Decimal("0.50")
    }
    TOLERANCE = Decimal("0.01") 

    @staticmethod
    def verify_grounding(md_content: str, evidence: Any) -> Tuple[bool, List[VerificationError]]:
        errors = []
        md_upper = md_content.upper()

        # 1. فحص الهيكل الإلزامي
        for section in ["EXECUTIVE SUMMARY", "PHYSICS", "LEGAL", "TEMPORAL"]:
            if section not in md_upper:
                errors.append(VerificationError("MISSING_SECTION", section, f"Section header {section} missing."))

        # 2. فحص الاستشهادات القانونية (Strict Protocol)
        if evidence.legal_citations:
            # يبحث عن صيغة [Source | ID]
            if not re.search(r"\[.*\|.*\]", md_content):
                errors.append(VerificationError("CITATION_MISSING", "legal_citations", "Articles found in data but missing or malformed in report text."))

        # 3. كاشف الهلوسة الرقمية (Decimal Math logic)
        found_floats = re.findall(r"(\d+\.\d+)", md_content)
        raw_text = str(evidence.raw_snapshot) + str(evidence.forensic_drivers)
        raw_floats = re.findall(r"(\d+\.\d+)", raw_text)
        
        raw_decimals = {Decimal(n).quantize(Decimal("0.01")) for n in raw_floats}

        for num in found_floats:
            try:
                val = Decimal(num).quantize(Decimal("0.01"))
                if val in ReportVerifier.GENERAL_NUMBERS:
                    continue
                
                # مطابقة مع هامش خطأ (لحالات التقريب البسيطة من الـ LLM)
                if not any(abs(val - raw) <= ReportVerifier.TOLERANCE for raw in raw_decimals):
                    errors.append(VerificationError("NUMERICAL_HALLUCINATION", "telemetry", f"Value '{num}' not grounded in raw telemetry."))
            except: continue

        # 4. مطابقة القرار النهائي
        expected_verdict = str(evidence.decision.value).upper()
        if expected_verdict not in md_upper:
            errors.append(VerificationError("DECISION_MISMATCH", "final_decision", f"Report failed to state the official '{expected_verdict}' verdict."))

        return len(errors) == 0, errors