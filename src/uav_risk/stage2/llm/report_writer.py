"""
Aviation Safety Report Writer (V14.0 - Full Audit & RAG Integration)
======================================================================
- 4-Agent Awareness: يحلل الآن الفيزياء، القانون، الزمن، والـ ML (10%).
- Full Data Audit: يعرض الـ 50+ عاموداً بالكامل لضمان الشفافية.
- RAG Evidence: يظهر النصوص القانونية والاستشهادات [Source | Article].
- Integrity Shield: يحافظ على نظام التشفير والتعقيم الأصلي (V7.2).
"""

from __future__ import annotations
import asyncio
import datetime
import hashlib
import json
import re
import textwrap
import unicodedata
from typing import TYPE_CHECKING, List, Dict, Any

if TYPE_CHECKING:
    from uav_risk.stage2.agents.consensus_agent import ConsensusReport
    from uav_risk.stage2.llm.groq_client import GroqAsyncClient

class SafetyReportWriter:
    def __init__(self, llm_client: "GroqAsyncClient"):
        self.llm = llm_client

    # --- 🛡️ حماية البيانات والتعقيم (من نسختك V7.2) ---
    @staticmethod
    def _sanitize_for_display(text: str) -> str:
        if not isinstance(text, str): text = str(text)
        text = unicodedata.normalize("NFKC", text).replace("`", "'")
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)

    @staticmethod
    def _compute_fingerprint(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    # --- 🧠 بناء المحفز الذكي (Prompt) ---
    def _build_summary_prompt(self, report: "ConsensusReport", telemetry: Dict[str, Any]) -> str:
        return textwrap.dedent(f"""\
            SYSTEM: You are a Senior UAV Safety Auditor. Produce a professional executive summary.
            
            [FLIGHT CONTEXT]
            - Aircraft Mass: {telemetry.get('uav.mass_kg', 'Dynamic')} kg
            - Decision: {report.final_decision.value}
            - ML Consultant Score: {report.ml_nrs:.3f} (10% influence)
            
            [LEGAL RAG EVIDENCE]
            - Articles Cited: {", ".join(report.legal_violations)}
            
            [PHYSICS FINDINGS]
            - Failure Prob: {report.physics_nrs:.2%}
            - Warnings: {report.physics_warnings}
            
            TASK: Explain the interaction between physical specs and legal RAG data in 3 sentences.
            """)

    # --- 📄 توليد التقرير الشامل (Markdown) ---
    async def generate_comprehensive_report(
        self,
        report: "ConsensusReport",
        flight_id: str,
        full_telemetry: Dict[str, Any] # الـ 50 عاموداً
    ) -> str:
        # 1. توليد الملخص التنفيذي عبر LLM
        prompt = self._build_summary_prompt(report, full_telemetry)
        exec_summary = await self.llm.generate(prompt)
        
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # 2. بناء هيكل التقرير (النسخة الاحترافية)
        md = textwrap.dedent(f"""\
            # ✈️ UAV Flight Safety Case: {flight_id}
            > **Generated:** {timestamp} | **Verdict:** `[{report.final_decision.value}]`

            ## ⚖️ Executive Summary
            {exec_summary}

            ---
            ## 📊 Multi-Agent Risk Matrix
            | Council Member | Decision | Risk Score | Key Insight |
            |:---|:---|:---|:---|
            | **Physics Guardian** | `{report.physics_decision}` | `{report.physics_nrs:.3f}` | MC Simulation |
            | **Legal Investigator** | `{report.legal_decision}` | `{report.legal_nrs:.3f}` | RAG Citations |
            | **Temporal Predictor** | `{report.temporal_decision}` | `{report.temporal_nrs:.3f}` | Trend Analysis |
            | **ML Consultant (10%)** | `{report.ml_decision}` | `{report.ml_nrs:.3f}` | Stage-1 Logic |

            ---
            ## 📜 Regulatory Compliance (RAG Evidence)
            """)
        
        if report.legal_violations:
            for violation in report.legal_violations:
                md += f"- ⚖️ **Cited Article:** `{self._sanitize_for_display(violation)}`\n"
        else:
            md += "- ✅ No regulatory breaches identified in RAG database.\n"

        md += "\n--- \n## 🔍 Full Operational Data Audit (50+ Columns)\n"
        md += "The following data points were used for this assessment:\n\n"
        
        # عرض الـ 50 عاموداً بشكل جدول أو قائمة منظمة
        for key, value in sorted(full_telemetry.items()):
            if value is not None:
                md += f"- **{key}**: `{value}`\n"

        md += f"\n---\n### 📜 Audit Log & Integrity\n"
        body_hash = self._compute_fingerprint(md)
        md += f"**Report Fingerprint:** `SHA256:{body_hash}` | *Automated Audit Ready*"

        return md

    # --- 💾 توليد التقرير بصيغة JSON (للسيرفر والـ UI) ---
    async def generate_json_report(
        self,
        report: "ConsensusReport",
        flight_id: str,
        full_telemetry: Dict[str, Any]
    ) -> Dict[str, Any]:
        md_report = await self.generate_comprehensive_report(report, flight_id, full_telemetry)
        
        return {
            "flight_id": flight_id,
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "decision": str(report.final_decision.value),
            "full_markdown": md_report,
            "data_snapshot": full_telemetry, # حفظ الـ 50 عاموداً للتدقيق مستقبلاً
            "metrics": {
                "confidence": round(report.calibrated_confidence_score, 4),
                "ml_impact": round(report.ml_nrs, 4)
            }
        }