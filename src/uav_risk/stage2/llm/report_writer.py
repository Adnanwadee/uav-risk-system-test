"""
Aviation Safety Report Writer (V7.2 - Certified Final)
======================================================
Defense:      Deep sanitization (Unicode + Null Bytes + HTML Entities).
Validation:   Schema validation للـ ConsensusReport قبل كتابة أي تقرير.
Isolation:    Prompt templates منفصلة عن منطق التقرير.
Resilience:   Async timeout صريح على LLM call.
Traceability: SHA-256 fingerprint + UTC Timezone Safe timestamps.
FIX V7.2:     Nested Backtick Injection Shield (يحوّل ` → ' لمنع كسر Markdown).
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SUMMARY_TIMEOUT_SECONDS = 15.0
_FLIGHT_ID_PATTERN = re.compile(r"^[A-Za-z0-9-_]{1,64}$")
_MAX_WARNING_LENGTH = 500

# ---------------------------------------------------------------------------
# SafetyReportWriter
# ---------------------------------------------------------------------------

class SafetyReportWriter:
    def __init__(
        self,
        llm_client: "GroqAsyncClient",
        summary_timeout: float = _SUMMARY_TIMEOUT_SECONDS,
    ):
        self.llm = llm_client
        self.summary_timeout = summary_timeout

    # ------------------------------------------------------------------
    # Input Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_flight_id(flight_id: str) -> str:
        if not flight_id or not isinstance(flight_id, str):
            raise ValueError("flight_id must be a non-empty string.")
        if not _FLIGHT_ID_PATTERN.match(flight_id):
            raise ValueError(
                f"Invalid flight_id format: {flight_id!r}. "
                "Only alphanumeric, hyphens, and underscores allowed (max 64 chars)."
            )
        return flight_id

    @staticmethod
    def _validate_report(report: "ConsensusReport") -> None:
        if not hasattr(report, "final_decision"):
            raise AttributeError("ConsensusReport missing 'final_decision'.")
        if not (0.0 <= report.calibrated_confidence_score <= 1.0):
            raise ValueError(f"calibrated_confidence_score out of range [0,1]: {report.calibrated_confidence_score}")
        if not (0.0 <= report.metrics.normalized_entropy <= 1.0):
            raise ValueError(f"normalized_entropy out of range [0,1]: {report.metrics.normalized_entropy}")
        if not (0.0 <= report.physics_nrs <= 1.0):
            raise ValueError(f"physics_nrs out of range [0,1]: {report.physics_nrs}")

    # ------------------------------------------------------------------
    # Context-Aware Sanitization
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_for_prompt(text: str) -> str:
        """تعقيم للحقن في الـ Prompt (LLM Context) باستخدام HTML Entities."""
        if not isinstance(text, str): text = str(text)
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)
        bidi_chars = "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\u200f\u200e"
        for c in bidi_chars: text = text.replace(c, " ")
        
        return (
            text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
        )

    @staticmethod
    def _sanitize_for_display(text: str) -> str:
        """
        تعقيم للعرض البشري في Markdown.
        نستخدم Backticks لعزل النص، لذا نزيل الـ Backticks الداخلية لمنع كسر التنسيق.
        """
        if not isinstance(text, str): text = str(text)
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)
        bidi_chars = "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\u200f\u200e"
        for c in bidi_chars: text = text.replace(c, " ")
        
        # [FIX V7.2] تحويل الـ Backticks الداخلية إلى علامات اقتباس مفردة
        # لمنع كسر هيكل Markdown عند التغليف بـ `...`
        text = text.replace("`", "'")
        
        if len(text) > _MAX_WARNING_LENGTH:
            return text[:_MAX_WARNING_LENGTH] + "… [TRUNCATED]"
        return text

    # ------------------------------------------------------------------
    # Prompt Engineering
    # ------------------------------------------------------------------

    def _build_summary_prompt(self, report: "ConsensusReport") -> str:
        safe_physics = [self._sanitize_for_prompt(w) for w in report.physics_warnings]
        safe_legal = [self._sanitize_for_prompt(v) for v in report.legal_violations]
        safe_decision = self._sanitize_for_prompt(str(report.final_decision.value))

        return textwrap.dedent(f"""\
        SYSTEM: You are a UAV Safety Auditor. Produce a concise, technical executive summary.
        
        CONSTRAINTS:
        - Use professional, dry, technical aviation language.
        - Maximum 3 sentences.
        - Do not add recommendations not supported by the data.
        
        ASSESSMENT DATA:
        - Decision: {safe_decision}
        - Safety Score: {report.calibrated_confidence_score:.3f} (0=Safe, 1=Danger)
        - Physics Warnings: {safe_physics}
        - Legal Constraints: {safe_legal}
        
        SUMMARY: """)

    # ------------------------------------------------------------------
    # LLM Summary Generation
    # ------------------------------------------------------------------

    async def _generate_executive_summary(self, report: "ConsensusReport") -> str:
        prompt = self._build_summary_prompt(report)
        try:
            return await asyncio.wait_for(
                self.llm.generate(prompt),
                timeout=self.summary_timeout,
            )
        except asyncio.TimeoutError:
            return (
                f"[SUMMARY TIMEOUT after {self.summary_timeout}s] "
                f"Decision: {self._sanitize_for_display(str(report.final_decision.value))}. "
                f"Manual review required."
            )
        except Exception as e:
            return (
                f"[SUMMARY GEN FAILURE: {type(e).__name__}] "
                f"Decision: {self._sanitize_for_display(str(report.final_decision.value))}."
            )

    # ------------------------------------------------------------------
    # Report Fingerprinting
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_fingerprint(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Markdown Report
    # ------------------------------------------------------------------

    async def generate_markdown_report(
        self,
        report: "ConsensusReport",
        flight_id: str,
    ) -> str:
        flight_id = self._validate_flight_id(flight_id)
        self._validate_report(report)

        exec_summary = await self._generate_executive_summary(report)
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        ccs     = f"{report.calibrated_confidence_score:.3f}"
        entropy = f"{report.metrics.normalized_entropy:.3f}"
        p_fail  = f"{report.physics_nrs:.3f}"

        md = textwrap.dedent(f"""\
        # ✈️ UAV Flight Safety Case: {flight_id}
        > **Generated:** {timestamp} | **System Status:** `[{self._sanitize_for_display(str(report.final_decision.value))}]`

        ## ⚖️ Executive Summary
        {exec_summary}

        ---
        ## 📊 Council Risk Metrics (Consolidated)
        | Safety Parameter | Value | Interpretation |
        |:---|:---|:---|
        | **Calibrated Confidence (CCS)** | `{ccs}` | (0=Safe, 1=Danger) |
        | **Semantic Disagreement (H)**   | `{entropy}` | Council Entropy |
        | **Monte Carlo Physics Risk**    | `{p_fail}` | Failure Probability |
        | **Regulatory Compliance**       | `{self._sanitize_for_display(str(report.legal_decision))}` | Legal Gate |

        ---
        ## ❌ Disqualifying Hard Stops
        """)

        if report.disqualifying_conditions:
            for stop in report.disqualifying_conditions:
                md += f"- **`{self._sanitize_for_display(stop)}`**\n"
        else:
            md += "- ✅ No hard-stop limits exceeded.\n"

        md += "\n## 🛡️ Warnings & Mitigation Requirements\n"
        
        for warn in report.all_warnings:
            md += f"- ⚠️ `{self._sanitize_for_display(warn)}`\n"
        for mit in report.required_mitigations:
            md += f"- 🛠️ **MITIGATION**: `{self._sanitize_for_display(mit)}`\n"

        md += f"\n---\n### 📜 Audit Log (Deliberation Trace)\n```text\n"
        for step in report.deliberation_steps:
            md += f"{self._sanitize_for_display(step)}\n"

        body_hash = self._compute_fingerprint(md)
        md += (
            f"```\n"
            f"\n---\n"
            f"**Report Integrity:** `SHA256:{body_hash}` | "
            f"*Automated safety report. Final responsibility rests with the PIC.*"
        )

        return md

    # ------------------------------------------------------------------
    # JSON Export
    # ------------------------------------------------------------------

    async def generate_json_report(
        self,
        report: "ConsensusReport",
        flight_id: str,
    ) -> Dict[str, Any]:
        flight_id = self._validate_flight_id(flight_id)
        self._validate_report(report)

        exec_summary = await self._generate_executive_summary(report)
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z"

        payload = {
            "schema_version": "2.0",
            "flight_id": flight_id,
            "generated_at": timestamp,
            "decision": str(report.final_decision.value),
            "metrics": {
                "calibrated_confidence_score": round(report.calibrated_confidence_score, 6),
                "normalized_entropy": round(report.metrics.normalized_entropy, 6),
                "physics_failure_probability": round(report.physics_nrs, 6),
                "legal_decision": str(report.legal_decision),
            },
            "executive_summary": exec_summary,
            "disqualifying_conditions": [self._sanitize_for_display(c) for c in report.disqualifying_conditions],
            "warnings": [self._sanitize_for_display(w) for w in report.all_warnings],
            "required_mitigations": [self._sanitize_for_display(m) for m in report.required_mitigations],
            "deliberation_steps": [self._sanitize_for_display(s) for s in report.deliberation_steps],
        }

        payload["report_fingerprint"] = self._compute_fingerprint(json.dumps(payload, sort_keys=True))
        return payload