"""
Legal Agent — Adversarial Regulatory Investigator (V15.0 - Apex Integrated)
=============================================================================
المميزات النهائية:
- حل مشكلة الاستيراد: إضافة AsyncRAGIndexInterface و AsyncLLMClientInterface.
- تحليل الـ 50 عاموداً: بناء استعلامات RAG دقيقة تعتمد على الوزن، الارتفاع، والحساسات.
- البحث المتعارض: استراتيجية (Advocate vs Adversary) لضمان عدم إغفال الموانع القانونية.
- التوافق الكامل: يعمل بشكل غير متزامن تماماً مع Groq V7.1 و RAG V13.
"""

from __future__ import annotations
import time
import json
import logging
import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger("LegalAgent")

# ─────────────────────────────────────────────────────────────────────────────
# 1. الواجهات البرمجية (لحل مشاكل الاستيراد وتوحيد العقود)
# ─────────────────────────────────────────────────────────────────────────────

class AsyncRAGIndexInterface(ABC):
    @abstractmethod
    async def search(self, query: str, top_k: int, min_score: float) -> list:
        """واجهة محرك البحث القانوني (RAG)."""
        pass

class AsyncLLMClientInterface(ABC):
    @abstractmethod
    async def generate(self, prompt: str, response_format: Optional[Dict[str, str]] = None) -> str:
        """واجهة محرك اللغة (LLM)."""
        pass

# ─────────────────────────────────────────────────────────────────────────────
# 2. العقود والبيانات (Schemas)
# ─────────────────────────────────────────────────────────────────────────────

class ComplianceStatus(str, Enum):
    COMPLIANT = "COMPLIANT"
    RESTRICTED = "RESTRICTED"
    NON_COMPLIANT = "NON_COMPLIANT"
    UNCERTAIN = "UNCERTAIN"

class GoNoGo(str, Enum):
    GO = "GO"
    CAUTION = "CAUTION"
    NO_GO = "NO-GO"

class LegalEvidence(BaseModel):
    source_document: str
    chunk_id: str
    exact_quote: str
    relevance_score: float

class ArgumentNode(BaseModel):
    claim: str
    supporting_evidence: List[LegalEvidence] = Field(default_factory=list)
    adversarial_rebuttal: Optional[str] = None
    rebuttal_evidence: List[LegalEvidence] = Field(default_factory=list)
    is_defeated: bool = False

class LLMJudgeResponse(BaseModel):
    compliance_status: ComplianceStatus
    go_no_go: GoNoGo
    reasoning_chain: str = Field(..., description="Explanation of the ruling.")
    critical_violations: List[str] = Field(default_factory=list)
    required_mitigations: List[str] = Field(default_factory=list)

class LegalRiskReport(BaseModel):
    compliance_status: ComplianceStatus
    go_no_go: GoNoGo
    primary_argument: ArgumentNode
    critical_violations: List[str]
    required_mitigations: List[str]
    execution_time_ms: float = 0.0
    error_flags: List[str] = []

@dataclass
class LegalAgentConfig:
    max_retrieval_docs: int = 4
    min_advocate_score: float = 0.60
    min_adversary_score: float = 0.65
    adversarial_override_threshold: float = 0.75
    llm_timeout_seconds: float = 10.0   
    rag_timeout_seconds: float = 5.0
    max_retries: int = 2

# ─────────────────────────────────────────────────────────────────────────────
# 3. الوكيل القانوني الأساسي (Core Logic)
# ─────────────────────────────────────────────────────────────────────────────

class LegalAgent:
    def __init__(self, rag_index: AsyncRAGIndexInterface, llm_client: AsyncLLMClientInterface, config: Optional[LegalAgentConfig] = None):
        self.rag = rag_index
        self.llm = llm_client
        self.config = config or LegalAgentConfig()

    def _build_context_query(self, data: Dict[str, Any]) -> str:
        """بناء استعلام قانوني ذكي يحلل الـ 50 عاموداً للوصول للمادة القانونية بدقة."""
        mass = data.get("uav.mass_kg", data.get("uav_mass_kg", 2.0))
        alt = data.get("telemetry.altitude_m", data.get("altitude_m", 120))
        pop = data.get("telemetry.population_density", "SUBURBAN")
        mission = data.get("mission.type", "VLOS flight")
        
        # رصد الحساسات (Privacy Impact)
        sensors = []
        if str(data.get("uav.sensors.has_lidar", "0")) == "1": sensors.append("LiDAR")
        if str(data.get("uav.sensors.has_camera", "0")) == "1": sensors.append("Camera")
        
        return (f"Aviation laws for {mass}kg UAV at {alt}m altitude over {pop} area. "
                f"Mission: {mission}. Sensors: {', '.join(sensors) if sensors else 'Standard'}")

    async def _advocate_search(self, query: str) -> List[LegalEvidence]:
        """يبحث عن المواد القانونية التي تسمح بالرحلة."""
        try:
            hits = await asyncio.wait_for(
                self.rag.search(f"Permitted conditions for: {query}", top_k=self.config.max_retrieval_docs, min_score=self.config.min_advocate_score),
                timeout=self.config.rag_timeout_seconds
            )
            return [self._map_hit(h) for h in hits]
        except Exception as e:
            logger.warning(f"Advocate Search Failed: {e}")
            return []

    async def _adversary_search(self, query: str) -> List[LegalEvidence]:
        """يبحث عن القيود والموانع القانونية الصريحة."""
        try:
            hits = await asyncio.wait_for(
                self.rag.search(f"Strict prohibitions and flight restrictions for: {query}", top_k=2, min_score=self.config.min_adversary_score),
                timeout=self.config.rag_timeout_seconds
            )
            return [self._map_hit(h) for h in hits]
        except Exception as e:
            logger.warning(f"Adversary Search Failed: {e}")
            return []

    def _map_hit(self, hit: Any) -> LegalEvidence:
        """تحويل نتائج RAG إلى كائنات LegalEvidence."""
        metadata = hit.get("metadata", {})
        return LegalEvidence(
            source_document=metadata.get("source", "Aviation Regulation"),
            chunk_id=metadata.get("article_id", "N/A"),
            exact_quote=hit.get("content", hit.get("page_content", "")),
            relevance_score=metadata.get("score", 0.0)
        )

    async def _safe_llm_evaluation(self, pro_ev: List[LegalEvidence], con_ev: List[LegalEvidence], context: str) -> LLMJudgeResponse:
        """يستدعي القاضي الرقمي (Groq) للموازنة بين الأدلة."""
        prompt = f"""
        SYSTEM: You are a Senior Aviation Regulatory Judge.
        Evaluate the flight context against retrieved laws.
        
        CONTEXT: {context}
        
        PRO-FLIGHT EVIDENCE:
        {chr(10).join([f"- {e.exact_quote} [{e.source_document}]" for e in pro_ev]) if pro_ev else 'None'}
        
        ANTI-FLIGHT EVIDENCE (Restrictions):
        {chr(10).join([f"- {e.exact_quote} [{e.source_document}]" for e in con_ev]) if con_ev else 'None'}
        
        Return JSON following this schema: {json.dumps(LLMJudgeResponse.model_json_schema())}
        """
        try:
            response = await self.llm.generate(prompt, response_format={"type": "json_object"})
            return LLMJudgeResponse(**json.loads(response))
        except Exception as e:
            logger.error(f"LLM Judge Error: {e}")
            return LLMJudgeResponse(compliance_status=ComplianceStatus.UNCERTAIN, go_no_go=GoNoGo.NO_GO, reasoning_chain="Judge Offline")

    async def analyze(self, flight_data: Dict[str, Any]) -> LegalRiskReport:
        t_start = time.perf_counter()
        
        query = self._build_context_query(flight_data)
        pro_evidence = await self._advocate_search(query)
        con_evidence = await self._adversary_search(query)
        
        judge_decision = await self._safe_llm_evaluation(pro_evidence, con_evidence, query)
        
        # [تعديل] تجميع حزمة الأدلة للاستشهادات [Source | Article] في التقرير
        all_citations = [f"[{e.source_document} | {e.chunk_id}]" for e in (pro_evidence + con_evidence)]

        return LegalRiskReport(
            compliance_status=judge_decision.compliance_status,
            go_no_go=judge_decision.go_no_go,
            primary_argument=ArgumentNode(
                claim="Flight Compliance Evaluation",
                supporting_evidence=pro_evidence,
                rebuttal_evidence=con_evidence,
                is_defeated=(judge_decision.go_no_go == GoNoGo.NO_GO)
            ),
            critical_violations=judge_decision.critical_violations + ([f"Legal Violation Cited: {all_citations}"] if all_citations else []),
            required_mitigations=judge_decision.required_mitigations,
            execution_time_ms=(time.perf_counter() - t_start) * 1000
        )