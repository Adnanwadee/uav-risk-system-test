"""
Enhanced Legal Agent for UAV RAG System (V3.0 - Fixed)
======================================================
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from .prompts import (
    LEGAL_COMPARISON_PROMPT,
    FINAL_ANSWER_TEMPLATE,
    SYSTEM_PROMPT,
)
from .groq_llm import GroqLLM
from .config import RAGConfig

logger = logging.getLogger("EnhancedLegalAgent")


class RegulationSource(Enum):
    FAA = "FAA (US)"
    EASA = "EASA (EU)"
    UNKNOWN = "Unknown"


@dataclass
class LegalCitation:
    """الاقتباس القانوني"""
    text: str
    source: RegulationSource
    section_number: Optional[str] = None
    
    def __str__(self) -> str:
        if self.section_number:
            if self.source == RegulationSource.FAA:
                return f"§ {self.section_number}"
            elif self.source == RegulationSource.EASA:
                return f"Article {self.section_number}"
        return self.source.value


@dataclass
class LegalAnswer:
    """الإجابة القانونية"""
    answer: str
    citations: List[LegalCitation]
    confidence_score: float = 0.0
    debug_info: Dict[str, Any] = field(default_factory=dict)


class EnhancedLegalAgent:
    """الوكيل القانوني المحسن"""
    
    def __init__(self, llm: GroqLLM, config: Optional[RAGConfig] = None):
        self.llm = llm
        self.config = config or RAGConfig()
        self.debug_mode = self.config.DEBUG_MODE
        
        self.section_pattern = re.compile(r'§\s*([0-9\.]+)')
        self.easa_article_pattern = re.compile(r'Article\s*([0-9]+)', re.IGNORECASE)
        self.easa_point_pattern = re.compile(r'UAS\.([A-Z0-9\.]+)', re.IGNORECASE)
    
    def _extract_section(self, text: str) -> Optional[str]:
        """استخراج رقم المقطع"""
        match = self.section_pattern.search(text)
        if match:
            return match.group(1)
        match = self.easa_article_pattern.search(text)
        if match:
            return f"Article {match.group(1)}"
        match = self.easa_point_pattern.search(text)
        if match:
            return match.group(1)
        return None
    
    def extract_citations(self, text: str, source_name: str) -> List[LegalCitation]:
        """استخراج الاقتباسات"""
        citations = []
        
        if "FAA" in source_name or "107" in text:
            source = RegulationSource.FAA
        elif "EASA" in source_name or "Regulation (EU)" in text:
            source = RegulationSource.EASA
        else:
            source = RegulationSource.UNKNOWN
        
        section = self._extract_section(text)
        citations.append(LegalCitation(
            text=text[:200],
            source=source,
            section_number=section
        ))
        
        return citations
    
    async def compare_regulations(self, faa_context: str, easa_context: str, topic: str) -> str:
        """مقارنة بين FAA و EASA"""
        prompt = LEGAL_COMPARISON_PROMPT.format(
            topic=topic,
            faa_context=faa_context[:3000],
            easa_context=easa_context[:3000]
        )
        return await self.llm.generate(prompt, include_system=True)
    
    async def analyze_direct(self, query: str, context: str) -> str:
        """تحليل مباشر"""
        prompt = f"""Answer based ONLY on this legal context:

CONTEXT:
{context[:4000]}

QUESTION: {query}

Cite specific sections. If not found, say so."""
        return await self.llm.generate(prompt, include_system=True)
    
    def _deduplicate_citations(self, citations: List[LegalCitation]) -> List[LegalCitation]:
        """إزالة الاقتباسات المكررة"""
        seen = set()
        unique = []
        for cit in citations:
            key = f"{cit.source.value}_{cit.section_number}"
            if key not in seen:
                seen.add(key)
                unique.append(cit)
        return unique
    
    async def build_final_answer(self, query: str, rag_results: List[Dict]) -> LegalAnswer:
        """بناء الإجابة النهائية مع دمج المزيد من المصادر"""
        
        if not rag_results:
            return LegalAnswer(
                answer="No relevant information found.",
                citations=[],
                confidence_score=0.0,
                debug_info={"results_count": 0}
            )
        
        if self.debug_mode:
            logger.info(f"Building answer from {len(rag_results)} chunks")
        
        context_parts = []
        all_citations = []
        faa_context = []
        easa_context = []
        
        for result in rag_results[:8]:
            content = result["page_content"]
            source = result["metadata"].get("source", "Unknown")
            score = result.get("relevance", result["metadata"].get("score", 0))
            
            context_parts.append(f"[Source: {source}] [Score: {score:.3f}]\n{content}\n")
            
            citations = self.extract_citations(content, source)
            all_citations.extend(citations)
            
            if "FAA" in source:
                faa_context.append(content)
            elif "EASA" in source:
                easa_context.append(content)
        
        context = "\n---\n".join(context_parts[:8])
        
        if len(faa_context) >= 2 or len(easa_context) >= 2:
            answer = await self.compare_regulations(
                "\n".join(faa_context[:4]),
                "\n".join(easa_context[:4]),
                query
            )
            strategy = "comparison_enhanced"
        else:
            answer = await self.analyze_direct(query, context)
            strategy = "direct"
        
        # حساب ثقة محسن
        weighted_scores = []
        for i, r in enumerate(rag_results[:8]):
            score = r.get("relevance", 0)
            weight = 1.0 if i == 0 else 0.8 if i == 1 else 0.6 if i == 2 else 0.4
            weighted_scores.append(score * weight)
        
        avg_score = sum(weighted_scores) / max(len(weighted_scores), 1)
        
        unique_citations = self._deduplicate_citations(all_citations)
        
        debug_info = {
            "strategy": strategy,
            "results_count": len(rag_results),
            "faa_chunks": len(faa_context),
            "easa_chunks": len(easa_context),
            "avg_score": avg_score,
            "total_citations": len(all_citations),
            "unique_citations": len(unique_citations)
        }
        
        if self.debug_mode:
            logger.info(f"Strategy: {strategy}, Confidence: {avg_score:.3f}, Citations: {len(unique_citations)}")
        
        return LegalAnswer(
            answer=answer,
            citations=unique_citations[:8],
            confidence_score=avg_score,
            debug_info=debug_info
        )
    
    def format_answer(self, legal_answer: LegalAnswer) -> str:
        """تنسيق الإجابة النهائية"""
        citations_text = "\n".join([f"- {str(cit)}" for cit in legal_answer.citations[:8]])
        
        return FINAL_ANSWER_TEMPLATE.format(
            answer=legal_answer.answer[:2000],
            citations=citations_text if citations_text else "No specific citations",
            compliance_note="Review cited regulations for complete requirements.",
            document_date="2024",
        )