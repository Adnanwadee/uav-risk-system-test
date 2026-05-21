"""
Module: src/uav_risk/stage2/rag/enhanced_legal_agent.py
Author: Elite Technical Partner
Description: Refactored legal agent that processes strict RetrievedChunk objects, 
             extracts accurate citations (FAA/SORA), and synthesizes validated compliance text.
"""

import re
from typing import List, Optional
import structlog

# الاستيراد المطلق للعقود لضمان التكاملية التامة ومنع تعارض الأنواع
from uav_risk.stage2.rag.schemas import RetrievedChunk, LegalCitation, LegalAnswer
from uav_risk.stage2.rag.groq_llm import GroqLLM
from uav_risk.stage2.rag.config import RAGConfig
from uav_risk.stage2.rag.prompts import LEGAL_COMPARISON_PROMPT, FINAL_ANSWER_TEMPLATE

logger = structlog.get_logger()


class EnhancedLegalAgent:
    """Aviation compliance intelligence core responsible for cross-referencing FAA and SORA documents."""

    def __init__(self, llm: GroqLLM, config: Optional[RAGConfig] = None):
        self.llm = llm
        self.config = config or RAGConfig()
        self.debug_mode = self.config.DEBUG_MODE
        
        # قوالب صيد نصوص المواد الفيدرالية وبنود SORA v2.5 الدولية بدقة
        self.faa_pattern = re.compile(r'(§+|Part)\s*([0-9\.]+)', re.IGNORECASE)
        self.sora_article_pattern = re.compile(r'(Article|Annex|UAS\.SPEC\.)\s*([A-Z0-9\._\-]+)', re.IGNORECASE)

    def _extract_section(self, text: str) -> Optional[str]:
        """Extracts formal specific clause numbers or article tags from standard regulatory phrasing."""
        faa_match = self.faa_pattern.search(text)
        if faa_match:
            return f"§ {faa_match.group(2)}"
            
        sora_match = self.sora_article_pattern.search(text)
        if sora_match:
            return f"{sora_match.group(1)} {sora_match.group(2)}"
            
        return None

    def extract_legal_citation(self, chunk: RetrievedChunk) -> LegalCitation:
        """Transforms a validated text chunk into a clean, unified LegalCitation structure."""
        extracted_clause = self._extract_section(chunk.content)
        source_label = chunk.source_file
        if extracted_clause:
            source_label = f"{chunk.source_file} ({extracted_clause})"
            
        return LegalCitation(
            source_file=source_label,
            page_number=chunk.page_number,
            full_text=chunk.content.strip()
        )

    async def compare_regulations(self, faa_context: str, easa_context: str, topic: str) -> str:
        """Executes a dual-track comparative analysis prompting for FAA and SORA variations."""
        prompt = LEGAL_COMPARISON_PROMPT.format(
            topic=topic,
            faa_context=faa_context[:3500],
            easa_context=easa_context[:3500]
        )
        return await self.llm.generate(prompt, include_system=True)

    async def analyze_direct(self, query: str, context: str) -> str:
        """Generates a straightforward compliance evaluation against unified background facts."""
        prompt = (
            f"Answer the query based ONLY on the following extracted legal context:\n\n"
            f"CONTEXT:\n{context[:4000]}\n\n"
            f"QUESTION: {query}\n\n"
            f"Cite specific subsections directly. If the context does not answer the question, state that clearly."
        )
        return await self.llm.generate(prompt, include_system=True)

    async def build_final_answer(self, query: str, chunks: List[RetrievedChunk]) -> LegalAnswer:
        """
        Orchestrates full response synthesis from strict structural RetrievedChunk formats.
        Drives offline dynamic calibration to score query precision thresholds.
        """
        if not chunks:
            logger.warning("agent_aborted_answer_synthesis_empty_chunks_pool")
            return LegalAnswer(
                query=query,
                answer="No relevant aviation regulatory constraints found in current index.",
                citations=[],
                confidence_score=0.0,
                rag_available=True
            )

        logger.info("agent_processing_retrieved_chunks", counts=len(chunks))
        
        context_blocks = []
        citations_pool = []
        faa_texts = []
        sora_texts = []

        # تفكيك ومعايرة الكتل المسترجعة حياً بناء على عقود الـ Object الجديدة
        for idx, chunk in enumerate(chunks[:8]):
            context_blocks.append(
                f"[Source File: {chunk.source_file}] [Page: {chunk.page_number}] [Weight: {chunk.relevance_score:.2f}]\n"
                f"{chunk.content}\n"
            )
            
            # استخراج الاقتباس وتجهيزه لخط التقرير الجنائي الموحد
            citation_obj = self.extract_legal_citation(chunk)
            citations_pool.append(citation_obj)
            
            # فرز النصوص دلالياً لتحديد استراتيجية الـ LLM المعرفية
            lowered_source = chunk.source_file.lower()
            if "faa" in lowered_source or "part 107" in lowered_source or "ac_" in lowered_source:
                faa_texts.append(chunk.content)
            elif "sora" in lowered_source or "easa" in lowered_source:
                sora_texts.append(chunk.content)

        aggregated_context = "\n---\n".join(context_blocks)
        
        # تحديد الإستراتيجية: مقارنة دولية ثنائية أم تحليل أحادي مباشر
        if len(faa_texts) >= 1 and len(sora_texts) >= 1:
            strategy = "international_comparative"
            synthesized_response = await self.compare_regulations(
                faa_context="\n".join(faa_texts[:3]),
                easa_context="\n".join(sora_texts[:3]),
                topic=query
            )
        else:
            strategy = "direct_compliance_lookup"
            synthesized_response = await self.analyze_direct(query, aggregated_context)

        # حساب معدل الثقة الموزون هندسياً لحماية التنبؤات من التشتت
        weighted_scores = []
        for i, chunk in enumerate(chunks[:8]):
            decay_weight = 1.0 if i == 0 else 0.8 if i == 1 else 0.5
            weighted_scores.append(chunk.relevance_score * decay_weight)
        
        final_confidence = sum(weighted_scores) / max(len(weighted_scores), 1)
        
        logger.info("agent_answer_synthesis_complete", strategy=strategy, confidence=f"{final_confidence:.3f}")
        
        return LegalAnswer(
            query=query,
            answer=synthesized_response.strip(),
            citations=citations_pool[:8],
            confidence_score=max(0.0, min(1.0, float(final_confidence))),
            rag_available=True
        )

    def format_answer(self, legal_answer: LegalAnswer) -> str:
        """Standardizes layout into presentation-ready templates for the automated Report Generator."""
        citations_block = "\n".join([f"- {cit.source_file}, Page {cit.page_number}" for cit in legal_answer.citations])
        return FINAL_ANSWER_TEMPLATE.format(
            answer=legal_answer.answer,
            citations=citations_block if citations_block else "No formal citations recorded.",
            compliance_note="Review official FAA Part 107 and JARUS SORA frameworks for complete legal definitions.",
            document_date="2026"
        )

# =====================================================================
# Stage 2 Architectural Dependency Comment Block:
# Dynamic text generation layer acting as the direct legal brain for RAG routing.
# Dependencies: src/uav_risk/stage2/rag/schemas.py -> RetrievedChunk, LegalAnswer, LegalCitation
# Dependent Files: Wired seamlessly into src/uav_risk/stage2/rag/rag_core.py
# =====================================================================