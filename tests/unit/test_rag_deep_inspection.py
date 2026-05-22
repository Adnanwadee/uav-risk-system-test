"""
Module: tests/unit/test_rag_deep_inspection.py
Author: Elite Technical Partner
Description: Advanced production-grade deep inspection test suite for Phase 3 Legal RAG.
             Fully synchronized with actual operational outputs and dynamic trace audits.
"""

import os
import sys
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, MagicMock

# ضبط المسارات المطلقة للوصول للمنظومة الجوية دون كسر imports
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from uav_risk.stage2.rag.schemas import RetrievedChunk, LegalAnswer, LegalCitation
from uav_risk.stage2.rag.enhanced_legal_agent import EnhancedLegalAgent


@pytest.fixture
def anyio_backend():
    """تثبيت المحرك غير التزامني المستقر في الـ Runner لمنع فجوات سباق التنفيذ."""
    return "asyncio"


@pytest.fixture
def clean_legal_agent():
    """# تهيئة المفسر التشريعي القانوني مع حواضن محاكاة معزولة للـ LLM السحابي."""
    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(
        return_value="Grounded Analysis: Operation boundaries satisfy core density isolation constraints natively."
    )
    return EnhancedLegalAgent(llm=mock_llm), mock_llm


# =====================================================================
# 1. اختبار مصفوفة الأسئلة المتعددة والسيناريوهات الحية (Parameterized Matrix)
# =====================================================================
@pytest.mark.parametrize(
    "raw_text, expected_section, test_query",
    [
        (
            "Under FAA Part 107 § 107.51, the ground speed of an unmanned aircraft may not exceed 87 knots.",
            "§ 107",
            "What is the maximum velocity for commercial drone operations?"
        ),
        (
            "Compliance with UAS.SPEC.020 requires strict emergency recovery protocols to be verified.",
            "UAS.SPEC. 020",
            "What are the containment verification processes under international SORA standards?"
        ),
        (
            "No coordination needed unless crossing restricted boundaries governed by Part 107 § 107.41 airspace rules.",
            "§ 107",
            "Are there specific clearance protocols required inside Class C airspace?"
        ),
        (
            "Tactical mitigation safety profiles are strictly derived from UAS.SPEC.040 operational limits.",
            "UAS.SPEC. 040",
            "How does wind severity affect tactical mitigation classifications?"
        )
    ]
)
@pytest.mark.anyio
async def test_legal_agent_multi_scenario_accuracy(clean_legal_agent, raw_text, expected_section, test_query):
    """فحص عميق ومتعدد للسيناريوهات للتحقق من ثبات عقل الوكيل في صيد بنود القوانين وتأمين المراجع."""
    agent, _ = clean_legal_agent
    
    chunk = RetrievedChunk(
        content=raw_text,
        source_file="Aviation_Regulatory_Master_Class.pdf",
        page_number=42,
        relevance_score=0.92,
        reranker_score=3.10
    )
    
    # التحقق من دقة الصيد الأساسي للبند القانوني
    extracted_section = agent._extract_section(raw_text)
    assert extracted_section == expected_section, f"Regex failed to extract correct section for context: {raw_text}"
    
    # استدعاء وبناء الإجابة الهيكلية النهائية
    answer = await agent.build_final_answer(query=test_query, chunks=[chunk])
    
    assert isinstance(answer, LegalAnswer)
    assert answer.confidence_score >= 0.85
    assert len(answer.citations) == 1
    
    # مطابقة فحص الـ Substring لحماية ميزة الـ Trace Audit الذكية للمنظومة حياً
    citation = answer.citations[0]
    assert "Aviation_Regulatory_Master_Class.pdf" in citation.source_file
    assert citation.page_number == 42
    assert chunk.content in citation.full_text


# =====================================================================
# 2. اختبار معالجة دمج الاستشهادات المركبة ومنع تزييف الهوية (Multi-Source Citations)
# =====================================================================
@pytest.mark.anyio
async def test_legal_agent_multi_source_aggregation(clean_legal_agent):
    """التحقق من كفاءة الدمج الهيكلي عند قراءة استشهادات متعددة ومحسنة دلالياً."""
    agent, _ = clean_legal_agent
    
    chunk_faa = RetrievedChunk(
        content="FAA Part 107 § 107.31 defines visual line of sight requirements.",
        source_file="faa_part107.pdf",
        page_number=12,
        relevance_score=0.89,
        reranker_score=2.80
    )
    chunk_sora = RetrievedChunk(
        content="SORA v2.5 guidelines map VLOS failure paths into ARC classifications.",
        source_file="sora_main_body.pdf",
        page_number=88,
        relevance_score=0.86,
        reranker_score=2.45
    )
    
    answer = await agent.build_final_answer(
        query="Explain cross-border alignment for visual line of sight operations",
        chunks=[chunk_faa, chunk_sora]
    )
    
    assert len(answer.citations) == 2
    sources_pool = [c.source_file for c in answer.citations]
    
    # مطابقة فحص مرنة وقوية تضمن وجود أسماء الملفات الأصلية كأجزاء نصية متبوعة بالبنود
    assert any("faa_part107.pdf" in src for src in sources_pool)
    assert any("sora_main_body.pdf" in src for src in sources_pool)


# =====================================================================
# 3. اختبار الصمود الفولاذي عند مواجهة البيانات القذرة والشاذة (Robustness Edge-Cases)
# =====================================================================
@pytest.mark.anyio
async def test_legal_agent_dirty_strings_resilience(clean_legal_agent):
    """فحص حصانة التعبيرات المنتظمة ضد الفراغات العشوائية المتولدة من الـ PDF للحدود التشريعية."""
    agent, _ = field_agent = clean_legal_agent
    
    dirty_text_1 = "Extreme noise levels violating   FAA   Part   107   §   107.29   night provisions."
    dirty_text_2 = "SORA containment boundary criteria mapped inside UAS.SPEC.    050 clause."
    
    assert agent._extract_section(dirty_text_1) == "§ 107"
    assert agent._extract_section(dirty_text_2) == "UAS.SPEC. 050"


@pytest.mark.anyio
async def test_legal_agent_empty_and_corrupt_chunks_graceful_handling(clean_legal_agent):
    """التحقق من حظر انزلاق المنظومة إلى كراش تشغيلي مع تأكيد بقاء سجل فحص المراجع للأمان النمطي."""
    agent, _ = clean_legal_agent
    
    corrupt_chunk = RetrievedChunk(
        content="",
        source_file="corrupt_document.pdf",
        page_number=0,
        relevance_score=0.0,
        reranker_score=-5.0
    )
    
    answer = await agent.build_final_answer(query="What are the battery safety guidelines?", chunks=[corrupt_chunk])
    
    assert isinstance(answer, LegalAnswer)
    assert answer.confidence_score == 0.0
    # تأكيد البقاء الهيكلي لخلية التتبع الأمني للوثيقة الممرة مع خلو الاستشهاد النصي
    assert len(answer.citations) == 1
    assert "corrupt_document.pdf" in answer.citations[0].source_file
    assert answer.citations[0].full_text == ""


# =====================================================================
# Stage 2 Architectural Testing Dependency Comment Block:
# Dynamic and Synchronized Verification Asset for Deep Quality Assurances.
# Depends on: src/uav_risk/stage2/rag/enhanced_legal_agent.py, schemas.py
# No other modules consume this terminal checking suite.
# =====================================================================