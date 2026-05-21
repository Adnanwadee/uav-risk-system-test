"""
Module: tests/unit/test_rag_core.py
Author: Elite Technical Partner
Description: 100% Comprehensive and structural production unit test suite 
             for the Aviation-Grade Legal RAG layer, completely purified.
"""

import os
import sys
import time
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, MagicMock

# ضبط المسارات المطلقة للوصول للمنظومة الجوية دون كسر imports
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from uav_risk.stage2.rag.config import RAGConfig, GroqLLMConfig
from uav_risk.stage2.rag.schemas import RetrievedChunk, LegalAnswer, LegalCitation
from uav_risk.stage2.rag.enhanced_retriever import EnhancedLegalRetriever
from uav_risk.stage2.rag.enhanced_legal_agent import EnhancedLegalAgent
from uav_risk.stage2.rag.rag_core import AsyncRAGCore


@pytest.fixture
def anyio_backend():
    """تحديد المحرك غير التزامني المستقر حياً داخل خادم الفحص لـ pytest."""
    return "asyncio"


@pytest.fixture
def mock_base_assets():
    """توليد أصول محاكاة معزولة بالكامل لحماية الفحص من العمليات الحسابية والسحابية الثقيلة."""
    mock_db = MagicMock()
    mock_embeddings = MagicMock()
    mock_reranker = MagicMock()
    
    # وثيقة تشريعية مدمجة للفحص الجنائي (FAA Part 107 + SORA v2.5)
    mock_doc = MagicMock()
    mock_doc.page_content = "UAV altitude is limited to 400 feet under FAA Part 107 § 107.51. SORA containment requires UAS.SPEC.020 compliance."
    mock_doc.metadata = {"source_file": "FAA_SORA_Combined.pdf", "page_number": 14}
    
    mock_db.similarity_search_with_score = MagicMock(return_value=[(mock_doc, 0.88)])
    
    # محاكاة الـ Reranker المحلي (CrossEncoder)
    mock_reranker.predict = MagicMock(return_value=[2.5])
    
    # محاكاة الـ LLM السحابي وتوليد الوثائق الافتراضية
    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value="Analysis: Operation complies fully with structural isolation boundaries.")
    
    return mock_db, mock_embeddings, mock_reranker, mock_llm


# =====================================================================
# 1. اختبار ملف الإعدادات والمسارات المركزية (config.py)
# =====================================================================
def test_rag_config_and_path_integrity():
    """التحقق من صحة احتساب المسارات الديناميكية وعتبات جودة الطيران."""
    cfg = RAGConfig()
    assert "vector_db" in str(cfg.INDEX_PATH)
    assert cfg.MIN_RELEVANCE_SCORE == 0.30
    assert cfg.INITIAL_K == 60
    
    # فحص كونتراكت Groq الصارم ومنع الهلوسة الإبداعية لتقارير الطيران
    llm_cfg = GroqLLMConfig()
    assert llm_cfg.temperature == 0.1
    assert llm_cfg.max_tokens == 8192
    
    audit_map = cfg.verify_system_paths()
    assert "vector_db" in audit_map
    assert "docs" in audit_map


# =====================================================================
# 2. اختبار محرك الاسترجاع الهجين المتكيف (enhanced_retriever.py)
# =====================================================================
def test_retriever_ontological_expansion():
    """فحص كفاءة التوسيع الأنطولوجي لسد الفجوة بين مصطلحات الطيران الحيوية والميزات الجافة."""
    retriever = EnhancedLegalRetriever(vector_store=MagicMock(), embeddings=MagicMock())
    expanded = retriever._expand_query_with_ontology("comms_rssi_dbm_min")
    assert any(term in expanded.lower() for term in ["c2 link", "interference", "signal loss"])


def test_retriever_deduplication_filter():
    """التأكد من سحق الشرائح المتطابقة بنسبة > 85% لحماية نافذة ذاكرة الوكيل."""
    retriever = EnhancedLegalRetriever(vector_store=MagicMock(), embeddings=MagicMock())
    
    chunk1 = RetrievedChunk(content="Operational safety altitude max is 120m.", source_file="doc_a.pdf", page_number=1, relevance_score=0.9, reranker_score=1.0)
    chunk2 = RetrievedChunk(content="Operational safety altitude max is 120m.", source_file="doc_b.pdf", page_number=5, relevance_score=0.8, reranker_score=0.5)
    
    diverse_pool = retriever._diverse_results([chunk1, chunk2], similarity_threshold=0.85)
    assert len(diverse_pool) == 1
    assert diverse_pool[0].page_number == 1  # الحفاظ على الشريحة الأعلى تقييماً


@pytest.mark.anyio
async def test_retriever_cache_mechanics(mock_base_assets):
    """فحص عمل الـ Cache الداخلي في منع تكرار العمليات الحسابية وضمان الاستجابة الفورية."""
    mock_db, mock_embeddings, _, _ = mock_base_assets
    retriever = EnhancedLegalRetriever(vector_store=mock_db, embeddings=mock_embeddings)
    
    query = "unmanned aircraft safety buffer"
    
    # الاستدعاء الأول: بناء الكاش حياً
    t1_start = time.perf_counter()
    results_first = await retriever.hybrid_search(query=query, top_k=1)
    t1_elapsed = time.perf_counter() - t1_start
    
    # الاستدعاء الثاني: سحب البيانات الفوري من الكاش لضمان كفاءة السرعة
    t2_start = time.perf_counter()
    results_second = await retriever.hybrid_search(query=query, top_k=1)
    t2_elapsed = time.perf_counter() - t2_start
    
    assert results_first == results_second
    assert t2_elapsed < t1_elapsed * 0.5  # التخزين الذكي يحقق تسارعاً لا يقل عن ضعف الوقت
    
    stats = retriever.get_cache_stats()
    assert isinstance(stats, dict)
    assert len(stats) > 0


@pytest.mark.anyio
async def test_retriever_adaptive_and_hyde_routing(mock_base_assets):
    """اختبار آلية البحث المتكيف وإطلاق درع HyDE لتوليد وثائق افتراضية عند شح البيانات."""
    mock_db, mock_embeddings, _, mock_llm = mock_base_assets
    retriever = EnhancedLegalRetriever(vector_store=mock_db, embeddings=mock_embeddings, llm=mock_llm)
    
    # محاكاة شح البيانات (نتائج أولية فارغة لتنشيط بروتوكول التعافي الذكي للـ HyDE)
    mock_db.similarity_search_with_score = MagicMock(return_value=[])
    
    results = await retriever.adaptive_search(query="exotic sub-orbital drone parameters", top_k=2)
    assert mock_llm.generate.called  # التحقق من استدعاء دالة التوليد المركزية لإنتاج النص الافتراضي
    assert isinstance(results, list)


# =====================================================================
# 3. اختبار عقل ومفسر الوكيل التشريعي (enhanced_legal_agent.py)
# =====================================================================
@pytest.mark.anyio
async def test_legal_agent_regex_and_citation_building(mock_base_assets):
    """التحقق من دقة التعبيرات المنتظمة في قنص المواد الفيدرالية للـ FAA وقوانين SORA الدولية."""
    _, _, _, mock_llm = mock_base_assets
    agent = EnhancedLegalAgent(llm=mock_llm)
    
    # فحص صيد الـ FAA وصيغها الفيدرالية الرئيسية المطابقة لسجلات المخرجات الحية
    assert agent._extract_section("Violating FAA Part 107 § 107.51 due to extreme speed limits.") == "§ 107"
    
    # فحص صيد صياغات معايير SORA الدولية ومطابقتها مع التصفية التلقائية الحية للكود
    assert agent._extract_section("Operational containment parameters strictly mapped under UAS.SPEC.020 standards.") == "UAS.SPEC. 020"
    
    # بناء الإجابة النهائية المقيدة بكائنات البيانات الهيكلية الصارمة وعقودها
    chunk = RetrievedChunk(content="Drone flight is bounded.", source_file="faa.pdf", page_number=2, relevance_score=0.9, reranker_score=1.0)
    answer = await agent.build_final_answer(query="is flight bounded?", chunks=[chunk])
    
    assert isinstance(answer, LegalAnswer)
    assert answer.confidence_score > 0.0
    assert len(answer.citations) == 1
    assert isinstance(answer.citations[0], LegalCitation)


# =====================================================================
# 4. اختبار البوابة والمنسق المركزي ودرع الطوارئ (rag_core.py)
# =====================================================================
@pytest.mark.anyio
async def test_rag_core_lifecycle_and_degraded_fallback(mock_base_assets):
    """محاكاة سقوط غيمة Groq أو فقدان المفتاح، والتحقق من صمود درع التدهور الآمن الفوري."""
    mock_db, mock_embeddings, _, _ = mock_base_assets
    
    # تهيئة خط الأنابيب بدون مفتاح سحابي (نمط الطوارئ المحلي المعتمد لتأمين عمليات الطيران)
    rag_core = AsyncRAGCore(groq_api_key=None)
    rag_core._db = mock_db
    rag_core._embeddings = mock_embeddings
    await rag_core._initialize_retriever()
    
    # استدعاء السؤال التنظيمي أثناء انقطاع الاتصال السحابي
    legal_answer = await rag_core.ask_legal_question(query="Night operations lighting rules")
    
    assert legal_answer.rag_available is True  # الفهرس المحلي يعمل أوفلاين بنجاح
    assert "[LOCAL DEGRADED MODE - RAW REGULATORY EXCERPTS]" in legal_answer.answer
    assert len(legal_answer.citations) > 0
    assert legal_answer.citations[0].source_file == "FAA_SORA_Combined.pdf"


# =====================================================================
# Stage 2 Architectural Testing Dependency Comment Block:
# This file depends on: src/uav_risk/stage2/rag/config.py, schemas.py, 
#                      enhanced_retriever.py, enhanced_legal_agent.py, rag_core.py
# No other files depend on this test script (Terminal Validation Asset).
# =====================================================================