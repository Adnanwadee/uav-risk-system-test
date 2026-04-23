# src/uav_risk/stage2/tools/toolbox.py
from __future__ import annotations
from typing import Dict, Any, List
import logging
from langchain_core.tools import tool

from uav_risk.stage1.infer import run_stage1_inference
from uav_risk.stage2.schemas import MLResult

# 1. استدعاء محرك الـ RAG الذكي الخاص بصديقك
from uav_risk.stage2.rag.rag_core import RAGCore 

logger = logging.getLogger(__name__)

# 2. تهيئة المحرك مرة واحدة فقط في الذاكرة (لتجنب بطء تحميل الموديلات مع كل استعلام)
try:
    rag_engine = RAGCore()
except Exception as e:
    logger.error(f"Failed to load FAISS RAG Engine: {e}")
    rag_engine = None

# ==========================================
# 1. ML Oracle Tool (Type-Safe & Isolated)
# ==========================================
@tool("get_ml_risk_prediction")
def get_ml_risk_prediction(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls the XGBoost Statistical ML Model to predict flight risk.
    """
    try:
        ml_result: MLResult = run_stage1_inference(scenario)
        return {
            "predicted_class": ml_result.predicted_class,
            "risk_score": ml_result.risk_score,
            "confidence": ml_result.confidence
        }
    except Exception as e:
        logger.error(f"[TOOL_ML_ERROR] Inference failed: {e}")
        # تم تصحيح الثغرة الأمنية هنا: الرفض المباشر عند الفشل
        return {
            "predicted_class": "HIGH_RISK",
            "risk_score": 1.0,
            "confidence": 0.0,
            "error": str(e)
        }

# ==========================================
# 2. Legal Oracle Tool (The Real Hybrid RAG)
# ==========================================
@tool("search_aviation_regulations")
def search_aviation_regulations(query: str, context_tags: List[str] | None = None) -> List[Dict[str, Any]]:
    """
    Searches the FAISS Vector database for aviation regulations using Semantic Search & Re-ranking.
    """
    if not rag_engine:
        return [{"article_id": "SYS_ERROR", "content": "RAG Engine offline. Abort.", "relevance_score": 0.0}]

    try:
        logger.info(f"[RAG SEARCH] Query: {query}")
        
        # 3. استدعاء المحرك الذكي (البحث + إعادة الترتيب)
        docs = rag_engine.retrieve_optimized_context(query)
        
        results = []
        for doc in docs:
            # 4. استخراج الميتاداتا بدقة كما صممها صديقك
            source_file = doc.metadata.get('source', 'Unknown').split('/')[-1]
            page_num = doc.metadata.get('page', 'N/A')
            score = doc.metadata.get('rerank_score', 0.0)
            
            # 5. تحويلها للصيغة التي يقبلها Agent LangGraph (RegulationChunk)
            # دمج اسم الملف والصفحة لتكوين معرف فريد (article_id)
            results.append({
                "article_id": f"[{source_file} - Pg:{page_num}]",
                "content": doc.page_content,
                "relevance_score": float(score)
            })
            
        return results
        
    except Exception as e:
        logger.error(f"[TOOL_RAG_ERROR] Regulation search failed: {e}")
        return []

# ==========================================
# Tool Registry
# ==========================================
AGENT_TOOLS = [get_ml_risk_prediction, search_aviation_regulations]