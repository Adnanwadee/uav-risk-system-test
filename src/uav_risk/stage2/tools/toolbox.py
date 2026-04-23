# src/uav_risk/stage2/tools/toolbox.py
from __future__ import annotations
from typing import Dict, Any, List
import json
import logging
from langchain_core.tools import tool

from uav_risk.stage1.infer import run_stage1_inference
from uav_risk.stage2.schemas import MLResult, RegulationChunk

logger = logging.getLogger(__name__)

# ==========================================
# 1. ML Oracle Tool (Type-Safe & Isolated)
# ==========================================
@tool("get_ml_risk_prediction")
def get_ml_risk_prediction(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls the XGBoost Statistical ML Model to predict flight risk.
    MUST only be used if DataQualityProfile.is_ml_reliable == True.
    
    Args:
        scenario: Dictionary of UAVScenario parameters.
        
    Returns:
        Dict containing predicted_class, risk_score, and confidence.
    """
    try:
        # run_stage1_inference now returns a Pydantic MLResult
        ml_result: MLResult = run_stage1_inference(scenario)
        
        # Return plain dict for LangGraph compatibility
        return {
            "predicted_class": ml_result.predicted_class,
            "risk_score": ml_result.risk_score,
            "confidence": ml_result.confidence
        }
    except Exception as e:
        logger.error(f"[TOOL_ML_ERROR] Inference failed: {e}")
        return {
            "predicted_class": "UNKNOWN",
            "risk_score": 0.5,
            "confidence": 0.0,
            "error": str(e)
        }

# ==========================================
# 2. Legal Oracle Tool (Hybrid RAG Interface)
# ==========================================
@tool("search_aviation_regulations")
def search_aviation_regulations(query: str, context_tags: List[str] | None = None) -> List[Dict[str, Any]]:
    """
    Searches the Hybrid RAG database for aviation regulations & safety thresholds.
    
    Args:
        query: The specific legal/safety question.
        context_tags: Optional metadata tags (e.g., ["BVLOS", "DENSE", "CLASS_C"]) to filter results.
        
    Returns:
        List of relevant regulation chunks with article_id and content.
    """
    try:
        # ---------------------------------------------------------
        # TODO: Replace mock with actual Hybrid RAG call
        # Example: rag_service.hybrid_search(query, metadata=context_tags)
        # ---------------------------------------------------------
        
        # Mock response structured to match RegulationChunk
        mock_results = [
            {
                "article_id": "EASA-UAS-01",
                "content": "For UAVs <25kg, max sustained wind must not exceed 10 m/s unless certified.",
                "relevance_score": 0.95
            },
            {
                "article_id": "MANUAL-SEC-4",
                "content": "If GNSS jamming > -75 dBm, initiate immediate RTH or controlled landing.",
                "relevance_score": 0.88
            }
        ]
        return mock_results
        
    except Exception as e:
        logger.error(f"[TOOL_RAG_ERROR] Regulation search failed: {e}")
        return []

# ==========================================
# Tool Registry
# ==========================================
AGENT_TOOLS = [get_ml_risk_prediction, search_aviation_regulations]