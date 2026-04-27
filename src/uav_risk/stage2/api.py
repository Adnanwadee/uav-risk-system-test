"""
Stage 2 API Router (ACE Pipeline Gateway V4.2)
==============================================
Fixes applied:
- Dependency Injection: Extracts the compiled safety graph and report writer 
  from the app state and injects them into the pipeline.
- Custom NaN Handling: Maintains the SafeNaNJSONResponse for telemetry safety.

Author: Stage 2 — ACE System
"""

import json
import time
import uuid
import math
import asyncio
import logging
from typing import Any
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import JSONResponse

from uav_risk.stage2.input_contract import MasterFlightPayload
from uav_risk.stage2.pipeline import run_ace_pipeline

logger = logging.getLogger("ACE_API_Router")

router = APIRouter(prefix="/stage2", tags=["ACE Stage 2"])

# ============================================================
# 1. Architectural Safeguards (The NaN Defuser)
# ============================================================
class SafeNaNJSONResponse(JSONResponse):
    """استجابة مخصصة تمنع انهيار JSON بسبب قيم NaN المتداخلة."""
    @staticmethod
    def _sanitize_nan(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: SafeNaNJSONResponse._sanitize_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [SafeNaNJSONResponse._sanitize_nan(i) for i in obj]
        elif isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj

    def render(self, content: Any) -> bytes:
        sanitized = self._sanitize_nan(content)
        return json.dumps(
            sanitized,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")

# ============================================================
# 2. Main Evaluation Endpoint (The Bridge)
# ============================================================

@router.post("/evaluate", response_class=SafeNaNJSONResponse)
async def evaluate_flight(request: Request, payload: MasterFlightPayload):
    """
    نقطة النهاية الرئيسية: تربط حالة التطبيق (App State) بالـ Pipeline المحصن.
    """
    flight_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    # [FIX]: استخراج المحركات التي تم بناؤها في lifespan بملف main.py
    graph_app = request.app.state.safety_agent_app
    report_writer = request.app.state.report_writer
    
    if not graph_app:
        logger.error(f"[{flight_id}] Critical: Safety Agent Graph not found in App State.")
        raise HTTPException(status_code=500, detail="ACE System Engine not initialized.")

    try:
        t0 = time.perf_counter()
        logger.info(f"[{flight_id}] Route received evaluation request.")

        # [FIX]: تمرير الـ graph_app والـ report_writer للـ Pipeline
        result = await run_ace_pipeline(
            flight_id=flight_id,
            payload=payload,
            graph_app=graph_app, 
            report_writer=report_writer
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"[{flight_id}] Route processing completed in {latency_ms:.1f}ms.")
        
        return result

    except ValueError as ve:
        logger.warning(f"[{flight_id}] Validation Failure: {ve}")
        raise HTTPException(
            status_code=400, 
            detail={"error": "Telemetry Validation Failed", "reason": str(ve)},
            headers={"X-Request-ID": flight_id}
        )
            
    except asyncio.TimeoutError:
        logger.error(f"[{flight_id}] Gateway Timeout during agents consensus.")
        raise HTTPException(
            status_code=504, 
            detail="Timeout: The agents took too long to reach consensus.",
            headers={"X-Request-ID": flight_id}
        )
        
    except Exception as e:
        logger.critical(f"[{flight_id}] UNHANDLED SYSTEM CRASH: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="ACE Internal System Failure. Incident has been logged.",
            headers={"X-Request-ID": flight_id}
        )