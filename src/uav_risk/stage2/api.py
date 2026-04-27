"""
Stage 2 API Router (ACE Pipeline Gateway V4.1)
==============================================
Fixes applied from final SRE audit:
- Defused the NaN Timebomb: Implemented recursive `_sanitize_nan` to bypass Python's C-level `json` module bypass for floats.
- Stringly-Typed Error Fix: Added a TODO and architectural marker for `TelemetryValidationError` dependency inversion.

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

def get_report_writer(request: Request):
    return request.app.state.report_writer

# ============================================================
# 1. Architectural Safeguards (The True NaN Defuser)
# ============================================================
class SafeNaNJSONResponse(JSONResponse):
    """
    استجابة مخصصة لمنع انهيار التسلسل بسبب قيم NaN.
    تطهر القاموس عودياً (Recursively) قبل إرساله لمكتبة json المدمجة.
    """
    @staticmethod
    def _sanitize_nan(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: SafeNaNJSONResponse._sanitize_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [SafeNaNJSONResponse._sanitize_nan(i) for i in obj]
        elif isinstance(obj, float) and not math.isfinite(obj):
            return None # تحويل صريح وقطعي لـ null
        return obj

    def render(self, content: Any) -> bytes:
        # 1. التطهير العميق للبيانات
        safe_content = self._sanitize_nan(content)
        # 2. التسلسل الآمن
        return json.dumps(
            safe_content, 
            ensure_ascii=False, 
            allow_nan=False
        ).encode("utf-8")


# TODO: Replace string-based error checking with this Custom Exception 
# once `toolbox.py` is updated to raise it. (Dependency Inversion for Exceptions)
class TelemetryValidationError(ValueError):
    pass


# ============================================================
# 2. Evaluation Endpoint
# ============================================================
@router.post(
    "/evaluate", 
    response_class=SafeNaNJSONResponse, 
    responses={
        200: {"description": "Flight evaluated successfully."},
        422: {"description": "Validation Error (Corrupted or Missing Critical Telemetry)."},
        504: {"description": "Gateway Timeout (Agents took too long)."},
        500: {"description": "Internal ACE System Failure."}
    }
)
async def evaluate_flight_risk(
    payload: MasterFlightPayload, 
    request: Request,
    report_writer = Depends(get_report_writer)
):
    start_time = time.perf_counter()
    flight_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    try:
        result_dict = await run_ace_pipeline(
            flight_id=flight_id,
            payload=payload,
            report_writer=report_writer
        )
        
        process_time_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"Request {flight_id} processed successfully in {process_time_ms:.1f}ms.")
        
        return SafeNaNJSONResponse(status_code=200, content=result_dict)
        
    except TelemetryValidationError as tve:
        # Future-proofed Explicit Catching
        logger.warning(f"[{flight_id}] Client Validation Error: {tve}")
        raise HTTPException(
            status_code=422, 
            detail={"error": "Telemetry Validation Failed", "reason": str(tve)},
            headers={"X-Request-ID": flight_id}
        )
        
    except ValueError as ve:
        # Legacy Stringly-Typed Fallback
        error_str = str(ve).lower()
        if "telemetry" in error_str or "sanitize" in error_str or "validation" in error_str:
            logger.warning(f"[{flight_id}] Client Validation Error: {ve}")
            raise HTTPException(
                status_code=422, 
                detail={"error": "Telemetry Validation Failed", "reason": str(ve)},
                headers={"X-Request-ID": flight_id}
            )
        else:
            logger.critical(f"[{flight_id}] Business Logic ValueError: {ve}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail="ACE Internal System Failure. The operation was halted.",
                headers={"X-Request-ID": flight_id}
            )
            
    except TimeoutError as te:
        if isinstance(te, asyncio.TimeoutError):
            logger.error(f"[{flight_id}] Async timeout during Graph execution.")
            raise HTTPException(
                status_code=504, 
                detail="Gateway Timeout: The autonomous agents took too long to reach consensus.",
                headers={"X-Request-ID": flight_id}
            )
        else:
            logger.error(f"[{flight_id}] General timeout error: {te}")
            raise HTTPException(
                status_code=500, 
                detail="ACE Internal System Failure (General Timeout).",
                headers={"X-Request-ID": flight_id}
            )
        
    except Exception as e:
        logger.critical(f"[{flight_id}] UNHANDLED SYSTEM CRASH: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="ACE Internal System Failure. The operation was halted. Incident has been logged.",
            headers={"X-Request-ID": flight_id}
        )