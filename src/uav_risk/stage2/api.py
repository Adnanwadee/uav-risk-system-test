"""
Stage 2 API Router (ACE Pipeline Gateway V14.0)
==============================================
التعديلات الجوهرية:
1. توحيد التطهير: استخدام sanitize_for_json المركزي لضمان سلامة الـ JSON.
2. ضخ البيانات الكاملة: استخدام flatten_for_ml لضمان وصول الـ 50 عاموداً للوكلاء دون نقص.
3. حقن التبعيات: استخراج الـ Graph و ReportWriter مباشرة من حالة التطبيق (App State).
"""

import time
import uuid
import logging
import asyncio
from typing import Dict, Any
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import JSONResponse

# استيراد العقود والأدوات المحدثة
from uav_risk.stage2.input_contract import MasterFlightPayload
from uav_risk.stage2.pipeline import run_ace_pipeline
from uav_risk.utils.json_sanitize import sanitize_for_json

logger = logging.getLogger("ACE_API_Router")

# تعريف الرواتر ببادئة /stage2
router = APIRouter(prefix="/stage2", tags=["ACE Stage 2"])

def get_flight_id() -> str:
    """توليد معرف فريد لكل عملية تدقيق لضمان التتبع الجنائي."""
    return f"FLIGHT-{uuid.uuid4().hex[:8].upper()}"

@router.post("/evaluate")
async def evaluate_flight(
    payload: MasterFlightPayload,
    request: Request,
    flight_id: str = Depends(get_flight_id)
):
    """
    نقطة الدخول الرئيسية لتقييم مخاطر الرحلة.
    تستلم الـ 50 عاموداً وتشغل الوكلاء.
    """
    # 1. استخراج المحركات من حالة التطبيق لضمان استمرارية الذاكرة
    graph_app = getattr(request.app.state, "safety_graph", None)
    report_writer = getattr(request.app.state, "report_writer", None)

    if not graph_app or not report_writer:
        logger.critical("ACE Engine components missing in App State.")
        raise HTTPException(status_code=500, detail="ACE System Engines are offline.")

    try:
        t0 = time.perf_counter()
        logger.info(f"[{flight_id}] Received evaluation request for {payload.uav.mass_kg}kg UAV.")

        # 2. تفعيل الـ 50 عاموداً عبر التسطيح (Flattening)
        full_telemetry = payload.flatten_for_ml()

        # 3. تشغيل الـ Pipeline المطور الذي يربط الوكلاء الأربعة
        result = await run_ace_pipeline(
            flight_id=flight_id,
            payload=payload,            
            full_telemetry=full_telemetry, 
            graph_app=graph_app, 
            report_writer=report_writer
        )

        # 4. التطهير النهائي لحماية الـ API من قيم NaN
        sanitized_result = sanitize_for_json(result)

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"[{flight_id}] Risk Assessment completed in {latency_ms:.1f}ms.")
        
        return JSONResponse(content=sanitized_result)

    except ValueError as ve:
        logger.warning(f"[{flight_id}] Data Integrity Error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
            
    except asyncio.TimeoutError:
        logger.error(f"[{flight_id}] Consensus Timeout - Agents took too long.")
        raise HTTPException(status_code=504, detail="Agents consensus timeout.")
        
    except Exception as e:
        logger.critical(f"[{flight_id}] UNHANDLED SYSTEM CRASH: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal ACE Engine Failure.")