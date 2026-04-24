# src/uav_risk/api/main.py
from __future__ import annotations

import os
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# استيراد المكونات الأساسية
from uav_risk.stage1.loader import load_stage1_artifacts
from uav_risk.stage1.infer import run_stage1_inference
from uav_risk.utils.json_sanitize import sanitize_for_json

# إعداد السجلات المهيكلة
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s')
logger = logging.getLogger("UAV_CORE_API")

# ============================================================
# 1. إدارة دورة حياة التطبيق (Lifespan Management)
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """إدارة الإقلاع والإغلاق لضمان تحميل النماذج مرة واحدة وبشكل آمن."""
    request_id = str(uuid.uuid4())
    logger.info("Initializing UAV Risk Engine artifacts...", extra={"request_id": request_id})
    
    try:
        artifacts_dir = os.getenv("UAV_ARTIFACTS_DIR", "artifacts")
        # تحميل النماذج في حالة التطبيق (App State) لضمان الوصول العالمي
        app.state.artifacts = load_stage1_artifacts(artifacts_dir)
        logger.info("Aviation artifacts loaded successfully.", extra={"request_id": request_id})
    except Exception as e:
        logger.critical(f"FAILED TO LOAD AVIATION ARTIFACTS: {e}", extra={"request_id": request_id})
        raise SystemExit("Application cannot start without ML Artifacts.")
    
    yield
    # عمليات التنظيف عند الإغلاق (إن وجدت)
    logger.info("Shutting down UAV Risk Engine...", extra={"request_id": request_id})

# ============================================================
# 2. إعداد التطبيق والتحقق من المدخلات
# ============================================================
app = FastAPI(
    title="UAV Risk System (Generation 2)",
    version="2.2.0",
    lifespan=lifespan
)

class FlightScenario(BaseModel):
    uav_mass_kg: float = Field(..., ge=0.1, le=150.0)
    environment_weather_wind_mps: float = Field(..., ge=0.0, le=50.0)
    environment_gnss_jam_dbm: float = Field(default=-100.0, ge=-140.0, le=0.0)
    uav_battery_model_hover_power_W: float | None = Field(None, ge=0.0)
    environment_weather_gust_mps: float | None = Field(None, ge=0.0)
    airspace_altitude_agl_m: float = Field(default=30.0, ge=0.0, le=500.0)

class ScenarioPayload(BaseModel):
    scenario: FlightScenario

# حماية استيراد المرحلة الثانية لتجنب الاعتمادية الدائرية
try:
    from uav_risk.stage2.api import router as stage2_router
    app.include_router(stage2_router, prefix="/v2", tags=["Stage 2 Agent"])
except ImportError as e:
    logger.warning(f"Stage 2 Router not loaded: {e}", extra={"request_id": "INIT"})

# ============================================================
# 3. نقاط النهاية (Endpoints)
# ============================================================

@app.middleware("http")
async def add_process_id_header(request: Request, call_next):
    """إضافة ID فريد لكل طلب لتسهيل تتبع الأخطاء (Observability)."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

@app.get("/health", tags=["Infrastructure"])
async def health_check():
    is_ready = hasattr(app.state, "artifacts")
    return {"status": "online" if is_ready else "initializing", "build": "Aviation-Ready-V2"}

@app.post("/stage1/infer", tags=["Stage 1"])
async def stage1_inference(payload: ScenarioPayload, request: Request):
    """نقطة التقييم الإحصائي والمعايرة."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.info(f"Processing inference for scenario...", extra={"request_id": request_id})
    
    # استخدام model_dump لضمان التوافق مع Pydantic V2
    scenario_dict = payload.scenario.model_dump(mode='python')
    
    # تمرير النماذج المحملة مسبقاً (مباشرة أو عبر Cache)
    result = run_stage1_inference(scenario_dict)
    
    # تعقيم المخرجات لضمان JSON سليم
    return sanitize_for_json(result.model_dump(mode='python'))

@app.get("/stage1/expected-columns", tags=["Stage 1"])
async def get_expected_features():
    """يرجع الميزات التي يتوقعها الـ Preprocessor فعلياً."""
    art = app.state.artifacts
    cols = getattr(art.preprocessor, "feature_names_in_", [])
    return {"count": len(cols), "columns": list(cols)}