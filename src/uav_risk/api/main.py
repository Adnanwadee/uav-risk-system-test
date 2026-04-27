"""
ACE System - Main API Entry Point (V4.1 - The Unbreakable Core)
===============================================================
Fixes applied from SRE Audit:
- Concurrency Shield: Integrated `asyncio.Semaphore` to cap max parallel requests and prevent Resource Exhaustion (OOM/CPU locks).
- Consistent Lifecycle Tracing: Fixed the shutdown trace ID to prevent disconnected log trails.

Author: Stage 2 — ACE System
"""

import os
import uuid
import logging
import asyncio
from contextvars import ContextVar
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from uav_risk.stage1.loader import load_stage1_artifacts
from uav_risk.stage2.llm.groq_client import GroqAsyncClient
from uav_risk.stage2.llm.report_writer import SafetyReportWriter

# ============================================================
# 1. Advanced Logging & Concurrency Setup
# ============================================================
request_id_context_var: ContextVar[str] = ContextVar("request_id", default="SYSTEM")

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_context_var.get()
        return True

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [%(request_id)s] - %(name)s - %(levelname)s - %(message)s'
)
for handler in logging.root.handlers:
    handler.addFilter(RequestIdFilter())

logger = logging.getLogger("ACE_CORE_API")

# [FIX] Semaphore لحماية النظام من الانهيار تحت الضغط العالي (Max 100 concurrent risk evaluations)
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "100"))
global_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


# ============================================================
# 2. Application Lifespan (Resource Management)
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # [FIX] توحيد بصمة التشغيل لسهولة التتبع
    request_id_context_var.set("SYS-BOOT")
    logger.info("Initializing ACE System Engines (V4.1)...")
    
    try:
        artifacts_dir = os.getenv("UAV_ARTIFACTS_DIR", "artifacts")
        app.state.artifacts = load_stage1_artifacts(artifacts_dir)
        logger.info("Stage 1 ML Artifacts loaded successfully.")
        
        model_name = os.getenv("LLM_MODEL_NAME", "llama3-70b-8192")
        app.state.llm_client = GroqAsyncClient(model_name=model_name, temperature=0.0)
        app.state.report_writer = SafetyReportWriter(llm_client=app.state.llm_client)
        logger.info(f"Stage 2 Cognitive Engine ({model_name}) instantiated.")
        
    except Exception as e: 
        logger.critical(f"FATAL BOOT ERROR: Failed to load system engines: {e}", exc_info=True)
        raise  
        
    logger.info("ACE System is FLIGHT-READY.")
    
    try:
        yield
    except asyncio.CancelledError:
        logger.warning("Lifespan task cancelled. Cleaning up...")
    finally:
        # [FIX] بصمة صريحة للإغلاق بدلاً من توليد UUID جديد يضيع السياق
        request_id_context_var.set("SYS-SHUTDOWN")
        logger.info("Initiating ACE System graceful shutdown...")
        if hasattr(app.state, 'llm_client') and app.state.llm_client:
            try:
                close_method = getattr(app.state.llm_client, 'close', getattr(app.state.llm_client, 'shutdown', None))
                if close_method:
                    if asyncio.iscoroutinefunction(close_method):
                        await close_method()
                    else:
                        close_method()
                    logger.info("LLM Client session closed successfully.")
            except Exception as e:
                logger.warning(f"Error closing LLM Client: {e}")


# ============================================================
# 3. Application Setup & Security Middleware
# ============================================================
app = FastAPI(
    title="ACE (Autonomous Control Engine) API",
    version="4.1.0",
    lifespan=lifespan
)

_origins_raw = os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else ["*"]
allowed_origins = list(set([o.strip() for o in _origins_raw if o.strip()]))
is_production = os.getenv("ENVIRONMENT", "development").lower() == "production"

if "*" in allowed_origins and is_production:
    logger.warning("SECURITY ALERT: Wildcard CORS origin (*) detected in PRODUCTION.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.middleware("http")
async def security_and_trace_middleware(request: Request, call_next):
    # 1. DoS Protection (Size Limit)
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > 10_000_000:
                logger.warning(f"DoS Protection: Rejected oversized payload ({content_length} bytes).")
                return JSONResponse(status_code=413, content={"detail": "Payload too large."})
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length."})

    # 2. Trace Injection
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    request_id_context_var.set(request_id)
    
    # 3. [FIX] Concurrency Shield (Semaphore)
    try:
        async with global_semaphore:
            response = await call_next(request)
    except Exception as e:
        # If the queue itself fails, fallback gracefully
        logger.error(f"Middleware Error: {e}")
        return JSONResponse(status_code=500, content={"detail": "Internal Server Overload."})

    response.headers["X-Request-ID"] = request_id
    return response

# ============================================================
# 4. Routing & Health Checks
# ============================================================

from uav_risk.stage2.api import router as stage2_router
app.include_router(stage2_router, prefix="/v2")

@app.get("/health", tags=["Infrastructure"])
async def health_check():
    artifacts_ok = hasattr(app.state, "artifacts") and app.state.artifacts is not None
    llm_ok = hasattr(app.state, "llm_client") and app.state.llm_client is not None
    
    is_healthy = artifacts_ok and llm_ok
    
    return {
        "status": "ONLINE" if is_healthy else "DEGRADED",
        "version": "4.1.0",
        "components": {
            "ml_artifacts": "OK" if artifacts_ok else "OFFLINE",
            "llm_client": "OK" if llm_ok else "OFFLINE"
        },
        # Monitoring current load
        "active_requests": MAX_CONCURRENT_REQUESTS - global_semaphore._value 
    }