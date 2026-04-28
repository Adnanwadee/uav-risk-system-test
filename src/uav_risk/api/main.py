"""
ACE System - Main API Entry Point (V15.0 - Apex Unified)
===========================================================
التحديثات الجوهرية:
1. توحيد الربط: تم تعديل المسميات (safety_graph) لتتوافق مع ملف api.py المطور.
2. دعم الـ 50 عاموداً: تهيئة الوكلاء لاستقبال البيانات الحية وتجاوز القيم الافتراضية.
3. تكامل التقارير: ربط SafetyReportWriter المحدث لإنتاج تقارير RAG الاحترافية.
4. استدامة الذاكرة: الحفاظ على MemorySaver لدعم التدخل البشري (HITL).
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

# LangGraph Memory
from langgraph.checkpoint.memory import MemorySaver

# 1. الأساسيات (Stage 1)
from uav_risk.stage1.loader import load_stage1_artifacts

# 2. مكونات الذكاء والوكلاء (Stage 2)
from uav_risk.stage2.llm.groq_client import GroqAsyncClient
from uav_risk.stage2.llm.report_writer import SafetyReportWriter
from uav_risk.stage2.rag.rag_core import AsyncRAGCore
from uav_risk.stage2.agents.physics_agent import PhysicsAgent, DronePhysicalSpec
from uav_risk.stage2.agents.temporal_agent import TemporalAgent
from uav_risk.stage2.agents.legal_agent import LegalAgent
from uav_risk.stage2.agents.consensus_agent import ConsensusAgent
from uav_risk.stage2.graph.safety_agent import ACESafetyGraph 

# ============================================================
# 1. Advanced Logging Setup
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

MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "100"))
global_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


# ============================================================
# 2. Application Lifespan (The Brain's Delivery Room)
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    request_id_context_var.set("SYS-BOOT")
    logger.info("Initializing ACE Multi-Agent Ecosystem (V15.0 Apex)...")
    
    try:
        # أ. تحميل نماذج المرحلة الأولى
        artifacts_dir = os.getenv("UAV_ARTIFACTS_DIR", "artifacts")
        app.state.artifacts = load_stage1_artifacts(artifacts_dir)
        logger.info("Stage 1 ML Artifacts loaded.")

        # ب. تهيئة البنية التحتية للمرحلة الثانية (LLM & RAG)
        model_name = os.getenv("LLM_MODEL_NAME", "llama3-70b-8192")
        app.state.llm_client = GroqAsyncClient(model_name=model_name, temperature=0.0)
        app.state.rag_core = AsyncRAGCore() 
        logger.info(f"Cognitive Engine ({model_name}) & RAG Core online.")

        # ج. إعداد مواصفات الطائرة الافتراضية
        # [ملاحظة]: سيتم تجاوز هذه القيم ديناميكياً من الـ 50 عاموداً في كل رحلة
        default_spec = DronePhysicalSpec(
            mass_kg=1.3, max_thrust_n=45.0, rotor_area_m2=0.25,
            drag_coefficient=0.8, frontal_area_m2=0.05, max_wind_tolerance_ms=12.0,
            battery_capacity_wh=50.0, hover_power_w=220.0, structural_load_limit_n=100.0
        )

        # د. تهيئة الوكلاء الأربعة (نظام الأوزان الجديدة)
        p_agent = PhysicsAgent(spec=default_spec)
        t_agent = TemporalAgent()
        l_agent = LegalAgent(rag_index=app.state.rag_core, llm_client=app.state.llm_client)
        c_agent = ConsensusAgent()

        # هـ. بناء الـ LangGraph وتزويده بالذاكرة!
        orchestrator = ACESafetyGraph(
            physics_agent=p_agent,
            temporal_agent=t_agent,
            legal_agent=l_agent,
            consensus_agent=c_agent
        )
        
        # [تعديل حاسم]: تغيير المسمى إلى safety_graph ليتوافق مع الـ API المطور
        memory = MemorySaver()
        app.state.safety_graph = orchestrator.compile(checkpointer=memory)
        
        # و. تهيئة كاتب التقارير المطور (V14.0)
        app.state.report_writer = SafetyReportWriter(llm_client=app.state.llm_client)
        
        logger.info("ACE Multi-Agent Graph is fully synchronized, stateful, and FLIGHT-READY.")
        
    except Exception as e: 
        logger.critical(f"FATAL BOOT ERROR: Synchronization failed: {e}", exc_info=True)
        raise  
        
    try:
        yield
    finally:
        request_id_context_var.set("SYS-SHUTDOWN")
        logger.info("Initiating ACE System graceful shutdown...")
        if hasattr(app.state, 'rag_core'):
            app.state.rag_core.shutdown()
        if hasattr(app.state, 'llm_client') and app.state.llm_client:
            try:
                close_op = getattr(app.state.llm_client, 'close', getattr(app.state.llm_client, 'shutdown', None))
                if close_op:
                    if asyncio.iscoroutinefunction(close_op): asyncio.create_task(close_op())
                    else: close_op()
            except Exception as e: logger.warning(f"Error during LLM cleanup: {e}")

# ============================================================
# 3. Application Setup & Security Middleware
# ============================================================
app = FastAPI(
    title="ACE (Autonomous Control Engine) API",
    version="15.0.0",
    lifespan=lifespan
)

_origins_raw = os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else ["*"]
allowed_origins = list(set([o.strip() for o in _origins_raw if o.strip()]))
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.middleware("http")
async def security_and_trace_middleware(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 10_000_000:
        return JSONResponse(status_code=413, content={"detail": "Payload too large."})

    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    request_id_context_var.set(request_id)
    
    async with global_semaphore:
        response = await call_next(request)

    response.headers["X-Request-ID"] = request_id
    return response

# ============================================================
# 4. Routing & Health Checks
# ============================================================
from uav_risk.stage2.api import router as stage2_router

# [تعديل]: الحفاظ على مسار /v2 ليتوافق مع تطبيق الـ UI (app.py)
app.include_router(stage2_router, prefix="/v2")

@app.get("/health", tags=["Infrastructure"])
async def health_check():
    state = app.state
    components = {
        "ml_artifacts": hasattr(state, "artifacts") and state.artifacts is not None,
        "llm_client": hasattr(state, "llm_client") and state.llm_client is not None,
        "rag_core": hasattr(state, "rag_core") and state.rag_core.get_health_status()["status"] == "healthy",
        "agent_graph": hasattr(state, "safety_graph") and state.safety_graph is not None
    }
    
    is_healthy = all(components.values())
    
    return {
        "status": "ONLINE" if is_healthy else "DEGRADED",
        "version": "15.0.0",
        "components": components,
        "active_concurrency": MAX_CONCURRENT_REQUESTS - global_semaphore._value 
    }