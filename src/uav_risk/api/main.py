"""
ACE System - Main API Entry Point (V15.5 - Apex Unified)
===========================================================
التحديثات:
1. توحيد الربط: المسمى المعتمد الآن هو safety_graph ليتوافق مع api.py.
2. النموذج المعتمد: الانتقال رسمياً لنموذج llama-3.3-70b-versatile.
3. التوجيه المحدث: ربط الرواتر ببادئة /v2 لخدمة تطبيق Streamlit.
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
from langgraph.checkpoint.memory import MemorySaver

from uav_risk.stage1.loader import load_stage1_artifacts
from uav_risk.stage2.llm.groq_client import GroqAsyncClient
from uav_risk.stage2.llm.report_writer import SafetyReportWriter
from uav_risk.stage2.rag.rag_core import AsyncRAGCore
from uav_risk.stage2.agents.physics_agent import PhysicsAgent, DronePhysicalSpec
from uav_risk.stage2.agents.temporal_agent import TemporalAgent
from uav_risk.stage2.agents.legal_agent import LegalAgent
from uav_risk.stage2.agents.consensus_agent import ConsensusAgent
from uav_risk.stage2.graph.safety_agent import ACESafetyGraph 

# إعداد السجلات المتقدمة
request_id_context_var: ContextVar[str] = ContextVar("request_id", default="SYSTEM")

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_context_var.get()
        return True

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(request_id)s] - %(name)s - %(levelname)s - %(message)s')
for handler in logging.root.handlers:
    handler.addFilter(RequestIdFilter())

logger = logging.getLogger("ACE_CORE_API")
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "100"))
global_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

@asynccontextmanager
async def lifespan(app: FastAPI):
    request_id_context_var.set("SYS-BOOT")
    logger.info("Initializing ACE Multi-Agent Ecosystem (V15.5 Apex)...")
    
    try:
        # أ. تحميل نماذج المرحلة الأولى
        artifacts_dir = os.getenv("UAV_ARTIFACTS_DIR", "artifacts")
        app.state.artifacts = load_stage1_artifacts(artifacts_dir)
        logger.info("Stage 1 ML Artifacts loaded.")

        # ب. تهيئة البنية التحتية للمرحلة الثانية
        model_name = os.getenv("LLM_MODEL_NAME", "llama-3.3-70b-versatile")
        app.state.llm_client = GroqAsyncClient(model_name=model_name, temperature=0.0)
        app.state.rag_core = AsyncRAGCore() 
        logger.info(f"Cognitive Engine ({model_name}) & RAG Core online.")

        # ج. مواصفات الطائرة
        default_spec = DronePhysicalSpec(
            mass_kg=1.3, max_thrust_n=45.0, rotor_area_m2=0.25,
            drag_coefficient=0.8, frontal_area_m2=0.05, max_wind_tolerance_ms=12.0,
            battery_capacity_wh=50.0, hover_power_w=220.0, structural_load_limit_n=100.0
        )

        # د. تهيئة الوكلاء
        p_agent = PhysicsAgent(spec=default_spec)
        t_agent = TemporalAgent()
        l_agent = LegalAgent(rag_index=app.state.rag_core, llm_client=app.state.llm_client)
        c_agent = ConsensusAgent()

        # هـ. بناء الـ LangGraph
        orchestrator = ACESafetyGraph(
            physics_agent=p_agent, temporal_agent=t_agent,
            legal_agent=l_agent, consensus_agent=c_agent
        )
        
        # [إصلاح حاسم]: استخدام safety_graph ليتوافق مع استدعاء الـ API
        memory = MemorySaver()
        app.state.safety_graph = orchestrator.compile(checkpointer=memory)
        
        app.state.report_writer = SafetyReportWriter(llm_client=app.state.llm_client)
        logger.info("ACE Multi-Agent Graph is fully synchronized and FLIGHT-READY.")
        
    except Exception as e: 
        logger.critical(f"FATAL BOOT ERROR: {e}", exc_info=True)
        raise  
    yield

app = FastAPI(title="ACE API", version="15.5.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# دمج الرواتر ببادئة النسخة الثانية لخدمة الـ UI المطور
from uav_risk.stage2.api import router as stage2_router
app.include_router(stage2_router, prefix="/v2")

@app.get("/health")
async def health_check():
    state = app.state
    components = {
        "ml_artifacts": hasattr(state, "artifacts"),
        "llm_client": hasattr(state, "llm_client"),
        "rag_core": hasattr(state, "rag_core"),
        "agent_graph": hasattr(state, "safety_graph")
    }
    return {"status": "ONLINE", "components": components}
# أضف هذا في آخر سطر في ملف main.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)