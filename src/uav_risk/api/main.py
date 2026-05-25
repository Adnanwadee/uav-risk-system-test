# File Path: src/uav_risk/api/main.py
from contextlib import asynccontextmanager
import os
from typing import Dict, Any, Optional
import structlog
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

# الاستيرادات المطلقة من كافة طبقات النظام الجوي الموحد لضمان التوافق المعماري الحتمي
from uav_risk.core.config import get_settings
from uav_risk.core.logging import setup_logging, set_flight_id, set_request_id
from uav_risk.core.contracts import MasterFlightPayload
from uav_risk.core.data_validator import DataValidator
from uav_risk.ml.loader import Stage1Bundle, load_stage1_bundle, assemble_feature_vector_from_dict, ModelLoadError
from uav_risk.ml.feature_defs import FEATURE_DEFINITIONS
from uav_risk.ml.feature_defs import get_all_feature_names, get_core_features
from uav_risk.stage2.rag.rag_core import AsyncRAGCore
from uav_risk.stage2.rag.config import GroqLLMConfig
from uav_risk.stage2.rag.groq_llm import GroqLLM
from uav_risk.stage2.llm.report_writer import ReportWriter
from uav_risk.stage2.pipeline import run_ace_pipeline

# تهيئة نظام السجلات الموحد للمنظومة بناءً على بيئة التشغيل
settings = get_settings()
setup_logging(log_level=settings.LOG_LEVEL, environment=settings.ENVIRONMENT)
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """مدير دورة حياة السيرفر الذكي المسؤول عن تهيئة المكونات الثقيلة والفهارس الحقيقية."""
    logger.info("Starting ACE UAV Risk Assessment System v4.5 [Aviation Production Mode]...", environment=settings.ENVIRONMENT)
    
    # ─── 1. تحميل حزمة نموذج التعلم الآلي LIGHTGBM الحقيقية والأوزان المستقرة ───
    try:
        artifacts_dir = settings.UAV_ARTIFACTS_DIR
        logger.info("loading_production_stage1_machine_learning_bundle", target_path=artifacts_dir)
        app.state.stage1_bundle = load_stage1_bundle(artifacts_dir)
        logger.info("stage1_machine_learning_bundle_loaded_success", feature_count=len(app.state.stage1_bundle.feature_names))
    except Exception as bundle_err:
        logger.critical("fatal_blocker_stage1_bundle_failed_to_initialize_crashing_server", error=str(bundle_err))
        raise bundle_err

    # ─── 2. تحميل دستور ومواصفات الميزات الفيزيائية المعتمدة ───
    app.state.feature_defs = FEATURE_DEFINITIONS
    logger.info("aviation_constitutional_features_defs_anchored_successfully")

    # ─── 3. تهيئة وبناء محرك الاسترجاع المعزز بالقوانين الجوية حياً (RAG Core) ───
    try:
        groq_key = settings.GROQ_API_KEY.get_secret_value() if settings.GROQ_API_KEY else os.getenv("GROQ_API_KEY")
        app.state.rag_core = AsyncRAGCore(groq_api_key=groq_key)
        
        if hasattr(app.state.rag_core, "initialize") and callable(app.state.rag_core.initialize):
            logger.info("assembling_local_faiss_vector_databases_and_ontology_rules")
            await app.state.rag_core.initialize()
            
        status_payload = app.state.rag_core.get_status()
        logger.info("rag_legislative_core_diagnostics_collected", status=status_payload)
    except Exception as rag_err:
        logger.warning("rag_initialization_failed_activating_autonomous_null_fallback", error=str(rag_err))
        app.state.rag_core = AsyncRAGCore(groq_api_key=None)

    # ─── 4. تهيئة بوابة معالجة اللغات الطبيعية سحابياً (Groq LLM) ───
    try:
        groq_key = settings.GROQ_API_KEY.get_secret_value() if settings.GROQ_API_KEY else os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("Groq Credentials Token string is null or missing from cloud cluster config environment.")
            
        # ✅ حل مشكلة الـ Deprecation: استبدال التسمية المُلغاة بنموذج التوليد والتقارير المستقر حياً
        target_model = settings.AGENT_MODEL_NAME
        if target_model == "llama3-70b-8192":
            target_model = "llama-3.3-70b-versatile"
            
        llm_config = GroqLLMConfig(api_key=groq_key, model=target_model)
        app.state.groq_llm = GroqLLM(config=llm_config)
        logger.info("groq_cloud_llm_gateway_established_successfully", model_target=target_model)
    except Exception as llm_err:
        logger.error("groq_llm_handshake_failed_transitioning_to_template_backup_mode", error=str(llm_err))
        app.state.groq_llm = None

    # ─── 5. تهيئة مصنف وصائغ التقارير الهيكلية ───
    app.state.report_writer = ReportWriter(llm=app.state.groq_llm)
    
    logger.info("ALL_ACE_SUB_SYSTEM_GATES_PASSED_SERVER_OPERATIONAL_READY")
    yield
    
    logger.info("Shutting down ACE UAV Risk Assessment System pipeline gates...")
    if hasattr(app.state, 'rag_core') and app.state.rag_core:
        await app.state.rag_core.shutdown()
    logger.info("ACE_SYSTEM_SHUTDOWN_SEQUENCE_SUCCESSFULLY_TERMINATED")


app = FastAPI(
    title="ACE UAV Autonomous Risk Assessment Compliance API",
    version="v4.5.0-Production",
    lifespan=lifespan
)


# Include modular routers for profiles and lightweight validation used by frontend
from uav_risk.api.profiles import router as profiles_router
from uav_risk.api.validate import router as validate_router

app.include_router(profiles_router, prefix="/api", tags=["profiles"])
app.include_router(validate_router, prefix="/api", tags=["validation"])


@app.get("/health", status_code=status.HTTP_200_OK)
async def health(request: Request) -> Dict[str, Any]:
    app_state = request.app.state
    stage1_loaded = hasattr(app_state, 'stage1_bundle') and app_state.stage1_bundle is not None
    rag_ready = hasattr(app_state, 'rag_core') and app_state.rag_core.is_ready() if hasattr(app_state, 'rag_core') else False
    llm_ready = app_state.groq_llm is not None if hasattr(app_state, 'groq_llm') else False
    
    return {
        "status": "healthy" if stage1_loaded and rag_ready and llm_ready else "degraded",
        "stage1_loaded": stage1_loaded,
        "rag_ready": rag_ready,
        "llm_ready": llm_ready,
        "feature_count": len(app_state.stage1_bundle.feature_names) if stage1_loaded else 0,
        "system_version": "ACE-v4.5.0-Prod"
    }


@app.post("/v2/evaluate", status_code=status.HTTP_200_OK)
async def evaluate_flight(payload: MasterFlightPayload, request: Request) -> JSONResponse:
    app_state = request.app.state
    flight_id = payload.get_flight_id() or "FLIGHT-UNKNOWN-GATEWAY-INBOUND"
    
    with set_flight_id(flight_id), set_request_id(flight_id):
        logger.info("inbound_flight_evaluation_request_received", flight_id=flight_id)
        
        if not hasattr(app_state, 'stage1_bundle') or app_state.stage1_bundle is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Aviation Engine Error: Stage-1 ML LightGBM model is not mounted."
            )
            
        try:
            flat_all = payload.flatten_for_ml(primary_only=False)
            allowed_features = set(get_all_feature_names())
            input_map = {key: value for key, value in flat_all.items() if key in allowed_features}

            # Deterministic pre-flight veto (quick reject for blatantly invalid submissions)
            from uav_risk.stage2.pipeline import DeterministicCore
            veto_checker = DeterministicCore()
            veto = veto_checker.pre_flight_veto_check(payload.to_tier0_dict())
            if veto.vetoed:
                return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={
                    "veto": True,
                    "reasons": [{"code": "tier0_veto", "message": veto.reason, "detail": "deterministic_tier0"}],
                    "missing_cores": [],
                    "warnings": [],
                    "is_usable": False,
                    "data_quality_score": 0.0,
                })

            feature_vec, fv_meta = assemble_feature_vector_from_dict(input_map, app_state.stage1_bundle)
            feature_map = fv_meta.get("feature_map", {})

            policy_flag = None
            if hasattr(app_state, 'stage1_bundle') and getattr(app_state.stage1_bundle, 'policy_config', None):
                policy_flag = app_state.stage1_bundle.policy_config.get('fail_on_imputed_core')

            validator = DataValidator(fail_on_imputed_core=policy_flag)
            validation_result = validator.validate_and_store(feature_map)

            if not validation_result.is_usable:
                reasons = []
                if validation_result.missing_core_features:
                    reasons.append({"code": "missing_cores", "message": "Missing or critical core feature violations", "detail": ",".join(validation_result.missing_core_features)})
                for w in validation_result.warnings:
                    reasons.append({"code": "warning", "message": w, "detail": "validator"})

                return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={
                    "veto": False,
                    "reasons": reasons,
                    "missing_cores": validation_result.missing_core_features,
                    "warnings": validation_result.warnings,
                    "is_usable": validation_result.is_usable,
                    "data_quality_score": validation_result.overall_data_quality_score,
                })

            pipeline_result = await run_ace_pipeline(
                flight_id=flight_id,
                payload=payload,
                full_telemetry=payload.model_dump(),
                stage1_bundle=app_state.stage1_bundle,
                rag_core=app_state.rag_core,
                groq_llm=app_state.groq_llm,
                feature_defs=app_state.feature_defs,
                report_writer=app_state.report_writer,
                precomputed_feature_vector=feature_vec,
                precomputed_validation_result=validation_result
            )
            return JSONResponse(content=jsonable_encoder(pipeline_result))

        except ModelLoadError as mle:
            logger.warning("model_load_error_in_api_evaluation", flight_id=flight_id, error=str(mle))
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": str(mle)})

        except HTTPException:
            # Re-raise HTTP exceptions to preserve intended status codes
            raise

        except Exception as system_failure:
            logger.critical("unhandled_critical_crash_inside_api_evaluation_gate", flight_id=flight_id, error=str(system_failure))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Sovereign Core Crash Sequence Activated: {str(system_failure)}"
            )

# =====================================================================
# Consumed by: Production WSGI / ASGI Servers (Uvicorn / Gunicorn)
# =====================================================================