from __future__ import annotations

from contextlib import asynccontextmanager

from uav_risk.core.env import load_project_env

load_project_env()

from fastapi import FastAPI

from uav_risk.api.dependencies import load_cached_stage1_bundle
from uav_risk.api.routes.assessments import (
    history_router as assessment_history_router,
    router as assessments_router,
)
from uav_risk.api.routes.features import router as features_router
from uav_risk.api.routes.health import router as health_router
from uav_risk.api.routes.profiles import router as profiles_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.stage1_bundle = load_cached_stage1_bundle()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="UAV Risk API",
        version="0.7.0-raw-core-ml",
        description="Raw-first Core/ML API for UAV flight risk assessment.",
        lifespan=lifespan,
    )
    app.include_router(health_router)
    app.include_router(features_router)
    app.include_router(profiles_router)
    app.include_router(assessments_router)
    app.include_router(assessment_history_router)
    return app


app = create_app()
