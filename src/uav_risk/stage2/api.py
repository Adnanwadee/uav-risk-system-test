from __future__ import annotations

from fastapi import APIRouter

from uav_risk.stage2.pipeline import run_stage2_report
from uav_risk.stage2.schemas import Stage2Request, Stage2Response

router = APIRouter(prefix="/stage2", tags=["stage2"])


@router.post("/report", response_model=Stage2Response)
def stage2_report(payload: Stage2Request):
    return run_stage2_report(payload.scenario, artifacts_dir="artifacts")
