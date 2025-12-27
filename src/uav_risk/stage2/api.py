#stage2/api,py
from __future__ import annotations
from typing import Any, Dict

from fastapi import APIRouter
from pydantic import BaseModel, Field

from uav_risk.stage2.pipeline import run_stage2_report
from uav_risk.stage2.schemas import Stage2Response

router = APIRouter(prefix="/stage2", tags=["stage2"])


class Stage2Request(BaseModel):
    scenario: Dict[str, Any] = Field(default_factory=dict)


@router.post("/report", response_model=Stage2Response)
def stage2_report(payload: Stage2Request):
    res = run_stage2_report(payload.scenario, artifacts_dir="artifacts")

    # إذا رجعت dict لأي سبب، نحوّلها لـ Stage2Response
    if isinstance(res, dict):
        return Stage2Response(**res)

    # إذا رجعت Stage2Response بالفعل
    return res
