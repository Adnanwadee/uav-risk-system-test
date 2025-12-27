from __future__ import annotations
from typing import Any, Dict
from pydantic import BaseModel, Field
from fastapi import APIRouter

from .report.service import run_stage2_report

router = APIRouter(prefix="/stage2", tags=["stage2"])


class Stage2Request(BaseModel):
    scenario: Dict[str, Any] = Field(default_factory=dict)


@router.post("/report")
def stage2_report(payload: Stage2Request):
    return run_stage2_report(payload.scenario, artifacts_dir="artifacts").model_dump()
