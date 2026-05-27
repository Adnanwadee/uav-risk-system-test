from __future__ import annotations

from fastapi import APIRouter, Depends

from uav_risk.api.dependencies import get_stage1_bundle
from uav_risk.ml.schemas import Stage1Bundle

router = APIRouter()


@router.get("/health")
def health(bundle: Stage1Bundle = Depends(get_stage1_bundle)) -> dict[str, object]:
    return {"status": "ok", "service": "uav-risk-api", "ml_bundle_loaded": bundle is not None}
