from typing import Dict, Any, List
from fastapi import APIRouter, Request, HTTPException, status
from pydantic import BaseModel

from uav_risk.stage2.pipeline import DeterministicCore
from uav_risk.core.data_validator import DataValidator
from uav_risk.ml.loader import assemble_feature_vector_from_dict
from uav_risk.ml.feature_defs import get_all_feature_names, get_core_features

router = APIRouter()


class Reason(BaseModel):
    code: str
    message: str
    detail: str = ""


class ValidateRequest(BaseModel):
    payload: Dict[str, Any]


class ValidateResponse(BaseModel):
    veto: bool
    reasons: List[Reason]
    missing_cores: List[str]
    warnings: List[str]
    is_usable: bool
    data_quality_score: float


def _reason(code: str, message: str, detail: str = "") -> Dict[str, str]:
    return {"code": code, "message": message, "detail": detail}


@router.post("/flight/validate", response_model=ValidateResponse)
def validate_flight(req: ValidateRequest, request: Request):
    tier0 = req.payload

    # Accept the authoritative 198-feature contract and ignore legacy extras.
    sso = set(get_all_feature_names())
    tier0 = {key: value for key, value in tier0.items() if key in sso}

    veto_checker = DeterministicCore()
    veto = veto_checker.pre_flight_veto_check(tier0)
    if veto.vetoed:
        return {
            "veto": True,
            "reasons": [_reason("tier0_veto", veto.reason, "deterministic_tier0")],
            "missing_cores": [],
            "warnings": [],
            "is_usable": False,
            "data_quality_score": 0.0,
        }

    app_state = request.app.state
    if not hasattr(app_state, 'stage1_bundle') or app_state.stage1_bundle is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Stage-1 bundle not mounted")

    feature_vec, feature_meta = assemble_feature_vector_from_dict(tier0, app_state.stage1_bundle)
    feature_map = feature_meta.get("feature_map", tier0)

    validator = DataValidator()
    validation_result = validator.validate_and_store(feature_map)

    reasons: List[Dict[str, str]] = []
    if not validation_result.is_usable:
        reasons.append(_reason("missing_cores", "Missing or critical core feature violations", ",".join(validation_result.missing_core_features)))

    # include warnings as structured reasons
    for w in validation_result.warnings:
        reasons.append(_reason("warning", w, "validator"))

    response = {
        "veto": False,
        "reasons": reasons,
        "missing_cores": validation_result.missing_core_features,
        "warnings": validation_result.warnings,
        "is_usable": validation_result.is_usable,
        "data_quality_score": validation_result.overall_data_quality_score,
    }

    return response


@router.post("/flight/assemble")
def assemble_flight(req: ValidateRequest, request: Request):
    """Assemble numeric feature vector and return metadata for frontend diagnostics.

    This endpoint performs deterministic preflight and structured validation, then
    returns the assembled vector length and metadata (imputed/provided/free_text).
    """
    input_map = req.payload

    sso = set(get_all_feature_names())
    input_map = {key: value for key, value in input_map.items() if key in sso}

    app_state = request.app.state
    if not hasattr(app_state, 'stage1_bundle') or app_state.stage1_bundle is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Stage-1 bundle not mounted")

    # quick deterministic veto
    veto_checker = DeterministicCore()
    veto = veto_checker.pre_flight_veto_check(input_map)
    if veto.vetoed:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Tier0 veto: {veto.reason}")

    # structured validation
    policy_flag = None
    if getattr(app_state.stage1_bundle, 'policy_config', None):
        policy_flag = app_state.stage1_bundle.policy_config.get('fail_on_imputed_core')

    validator = DataValidator(fail_on_imputed_core=policy_flag)
    feature_vec, feature_meta = assemble_feature_vector_from_dict(input_map, app_state.stage1_bundle)
    validation_result = validator.validate_and_store(feature_meta.get("feature_map", input_map))
    if not validation_result.is_usable:
        return {
            "ok": False,
            "missing_cores": validation_result.missing_core_features,
            "warnings": validation_result.warnings,
            "is_usable": validation_result.is_usable,
            "data_quality_score": validation_result.overall_data_quality_score,
        }

    return {"ok": True, "feature_vector_length": len(feature_vec), "metadata": feature_meta}
