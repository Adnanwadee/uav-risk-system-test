from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.encoders import jsonable_encoder

from uav_risk.api.dependencies import get_profile_storage_root, get_stage1_bundle
from uav_risk.api.schemas import AssessmentRequest
from uav_risk.api.storage import LocalProfileStorage
from uav_risk.core.contracts import AssessmentCoreInput, DroneProfileRaw
from uav_risk.core.data_validator import run_structural_hard_veto
from uav_risk.ml.inference import run_stage1_inference
from uav_risk.ml.loader import ModelLoadError, assemble_raw_feature_vector
from uav_risk.ml.schemas import Stage1Bundle

router = APIRouter(prefix="/users/{user_id}/profiles/{profile_id}/assessments", tags=["assessments"])


def _issue_payload(validation_result) -> list[dict[str, object]]:
    return [
        {
            "code": issue.code,
            "field": issue.field,
            "message": issue.message,
            "details": issue.details,
        }
        for issue in validation_result.issues
    ]


def _blocked(user_id: str, profile_id: str, issues: list[dict[str, object]]) -> dict[str, object]:
    return {"status": "blocked", "user_id": user_id, "profile_id": profile_id, "issues": issues, "ml": None}


@router.post("")
def create_assessment(
    user_id: str,
    profile_id: str,
    request_body: AssessmentRequest,
    bundle: Stage1Bundle = Depends(get_stage1_bundle),
) -> dict[str, object]:
    stored_profile = LocalProfileStorage(get_profile_storage_root()).get_profile(user_id, profile_id)
    if stored_profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")

    profile = DroneProfileRaw.model_validate(stored_profile)
    assessment = AssessmentCoreInput(
        user_id=user_id,
        profile_id=profile_id,
        drone_profile=profile,
        scenario=request_body.scenario,
        secondary_overrides=request_body.secondary_overrides,
    )
    veto = run_structural_hard_veto(assessment)
    if not veto.passed:
        return _blocked(user_id, profile_id, _issue_payload(veto))

    override_values = request_body.secondary_overrides.values
    try:
        raw_vector, metadata = assemble_raw_feature_vector(
            profile,
            request_body.scenario,
            overrides=override_values,
            bundle=bundle,
        )
    except ModelLoadError as exc:
        return _blocked(
            user_id,
            profile_id,
            [{"code": "RAW_ASSEMBLY_FAILED", "field": None, "message": str(exc), "details": {}}],
        )

    try:
        ml_result = run_stage1_inference(bundle, raw_vector, compute_shap=True)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="ML inference failed") from exc

    raw_feature_map = metadata["raw_feature_map"]
    applied_overrides = {key: raw_feature_map[key] for key in override_values if key in raw_feature_map}
    response = {
        "status": "completed",
        "user_id": user_id,
        "profile_id": profile_id,
        "ml": {
            "predicted_class": ml_result.risk_class.value,
            "probabilities": ml_result.probabilities,
        },
        "shap": {"top_features": [feature.to_dict() for feature in ml_result.top_features]},
        "raw_feature_count": metadata["raw_feature_vector_length"],
        "processed_feature_count": metadata["processed_feature_vector_length"],
        "warnings": [],
        "operator_notes": request_body.operator_notes,
        "raw_feature_diagnostics": {
            "applied_secondary_overrides": applied_overrides,
        },
    }
    return jsonable_encoder(response)
