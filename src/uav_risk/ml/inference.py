"""
Module: uav_risk.ml.inference
Purpose: Execution engine for LightGBM, binding feature vector to SHAP attribution,
         with mandatory preprocessing and enhanced result fidelity.
"""

import time
import hashlib
import numpy as np
import pandas as pd
import structlog
from uav_risk.ml.schemas import MLResult, RiskClass, probabilities_to_dict, calculate_risk_score, Stage1Bundle
from uav_risk.ml.shap_explain import ShapExplainer
from uav_risk.ml.raw_schema import get_raw_feature_names

logger = structlog.get_logger(__name__)


def _hash_vector(vec: np.ndarray) -> str:
    """إنشاء بصمة سريعة للمتجه لأغراض التتبع."""
    return hashlib.sha256(vec.tobytes()).hexdigest()[:12]


def _model_frame(bundle: Stage1Bundle, processed_matrix: np.ndarray) -> pd.DataFrame | np.ndarray:
    columns = getattr(bundle.model, "feature_names_in_", None)
    if columns is not None and len(columns) == processed_matrix.shape[1]:
        return pd.DataFrame(processed_matrix, columns=list(columns))
    return processed_matrix


def _build_ml_result(
    bundle: Stage1Bundle,
    processed_vector: np.ndarray,
    start: float,
    compute_shap: bool,
) -> MLResult:
    X_for_model = processed_vector.reshape(1, -1)
    X_model_input = _model_frame(bundle, X_for_model)

    probs_raw = bundle.model.predict_proba(X_model_input)[0]
    probs = probabilities_to_dict(probs_raw, bundle.class_names)
    predicted_idx = int(np.argmax(probs_raw))
    risk_class = RiskClass.from_string(bundle.class_names[predicted_idx])
    score = calculate_risk_score(probs)

    explainer = ShapExplainer(bundle.model, bundle.feature_names)
    top_features = []
    if compute_shap:
        try:
            top_features = explainer.explain(
                X_for_model,
                predicted_class_idx=predicted_idx,
                class_names=bundle.class_names,
                raw_values=processed_vector,
            )
        except Exception as exc:
            logger.exception("SHAP explanation failed; returning empty top_features", exc_info=exc)
            top_features = []

    return MLResult(
        risk_class=risk_class,
        risk_score=score,
        confidence=float(probs_raw[predicted_idx]),
        probabilities=probs,
        top_features=top_features,
        processing_time_ms=(time.perf_counter() - start) * 1000,
        model_version=bundle.get_model_version(),
        feature_vector_hash=_hash_vector(processed_vector),
    )


def predict_processed_vector(
    bundle: Stage1Bundle,
    processed_vector: np.ndarray,
    compute_shap: bool = True,
) -> MLResult:
    """Explicit test/diagnostic path for vectors already in processed 198-feature space."""
    start = time.perf_counter()
    processed_vector = np.asarray(processed_vector, dtype=np.float64).reshape(-1)
    if processed_vector.shape != (len(bundle.feature_names),):
        raise ValueError(
            f"Processed vector shape mismatch: expected ({len(bundle.feature_names)},), got {processed_vector.shape}"
        )
    return _build_ml_result(bundle, processed_vector, start, compute_shap)


def run_stage1_inference(bundle: Stage1Bundle, feature_vector: np.ndarray, feature_names=None, compute_shap: bool = True) -> MLResult:
    """
    Run production Stage-1 inference from the raw 197-feature serving contract.

    The input vector must be ordered as bundle.preprocessor.feature_names_in_. It is
    transformed by the fitted ColumnTransformer before LightGBM prediction.
    """
    start = time.perf_counter()
    raw_names = get_raw_feature_names(bundle)
    raw_vector = np.asarray(feature_vector, dtype=object).reshape(-1)
    if raw_vector.shape != (len(raw_names),):
        raise ValueError(f"Raw vector shape mismatch: expected ({len(raw_names)},), got {raw_vector.shape}")

    raw_frame = pd.DataFrame([raw_vector.tolist()], columns=raw_names)
    processed = bundle.preprocessor.transform(raw_frame)
    processed_vector = np.asarray(processed, dtype=np.float64).reshape(-1)
    if processed_vector.shape != (len(bundle.feature_names),):
        raise ValueError(
            f"Processed vector shape mismatch: expected ({len(bundle.feature_names)},), got {processed_vector.shape}"
        )

    return _build_ml_result(bundle, processed_vector, start, compute_shap)
