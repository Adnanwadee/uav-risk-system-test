"""
Stage 1 Inference Engine (V1.9 - Fixed XGBoost Compatibility)
==============================================================
Fixes both:
- use_label_encoder
- gpu_id
"""

from __future__ import annotations
import logging
import numpy as np
import joblib
import xgboost as xgb
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# =============================================================================
# FIXES FOR XGBOOST 1.7.6 COMPATIBILITY WITH OLDER MODELS
# =============================================================================

# Fix 1: Add missing 'use_label_encoder' attribute to XGBClassifier
if not hasattr(xgb.XGBClassifier, 'use_label_encoder'):
    xgb.XGBClassifier.use_label_encoder = property(lambda self: False)

# Fix 2: Add missing 'gpu_id' attribute to XGBModel
if not hasattr(xgb.XGBModel, 'gpu_id'):
    xgb.XGBModel.gpu_id = 0


@dataclass
class MLResult:
    risk_score: float
    risk_category: str
    confidence: float
    drift_detected: bool = False
    drift_score: float = 0.0
    features_exceeding_threshold: int = 0
    top_offending_features: List[Tuple[str, float]] = field(default_factory=list)
    status: str = "OK"
    message: Optional[str] = None


def load_bundle(bundle_path: str = "artifacts/stage1_bundle_v2.pkl") -> Dict[str, Any]:
    bundle = joblib.load(bundle_path)
    logger.info(f"✅ Loaded bundle from {bundle_path}")
    logger.info(f"   Features: {len(bundle.get('feature_names', []))}")
    return bundle


def compute_drift_score(
    feature_vector: np.ndarray,
    training_stats: Dict[str, Dict[str, float]],
    method: str = "max",
    z_threshold: float = 3.0
) -> Tuple[float, bool, int, List[Tuple[str, float]], Dict[str, float]]:
    
    if not training_stats or "error" in training_stats:
        return 0.0, False, 0, [], {}
    
    expected_features = list(training_stats.keys())
    n_features = len(expected_features)
    
    if hasattr(feature_vector, 'ndim'):
        if feature_vector.ndim == 2:
            if feature_vector.shape[0] == 1:
                feature_vector = feature_vector.flatten()
            elif feature_vector.shape[1] == n_features:
                logger.warning(f"Multiple samples ({feature_vector.shape[0]}), using first sample")
                feature_vector = feature_vector[0]
            else:
                return 0.0, False, 0, [], {}
    
    if len(feature_vector) != n_features:
        logger.error(f"Feature count mismatch: {len(feature_vector)} vs {n_features}")
        return 0.0, False, 0, [], {}
    
    z_scores = {}
    z_values_list = []
    features_exceeding_threshold = 0
    offenders = []
    
    for i, feature_name in enumerate(expected_features):
        mean = training_stats[feature_name].get("mean", 0.0)
        std = training_stats[feature_name].get("std", 1.0)
        if std == 0 or np.isnan(std):
            std = 1.0
        
        value = feature_vector[i]
        
        if np.isnan(value):
            z = 0.0
        elif np.isinf(value):
            z = 100.0
        else:
            z = abs(value - mean) / std
        
        z_scores[feature_name] = float(z)
        z_values_list.append(float(z))
        
        if z > z_threshold:
            features_exceeding_threshold += 1
            offenders.append((feature_name, float(z)))
    
    if method == "max":
        drift_score = float(np.max(z_values_list)) if z_values_list else 0.0
        drift_detected = drift_score > z_threshold
    elif method == "count":
        drift_score = float(features_exceeding_threshold)
        drift_detected = drift_score > 0
    else:
        drift_score = float(np.mean(z_values_list)) if z_values_list else 0.0
        drift_detected = drift_score > z_threshold
    
    sorted_features = sorted(z_scores.items(), key=lambda x: x[1], reverse=True)
    top_offending_features = sorted_features[:5]
    
    return drift_score, drift_detected, features_exceeding_threshold, top_offending_features, z_scores


def predict_with_booster(model, X: np.ndarray) -> np.ndarray:
    try:
        return model.predict(X)
    except AttributeError as e:
        if 'gpu_id' in str(e):
            booster = model.get_booster()
            dmatrix = xgb.DMatrix(X)
            return booster.predict(dmatrix)
        raise


def infer_from_bundle(
    feature_vector: np.ndarray,
    bundle: Dict[str, Any],
    drift_method: str = "max",
    drift_z_threshold: float = 3.0
) -> MLResult:
    
    if feature_vector is None:
        return MLResult(risk_score=0.0, risk_category="UNKNOWN", confidence=0.0, status="ERROR", message="feature_vector is None")
    
    if feature_vector.ndim == 1:
        X = feature_vector.reshape(1, -1)
    else:
        X = feature_vector
        if X.shape[0] > 1:
            logger.warning(f"Multiple samples ({X.shape[0]}), using first sample only")
            X = X[0:1]
    
    regressor = bundle["regressor"]
    calibrator = bundle["calibrator"]
    label_encoder = bundle["label_encoder"]
    training_stats = bundle.get("training_stats", {})
    policy = bundle.get("policy", {})
    
    drift_score, drift_detected, features_exceeding, top_offenders, _ = compute_drift_score(
        feature_vector, training_stats, method=drift_method, z_threshold=drift_z_threshold
    )
    
    try:
        risk_score = float(predict_with_booster(regressor, X)[0])
        
        probabilities = calibrator.predict_proba(X)[0]
        class_index = int(np.argmax(probabilities))
        confidence = float(probabilities[class_index])
        risk_category = label_encoder.inverse_transform([class_index])[0]
        
        min_confidence = policy.get("min_confidence_any_decision", 0.55)
        if confidence < min_confidence:
            logger.warning(f"Low confidence: {confidence:.3f} < {min_confidence}")
        
    except Exception as e:
        logger.error(f"ML prediction failed: {e}")
        return MLResult(
            risk_score=0.0,
            risk_category="UNKNOWN",
            confidence=0.0,
            drift_detected=drift_detected,
            drift_score=drift_score,
            features_exceeding_threshold=features_exceeding,
            top_offending_features=top_offenders,
            status="ERROR",
            message=f"Prediction error: {str(e)}"
        )
    
    status = "DRIFT_WARNING" if drift_detected else "OK"
    
    logger.info(f"✅ Inference: risk={risk_score:.3f}, category={risk_category}, confidence={confidence:.3f}, drift={drift_score:.3f}")
    
    return MLResult(
        risk_score=round(risk_score, 3),
        risk_category=risk_category,
        confidence=round(confidence, 3),
        drift_detected=drift_detected,
        drift_score=round(drift_score, 3),
        features_exceeding_threshold=features_exceeding,
        top_offending_features=top_offenders,
        status=status
    )


def predict_risk(
    feature_vector: np.ndarray,
    bundle_path: str = "artifacts/stage1_bundle_v2.pkl",
    drift_method: str = "max"
) -> MLResult:
    bundle = load_bundle(bundle_path)
    return infer_from_bundle(feature_vector, bundle, drift_method=drift_method)