"""
ML Inference Module - Run predictions using loaded Stage1 model.

This module handles:
- Preprocessing raw telemetry data
- Running model inference
- Extracting SHAP explanations (top features)
- Returning structured MLResult

IMPORTANT: This module expects RAW flattened telemetry data.
All required features must be present or filled with defaults.
"""

import logging
import hashlib
import signal
from contextlib import contextmanager
from typing import Dict, Any, List, Optional, Union

import numpy as np
import pandas as pd

from .schemas import MLResult, RiskClass, FeatureImportance
from .loader import ModelBundle

logger = logging.getLogger(__name__)


class InferenceError(Exception):
    """Raised when inference fails."""
    pass


class TimeoutError(Exception):
    """Raised when inference times out."""
    pass


@contextmanager
def timeout(seconds: int):
    """
    Context manager for inference timeout.
    
    Args:
        seconds: Timeout in seconds
    
    Raises:
        TimeoutError: If operation exceeds timeout
    """
    def handler(signum, frame):
        raise TimeoutError(f"Inference timed out after {seconds}s")
    
    # Set the signal handler
    original_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


def _validate_input_values(df: pd.DataFrame) -> None:
    """
    Check for NaN/Inf values in input features (numeric columns only).
    
    Args:
        df: Input DataFrame
    
    Raises:
        InferenceError: If NaN or Inf values are found in numeric columns
    """
    # Select only numeric columns for NaN/Inf check
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.isnull().any().any():
        nan_cols = numeric_df.columns[numeric_df.isnull().any()].tolist()
        raise InferenceError(f"Input contains NaN values in numeric columns: {nan_cols}")
    
    if np.isinf(numeric_df.values).any():
        inf_cols = numeric_df.columns[np.isinf(numeric_df).any()].tolist()
        raise InferenceError(f"Input contains infinite values in numeric columns: {inf_cols}")
    
    
def _ensure_all_features_present(
    df: pd.DataFrame,
    expected_features: List[str],
    fill_value: float = 0.0
) -> pd.DataFrame:
    """
    Ensure all expected features are present (optimized version).
    Missing columns are filled with default value.
    
    Args:
        df: Input DataFrame (single row typically)
        expected_features: List of feature names the model expects
        fill_value: Value to fill missing columns (default: 0.0)
    
    Returns:
        DataFrame with all expected features in correct order
    """
    missing = set(expected_features) - set(df.columns)
    if missing:
        logger.warning(f"Missing {len(missing)} features, filling with {fill_value}")
        # Add all missing columns at once using assign
        df = df.assign(**{feat: fill_value for feat in missing})
    
    # Ensure correct column order
    return df.reindex(columns=expected_features)


def _validate_feature_shapes(df: pd.DataFrame, expected_features: List[str]) -> None:
    """
    Validate that DataFrame has correct shape.
    
    Args:
        df: Input DataFrame
        expected_features: List of expected feature names
    
    Raises:
        InferenceError: If shape validation fails
    """
    if df.shape[0] != 1:
        raise InferenceError(f"Expected 1 row for inference, got {df.shape[0]}")
    
    if df.shape[1] != len(expected_features):
        raise InferenceError(
            f"Feature count mismatch: got {df.shape[1]}, expected {len(expected_features)}"
        )


def _compute_feature_vector_hash(feature_vector: np.ndarray) -> str:
    """
    Compute SHA256 hash of feature vector for audit trail.
    
    Args:
        feature_vector: Numpy array of features
    
    Returns:
        SHA256 hash string (first 16 characters)
    """
    vector_bytes = feature_vector.tobytes()
    return hashlib.sha256(vector_bytes).hexdigest()[:16]
def _get_default_value_for_column(col_name: str, preprocessor) -> Any:
    """
    Determine appropriate default value for a column based on its transformer.
    
    Args:
        col_name: Name of the column
        preprocessor: ColumnTransformer from the bundle
    
    Returns:
        Appropriate default value (0 for numeric, 'unknown' for categorical)
    """
    # Try to find which transformer handles this column
    for name, transformer, columns in preprocessor.transformers_:
        if columns and col_name in columns:
            if name == 'onehot':
                return 'unknown'  # Categorical column
            elif name == 'scaler':
                return 0.0  # Numeric column
            elif name == 'passthrough':
                return 0  # Binary/passthrough
    return 0  # Default fallback

def _extract_top_shap_features(
    shap_values: np.ndarray,
    feature_names: List[str],
    feature_vector: np.ndarray,
    top_k: int = 10,
    predicted_class_idx: int = 0,
    class_names: Optional[List[str]] = None
) -> List[FeatureImportance]:
    """
    Extract top K features by |SHAP| contribution for predicted class.
    
    Args:
        shap_values: SHAP values (various shapes: 2D or 3D)
        feature_names: List of feature names
        feature_vector: Raw feature vector (n_samples, n_features)
        top_k: Number of top features to return
        predicted_class_idx: Index of predicted class to extract SHAP for
        class_names: Optional list of class names for logging
    
    Returns:
        List of FeatureImportance objects sorted by |SHAP| descending
    """
    try:
        # Handle different SHAP output shapes robustly
        if shap_values.ndim == 3:
            # Multi-class: (n_samples, n_features, n_classes)
            shap_for_class = shap_values[0, :, predicted_class_idx]
        elif shap_values.ndim == 2:
            # Binary or reshaped output
            shap_for_class = shap_values[0, :]
        else:
            shap_for_class = np.asarray(shap_values).flatten()
        
        # Verify length matches feature count
        if len(shap_for_class) != len(feature_names):
            logger.warning(
                f"SHAP length mismatch: {len(shap_for_class)} vs {len(feature_names)}. "
                f"Returning empty list."
            )
            return []
        
        # Get top k indices by absolute SHAP value
        top_indices = np.argsort(np.abs(shap_for_class))[-top_k:][::-1]
        
        top_features = []
        for idx in top_indices:
            if idx < len(feature_names):
                feature_value = float(feature_vector[0, idx]) if feature_vector.ndim > 1 else float(feature_vector[idx])
                top_features.append(FeatureImportance(
                    feature_name=feature_names[idx],
                    shap_value=float(shap_for_class[idx]),
                    feature_value=feature_value
                ))
        
        return top_features
    except Exception as e:
        logger.warning(f"Failed to extract SHAP features: {e}")
        return []


def run_inference(
    bundle: ModelBundle,
    flat_telemetry: Dict[str, Any],
    return_shap: bool = True,
    shap_explainer: Optional[Any] = None,
    top_k_features: int = 10,
    inference_timeout_seconds: Optional[int] = 30
) -> MLResult:
    """
    Run inference on a single flight scenario.
    
    Args:
        bundle: Loaded ModelBundle from loader
        flat_telemetry: Flattened telemetry dictionary from input_contract
        return_shap: Whether to compute and return SHAP explanations
        shap_explainer: Optional pre-loaded SHAP explainer (for performance)
        top_k_features: Number of top features to return in SHAP explanation
        inference_timeout_seconds: Timeout in seconds (None = no timeout)
    
    Returns:
        MLResult object with prediction and explanations
    
    Raises:
        InferenceError: If inference fails at any step
        TimeoutError: If inference exceeds timeout
    """
    logger.debug("Starting inference...")
    
    def _run():
        # Step 1: Convert telemetry to DataFrame
        df = pd.DataFrame([flat_telemetry])
        logger.debug(f"Raw DataFrame shape: {df.shape}")
        
        # Step 2: Validate input values (NaN/Inf)
        _validate_input_values(df)
        logger.debug("Input validation passed")
        
        # Step 3: Ensure all features expected by preprocessor are present
        preprocessor_features = bundle.preprocessor.feature_names_in_
        df = _ensure_all_features_present(df, preprocessor_features, fill_value=0)
        logger.debug(f"After filling missing features: {df.shape}")
        
        # Step 4: Preprocess
        try:
            X_processed = bundle.preprocessor.transform(df)
            logger.debug(f"Preprocessed feature vector shape: {X_processed.shape}")
        except Exception as e:
            raise InferenceError(f"Preprocessing failed: {e}")
        
        # Step 5: Validate shape
        _validate_feature_shapes(pd.DataFrame(X_processed), bundle.feature_names)
        
        # Step 6: Run prediction
        try:
            probabilities = bundle.model.predict_proba(X_processed)[0]
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = bundle.class_names[predicted_class_idx]
            confidence = float(probabilities[predicted_class_idx])
        except Exception as e:
            raise InferenceError(f"Model prediction failed: {e}")
        
        # Step 7: Calculate risk score
        from .schemas import calculate_risk_score
        risk_score = calculate_risk_score({
            bundle.class_names[i]: float(probabilities[i])
            for i in range(len(bundle.class_names))
        })
        
        # Step 8: Compute feature vector hash for audit
        feature_vector_hash = _compute_feature_vector_hash(X_processed)
        
        # Step 9: Extract SHAP explanations (if requested)
        top_features = []
        shap_expected_values = None
        
        if return_shap and shap_explainer is not None:
            try:
                shap_values = shap_explainer.shap_values(X_processed)
                
                # Extract expected values with proper alignment to class order
                if hasattr(shap_explainer, 'expected_value'):
                    expected = shap_explainer.expected_value
                    if isinstance(expected, (list, np.ndarray)):
                        # Ensure alignment with bundle.class_names
                        shap_expected_values = [float(v) for v in expected[:len(bundle.class_names)]]
                    else:
                        # Single value: apply to predicted class only
                        shap_expected_values = [0.0] * len(bundle.class_names)
                        shap_expected_values[predicted_class_idx] = float(expected)
                
                # Extract top features
                top_features = _extract_top_shap_features(
                    shap_values=shap_values,
                    feature_names=bundle.feature_names,
                    feature_vector=X_processed,
                    top_k=top_k_features,
                    predicted_class_idx=predicted_class_idx,
                    class_names=bundle.class_names
                )
                logger.debug(f"Extracted {len(top_features)} top SHAP features")
            except Exception as e:
                logger.warning(f"SHAP extraction failed (continuing without): {e}")
        
        # Step 10: Build probabilities dictionary
        probabilities_dict = {
            bundle.class_names[i]: float(probabilities[i])
            for i in range(len(bundle.class_names))
        }
        
        # Step 11: Create and return MLResult
        result = MLResult(
            risk_class=RiskClass.from_string(predicted_class),
            risk_score=risk_score,
            confidence=confidence,
            probabilities=probabilities_dict,
            top_features=top_features,
            model_version=bundle.get_model_version(),
            mapping_version=bundle.metadata.get("feature_mapping_version", "1.0"),
            feature_vector_hash=feature_vector_hash,
            shap_expected_values=shap_expected_values
        )
        
        logger.info(f"Inference complete: {predicted_class} (confidence: {confidence:.3f})")
        return result
    
    # Apply timeout if specified
    if inference_timeout_seconds:
        with timeout(inference_timeout_seconds):
            return _run()
    else:
        return _run()


def run_inference_with_bundle_path(
    bundle_path: str,
    flat_telemetry: Dict[str, Any],
    model_card_path: Optional[str] = None,
    feature_mapping_path: Optional[str] = None,
    return_shap: bool = True,
    shap_explainer_path: Optional[str] = None,
    inference_timeout_seconds: Optional[int] = 30
) -> MLResult:
    """
    Convenience function to load bundle and run inference in one call.
    
    Args:
        bundle_path: Path to stage1_production_bundle.pkl
        flat_telemetry: Flattened telemetry dictionary
        model_card_path: Optional path to model_card.json
        feature_mapping_path: Optional path to stage1_feature_mapping.json
        return_shap: Whether to compute SHAP explanations
        shap_explainer_path: Optional path to pre-saved SHAP explainer
        inference_timeout_seconds: Timeout in seconds
    
    Returns:
        MLResult object
    """
    from .loader import load_stage1_bundle
    
    # Load bundle
    bundle = load_stage1_bundle(
        bundle_path=bundle_path,
        model_card_path=model_card_path,
        feature_mapping_path=feature_mapping_path
    )
    
    # Load SHAP explainer if requested
    shap_explainer = None
    if return_shap and shap_explainer_path:
        import joblib
        try:
            shap_explainer = joblib.load(shap_explainer_path)
            logger.info("SHAP explainer loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SHAP explainer: {e}")
    
    # Run inference
    return run_inference(
        bundle=bundle,
        flat_telemetry=flat_telemetry,
        return_shap=return_shap,
        shap_explainer=shap_explainer,
        inference_timeout_seconds=inference_timeout_seconds
    )


# Optional: Quick test function
def test_inference():
    """Quick test to verify inference works."""
    import json
    from pathlib import Path
    
    # Check if we have a test scenario
    test_scenario_path = Path("artifacts/test_scenario.json")
    if not test_scenario_path.exists():
        print("No test scenario found. Skipping inference test.")
        print("Creating minimal test with sample data...")
        
        # Create minimal test telemetry
        test_telemetry = {
            'uav_mass_kg': 1.5,
            'uav_battery_wh': 100,
            'uav_max_speed_mps': 20,
            'environment_weather_wind_mps': 5.0,
        }
        
        try:
            from .loader import load_stage1_bundle_from_artifacts
            bundle = load_stage1_bundle_from_artifacts("artifacts")
            result = run_inference(bundle, test_telemetry, return_shap=False)
            print(f"✅ Inference test passed!")
            print(f"   Risk: {result.risk_class.value}")
            print(f"   Score: {result.risk_score}")
            print(f"   Confidence: {result.confidence}")
        except Exception as e:
            print(f"❌ Inference test failed: {e}")
        return
    
    try:
        with open(test_scenario_path, 'r') as f:
            scenario = json.load(f)
        
        from .loader import load_stage1_bundle_from_artifacts
        from ..stage2.input_contract import flatten_telemetry
        
        bundle = load_stage1_bundle_from_artifacts("artifacts")
        flat_telemetry = flatten_telemetry(scenario)
        
        result = run_inference(bundle, flat_telemetry, return_shap=False)
        
        print(f"✅ Inference test passed!")
        print(f"   Risk: {result.risk_class.value}")
        print(f"   Score: {result.risk_score}")
        print(f"   Confidence: {result.confidence}")
    except Exception as e:
        print(f"❌ Inference test failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_inference()