"""
SHAP Explainability Module - Feature importance explanation for Stage1 model.

This module provides:
- SHAP explainer creation from loaded LightGBM model
- Feature importance extraction for predictions
- Human-readable explanations for top features using feature_defs

SHAP (SHapley Additive exPlanations) helps understand:
- Which features drove the prediction
- Whether each feature increased or decreased risk
- How confident we should be in the explanation
"""

import logging
import warnings
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import shap

from .schemas import FeatureImportance, MLResult
from .loader import ModelBundle
from .feature_defs import get_feature_definition

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP explainer wrapper for LightGBM model.
    
    Creates TreeExplainer from loaded model and provides:
    - Feature importance for single predictions
    - Top contributing features with human-readable descriptions
    """
    
    def __init__(self, bundle: ModelBundle, use_background_data: Optional[np.ndarray] = None):
        """
        Initialize SHAP explainer from model bundle.
        
        Args:
            bundle: Loaded ModelBundle with LightGBM model
            use_background_data: Optional background data for explainer (n_samples, n_features).
                                If None, uses a small subset of training data.
        """
        self.bundle = bundle
        self.model = bundle.model
        self.feature_names = bundle.feature_names
        self.class_names = bundle.class_names
        
        logger.info("Creating SHAP TreeExplainer...")
        
        # Suppress SHAP's verbose warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            
            if use_background_data is not None:
                # Use provided background data
                self.explainer = shap.TreeExplainer(
                    self.model,
                    data=use_background_data,
                    feature_names=self.feature_names,
                    model_output='raw'  # ✅ Fixed: use 'raw' instead of 'probability'
                )
                logger.info(f"SHAP explainer created with {len(use_background_data)} background samples")
            else:
                # Create explainer without background
                self.explainer = shap.TreeExplainer(
                    self.model,
                    feature_names=self.feature_names,
                    model_output='raw'  # ✅ Fixed: use 'raw' instead of 'probability'
                )
                logger.info("SHAP explainer created without background data")
        
        logger.info("✅ SHAP explainer ready")
    
    def explain_prediction(
        self,
        feature_vector: np.ndarray,
        predicted_class_idx: int,
        top_k: int = 10
    ) -> List[FeatureImportance]:
        """
        Explain a single prediction using SHAP.
        
        Args:
            feature_vector: Preprocessed feature vector (1, n_features)
            predicted_class_idx: Index of predicted class (0=High, 1=Low, 2=Medium)
            top_k: Number of top features to return
        
        Returns:
            List of FeatureImportance objects sorted by |SHAP| descending
        """
        try:
            # Get SHAP values for the prediction
            shap_values = self.explainer.shap_values(feature_vector)
            
            # Extract SHAP for predicted class
            if isinstance(shap_values, list):
                # Multi-class output
                shap_for_class = shap_values[predicted_class_idx][0]
            elif shap_values.ndim == 3:
                # Shape: (n_samples, n_features, n_classes)
                shap_for_class = shap_values[0, :, predicted_class_idx]
            else:
                # Binary or single class
                shap_for_class = shap_values[0, :] if shap_values.ndim > 1 else shap_values
            
            # Get top k indices by absolute SHAP value
            top_indices = np.argsort(np.abs(shap_for_class))[-top_k:][::-1]
            
            top_features = []
            for idx in top_indices:
                if idx < len(self.feature_names):
                    feature_name = self.feature_names[idx]
                    
                    # Get feature description from feature_defs if available
                    feature_def = get_feature_definition(feature_name)
                    description = feature_def.get("description") if feature_def else None
                    
                    top_features.append(FeatureImportance(
                        feature_name=feature_name,
                        shap_value=float(shap_for_class[idx]),
                        feature_value=float(feature_vector[0, idx]) if feature_vector.ndim > 1 else float(feature_vector[idx]),
                        description=description
                    ))
            
            return top_features
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return []
    
    def get_shap_summary(
        self,
        feature_vector: np.ndarray,
        predicted_class_idx: int
    ) -> Dict[str, Any]:
        """
        Get comprehensive SHAP summary for a prediction.
        
        Args:
            feature_vector: Preprocessed feature vector (1, n_features)
            predicted_class_idx: Index of predicted class
        
        Returns:
            Dictionary with SHAP summary including:
            - positive_contributors: Features pushing risk UP
            - negative_contributors: Features pushing risk DOWN
            - top_push_features: Most influential features
        """
        top_features = self.explain_prediction(feature_vector, predicted_class_idx, top_k=15)
        
        positive = [f for f in top_features if f.shap_value > 0]
        negative = [f for f in top_features if f.shap_value < 0]
        
        return {
            "top_features": [f.to_dict() for f in top_features],
            "positive_contributors": [f.to_dict() for f in positive[:5]],
            "negative_contributors": [f.to_dict() for f in negative[:5]],
            "total_features_explained": len(top_features)
        }
    
    def explain_ml_result(
        self,
        ml_result: MLResult,
        feature_vector: np.ndarray
    ) -> MLResult:
        """
        Augment MLResult with SHAP explanations.
        
        Args:
            ml_result: Existing MLResult from inference
            feature_vector: Preprocessed feature vector
        
        Returns:
            Same MLResult with top_features populated
        """
        risk_class_idx = ["High Risk", "Low Risk", "Medium Risk"].index(ml_result.risk_class.value)
        top_features = self.explain_prediction(feature_vector, risk_class_idx, top_k=10)
        
        ml_result.top_features = top_features
        return ml_result


def create_explainer_from_bundle(
    bundle: ModelBundle,
    sample_data: Optional[np.ndarray] = None
) -> SHAPExplainer:
    """
    Convenience function to create SHAP explainer from bundle.
    
    Args:
        bundle: Loaded ModelBundle
        sample_data: Optional sample data for background (helps with consistency)
    
    Returns:
        SHAPExplainer instance
    """
    return SHAPExplainer(bundle, use_background_data=sample_data)


def get_feature_importance_description(
    feature_importance: FeatureImportance
) -> str:
    """
    Generate human-readable description of a feature's impact.
    
    Args:
        feature_importance: FeatureImportance object
    
    Returns:
        Human-readable explanation
    """
    feature_def = get_feature_definition(feature_importance.feature_name)
    description = feature_def.get("description") if feature_def else feature_importance.feature_name
    
    impact = "increases" if feature_importance.shap_value > 0 else "decreases"
    magnitude = abs(feature_importance.shap_value)
    
    if magnitude > 0.1:
        strength = "strongly"
    elif magnitude > 0.05:
        strength = "moderately"
    else:
        strength = "slightly"
    
    return f"{description} {strength} {impact} risk (contribution: {feature_importance.shap_value:.3f})"


# ============================================================
# Quick Test
# ============================================================

def test_shap_explainer():
    """Quick test to verify SHAP explainer works."""
    from .loader import load_stage1_bundle_from_artifacts
    from .inference import run_inference
    
    print("=" * 60)
    print("Testing SHAP Explainer")
    print("=" * 60)
    
    # Load bundle
    bundle = load_stage1_bundle_from_artifacts("artifacts")
    
    # Create minimal test telemetry
    test_telemetry = {
        'uav_mass_kg': 1.5,
        'uav_battery_wh': 100,
        'uav_max_speed_mps': 20,
        'environment_weather_wind_mps': 5.0,
        'uav_energy_source': 'battery',
        'mission_pattern': 'grid',
        'controls_mode': 'continuous',
        'controls_actions_first': 'hold',
        'swarm_roles_first': 'single',
    }
    
    # Run inference without SHAP first
    result = run_inference(bundle, test_telemetry, return_shap=False)
    print(f"\n📊 Inference Result: {result.risk_class.value} (confidence: {result.confidence:.3f})")
    
    # Get feature vector from inference (we need to recreate preprocessing)
    # For test, we'll create explainer and get SHAP values manually
    from .inference import _ensure_all_features_present
    
    df = pd.DataFrame([test_telemetry])
    preprocessor_features = bundle.preprocessor.feature_names_in_
    df = _ensure_all_features_present(df, preprocessor_features, fill_value=0)
    X_processed = bundle.preprocessor.transform(df)
    
    # Create SHAP explainer
    explainer = SHAPExplainer(bundle)
    
    # Get top features
    risk_class_idx = ["High Risk", "Low Risk", "Medium Risk"].index(result.risk_class.value)
    top_features = explainer.explain_prediction(X_processed, risk_class_idx, top_k=10)
    
    print(f"\n🔍 Top 10 Features Influencing Decision:")
    print("-" * 60)
    for i, feat in enumerate(top_features[:10], 1):
        impact = "🔴 Increases" if feat.shap_value > 0 else "🟢 Decreases"
        desc = get_feature_definition(feat.feature_name)
        description = desc.get("description", feat.feature_name) if desc else feat.feature_name
        print(f"{i:2}. {feat.feature_name}")
        print(f"    {impact} risk by {abs(feat.shap_value):.4f}")
        print(f"    Value: {feat.feature_value:.2f}")
        print(f"    {description[:80]}...")
        print()
    
    # Get summary
    summary = explainer.get_shap_summary(X_processed, risk_class_idx)
    print(f"\n📈 SHAP Summary:")
    print(f"   Positive contributors (push UP): {len(summary['positive_contributors'])}")
    print(f"   Negative contributors (push DOWN): {len(summary['negative_contributors'])}")
    
    print("\n✅ SHAP explainer test complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_shap_explainer()