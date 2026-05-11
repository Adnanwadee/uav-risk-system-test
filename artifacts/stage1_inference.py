
"""
UAV Risk Stage-1 Inference Module
=================================
Environment-agnostic inference for drone risk classification.

Usage:
    from stage1_inference import RiskPredictor
    
    predictor = RiskPredictor("stage1_ml_bundle.pkl", "stage1_shap_explainer.pkl")
    result = predictor.predict("path/to/scenario.json")
    
    print(result["risk_category"])  # "High Risk"
    print(result["confidence"])     # 0.97
"""

import sys
import os
import json
import warnings
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class RiskPredictor:
    """
    Production-ready drone risk predictor.
    
    Works in any Python 3.9+ environment with standard ML libraries.
    """
    
    def __init__(self, bundle_path: str, explainer_path: Optional[str] = None):
        """
        Initialize predictor.
        
        Args:
            bundle_path: Path to stage1_ml_bundle.pkl
            explainer_path: Path to stage1_shap_explainer.pkl (optional)
        """
        logger.info("Loading Stage-1 Risk Predictor...")
        
        # Load core bundle
        self.bundle = joblib.load(bundle_path)
        self.model = self.bundle['model']
        self.preprocessor = self.bundle['preprocessor']
        self.label_encoder = self.bundle['label_encoder']
        self.feature_names = self.bundle['feature_names']
        self.class_names = self.bundle['class_names']
        
        logger.info(f"Model loaded: {self.bundle['metadata']['model_type']}")
        logger.info(f"Features: {len(self.feature_names)}")
        logger.info(f"Classes: {self.class_names}")
        
        # Load SHAP explainer (optional)
        self.explainer = None
        if explainer_path and os.path.exists(explainer_path):
            self.explainer = joblib.load(explainer_path)
            logger.info("SHAP explainer loaded")
        
        logger.info("Predictor ready!")
    
    def predict(self, data, return_shap: bool = False) -> Dict[str, Any]:
        """
        Predict risk for a scenario.
        
        Args:
            data: Can be:
                - Path to JSON file
                - Dict (already parsed JSON)
                - pd.DataFrame (already preprocessed)
            return_shap: If True, include SHAP values
        
        Returns:
            Dict with prediction results
        """
        # Step 1: Load data
        if isinstance(data, str):
            with open(data, 'r') as f:
                raw_data = json.load(f)
            X_processed = self._preprocess_single(raw_data)
        elif isinstance(data, dict):
            X_processed = self._preprocess_single(data)
        elif isinstance(data, pd.DataFrame):
            X_processed = self.preprocessor.transform(data)
        elif isinstance(data, np.ndarray):
            X_processed = data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Step 2: Predict
        pred_class_idx = self.model.predict(X_processed)[0]
        probabilities = self.model.predict_proba(X_processed)[0]
        
        # Step 3: Build result
        result = {
            "risk_category": self.class_names[pred_class_idx],
            "risk_label": int(pred_class_idx),
            "confidence": float(probabilities[pred_class_idx]),
            "probabilities": {
                self.class_names[i]: float(probabilities[i])
                for i in range(len(self.class_names))
            },
            "timestamp": datetime.now().isoformat(),
        }
        
        # Step 4: SHAP (optional)
        if return_shap and self.explainer is not None:
            shap_vals = self.explainer.shap_values(X_processed)
            # Get top contributing features for predicted class
            shap_for_class = shap_vals[:, :, pred_class_idx][0]
            top_idx = np.argsort(np.abs(shap_for_class))[-5:][::-1]
            
            result["shap_explanation"] = []
            for idx in top_idx:
                result["shap_explanation"].append({
                    "feature": self.feature_names[idx],
                    "value": float(X_processed[0, idx]),
                    "contribution": float(shap_for_class[idx]),
                })
        
        return result
    
    def _preprocess_single(self, raw_data: dict) -> np.ndarray:
        """Preprocess a single JSON scenario."""
        # Flatten JSON
        flat = self._flatten_json(raw_data)
        # Convert to DataFrame
        df = pd.DataFrame([flat])
        # Ensure all required columns exist
        for col in self.preprocessor.feature_names_in_:
            if col not in df.columns:
                df[col] = 0
        df = df[self.preprocessor.feature_names_in_]
        # Transform
        return self.preprocessor.transform(df)
    
    def _flatten_json(self, data: dict, parent_key: str = '', sep: str = '_') -> dict:
        """Flatten nested JSON (simplified version)."""
        import re
        items = {}
        for k, v in data.items():
            clean_k = re.sub(r'[^\w]', '_', str(k)).lower()
            new_key = f"{parent_key}{sep}{clean_k}" if parent_key else clean_k
            if isinstance(v, dict):
                items.update(self._flatten_json(v, new_key, sep=sep))
            elif isinstance(v, list):
                items[f"{new_key}_count"] = len(v)
            else:
                items[new_key] = v
        return items


# Simple test function
def test_predictor():
    """Quick test to verify predictor works."""
    print("Testing RiskPredictor...")
    predictor = RiskPredictor("stage1_ml_bundle.pkl")
    print("✅ Predictor initialized successfully!")
    print(f"   Classes: {predictor.class_names}")
    print(f"   Features: {len(predictor.feature_names)}")
    return predictor


if __name__ == "__main__":
    test_predictor()
