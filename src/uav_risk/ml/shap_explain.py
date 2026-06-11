"""
Module: uav_risk.ml.shap_explain
Purpose: High-performance SHAP explainability engine with dynamic class-aware attribution,
         fortified against static fallback and dimensionality drift.
         Now supports raw feature values for human-readable agent reports.
Dependencies: uav_risk.ml.schemas, uav_risk.ml.feature_defs
Source References: Lundberg & Lee (2017), LightGBM Multiclass Production Standards.
"""

import numpy as np
import shap
import structlog
from typing import List, Dict, Any, Optional

from uav_risk.ml.schemas import FeatureImportance
from uav_risk.ml.feature_defs import get_feature_definition

logger = structlog.get_logger(__name__)


class ShapExplainer:
  
    _cache: Dict[int, shap.TreeExplainer] = {}

    def __init__(self, model: Any, feature_names: List[str]):
        self.feature_names = feature_names
        model_id = id(model)
        
        if model_id in ShapExplainer._cache:
            self.explainer = ShapExplainer._cache[model_id]
        else:
            try:
                self.explainer = shap.TreeExplainer(model)
                ShapExplainer._cache[model_id] = self.explainer
            except Exception as e:
                logger.error("Failed to initialize SHAP TreeExplainer baseline graph", error=str(e))
                self.explainer = None

    def explain(
        self,
        X: np.ndarray,
        top_n: int = 10,
        predicted_class_idx: int = 0,
        class_names: List[str] = None,
        raw_values: Optional[np.ndarray] = None  
    ) -> List[FeatureImportance]:
       
        if self.explainer is None:
            logger.warning("SHAP core uninitialized; returning empty attribution list.")
            return []
            
        try:
            shap_output = self.explainer.shap_values(X)
            
            if isinstance(shap_output, list):
                shap_for_class = np.asarray(shap_output[predicted_class_idx])
            elif getattr(shap_output, 'ndim', 0) == 3:
                shap_for_class = shap_output[0, :, predicted_class_idx]
            elif getattr(shap_output, 'ndim', 0) == 2:
                shap_for_class = shap_output[0]
            else:
                shap_for_class = np.asarray(shap_output).flatten()

            if shap_for_class.ndim > 1:
                shap_for_class = shap_for_class.flatten()

            if len(shap_for_class) != len(self.feature_names):
                logger.error("SHAP dimension mismatch against registry", 
                             shap_len=len(shap_for_class), target_len=len(self.feature_names))
                return []

            abs_values = np.abs(shap_for_class)
            top_indices = np.argsort(abs_values)[-top_n:][::-1]
            
            output_drivers: List[FeatureImportance] = []
            target_class_name = class_names[predicted_class_idx] if class_names else None
            
            raw_flat = None
            if raw_values is not None:
                raw_flat = np.asarray(raw_values).flatten()
            
            for rank_idx, idx in enumerate(top_indices, start=1):
                feat_name = self.feature_names[idx]
                shap_val = float(shap_for_class[idx])
                
                if raw_flat is not None and idx < len(raw_flat):
                    feat_val = float(raw_flat[idx])
                else:
                    feat_val = float(X[0, idx]) if X.ndim > 1 else float(X[idx])
                
                feat_def = get_feature_definition(feat_name)
                
                driver = FeatureImportance(
                    feature_name=feat_name,
                    shap_value=shap_val,
                    feature_value=feat_val,
                    description=feat_def.get("description", feat_name),
                    rank=rank_idx,
                    predicted_class=target_class_name,
                    shap_values_all_classes={
                        class_names[c]: float(shap_output[c][0, idx])
                        for c in range(len(class_names))
                    } if isinstance(shap_output, list) else {}
                )
                output_drivers.append(driver)
                
            return output_drivers
            
        except Exception as e:
            logger.error("SHAP explanation sequence failed due to internal tensor error", error=str(e))
            return []

# =====================================================================
# Architectural Registry Block:
# This file serves as the strict SHAP engine for explainable AI attribution.
# This file depends on: src/uav_risk/ml/schemas.py, src/uav_risk/ml/feature_defs.py
# Files depending on this file: src/uav_risk/ml/inference.py
# =====================================================================
