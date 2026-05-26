"""
Feature Router (Gate 3 - Vectorization & Context Routing)
Maps the validated 198-feature dictionary into a strictly ordered NumPy array for ML.
Routes features into semantic categories for the ReAct Agent's context.
"""

import logging
import numpy as np
from typing import Dict, Tuple, List, Any
from uav_risk.ml.feature_defs import get_all_feature_names, get_features_by_category

logger = logging.getLogger(__name__)

class FeatureRouter:
    """تأكيد مصفوفة الاستنتاج النهائي وفصل الميزات دلالياً لصالح الوكيل الذكي."""
    
    def __init__(self) -> None:
        self.authoritative_names = get_all_feature_names()
        self._index_map = {name: idx for idx, name in enumerate(self.authoritative_names)}
        logger.info(f"FeatureRouter Synced with SSoT. Matrix Dimensions Locked at: ({len(self._index_map)},)")

    def route_to_vector(self, final_dag_features: Dict[str, float]) -> np.ndarray:
        """يحول الخريطة الكاملة الناتجة من الـ DAG إلى متجهة الاستنتاج الرقمي LightGBM."""
        vector = np.zeros(len(self._index_map), dtype=np.float64)
        
        for name, index in self._index_map.items():
            if name not in final_dag_features:
                raise KeyError(f"FeatureRouter Integrity Failure: Expected feature '{name}' missing from DAG pipeline outputs.")
                
            val = final_dag_features[name]
            if np.isnan(val) or np.isinf(val):
                raise ValueError(f"FeatureRouter Finite Contamination: Infinite or NaN vector element detected at '{name}': {val}")
                
            vector[index] = float(val)
            
        return vector

    def route_to_context_pool(self, final_dag_features: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """يصنف الميزات هندسياً في فئات دلالية محكمة لتمكين محرك تفكير الوكيل ReAct."""
        categories = ["aerodynamic", "environmental", "battery", "mission", "gps", "comms", "operator"]
        context_pool: Dict[str, Dict[str, float]] = {cat: {} for cat in categories}
        context_pool["other"] = {}
        
        routed_keys = set()
        for cat in categories:
            cat_features = get_features_by_category(cat)
            for feat in cat_features:
                if feat in final_dag_features:
                    context_pool[cat][feat] = float(final_dag_features[feat])
                    routed_keys.add(feat)
                    
        # عزل وضبط بقية الميزات المشتقة أو الإحصائية تحت لواء "other" لمنع ضياع الأدلة الجنائية
        for key, value in final_dag_features.items():
            if key not in routed_keys:
                context_pool["other"][key] = float(value)
                
        return context_pool

    def validate_vector(self, vector: np.ndarray) -> Tuple[bool, List[str]]:
        issues = []
        if vector.shape != (len(self._index_map),):
            issues.append(f"Invalid shape: expected ({len(self._index_map)},), got {vector.shape}")
        if np.any(np.isnan(vector)):
            issues.append("Vector contains NaN values")
        if np.any(np.isinf(vector)):
            issues.append("Vector contains Inf values")
        if vector.dtype != np.float64:
            issues.append(f"Invalid dtype: expected float64, got {vector.dtype}")
            
        return len(issues) == 0, issues

# =====================================================================
# Architectural Registry Block:
# This file serves as the strict mapping array vectorizer for LightGBM inference.
# This file depends on: src/uav_risk/ml/feature_defs.py
# Files depending on this file: src/uav_risk/stage2/pipeline.py
# =====================================================================