"""
Feature Router (Gate 3 - Vectorization & Context Routing)
Maps the validated 198-feature dictionary into a strictly ordered NumPy array for ML.
Routes features into semantic categories for the ReAct Agent's context.
"""

import logging
import numpy as np
from typing import Dict, Tuple, List, Any

# استدعاء الدوال المساعدة من الدستور
from uav_risk.ml.feature_defs import get_safe_value, get_features_by_category

logger = logging.getLogger(__name__)

class FeatureRouter:
    """
    Translates dictionary-based validated features into the exact mathematical 
    vector expected by the LightGBM model using stage1_feature_mapping.json.
    """
    
    def __init__(self, feature_defs: Dict[str, Any], feature_mapping: Any):
        """
        يستقبل تعريفات الميزات وملف الـ JSON الخاص بالترتيب (mapping).
        """
        self.feature_defs = feature_defs
        self._index_map = {}
        
        # ==========================================
        # 🧠 Precise JSON Parser based on actual artifact structure
        # ==========================================
        if isinstance(feature_mapping, dict) and "feature_names" in feature_mapping:
            # هذا هو الهيكل الحقيقي للملف!
            features_list = feature_mapping["feature_names"]
            self._index_map = {str(name): idx for idx, name in enumerate(features_list)}
            
        elif isinstance(feature_mapping, list):
            self._index_map = {str(name): idx for idx, name in enumerate(feature_mapping)}
            
        elif isinstance(feature_mapping, dict):
            if len(feature_mapping) > 0:
                first_key = next(iter(feature_mapping.keys()))
                if str(first_key).isdigit():
                    self._index_map = {str(v): int(k) for k, v in feature_mapping.items()}
                else:
                    self._index_map = {str(k): int(v) for k, v in feature_mapping.items()}
        # ==========================================
        
        # 1. Ensure mapping length consistency (use dynamic size derived from mapping)
        n_features = len(self._index_map)
        if n_features == 0:
            raise ValueError(f"CRITICAL: Feature mapping empty. Got {n_features} entries")

        # 2. Validate index range is contiguous from 0..n-1
        indices = list(self._index_map.values())
        if min(indices) != 0 or max(indices) != (n_features - 1) or len(set(indices)) != n_features:
            raise ValueError("CRITICAL: Feature mapping indices are corrupted! Indices must be contiguous 0..n-1.")

        logger.info("FeatureRouter initialized successfully.", total_dimensions=n_features)
    def route_to_vector(self, validated_features: Dict[str, Any]) -> np.ndarray:
        """
        يحول القاموس النظيف إلى مصفوفة رياضية بالترتيب الصارم.
        مصفح ضد أخطاء أنواع البيانات ومشاكل الفهارس.
        """
        # 1. إنشاء مصفوفة أصفار بحجم ديناميكي ونوع float64
        vector = np.zeros(len(self._index_map), dtype=np.float64)
        
        # 2. تعبئة المصفوفة بناءً على الـ mapping الصحيح
        for name, index in self._index_map.items():
            raw_value = validated_features.get(name)
            try:
                float_val = float(raw_value)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"FeatureRouter expected numeric value for '{name}', got {raw_value!r}") from exc

            if np.isnan(float_val) or np.isinf(float_val):
                raise ValueError(f"FeatureRouter received non-finite value for '{name}': {raw_value!r}")

            vector[index] = float_val
            
        # 3. الفحص النهائي الصارم جداً قبل التسليم للنموذج
        if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
            bad_indices = np.where(np.isnan(vector) | np.isinf(vector))[0]
            bad_names = [name for name, idx in self._index_map.items() if idx in bad_indices]
            raise RuntimeError(f"CRITICAL: route_to_vector produced NaN/Inf in features: {bad_names}")
            
        return vector

    def route_to_context_pool(self, validated_features: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        يصنف الميزات في فئات منطقية لتسهيل قراءتها وفحصها من قبل الوكيل (ReAct Agent).
        """
        categories = ["aerodynamic", "environmental", "battery", "mission", "gps", "comms", "operator"]
        context_pool: Dict[str, Dict[str, float]] = {cat: {} for cat in categories}
        context_pool["other"] = {} # شبكة أمان للميزات التي لم تُصنف
        
        routed_keys = set()
        
        # تصنيف الميزات الأساسية
        for cat in categories:
            cat_features = get_features_by_category(cat)
            for feat in cat_features:
                if feat in validated_features:
                    try:
                        context_pool[cat][feat] = float(validated_features[feat])
                    except (ValueError, TypeError):
                        context_pool[cat][feat] = float(get_safe_value(feat))
                    routed_keys.add(feat)
                    
        # وضع الباقي في "other" لضمان عدم ضياع أي معلومة إحصائية أو مشتقة
        for key, value in validated_features.items():
            if key not in routed_keys:
                try:
                    context_pool["other"][key] = float(value)
                except (ValueError, TypeError):
                    context_pool["other"][key] = float(get_safe_value(key))
                
        return context_pool

    def validate_vector(self, vector: np.ndarray) -> Tuple[bool, List[str]]:
        """
        يتأكد أن المصفوفة الجاهزة مثالية رياضياً لتدخل لنموذج الـ LightGBM.
        """
        issues = []
        
        if vector.shape != (len(self._index_map),):
            issues.append(f"Invalid shape: expected ({len(self._index_map)},), got {vector.shape}")
            
        if np.any(np.isnan(vector)):
            issues.append("Vector contains NaN (Not a Number) values")
            
        if np.any(np.isinf(vector)):
            issues.append("Vector contains Inf (Infinite) values")
            
        if vector.dtype != np.float64:
            issues.append(f"Invalid dtype: expected float64, got {vector.dtype}")
            
        is_valid = len(issues) == 0
        return is_valid, issues