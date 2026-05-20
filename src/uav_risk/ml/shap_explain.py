"""
Module: uav_risk.ml.shap_explain
Purpose: High-performance SHAP explainability engine with class-level explainer caching.
Dependencies: Imports FeatureImportance from uav_risk.ml.schemas and binds definitions from uav_risk.ml.feature_defs.
"""

import numpy as np
import shap
import structlog
from typing import List, Dict, Any, Optional

# استيراد المكونات المقفلة لضمان التوافق المطلق وعدم حدوث تعارض
from uav_risk.ml.schemas import FeatureImportance
from uav_risk.ml.feature_defs import get_feature_definition

# إعداد نظام التتبع والـ Logger للمرحلة الثانية
logger = structlog.get_logger(__name__)


class ShapExplainer:
    """
    Thread-safe wrapper for SHAP TreeExplainer optimized for LightGBM models.
    Implements class-level caching to guarantee sub-millisecond lookups on repeated cycles.
    """
    # التخزين المؤقت على مستوى الكلاس لمنع إعادة بناء الـ Explainer المستهلك للموارد
    _cache: Dict[int, shap.TreeExplainer] = {}

    def __init__(self, model: Any, feature_names: List[str]):
        """
        Initializes the SHAP Explainer by utilizing the global memory cache or creating a new tree graph.
        
        Args:
            model: The loaded LightGBM Classifier instance.
            feature_names: List of 198 feature identifiers matching model alignment.
        """
        self.feature_names = feature_names
        model_id = id(model)
        
        # التحقق من وجود الكائن في الـ Cache لمنع إعادة البناء
        if model_id in ShapExplainer._cache:
            logger.debug("SHAP TreeExplainer retrieved directly from class-level memory cache", model_id=model_id)
            self.explainer = ShapExplainer._cache[model_id]
        else:
            logger.info("Constructing new SHAP TreeExplainer instance for target model graph", model_id=model_id)
            try:
                # تفعيل TreeExplainer المناسب لهياكل الأشجار في LightGBM
                self.explainer = shap.TreeExplainer(model)
                ShapExplainer._cache[model_id] = self.explainer
            except Exception as e:
                logger.error("Failed to initialize SHAP TreeExplainer baseline graph", error=str(e))
                self.explainer = None

    def explain(self, X: np.ndarray, top_n: int = 10, predicted_class_idx: int = 0) -> List[FeatureImportance]:
        """
        Calculates the exact SHAP value matrix for a processed state vector and sorts drivers by absolute impact.
        
        Args:
            X (np.ndarray): Preprocessed feature matrix of exact shape (1, 198).
            top_n (int): Number of driving features to extract for the agent.
            predicted_class_idx (int): Index of the target class to explain (corresponds to highest probability).
            
        Returns:
            List[FeatureImportance]: Strictly sorted list of drivers descending by absolute mathematical contribution.
        """
        if self.explainer is None:
            logger.warning("SHAP execution requested but explainer core is uninitialized. Returning empty analysis.")
            return []
            
        try:
            logger.debug("Executing SHAP mathematical attribution vector pass", matrix_shape=X.shape)
            # 1. حساب مصفوفة مساهمات قيم شيب
            shap_values = self.explainer.shap_values(X)
            
            # 2. التعامل المرن والصارم مع أبعاد مخرجات SHAP في التصنيف المتعدد (Multi-class robust check)
            if isinstance(shap_values, list):
                # إذا كانت قائمة مصفوفات، نأخذ الفئة المستهدفة مباشرة
                shap_for_class = shap_values[predicted_class_idx][0]
            elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
                # شكل المصفوفة: (n_samples, n_features, n_classes)
                shap_for_class = shap_values[0, :, predicted_class_idx]
            elif hasattr(shap_values, 'ndim') and shap_values.ndim == 2:
                # حالة ثنائية الأبعاد أو مخرجات مستوية
                shap_for_class = shap_values[0]
            else:
                shap_for_class = np.asarray(shap_values).flatten()

            # التحقق الحرج لمنع حدوث إزاحة أو انهيار بسبب عدم تطابق طول الميزات
            if len(shap_for_class) != len(self.feature_names):
                logger.error("SHAP dimension mismatch encountered against registered protocol features", 
                             shap_len=len(shap_for_class), target_len=len(self.feature_names))
                return []

            # 3. ترتيب المؤثرات حسب القيمة المطلقة تنازلياً لتحديد أقوى الدوافع
            abs_values = np.abs(shap_for_class)
            top_indices = np.argsort(abs_values)[-top_n:][::-1]
            
            output_drivers: List[FeatureImportance] = []
            
            # 4. بناء كائنات عقود الأهمية وشحنها بالأوصاف الدلالية من دستور النظام
            for rank_idx, idx in enumerate(top_indices, start=1):
                feat_name = self.feature_names[idx]
                shap_val = float(shap_for_class[idx])
                
                # استخراج القيمة الحقيقية المدخلة للموديل لاستعراضها في التقرير
                feat_val = float(X[0, idx]) if X.ndim > 1 else float(X[idx])
                
                # جلب الدستور الدلالي للميزة لمساعدة الوكيل الذكي في التفكير المنطقي لاحقاً
                feat_def = get_feature_definition(feat_name)
                description = feat_def.get("description", feat_name) if feat_def else feat_name
                
                # إنشاء الكائن المعتمد والمقفل مع تحديد الرتبة والاتجاه ديناميكياً
                driver = FeatureImportance(
                    feature_name=feat_name,
                    shap_value=shap_val,
                    feature_value=feat_val,
                    description=description,
                    rank=rank_idx
                )
                output_drivers.append(driver)
                
            return output_drivers
            
        except Exception as e:
            logger.warning("SHAP calculation cycle encountered a non-fatal bypass sequence", error=str(e))
            return []

    def get_decision_drivers(self, X: np.ndarray, predicted_class_idx: int = 0) -> Dict[str, List[FeatureImportance]]:
        """
        Splits all calculated feature contributions into positive (risk increasing) and negative (risk decreasing) groups.
        This provides a definitive defense shield against high-risk bias by revealing exactly what forces pushed the model over.
        
        Returns:
            Dict containing two explicit lists: "risk_increasing" and "risk_decreasing".
        """
        # جلب كافة المؤثرات الـ 198 لتشريح كامل للرحلة
        all_features = self.explain(X, top_n=len(self.feature_names), predicted_class_idx=predicted_class_idx)
        
        drivers_map = {
            "risk_increasing": [],
            "risk_decreasing": []
        }
        
        for feat in all_features:
            if feat.shap_value > 0:
                drivers_map["risk_increasing"].append(feat)
            else:
                drivers_map["risk_decreasing"].append(feat)
                
        # إعادة ترتيب وتحديث قيم الرتب الداخلية لكل مجموعة مستقلة لضمان نظافة البيانات للوكيل
        for group in ["risk_increasing", "risk_decreasing"]:
            for sub_rank, feat in enumerate(drivers_map[group], start=1):
                feat.rank = sub_rank
                
        logger.debug("Decision driver groups assembled for agent audit", 
                     increasing_count=len(drivers_map["risk_increasing"]), 
                     decreasing_count=len(drivers_map["risk_decreasing"]))
                     
        return drivers_map

# =====================================================================
# Architectural Registry Block:
# This file depends on: src/uav_risk/ml/schemas.py, src/uav_risk/ml/feature_defs.py
# Files depending on this file: src/uav_risk/ml/inference.py
# =====================================================================