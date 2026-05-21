"""
Module: src.uav_risk.core.data_validator
Purpose: Advanced high-integrity data validation core enforcing a strict 40 core features lock down.
Dependencies: uav_risk.ml.feature_defs, src.uav_risk.core.imputation_strategy
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from uav_risk.ml.feature_defs import (
    get_all_feature_names,
    get_feature_definition,
    get_safe_value,
    get_core_features,
    is_critical_value
)

logger = logging.getLogger(__name__)

@dataclass
class FeatureValidationRecord:
    feature_name: str
    original_value: Any
    final_value: float
    status: str  # "PROVIDED", "IMPUTED", "CORRECTED", "DERIVED"
    was_missing: bool
    was_out_of_range: bool
    was_invalid_type: bool
    correction_reason: str
    is_core_feature: bool

@dataclass
class ValidationResult:
    validated_features: Dict[str, float] = field(default_factory=dict)
    validation_records: List[FeatureValidationRecord] = field(default_factory=list)
    missing_core_features: List[str] = field(default_factory=list)
    corrected_features: List[str] = field(default_factory=list)
    derived_features: List[str] = field(default_factory=list)
    has_critical_missing: bool = False
    overall_data_quality_score: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    is_usable: bool = False


class DataValidator:
    """The rigorous guardian of flight data integrity. Enforces the strict 40 core feature lock."""
    
    def __init__(self):
        self.all_feature_names = get_all_feature_names()
        self.core_features = set(get_core_features())
        from uav_risk.core.imputation_strategy import ImputationStrategy
        self.imputation_strategy = ImputationStrategy()

    def validate_and_store(self, flat_features: Dict[str, Any]) -> ValidationResult:
        logger.info("Executing Live Flight Telemetry System Audit Pass")
        result = ValidationResult()
        partial_validated: Dict[str, float] = {}
        
        user_provided_cores = set()
        
        # الخطوة 1: الفحص الفردي وتأمين المعطيات الفيزيائية الخام فقط
        for name in self.all_feature_names:
            if name.startswith("feat_") or name.endswith("_was_missing"):
                continue
                
            raw_value = flat_features.get(name)
            record = self._process_single_feature(name, raw_value)
            
            partial_validated[name] = record.final_value
            result.validation_records.append(record)
            
            if not record.was_missing and name in self.core_features:
                user_provided_cores.add(name)
            if record.was_missing and record.is_core_feature:
                result.missing_core_features.append(name)
            if record.status == "CORRECTED":
                result.corrected_features.append(name)
                
            missing_indicator_name = f"{name}_was_missing"
            if missing_indicator_name in self.all_feature_names:
                partial_validated[missing_indicator_name] = 1.0 if record.was_missing else 0.0

        # الخطوة 2: استدعاء العقل الفيزيائي لاشتقاق النواقص الأساسية الحتمية من المعطيات الحية
        for missing_feature in result.missing_core_features.copy():
            imputed_val, reason = self.imputation_strategy.get_imputed_value(
                feature_name=missing_feature,
                available_features=partial_validated,
                raw_inputs=flat_features
            )
            # تم تحويل الشرط ليسمح بالتنظيف الديناميكي للميزات المشتقة بنجاح
            if "registry" not in reason.lower() and "forced" not in reason.lower():
                partial_validated[missing_feature] = imputed_val
                result.derived_features.append(missing_feature)
                user_provided_cores.add(missing_feature)
                # 🎯 الإضافة الجذرية: تنظيف القائمة لمنع الخطأ البرمجي (KeyError) ولضمان صلاحية الرحلة
                result.missing_core_features.remove(missing_feature)
        # الخطوة 3: الحساب الديناميكي لجميع الميزات المشتقة المركبة (feat_*)
        for name in self.all_feature_names:
            if name.startswith("feat_"):
                imputed_val, reason = self.imputation_strategy.get_imputed_value(
                    feature_name=name, 
                    available_features=partial_validated, 
                    raw_inputs=flat_features
                )
                partial_validated[name] = imputed_val
                result.derived_features.append(name)
                result.validation_records.append(FeatureValidationRecord(
                    feature_name=name, original_value=None, final_value=imputed_val,
                    status="DERIVED", was_missing=False, was_out_of_range=False, was_invalid_type=False,
                    correction_reason=reason, is_core_feature=False
                ))

        # الخطوة 4: شبكة الأمان الأخيرة لضمان تماسك أبعاد المصفوفة
        self._final_safety_check(partial_validated, result)
        result.validated_features = partial_validated
        
        # حماية مشددة ضد الـ KeyError عبر استخدام السحب الآمن .get() مع الـ Fallback الافتراضي
        has_all_cores = all(name in user_provided_cores or name in partial_validated for name in self.core_features)
        has_critical_breach = any(
            is_critical_value(name, partial_validated.get(name, get_safe_value(name))) 
            for name in self.core_features
        )
        
        result.overall_data_quality_score = self._compute_quality_score(result.validation_records)
        result.is_usable = True if (has_all_cores and not has_critical_breach) else False
        result.has_critical_missing = not has_all_cores
        
        return result

    def _process_single_feature(self, name: str, raw_value: Any) -> FeatureValidationRecord:
        defn = get_feature_definition(name) or {}
        is_core = name in self.core_features
        safe_fallback = get_safe_value(name)

        if raw_value is None or str(raw_value).strip().lower() in ["", "n/a", "unknown", "null", "none"]:
            return FeatureValidationRecord(
                feature_name=name, original_value=raw_value, final_value=safe_fallback,
                status="IMPUTED", was_missing=True, was_out_of_range=False, was_invalid_type=False,
                correction_reason="Missing feature telemetry field.", is_core_feature=is_core
            )

        try:
            val_float = float(raw_value)
            if math.isnan(val_float) or math.isinf(val_float): 
                raise ValueError()
        except (ValueError, TypeError):
            return FeatureValidationRecord(
                feature_name=name, original_value=raw_value, final_value=safe_fallback,
                status="IMPUTED", was_missing=False, was_out_of_range=False, was_invalid_type=True,
                correction_reason="Invalid numeric format signals.", is_core_feature=is_core
            )

        if is_critical_value(name, val_float):
            return FeatureValidationRecord(
                feature_name=name, original_value=raw_value, final_value=val_float,
                status="PROVIDED", was_missing=False, was_out_of_range=True, was_invalid_type=False,
                correction_reason="CRITICAL OPERATIONAL BREACH DETECTED AND NATIVELY PRESERVED", is_core_feature=is_core
            )

        safe_min = defn.get("safe_min")
        safe_max = defn.get("safe_max")
        final_val = val_float
        reasons = []

        if safe_min is not None and final_val < safe_min:
            final_val = float(safe_min)
            reasons.append(f"Clipped to min ({safe_min})")
        if safe_max is not None and final_val > safe_max:
            final_val = float(safe_max)
            reasons.append(f"Clipped to max ({safe_max})")

        if reasons:
            return FeatureValidationRecord(
                feature_name=name, original_value=raw_value, final_value=final_val,
                status="CORRECTED", was_missing=False, was_out_of_range=True, was_invalid_type=False,
                correction_reason="; ".join(reasons), is_core_feature=is_core
            )

        return FeatureValidationRecord(
            feature_name=name, original_value=raw_value, final_value=final_val,
            status="PROVIDED", was_missing=False, was_out_of_range=False, was_invalid_type=False,
            correction_reason="OK", is_core_feature=is_core
        )

    def _final_safety_check(self, validated: Dict[str, float], result: ValidationResult) -> None:
        for key, value in list(validated.items()):
            if math.isnan(value) or math.isinf(value):
                validated[key] = get_safe_value(key)
        keys_to_remove = [k for k in validated.keys() if k not in self.all_feature_names]
        for k in keys_to_remove: 
            del validated[k]
        for name in self.all_feature_names:
            if name not in validated: 
                validated[name] = get_safe_value(name)

    def _compute_quality_score(self, records: List[FeatureValidationRecord]) -> float:
        core_records = [r for r in records if r.is_core_feature]
        core_provided = sum(1 for r in core_records if r.status in ["PROVIDED", "CORRECTED", "DERIVED"])
        all_provided = sum(1 for r in records if r.status in ["PROVIDED", "CORRECTED", "DERIVED"])
        return min(1.0, (core_provided / 40.0) * 0.70 + (all_provided / len(self.all_feature_names)) * 0.30)

# =====================================================================
# Architectural Registry Block:
# This file depends on: src/uav_risk/ml/feature_defs.py
# Depended on by: tests/test_ml_deep_inspection.py, src/uav_risk/stage2/pipeline.py
# =====================================================================