"""
Data Validator & Imputation Engine (Gate 1 - Processing)
The absolute guardian of data quality. Ensures 198 features are passed to ML
without any missing values, NaNs, or Infs. Applies safe fallbacks and physical limits.
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

# استيراد المراجع الثابتة من الدستور
from uav_risk.ml.feature_defs import (
    get_all_feature_names,
    get_feature_definition,
    get_safe_value,
    get_core_features
)

logger = logging.getLogger(__name__)

# ============================================================
# Data Structures
# ============================================================

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


# ============================================================
# Core Validator Class
# ============================================================

class DataValidator:
    """
    The main engine for validating, correcting, and imputing flight data.
    Guarantees a safe, 198-dimension vector output for the ML stage.
    """
    
    def __init__(self):
        # تحميل الأسماء والميزات الأساسية مرة واحدة عند الإقلاع لتحسين الأداء
        self.all_feature_names = get_all_feature_names()
        self.core_features = set(get_core_features())

    def validate_and_store(self, flat_features: Dict[str, Any]) -> ValidationResult:
        """
        المنسق الرئيسي. يمر على الميزات، يفحصها، ويشتق النواقص بالتعاون مع ImputationStrategy.
        """
        logger.info(f"Starting Data Validation. Received {len(flat_features)} raw input fields.")
        
        result = ValidationResult()
        partial_validated: Dict[str, float] = {}
        
        # 1. معالجة كل ميزة بشكل فردي
        for name in self.all_feature_names:
            raw_value = flat_features.get(name)
            record = self._process_single_feature(name, raw_value)
            
            partial_validated[name] = record.final_value
            result.validation_records.append(record)
            
            if record.was_missing and record.is_core_feature:
                result.missing_core_features.append(name)
            if record.status == "CORRECTED":
                result.corrected_features.append(name)
                
            missing_indicator_name = f"{name}_was_missing"
            if missing_indicator_name in self.all_feature_names:
                partial_validated[missing_indicator_name] = 1.0 if record.was_missing else 0.0
        
        # 2. الاشتقاق الذكي (استدعاء Imputation Strategy بشكل صحيح)
        for missing_feature in result.missing_core_features.copy():
            if hasattr(self, 'imputation_strategy'):
                imputed_val, reason = self.imputation_strategy.get_imputed_value(
                    feature_name=missing_feature, 
                    available_features=partial_validated, 
                    raw_inputs=flat_features
                )
                partial_validated[missing_feature] = imputed_val
                if missing_feature not in result.derived_features:
                    result.derived_features.append(missing_feature)
        
        # 3. الفحص الأمني النهائي
        self._final_safety_check(partial_validated, result)
        result.validated_features = partial_validated
        
        # 4 & 5. حساب جودة البيانات
        result.overall_data_quality_score = self._compute_quality_score(result.validation_records)
        result.is_usable = self._check_is_usable(result.validation_records)
        result.has_critical_missing = len(result.missing_core_features) > 0
        
        logger.info(f"Validation Complete. Usable: {result.is_usable} | Quality Score: {result.overall_data_quality_score:.2f}")
        return result

    def _process_single_feature(self, name: str, raw_value: Any) -> FeatureValidationRecord:
        """يعالج ميزة واحدة: يتأكد من النوع، يقص القيم الشاذة، ويحشو النواقص."""
        defn = get_feature_definition(name) or {}
        is_core = name in self.core_features
        safe_fallback = get_safe_value(name)

        # 1. إذا كانت القيمة مفقودة تماماً
        if raw_value is None or str(raw_value).strip().lower() in ["", "n/a", "unknown", "null", "none"]:
            return FeatureValidationRecord(
                feature_name=name, original_value=raw_value, final_value=safe_fallback,
                status="IMPUTED", was_missing=True, was_out_of_range=False, was_invalid_type=False,
                correction_reason="Value missing; used safe fallback.", is_core_feature=is_core
            )

        # 2. التحقق من النوع (Type validation) وأخطاء NaN
        try:
            val_float = float(raw_value)
            if math.isnan(val_float) or math.isinf(val_float):
                raise ValueError("Encountered NaN or Inf")
        except (ValueError, TypeError):
            return FeatureValidationRecord(
                feature_name=name, original_value=raw_value, final_value=safe_fallback,
                status="IMPUTED", was_missing=False, was_out_of_range=False, was_invalid_type=True,
                correction_reason="Invalid type/NaN/Inf; used safe fallback.", is_core_feature=is_core
            )

        # 3. التحقق من النطاق (Clipping)
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

        # 4. قيمة ممتازة وصحيحة
        return FeatureValidationRecord(
            feature_name=name, original_value=raw_value, final_value=final_val,
            status="PROVIDED", was_missing=False, was_out_of_range=False, was_invalid_type=False,
            correction_reason="OK", is_core_feature=is_core
        )

    
    def _final_safety_check(self, validated: Dict[str, float], result: ValidationResult) -> None:
        """
        شبكة الأمان الأخيرة. تضمن ألا يتسرب أي خطأ قاتل إلى نموذج الـ ML، والعدد 198 حصراً.
        """
        # 1. تنظيف NaN و Inf أولاً
        for key, value in list(validated.items()):
            if math.isnan(value) or math.isinf(value):
                safe_val = get_safe_value(key)
                validated[key] = safe_val
                msg = f"Removed NaN/Inf in '{key}'. Forced to {safe_val}."
                logger.error(msg)
                result.errors.append(msg)

        # 2. ضمان أن القاموس يحتوي حصراً على الـ 198 ميزة المعرفة (لا زيادة ولا نقصان)
        keys_to_remove = [k for k in validated.keys() if k not in self.all_feature_names]
        for k in keys_to_remove:
            del validated[k] # حذف أي ميزة عشوائية تسربت من الـ Payload
            
        for name in self.all_feature_names:
            if name not in validated:
                safe_val = get_safe_value(name)
                validated[name] = safe_val
                result.warnings.append(f"Hard-filled missing feature: {name}")

        # التأكيد النهائي
        assert len(validated) == len(self.all_feature_names), "CRITICAL: Output length is not 198!"
    def _compute_quality_score(self, records: List[FeatureValidationRecord]) -> float:
        """
        يحسب درجة جودة البيانات بناءً على الميزات الأساسية والفرعية.
        """
        core_records = [r for r in records if r.is_core_feature]
        core_provided = sum(1 for r in core_records if r.status in ["PROVIDED", "CORRECTED", "DERIVED"])
        
        all_provided = sum(1 for r in records if r.status in ["PROVIDED", "CORRECTED", "DERIVED"])
        
        # 70% من الوزن للميزات الأساسية الـ 40، و 30% لباقي الميزات
        core_score = (core_provided / max(1, len(self.core_features))) * 0.70
        all_score = (all_provided / len(self.all_feature_names)) * 0.30
        
        return min(1.0, core_score + all_score)

    def _check_is_usable(self, records: List[FeatureValidationRecord]) -> bool:
        """
        الشرط الصارم: يجب توفر 20 ميزة أساسية على الأقل ليكون التقييم موثوقاً.
        """
        core_provided = sum(
            1 for r in records 
            if r.is_core_feature and r.status in ["PROVIDED", "CORRECTED", "DERIVED"]
        )
        return core_provided >= 20