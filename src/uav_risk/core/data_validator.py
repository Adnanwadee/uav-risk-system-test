from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import math

from uav_risk.ml import feature_defs


@dataclass
class ValidationResult:
    is_usable: bool
    validated_features: Dict[str, float]
    errors: List[str]
    warnings: List[str]


def _to_float(value: Any) -> Tuple[Optional[float], Optional[str]]:
    try:
        num = float(value)
        if not math.isfinite(num):
            return None, "non-finite"
        return num, None
    except Exception:
        return None, "not-numeric"


def validate_core_numeric_fields(input_mapping: Dict[str, Any], strict_bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None) -> ValidationResult:
    """
    Validate the canonical core features are present and within bounds.

    - input_mapping: mapping of feature_name -> raw value (user or preprocessor)
    - strict_bounds: optional override of (safe_min, safe_max) or (critical_low, critical_high)

    Returns ValidationResult indicating usability, converted floats, errors and warnings.
    """
    cores = feature_defs.get_core_features()
    validated: Dict[str, float] = {}
    errors: List[str] = []
    warnings: List[str] = []

    for feat in cores:
        if feat not in input_mapping:
            errors.append(f"missing core feature: {feat}")
            continue

        raw = input_mapping[feat]
        num, err = _to_float(raw)
        if err is not None:
            errors.append(f"core feature '{feat}' invalid: {err} (value={raw})")
            continue

        # fetch definition
        defn = feature_defs.get_feature_definition(feat) or {}

        # allow override bounds
        safe_min = None
        safe_max = None
        critical_low = defn.get("critical_low")
        critical_high = defn.get("critical_high")
        if strict_bounds and feat in strict_bounds:
            safe_min, safe_max = strict_bounds[feat]
        else:
            safe_min = defn.get("safe_min")
            safe_max = defn.get("safe_max")

        # Check critical violations
        if critical_low is not None and num < critical_low:
            errors.append(f"CRITICAL: {feat}={num} < critical_low {critical_low}")
            continue
        if critical_high is not None and num > critical_high:
            errors.append(f"CRITICAL: {feat}={num} > critical_high {critical_high}")
            continue

        # Safe bounds -> warnings
        if safe_min is not None and num < safe_min:
            warnings.append(f"{feat}={num} below safe_min {safe_min}")
        if safe_max is not None and num > safe_max:
            warnings.append(f"{feat}={num} above safe_max {safe_max}")

        validated[feat] = num

    is_usable = len(errors) == 0
    return ValidationResult(is_usable=is_usable, validated_features=validated, errors=errors, warnings=warnings)


def solve_physical_consistency(validated_features: Dict[str, float]) -> Dict[str, float]:
    """Compute derived physical features and return an enriched features dict.

    Examples of derived features:
    - `air_density_kg_m3` approximated from altitude (simple exponential decay)
    - `uav_battery_wh` computed from `uav_battery_capacity_mah` and `uav_battery_voltage_v` if missing
    """
    out = dict(validated_features)

    # derive air density from altitude (AGL) using a simple scale height model
    alt_m = out.get("mission_altitude_m") or out.get("airspace_altitude_agl_min_m") or 0.0
    try:
        alt = float(alt_m)
    except Exception:
        alt = 0.0
    # scale height ~8500 m, sea-level density 1.225 kg/m^3
    rho = 1.225 * math.exp(-alt / 8500.0)
    out["air_density_kg_m3"] = round(rho, 4)

    # compute battery Wh if capacity (mAh) and voltage (V) present
    if "uav_battery_capacity_mah" in out and "uav_battery_voltage_v" in out:
        try:
            mah = float(out["uav_battery_capacity_mah"])
            v = float(out["uav_battery_voltage_v"])
            wh = (mah / 1000.0) * v
            out.setdefault("uav_battery_wh", round(wh, 2))
        except Exception:
            pass

    # reserve utilization: if uav_battery_wh and mission_time_budget_s and battery_model_hover_power_w present
    if "uav_battery_wh" in out and "mission_time_budget_s" in out and "uav_battery_model_hover_power_w" in out:
        try:
            wh = float(out["uav_battery_wh"])
            t_s = float(out["mission_time_budget_s"])
            power_w = float(out["uav_battery_model_hover_power_w"])
            # energy needed approx = power * time (hrs)
            needed_wh = power_w * (t_s / 3600.0)
            reserve_util = min(max((needed_wh / wh) if wh > 0 else 1.0, 0.0), 1.0)
            out["feat_reserve_utilization"] = round(reserve_util, 4)
        except Exception:
            pass

    return out


def validate_and_enrich(input_mapping: Dict[str, Any], strict_bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None) -> ValidationResult:
    """Convenience: validate core features and enrich derived fields.

    Returns ValidationResult with validated_features extended by derived values when usable.
    """
    vr = validate_core_numeric_fields(input_mapping, strict_bounds=strict_bounds)
    if not vr.is_usable:
        return vr

    enriched = solve_physical_consistency(vr.validated_features)
    vr.validated_features = enriched
    return vr
"""
Module: uav_risk.core.data_validator
Purpose: Advanced high-integrity data validation core enforcing a strict core features lock down
         aligned dynamically with the production model's architectural columns mapping.
Dependencies: uav_risk.ml.feature_defs, src.uav_risk.core.imputation_strategy
Source References: FAA Part 107 Small UAS Safety Certification Frameworks, EASA Gate 1 Standards.
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

# إعداد اللوجر المركزي لطبقة فحص وإجازة البيانات
logger = logging.getLogger(__name__)

@dataclass
class FeatureValidationRecord:
    """سجل التوثيق الجنائي للميزة يوضح أصلها والتعديلات والقصاصات التي تمت عليها."""
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
    """الوعاء الحاظر لنتائج البوابة الثانية يحمل مصفوفة البيانات ورايات صلاحية الإقلاع."""
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
    """الحارس الحديدي لسلامة البيانات الجوية، يمنع عبور الرحلات المزيفة فيزيائياً."""
    
    def __init__(self) -> None:
        self.all_feature_names = get_all_feature_names()
        # مواءمة القفل الصارم ديناميكياً مع الميزات الفعلية المتاحة في مصفوفة أعمدة النموذج
        self.core_features = set(get_core_features()).intersection(set(self.all_feature_names))
        from uav_risk.core.imputation_strategy import ImputationStrategy
        self.imputation_strategy = ImputationStrategy()

    def validate_and_store(self, flat_features: Dict[str, Any]) -> ValidationResult:
        """يقوم بتدقيق وتطهير الـ 198 ميزة بالكامل وفرض قفل الميزات الحتمية لمنع الاختراقات صامتة الخطورة."""
        logger.info("Executing Live Flight Telemetry System Audit Pass")
        result = ValidationResult()
        partial_validated: Dict[str, float] = {}
        
        user_provided_cores = set()
        missing_features_to_impute = []
        
        # الخطوة 1: الفحص الفردي وتأمين المعطيات الفيزيائية الخام غير المشتقة
        for name in self.all_feature_names:
            if name.startswith("feat_") or name.endswith("_was_missing"):
                continue
                
            raw_value = flat_features.get(name)
            record = self._process_single_feature(name, raw_value)
            
            partial_validated[name] = record.final_value
            result.validation_records.append(record)
            
            if not record.was_missing and name in self.core_features:
                user_provided_cores.add(name)
            if record.was_missing:
                missing_features_to_impute.append(name)
                if record.is_core_feature:
                    result.missing_core_features.append(name)
            if record.status == "CORRECTED":
                result.corrected_features.append(name)
                
            # تعبئة مؤشرات الفقد الحركي حياً لتغذية مصفوفة المدخلات
            missing_indicator_name = f"{name}_was_missing"
            if missing_indicator_name in self.all_feature_names:
                partial_validated[missing_indicator_name] = 1.0 if record.was_missing else 0.0

        # الخطوة 2: استدعاء العقل الفيزيائي لاشتقاق كافة النواقص
        for missing_feature in missing_features_to_impute:
            imputed_val, reason = self.imputation_strategy.get_imputed_value(
                feature_name=missing_feature,
                available_features=partial_validated,
                raw_inputs=flat_features
            )
            if "registry" not in reason.lower() and "fallback" not in reason.lower():
                partial_validated[missing_feature] = imputed_val
                result.derived_features.append(missing_feature)
                
                if missing_feature in self.core_features:
                    user_provided_cores.add(missing_feature)
                    if missing_feature in result.missing_core_features:
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

        # الخطوة 4: شبكة الأمان الأخيرة لضمان تماسك مصفوفة الـ 198 ميزة من القيم الشاذة (NaN)
        self._final_safety_check(partial_validated, result)
        result.validated_features = partial_validated
        
        # تحقق القفل الصارم المتوافق مع فضاء أعمدة النموذج
        has_all_cores = all(name in user_provided_cores for name in self.core_features)
        
        has_critical_breach = any(
            is_critical_value(name, partial_validated.get(name, get_safe_value(name))) 
            for name in self.core_features
        )
        
        # احتساب درجة جودة مدخلات البيانات عيارياً بناء على الطول الحقيقي للمجموعات المتوفرة
        result.overall_data_quality_score = self._compute_quality_score(result.validation_records)
        result.is_usable = True if (has_all_cores and not has_critical_breach) else False
        result.has_critical_missing = not has_all_cores
        
        if not has_all_cores:
            logger.error(f"STRICT CORE LOCK BREACH: Missing vital core features: {result.missing_core_features}")
            
        return result

    def _process_single_feature(self, name: str, raw_value: Any) -> FeatureValidationRecord:
        """تدقق جنائياً في جودة ونوع ومدى القيمة الممررة للميزة المفردة وتقليم الزوائد الثانوية."""
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

        # حماية فيزيائية: إذا كانت الميزة من الـ 40 الأساسية وهي خارج المدى الآمن (Warning) مررها كاملة بلا تعديل
        if is_core:
            if (safe_min is not None and final_val < safe_min) or (safe_max is not None and final_val > safe_max):
                return FeatureValidationRecord(
                    feature_name=name, original_value=raw_value, final_value=final_val,
                    status="PROVIDED", was_missing=False, was_out_of_range=True, was_invalid_type=False,
                    correction_reason="CORE OPERATIONAL WARNING: Preserved raw value natively for cognitive and ML safety review.",
                    is_core_feature=is_core
                )

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
        """فحص بارانويا أخير لضمان مطابقة شكل مصفوفة الـ 198 حقل."""
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
        """حساب معيار جودة البيانات الممررة عبر أوزان مرجعية مرنة تعتمد على الحجم الفعلي للمجموعات المتقاطعة."""
        core_records = [r for r in records if r.is_core_feature]
        if not core_records:
            return 0.0
        core_provided = sum(1 for r in core_records if r.status in ["PROVIDED", "CORRECTED", "DERIVED"])
        all_provided = sum(1 for r in records if r.status in ["PROVIDED", "CORRECTED", "DERIVED"])
        return min(1.0, (core_provided / len(core_records)) * 0.70 + (all_provided / len(self.all_feature_names)) * 0.30)

# =====================================================================
# Architectural Registry Block:
# This file serves as the strict pipeline gatekeeper enforcing dynamic core features lock compliance.
# This file depends on: src/uav_risk/ml/feature_defs.py, src/uav_risk/core/imputation_strategy.py
# Files depending on this file: tests/test_ml_deep_inspection.py, src/uav_risk/stage2/pipeline.py
# =====================================================================