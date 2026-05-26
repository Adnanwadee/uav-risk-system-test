from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import math

import logging

from uav_risk.ml import feature_defs

logger = logging.getLogger(__name__)

"""
Module: uav_risk.core.data_validator
Purpose: Advanced high-integrity data validation core enforcing a strict core features lock down
         aligned dynamically with the production model's architectural columns mapping.
Dependencies: uav_risk.ml.feature_defs, uav_risk.core.imputation_strategy
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
    imputed_core_features: List[str] = field(default_factory=list)
    has_critical_missing: bool = False
    overall_data_quality_score: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    is_usable: bool = False


class DataValidator:
    """الحارس الحديدي لسلامة البيانات الجوية، يمنع عبور الرحلات المزيفة فيزيائياً."""
    
    def __init__(self, fail_on_imputed_core: Optional[bool] = None) -> None:
        self.all_feature_names = get_all_feature_names()
        # Use the canonical 68-core contract; do not drop expected cores even
        # if they are absent from the model column list. This preserves the
        # validator contract used by tests and the rest of the pipeline.
        self.core_feature_order = list(get_core_features())
        self.core_features = set(self.core_feature_order)
        missing_from_artifact = [c for c in self.core_features if c not in self.all_feature_names]
        if missing_from_artifact:
            logger.warning(f"Core features absent from artifact ordering: {missing_from_artifact}")
        # Policy toggle: when True, any imputed canonical core feature will cause
        # the validator to mark the payload as unusable. The policy may be
        # injected by callers, or controlled via the environment variable
        # `UAV_FAIL_ON_IMPUTED_CORE` (1/true/yes). The default remains False to
        # preserve historical permissive behaviour.
        import os
        if fail_on_imputed_core is None:
            env_val = os.getenv("UAV_FAIL_ON_IMPUTED_CORE", "0").lower()
            self.fail_on_imputed_core = env_val in ("1", "true", "yes")
        else:
            self.fail_on_imputed_core = bool(fail_on_imputed_core)

    def validate_and_store(self, flat_features: Dict[str, Any]) -> ValidationResult:
        """يقوم بتدقيق وتطهير الـ 198 ميزة بالكامل وفرض قفل الميزات الحتمية لمنع الاختراقات صامتة الخطورة."""
        logger.info("Executing Live Flight Telemetry System Audit Pass")
        result = ValidationResult()
        partial_validated: Dict[str, float] = {}
        
        user_provided_cores = set()
        
        # الخطوة 1: الفحص الفردي وتأمين المعطيات الفيزيائية الخام غير المشتقة
        for name in self.all_feature_names:
            if name.endswith("_was_missing"):
                continue
                
            raw_value = flat_features.get(name)
            record = self._process_single_feature(name, raw_value)
            
            partial_validated[name] = record.final_value
            result.validation_records.append(record)
            
            if record.status in ["PROVIDED", "CORRECTED", "DERIVED"] and name in self.core_features:
                user_provided_cores.add(name)
        result.validated_features = partial_validated
        
        # Recompute imputed core features from validation records (covers NaN/Inf and invalid types)
        result.imputed_core_features = [r.feature_name for r in result.validation_records if r.is_core_feature and r.status == "IMPUTED"]
        result.missing_core_features = [name for name in self.core_feature_order if name not in user_provided_cores]

        # تحقق القفل الصارم المتوافق مع فضاء أعمدة النموذج
        has_all_cores = all(name in user_provided_cores for name in self.core_features)
        has_critical_breach = False
        for name in self.core_features:
            value = partial_validated.get(name)
            if value is None:
                continue
            if is_critical_value(name, value):
                has_critical_breach = True
                break
        
        # احتساب درجة جودة مدخلات البيانات عيارياً بناء على الطول الحقيقي للمجموعات المتوفرة
        result.overall_data_quality_score = self._compute_quality_score(result.validation_records)
        result.is_usable = True if (has_all_cores and not has_critical_breach) else False
        # Enforce policy: if configured to fail on imputed core features, set unusable.
        if getattr(self, "fail_on_imputed_core", False) and result.imputed_core_features:
            result.is_usable = False
            result.errors.append(f"IMPUTED_CORE_POLICY: imputed cores present: {result.imputed_core_features}")
        result.has_critical_missing = not has_all_cores
        
        if not has_all_cores:
            result.errors.append(f"MISSING_CORE_FEATURES: {result.missing_core_features}")
            logger.error(f"STRICT CORE LOCK BREACH: Missing vital core features: {result.missing_core_features}")
            
        # Final informational signal for test harnesses and operators
        logger.info("Validation Complete: computed feature validations and imputation summary")
        return result

    def _process_single_feature(self, name: str, raw_value: Any) -> FeatureValidationRecord:
        """تدقق جنائياً في جودة ونوع ومدى القيمة الممررة للميزة المفردة دون تعديلها."""
        defn = get_feature_definition(name) or {}
        is_core = name in self.core_features

        if name == "spawn_xyz_first" and isinstance(raw_value, (list, tuple)):
            if len(raw_value) != 3:
                return FeatureValidationRecord(
                    feature_name=name, original_value=raw_value, final_value=float("nan"),
                    status="IMPUTED", was_missing=False, was_out_of_range=False, was_invalid_type=True,
                    correction_reason="spawn_xyz_first must contain exactly 3 spatial values.", is_core_feature=is_core
                )
            try:
                return FeatureValidationRecord(
                    feature_name=name, original_value=raw_value, final_value=float(raw_value[0]),
                    status="PROVIDED", was_missing=False, was_out_of_range=False, was_invalid_type=False,
                    correction_reason="Accepted spatial triplet payload; canonical scalar anchor retained from first coordinate.", is_core_feature=is_core
                )
            except (TypeError, ValueError):
                return FeatureValidationRecord(
                    feature_name=name, original_value=raw_value, final_value=float("nan"),
                    status="IMPUTED", was_missing=False, was_out_of_range=False, was_invalid_type=True,
                    correction_reason="spawn_xyz_first triplet contains non-numeric values.", is_core_feature=is_core
                )

        if raw_value is None or str(raw_value).strip().lower() in ["", "n/a", "unknown", "null", "none"]:
            return FeatureValidationRecord(
                feature_name=name, original_value=raw_value, final_value=float("nan"),
                status="IMPUTED", was_missing=True, was_out_of_range=False, was_invalid_type=False,
                correction_reason="Missing feature telemetry field.", is_core_feature=is_core
            )

        try:
            val_float = float(raw_value)
            if math.isnan(val_float) or math.isinf(val_float): 
                raise ValueError()
        except (ValueError, TypeError):
            return FeatureValidationRecord(
                feature_name=name, original_value=raw_value, final_value=float("nan"),
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
        was_out_of_range = False
        if safe_min is not None and final_val < safe_min:
            was_out_of_range = True
        if safe_max is not None and final_val > safe_max:
            was_out_of_range = True

        if was_out_of_range:
            return FeatureValidationRecord(
                feature_name=name, original_value=raw_value, final_value=final_val,
                status="PROVIDED", was_missing=False, was_out_of_range=True, was_invalid_type=False,
                correction_reason="Value preserved outside safe envelope.", is_core_feature=is_core
            )

        return FeatureValidationRecord(
            feature_name=name, original_value=raw_value, final_value=final_val,
            status="PROVIDED", was_missing=False, was_out_of_range=False, was_invalid_type=False,
            correction_reason="OK", is_core_feature=is_core
        )

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