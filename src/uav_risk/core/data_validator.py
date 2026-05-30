# STAGE6_CLEANUP_REVIEW:
# Classification: MIXED_ACTIVE_LEGACY_VALIDATION
# Plan lineage: PLAN3_ACTIVE raw validation plus PLAN1/PLAN2 processed feature validator bridge.
# Runtime status: validate_*_raw helpers and run_structural_hard_veto() are active API/Core validation paths.
# Legacy signal: DataValidator validates processed/198-style feature maps and is marked compatibility-only.
# Replacement: Use raw contract validators and run_structural_hard_veto() for API/production paths.
# Action rule: Do not delete this file. Review DataValidator only after old processed-vector tests/callers are resolved.
from __future__ import annotations
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
from typing import Any, Dict, List, Mapping, Optional

from uav_risk.ml.feature_defs import (
    get_all_feature_names,
    get_feature_definition,
    get_core_features,
    is_critical_value
)
from uav_risk.ml.raw_schema import (
    DROPPED_RAW_METADATA_FEATURES,
    FORBIDDEN_USER_FEATURES,
    INTERNAL_ONLY_RAW_FEATURES,
    OPTIONAL_RAW_OVERRIDE_FEATURES,
    PROFILE_DERIVED_RAW_FEATURES,
    RAW_CATEGORICAL_FEATURES,
    SCENARIO_REQUIRED_RAW_FEATURES,
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


@dataclass
class ValidationIssue:
    code: str
    field: Optional[str]
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RawValidationResult:
    passed: bool = True
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)

    def add_issue(self, code: str, field: Optional[str], message: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.issues.append(ValidationIssue(code=code, field=field, message=message, details=details or {}))
        self.passed = False

    def extend(self, other: "RawValidationResult") -> None:
        self.issues.extend(other.issues)
        self.warnings.extend(other.warnings)
        self.passed = self.passed and other.passed


def _to_mapping(value: Any) -> Dict[str, Any]:
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _is_missing(mapping: Mapping[str, Any], field_name: str) -> bool:
    return field_name not in mapping or mapping[field_name] is None


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _is_boolean_like(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value)) and float(value) in {0.0, 1.0}
    return False


def _validate_required_fields(
    data: Mapping[str, Any],
    required: tuple[str, ...],
    code: str,
    result: RawValidationResult,
) -> None:
    for name in required:
        if _is_missing(data, name):
            result.add_issue(code, name, f"Missing required raw field: {name}")


def _validate_category(data: Mapping[str, Any], field_name: str, result: RawValidationResult) -> None:
    if _is_missing(data, field_name):
        return
    allowed = RAW_CATEGORICAL_FEATURES[field_name]
    if data[field_name] not in allowed:
        result.add_issue(
            "INVALID_CATEGORY",
            field_name,
            f"Invalid category for {field_name}: {data[field_name]!r}",
            {"allowed": list(allowed)},
        )


def _validate_finite_numeric_fields(
    data: Mapping[str, Any],
    field_names: tuple[str, ...],
    result: RawValidationResult,
) -> None:
    for name in field_names:
        if _is_missing(data, name) or name in RAW_CATEGORICAL_FEATURES:
            continue
        if not _is_finite_number(data[name]):
            result.add_issue("INVALID_NUMERIC", name, f"Field {name} must be a finite numeric scalar.")


def _validate_non_negative(
    data: Mapping[str, Any],
    field_names: tuple[str, ...],
    result: RawValidationResult,
) -> None:
    for name in field_names:
        if _is_missing(data, name) or name in RAW_CATEGORICAL_FEATURES:
            continue
        if _is_finite_number(data[name]) and float(data[name]) < 0.0:
            result.add_issue("INVALID_NUMERIC", name, f"Field {name} must be non-negative.")


def validate_drone_profile_raw(profile: Any | Mapping[str, Any]) -> RawValidationResult:
    data = _to_mapping(profile)
    result = RawValidationResult()

    for name in ("user_id", "profile_id", "profile_name"):
        if _is_missing(data, name):
            result.add_issue("MISSING_PROFILE_FIELD", name, f"Missing required profile identity field: {name}")

    _validate_required_fields(data, PROFILE_DERIVED_RAW_FEATURES, "MISSING_PROFILE_FIELD", result)
    _validate_category(data, "uav_energy_source", result)

    numeric_fields = tuple(name for name in PROFILE_DERIVED_RAW_FEATURES if name not in RAW_CATEGORICAL_FEATURES)
    _validate_finite_numeric_fields(data, numeric_fields, result)
    _validate_non_negative(data, numeric_fields, result)

    for sensor in (
        "uav_sensors_gnss",
        "uav_sensors_lidar",
        "uav_sensors_radar",
        "uav_sensors_camera_rgb",
        "uav_sensors_camera_thermal",
    ):
        if not _is_missing(data, sensor) and not _is_boolean_like(data[sensor]):
            result.add_issue("INVALID_BOOLEAN", sensor, f"Sensor flag {sensor} must be boolean-like 0/1 or bool.")

    for cap in ("max_payload_kg", "max_takeoff_mass_kg"):
        if cap in data and data[cap] is not None:
            if not _is_finite_number(data[cap]) or float(data[cap]) < 0.0:
                result.add_issue("INVALID_NUMERIC", cap, f"Capability field {cap} must be a non-negative finite number.")

    if data.get("max_swarm_size") is not None:
        value = data["max_swarm_size"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            result.add_issue("INVALID_NUMERIC", "max_swarm_size", "max_swarm_size must be an integer >= 1.")

    for cap in ("runway_capable", "swarm_capable"):
        if cap in data and not isinstance(data[cap], bool):
            result.add_issue("INVALID_BOOLEAN", cap, f"Capability field {cap} must be bool.")

    return result


def validate_scenario_raw(scenario: Any | Mapping[str, Any]) -> RawValidationResult:
    data = _to_mapping(scenario)
    result = RawValidationResult()

    _validate_required_fields(data, SCENARIO_REQUIRED_RAW_FEATURES, "MISSING_SCENARIO_FIELD", result)
    for name in ("mission_pattern", "controls_mode", "swarm_roles_first"):
        _validate_category(data, name, result)

    numeric_fields = tuple(name for name in SCENARIO_REQUIRED_RAW_FEATURES if name not in RAW_CATEGORICAL_FEATURES)
    _validate_finite_numeric_fields(data, numeric_fields, result)

    non_negative_fields = tuple(
        name for name in numeric_fields if name not in {"environment_gnss_jam_dbm", "comms_rssi_dbm_min"}
    )
    _validate_non_negative(data, non_negative_fields, result)

    for boolean_field in (
        "mission_runway_required",
        "swarm_enabled",
        "comms_uplink_ok",
        "comms_downlink_ok",
        "environment_gnss_multipath",
        "environment_em_interference",
    ):
        if not _is_missing(data, boolean_field) and not _is_boolean_like(data[boolean_field]):
            result.add_issue("INVALID_BOOLEAN", boolean_field, f"Field {boolean_field} must be boolean-like 0/1 or bool.")

    min_alt = data.get("airspace_altitude_agl_min_m")
    max_alt = data.get("airspace_altitude_agl_max_m")
    if _is_finite_number(min_alt) and _is_finite_number(max_alt) and float(max_alt) <= float(min_alt):
        result.add_issue(
            "ALTITUDE_RANGE_INVALID",
            "airspace_altitude_agl_max_m",
            "Maximum AGL altitude must be greater than minimum AGL altitude.",
            {"min": min_alt, "max": max_alt},
        )

    return result


def validate_secondary_overrides_raw(overrides: Any | Mapping[str, Any]) -> RawValidationResult:
    data = _to_mapping(overrides)
    values = data.get("values", data)
    if values is None:
        values = {}
    result = RawValidationResult()

    if not isinstance(values, Mapping):
        result.add_issue("INVALID_OVERRIDE_KEY", None, "Secondary overrides must be a mapping.")
        return result

    optional = set(OPTIONAL_RAW_OVERRIDE_FEATURES)
    for key, value in values.items():
        if key in FORBIDDEN_USER_FEATURES:
            result.add_issue("FORBIDDEN_PROCESSED_FEATURE", key, f"Processed feature cannot be overridden: {key}")
            continue
        if key in INTERNAL_ONLY_RAW_FEATURES:
            result.add_issue("INTERNAL_ONLY_OVERRIDE", key, f"Internal-only raw feature cannot be overridden: {key}")
            continue
        if key in DROPPED_RAW_METADATA_FEATURES:
            result.add_issue("DROPPED_METADATA_OVERRIDE", key, f"Dropped raw metadata cannot be overridden: {key}")
            continue
        if key not in optional:
            result.add_issue("INVALID_OVERRIDE_KEY", key, f"Unknown or non-overridable raw feature: {key}")
            continue

        if key == "controls_actions_first":
            allowed = RAW_CATEGORICAL_FEATURES["controls_actions_first"]
            if value not in allowed:
                result.add_issue(
                    "INVALID_CATEGORY",
                    key,
                    f"Invalid controls_actions_first override: {value!r}",
                    {"allowed": list(allowed)},
                )
            continue

        if not _is_finite_number(value):
            result.add_issue("INVALID_NUMERIC", key, f"Override {key} must be a finite numeric scalar.")

    return result


def validate_assessment_core_input_raw(assessment: Any | Mapping[str, Any]) -> RawValidationResult:
    data = _to_mapping(assessment)
    result = RawValidationResult()

    for name in ("user_id", "profile_id"):
        if _is_missing(data, name):
            result.add_issue("MISSING_ASSESSMENT_FIELD", name, f"Missing assessment field: {name}")

    profile = data.get("drone_profile", {})
    scenario = data.get("scenario", {})
    overrides = data.get("secondary_overrides", {})
    profile_data = _to_mapping(profile)

    result.extend(validate_drone_profile_raw(profile))
    result.extend(validate_scenario_raw(scenario))
    result.extend(validate_secondary_overrides_raw(overrides))

    if data.get("user_id") is not None and profile_data.get("user_id") is not None and data["user_id"] != profile_data["user_id"]:
        result.add_issue("USER_ID_MISMATCH", "user_id", "Assessment user_id must match drone_profile.user_id.")
    if data.get("profile_id") is not None and profile_data.get("profile_id") is not None and data["profile_id"] != profile_data["profile_id"]:
        result.add_issue("PROFILE_ID_MISMATCH", "profile_id", "Assessment profile_id must match drone_profile.profile_id.")

    return result


def _truthy_flag(value: Any) -> bool:
    return value is True or (_is_finite_number(value) and float(value) == 1.0)


def run_structural_hard_veto(assessment: Any | Mapping[str, Any]) -> RawValidationResult:
    data = _to_mapping(assessment)
    result = validate_assessment_core_input_raw(assessment)
    profile = _to_mapping(data.get("drone_profile", {}))
    scenario = _to_mapping(data.get("scenario", {}))

    payload = scenario.get("uav_payload_mass_kg")
    max_payload = profile.get("max_payload_kg")
    if max_payload is not None and _is_finite_number(payload) and _is_finite_number(max_payload) and float(payload) > float(max_payload):
        result.add_issue(
            "PAYLOAD_EXCEEDS_PROFILE_LIMIT",
            "uav_payload_mass_kg",
            "Payload mass exceeds profile max_payload_kg.",
            {"payload": payload, "max_payload_kg": max_payload},
        )

    mass = profile.get("uav_mass_kg")
    max_mass = profile.get("max_takeoff_mass_kg")
    if max_mass is not None and _is_finite_number(mass) and _is_finite_number(max_mass) and float(mass) > float(max_mass):
        result.add_issue(
            "MASS_EXCEEDS_PROFILE_LIMIT",
            "uav_mass_kg",
            "UAV mass exceeds profile max_takeoff_mass_kg.",
            {"mass": mass, "max_takeoff_mass_kg": max_mass},
        )

    max_alt = scenario.get("airspace_altitude_agl_max_m")
    ceiling = profile.get("uav_rotorcraft_hover_ceiling_m")
    if _is_finite_number(max_alt) and _is_finite_number(ceiling) and float(max_alt) > float(ceiling):
        result.add_issue(
            "ALTITUDE_EXCEEDS_HOVER_CEILING",
            "airspace_altitude_agl_max_m",
            "Mission altitude exceeds profile hover ceiling.",
            {"max_altitude": max_alt, "hover_ceiling": ceiling},
        )

    if _truthy_flag(scenario.get("swarm_enabled")) and profile.get("swarm_capable") is False:
        result.add_issue("SWARM_NOT_CAPABLE", "swarm_enabled", "Swarm requested but profile is not swarm capable.")

    swarm_size = scenario.get("swarm_size")
    max_swarm = profile.get("max_swarm_size")
    if max_swarm is not None and _is_finite_number(swarm_size) and isinstance(max_swarm, int) and float(swarm_size) > float(max_swarm):
        result.add_issue(
            "SWARM_SIZE_EXCEEDS_PROFILE_LIMIT",
            "swarm_size",
            "Scenario swarm_size exceeds profile max_swarm_size.",
            {"swarm_size": swarm_size, "max_swarm_size": max_swarm},
        )

    if _truthy_flag(scenario.get("mission_runway_required")) and profile.get("runway_capable") is False:
        result.add_issue("RUNWAY_NOT_CAPABLE", "mission_runway_required", "Runway mission requested but profile is not runway capable.")

    return result


class DataValidator:
    """Legacy compatibility only. Do not use in production raw path.

    The production validator is the raw contract helper set above:
    validate_drone_profile_raw(), validate_scenario_raw(),
    validate_secondary_overrides_raw(), validate_assessment_core_input_raw(),
    and run_structural_hard_veto().
    """
    
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