"""
MASTER FEATURE REGISTRY (V3.2 - Production-Ready)
====================================================
تم استخراج هذا الترتيب بشكل قطعي من مخرجات `ColumnTransformer` في
`stage-1-final-plan.ipynb`. عدد الميزات = 58 بالضبط.

تحذيرات MLOps هامة:
- Log-Transform: استخدم `np.log1p(x)` فقط عند إعادة حساب الميزات المشتقة يدوياً.
- Categorical Unknowns: الـ OHE في `uav_stage1_preprocessor_v2.pkl` مضبوط على
  `handle_unknown='ignore'`. أي فئة جديدة لم يرها النموذج ستصبح أصفاراً.
  يجب بناء `UnknownCategoryDetector` في الـ Input Contract لاحقاً.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger("FeatureRegistry")

# ==============================================================================
# 1. الترتيب القطعي للميزات (58 ميزة بالضبط)
# ==============================================================================
MODEL_EXPECTED_FEATURES: List[str] = [
    # Group 1: Log-transformed features (7 features)
    'uav.mass_kg',
    'uav.battery_model.hover_power_W',
    'feat_mission_dist_m',
    'feat_mission_climb_m',
    'feat_airspace_area_m2',
    'feat_power_to_weight',
    'feat_obstacle_density_per_km2',

    # Group 2: Robust-scaled features (9 features)
    'uav.max_speed_mps',
    'environment.weather.wind_mps',
    'environment.weather.gust_mps',
    'environment.gnss_jam_dbm',
    'daa.sep_threshold_m',
    'daa.ttc_threshold_s',
    'feat_weather_score',
    'feat_obstacle_avg_speed',
    'feat_mission_tortuosity',

    # Group 3: Binary & Missing Flags (17 features)
    'is_missing__environment.gnss_multipath',
    'is_missing__comms.uplink_ok',
    'is_missing__comms.downlink_ok',
    'is_missing__environment.em_interference',
    'is_missing__mission.runway_required',
    'has_lidar',
    'is_missing__has_lidar',
    'has_radar',
    'is_missing__has_radar',
    'has_gnss',
    'is_missing__has_gnss',
    'has_imu',
    'is_missing__has_imu',
    'has_camera_rgb',
    'is_missing__has_camera_rgb',
    'has_camera_thermal',
    'is_missing__has_camera_thermal',

    # Group 4: Data Quality features (6 features)
    'dq_core_present_pct',
    'dq_weather_present',
    'dq_uav_present',
    'dq_mission_present',
    'dq_comms_present',
    'dq_sensors_present_pct',

    # Group 5: One-Hot Encoded Categoricals (19 features)
    'mission.type_area_recon_fixedwing',
    'mission.type_convoy_escort',
    'mission.type_delivery',
    'mission.type_firefighting_drop',
    'mission.type_inspection',
    'mission.type_mapping',
    'mission.type_package_delivery',
    'mission.type_runway_touch_and_go',
    'mission.type_search_rescue',
    'mission.type_survey',
    'mission.type_infrequent_sklearn',
    'mission.pattern_corridor',
    'mission.pattern_grid',
    'mission.pattern_orbit',
    'mission.pattern_spiral',
    'mission.pattern_infrequent_sklearn',
    'environment.weather.visibility_good',
    'environment.weather.visibility_poor',
    'environment.weather.visibility_infrequent_sklearn'
]

assert len(MODEL_EXPECTED_FEATURES) == 58, \
    f"CRITICAL: Feature count mismatch! Expected 58, got {len(MODEL_EXPECTED_FEATURES)}"

# ==============================================================================
# 2. بيانات النطاق الوصفية (Metadata Ranges)
# ==============================================================================
FEATURES_METADATA: Dict[str, Dict[str, Any]] = {
    "uav.mass_kg": {
        "type": "float", "min": 0.5, "max": 2800.0, "unit": "kg",
        "warning": "Max value (2800 kg) is physically impossible for a civilian drone. Likely a data error or military-grade UAV."
    },
    "uav.max_speed_mps": {
        "type": "float", "min": 1.0, "max": 85.0, "unit": "m/s"
    },
    "uav.battery_model.hover_power_W": {
        "type": "float", "min": 20.0, "max": 1583983.1, "unit": "W",
        "warning": "Max value (1.58 MW) is equivalent to a small jet engine. Likely a sensor recording error."
    },
    "environment.weather.wind_mps": {
        "type": "float", "min": 0.0, "max": 18.5, "unit": "m/s"
    },
    "environment.weather.gust_mps": {
        "type": "float", "min": 0.0, "max": 9.0, "unit": "m/s"
    },
    "environment.gnss_jam_dbm": {
        "type": "float", "min": -150.0, "max": -35.0, "unit": "dBm", "missing_rate": 0.3778
    },
    "daa.sep_threshold_m": {
        "type": "float", "min": 1.0, "max": 1500.0, "unit": "m"
    },
    "daa.ttc_threshold_s": {
        "type": "float", "min": 3.0, "max": 60.0, "unit": "s"
    },
    "feat_mission_dist_m": {
        "type": "float", "min": 0.0, "max": 226353.71, "unit": "m",
        "transform_note": "Log-transformed with np.log1p"
    },
    "feat_mission_climb_m": {
        "type": "float", "min": 0.0, "max": 12000.0, "unit": "m",
        "transform_note": "Log-transformed with np.log1p"
    },
    "feat_mission_tortuosity": {
        "type": "float", "min": 0.0, "max": 555.52, "unit": "degrees"
    },
    "feat_obstacle_count": {
        "type": "int", "min": 0, "max": 1
    },
    "feat_obstacle_avg_speed": {
        "type": "float", "min": 0.0, "max": 50.0, "unit": "m/s"
    },
    "feat_airspace_area_m2": {
        "type": "float", "min": 60.0, "max": 40000000000.0, "unit": "m²",
        "warning": "Max value is geospatially anomalous (size of a small country)."
    },
    "feat_obstacle_density_per_km2": {
        "type": "float", "min": 0.0, "max": 16666.66, "unit": "km⁻²"
    },
    "feat_weather_score": {
        "type": "int", "min": 0, "max": 9
    },
    "feat_power_to_weight": {
        "type": "float", "min": 17.84, "max": 600.02, "unit": "W/kg"
    },
}

# ==============================================================================
# 3. Class FeatureRegistry (V3.2 - Production-Ready)
# ==============================================================================
class FeatureRegistry:
    """المرجع النهائي لترتيب الميزات والتحقق من صحتها."""

    def __init__(self):
        self.features = MODEL_EXPECTED_FEATURES
        self.expected_count = len(self.features)
        self.metadata = FEATURES_METADATA
        logger.info(f"FeatureRegistry V3.2 Initialized. Expecting {self.expected_count} features.")
        self._validate_internal_consistency()

    def _validate_internal_consistency(self):
        """يتحقق من اتساق القائمة الداخلية عند بدء التشغيل."""
        if len(self.features) != len(set(self.features)):
            logger.error("CRITICAL: Duplicate feature names found in FeatureRegistry!")
            raise ValueError("Duplicate feature names in MODEL_EXPECTED_FEATURES")

    def validate_vector(self, vector: Any) -> bool:
        """
        يتحقق مما إذا كانت المصفوفة المدخلة تحتوي على العدد الصحيح من الميزات.
        يدعم 1D arrays, 2D batches, و Python lists.
        """
        try:
            if hasattr(vector, 'shape'):
                # Numpy arrays / Pandas DataFrames
                if len(vector.shape) == 1:
                    length = vector.shape[0]
                elif len(vector.shape) == 2:
                    length = vector.shape[-1]  # البعد الأخير = عدد الميزات
                else:
                    logger.error(f"Unsupported vector shape: {vector.shape}")
                    return False
            elif isinstance(vector, list):
                # Python lists
                if len(vector) > 0 and isinstance(vector[0], (list, tuple)):
                    length = len(vector[0])  # 2D list
                else:
                    length = len(vector)     # 1D list
            else:
                logger.error(f"Unsupported vector type: {type(vector)}")
                return False

            if length != self.expected_count:
                logger.error(f"CRITICAL: Feature dimension mismatch. Expected {self.expected_count}, got {length}.")
                return False

            logger.debug(f"Feature dimension validation passed ({length} features).")
            return True

        except Exception as e:
            logger.error(f"CRITICAL: Vector validation failed. Error: {str(e)}")
            return False

    def validate_columns_with_order(self, columns: List[str]) -> bool:
        """يتحقق من تطابق أسماء الأعمدة وترتيبها بالضبط."""
        if len(columns) != self.expected_count:
            logger.error(f"CRITICAL: Column count mismatch. Expected {self.expected_count}, got {len(columns)}.")
            return False

        for i, col in enumerate(columns):
            if col != self.features[i]:
                logger.error(f"CRITICAL: Column order mismatch at index {i}. Expected '{self.features[i]}', got '{col}'.")
                return False

        logger.info("Column name and order validation passed 100%.")
        return True

    def get_feature_list(self) -> List[str]:
        """يعيد القائمة الرسمية للميزات."""
        return self.features.copy()

    def get_feature_index(self, feature_name: str) -> int:
        """يعيد موقع الميزة في القائمة الرسمية."""
        if feature_name not in self.features:
            raise KeyError(f"Feature '{feature_name}' not found in registry.")
        return self.features.index(feature_name)

    def get_metadata(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """يعيد البيانات الوصفية (النطاقات والوحدات) للميزة."""
        return self.metadata.get(feature_name)

    def get_warnings_for_values(self, feature_name: str, value: float) -> List[str]:
        """
        يفحص قيمة مقابل النطاقات المعروفة ويعيد تحذيرات إن وجدت.
        التحذيرات الوصفية (Domain Notes) تظهر فقط في حالة تجاوز النطاق فعلياً.
        """
        warnings = []
        meta = self.metadata.get(feature_name)
        if meta:
            min_val = meta.get("min")
            max_val = meta.get("max")

            is_anomalous = False
            if min_val is not None and value < min_val:
                warnings.append(f"Value {value} is below training minimum ({min_val})")
                is_anomalous = True
            if max_val is not None and value > max_val:
                warnings.append(f"Value {value} is above training maximum ({max_val})")
                is_anomalous = True

            # إرفاق التحذير الوصفي فقط إذا كانت القيمة شاذة فعلاً
            if is_anomalous and "warning" in meta:
                warnings.append(f"Domain Note: {meta['warning']}")

        return warnings