"""
Stage 1 Canonicalization (V2.0 - Strict & Safe)
==================================================
إعادة كتابة كاملة وفق خطة ACE المعمارية:

المبادئ:
1. لا تعبئة صامتة (No Silent Defaults): الحقول الحرجة المفقودة ترفض المهمة فوراً.
2. الحقول الاختيارية: يتم تعيين is_missing__flag = 1 بدلاً من التعبئة بصفر.
3. الميزات المشتقة: تُحسب مطابقة للنوت بوك (log1p، إلخ).
4. الربط مع FeatureRegistry: جميع أسماء الأعمدة وترتيبها مستمدة من Registry.
5. الشفافية: إخراج data_quality_report كامل مع warnings.

Author: Stage 1 — ACE System V2
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass, field

from uav_risk.schema.feature_registry import FeatureRegistry

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. تعريف الحقول الحرجة (Core Fields) — لا يمكن الاستغناء عنها أبداً
# ==============================================================================
CORE_FIELDS: List[str] = [
    "uav.mass_kg",
    "uav.max_speed_mps",
    "uav.battery_model.hover_power_W",
    "environment.weather.wind_mps",
    "environment.weather.gust_mps",
    "environment.gnss_jam_dbm",
    "daa.sep_threshold_m",
    "daa.ttc_threshold_s",
    "airspace.altitude_agl_max_m",
    "uav.type",
    "mission.type",
    "mission.pattern",
    "environment.weather.visibility",
]

# ==============================================================================
# 2. تعريف الحقول الاختيارية (Optional Fields) مع أعلام الفقدان
# ==============================================================================
BOOL_FIELDS_WITH_MISSING: Dict[str, str] = {
    "environment.gnss_multipath": "is_missing__environment.gnss_multipath",
    "comms.uplink_ok": "is_missing__comms.uplink_ok",
    "comms.downlink_ok": "is_missing__comms.downlink_ok",
    "environment.em_interference": "is_missing__environment.em_interference",
    "mission.runway_required": "is_missing__mission.runway_required",
}

SENSOR_KEYS: List[str] = [
    "lidar", "radar", "gnss", "imu", "camera_rgb", "camera_thermal"
]

# ==============================================================================
# 3. دوال مساعدة (Helper Functions) — مطابقة للنوت بوك
# ==============================================================================

WEATHER_SEVERITY_MAP = {
    'clear': 0, 'sun': 0, 'good': 0,
    'cloudy': 1, 'overcast': 1,
    'fog': 2, 'haze': 2, 'mist': 2,
    'rain': 3, 'light_rain': 3,
    'heavy_rain': 4, 'snow': 4,
    'storm': 5, 'thunderstorm': 5, 'windy': 3
}


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """تحويل آمن إلى float، يعيد None إذا فشل."""
    if value is None:
        return default
    try:
        if isinstance(value, bool):
            return float(int(value))
        return float(value)
    except (ValueError, TypeError):
        return default


def _tri_state_bool(value: Any) -> Tuple[int, int]:
    """
    معالجة القيم الثلاثية (True, False, None).
    تعيد (القيمة 0/1, علم الفقدان 0/1).
    مطابق لدالة tri_state_bool في النوت بوك.
    """
    if value is None:
        return 0, 1
    if isinstance(value, bool):
        return (1 if value else 0), 0
    try:
        v = str(value).strip()
        if v in ["0", "1"]:
            return (int(v), 0)
    except Exception:
        pass
    return 0, 1


def _weather_score(weather_obj: Any) -> int:
    """حساب درجة خطورة الطقس. مطابق للنوت بوك."""
    if not isinstance(weather_obj, dict):
        return 0
    score = 0
    phenomena = weather_obj.get("phenomena", [])
    if isinstance(phenomena, list):
        for p in phenomena:
            score += WEATHER_SEVERITY_MAP.get(str(p).lower().strip(), 1)
    vis = str(weather_obj.get("visibility", "")).lower()
    if "poor" in vis or "low" in vis:
        score += 2
    return score


def _mission_dist_climb(waypoints: Any) -> Tuple[float, float]:
    """حساب مسافة المهمة والتسلق. مطابق للنوت بوك."""
    if not isinstance(waypoints, list) or len(waypoints) < 2:
        return 0.0, 0.0
    dist, climb = 0.0, 0.0
    try:
        prev = np.array(waypoints[0], dtype=float)
        for pt in waypoints[1:]:
            cur = np.array(pt, dtype=float)
            dist += float(np.linalg.norm(cur - prev))
            if len(cur) > 2 and len(prev) > 2:
                dz = cur[2] - prev[2]
                if dz > 0:
                    climb += float(dz)
            prev = cur
    except Exception:
        return 0.0, 0.0
    return round(dist, 2), round(climb, 2)


def _tortuosity(waypoints: Any) -> float:
    """حساب تعرج المسار. مطابق للنوت بوك."""
    if not isinstance(waypoints, list) or len(waypoints) < 3:
        return 0.0
    try:
        pts = np.array([p[:3] for p in waypoints if isinstance(p, list) and len(p) >= 2], dtype=float)
        if len(pts) < 3:
            return 0.0
        vecs = pts[1:] - pts[:-1]
        total = 0.0
        for i in range(len(vecs) - 1):
            v1, v2 = vecs[i], vecs[i + 1]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 == 0 or n2 == 0:
                continue
            cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            total += float(np.degrees(np.arccos(cos_a)))
        return round(total, 2)
    except Exception:
        return 0.0


def _polygon_area(geofence: Any) -> Optional[float]:
    """حساب مساحة المضلع الجغرافي. مطابق للنوت بوك."""
    try:
        pts = []
        if isinstance(geofence, list) and len(geofence) > 0 and isinstance(geofence[0], dict):
            pts = geofence[0].get("points", [])
        elif isinstance(geofence, dict):
            pts = geofence.get("points", [])
        if not isinstance(pts, list) or len(pts) < 3:
            return None
        x = np.array([p[0] for p in pts], dtype=float)
        y = np.array([p[1] for p in pts], dtype=float)
        return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
    except Exception:
        return None


# ==============================================================================
# 4. هيكل نتيجة الـ Canonicalization
# ==============================================================================

@dataclass
class CanonicalizationResult:
    """نتيجة عملية الـ canonicalization الكاملة."""
    status: str  # "OK" أو "CORE_FIELD_MISSING" أو "CANONICAL_ERROR"
    df: Optional[pd.DataFrame] = None
    feature_vector: Optional[np.ndarray] = None
    data_quality_report: Dict[str, Any] = field(default_factory=dict)
    missing_core_fields: List[str] = field(default_factory=list)
    missing_optional_fields: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ==============================================================================
# 5. الدالة الرئيسية: canonicalize_scenario
# ==============================================================================

def canonicalize_scenario(
    flat_data: Dict[str, Any],
    feature_registry: FeatureRegistry,
    preprocessor: Any
) -> CanonicalizationResult:
    """
    تحويل القاموس الخام إلى Feature Vector جاهز للنموذج.

    الخطوات:
    1. التحقق من الحقول الحرجة (Core Fields).
    2. معالجة الحقول الاختيارية (Optional Fields) مع أعلام الفقدان.
    3. حساب الميزات المشتقة (Derived Features) — مطابق للنوت بوك.
    4. بناء DataFrame بالأعمدة الـ 51 الأصلية بالترتيب الصحيح.
    5. تطبيق الـ preprocessor للحصول على مصفوفة (58,).
    6. إخراج تقرير جودة البيانات.

    Args:
        flat_data: القاموس المسطح من input_contract.
        feature_registry: مرجع الميزات الرسمي.
        preprocessor: الـ ColumnTransformer المحمل من artifacts.

    Returns:
        CanonicalizationResult: النتيجة الكاملة.
    """
    warnings: List[str] = []
    missing_optional: List[str] = []

    # =========================================================================
    # STEP 1: Core Field Validation
    # =========================================================================
    missing_core: List[str] = []
    for field in CORE_FIELDS:
        if field not in flat_data or flat_data[field] is None:
            missing_core.append(field)

    if missing_core:
        logger.error(f"❌ CORE FIELD MISSING: {missing_core}")
        return CanonicalizationResult(
            status="CORE_FIELD_MISSING",
            missing_core_fields=missing_core,
            warnings=[f"Missing core fields: {missing_core}"]
        )

    # =========================================================================
    # STEP 2: Build Engineered Row (مطابق للنوت بوك)
    # =========================================================================
    row: Dict[str, Any] = {}

    # --- 2.1 Raw Numeric Fields ---
    raw_numeric_keys = [
        "uav.mass_kg", "uav.max_speed_mps", "uav.battery_model.hover_power_W",
        "environment.weather.wind_mps", "environment.weather.gust_mps",
        "environment.gnss_jam_dbm", "daa.sep_threshold_m", "daa.ttc_threshold_s",
        "airspace.altitude_agl_max_m"
    ]
    for k in raw_numeric_keys:
        row[k] = _safe_float(flat_data.get(k), 0.0)

    # --- 2.2 Raw Categorical Fields ---
    raw_categ_keys = ["uav.type", "mission.type", "mission.pattern", "environment.weather.visibility"]
    for k in raw_categ_keys:
        v = flat_data.get(k)
        row[k] = str(v).lower().strip() if v is not None else np.nan

    # --- 2.3 Boolean Fields (Tri-State → two columns each) ---
    for k in BOOL_FIELDS_WITH_MISSING:
        v01, miss = _tri_state_bool(flat_data.get(k))
        row[k] = v01
        row[f"is_missing__{k}"] = miss

    # --- 2.4 Sensors (Tri-State) ---
    sensors = flat_data.get("uav.sensors", {})
    sensors = sensors if isinstance(sensors, dict) else {}
    present_cnt = 0
    for s in SENSOR_KEYS:
        v = sensors.get(s, None)
        v01, miss = _tri_state_bool(v)
        row[f"has_{s}"] = v01
        row[f"is_missing__has_{s}"] = miss
        if miss == 0:
            present_cnt += 1

    # --- 2.5 Derived Physics/System Features ---
    waypoints = flat_data.get("mission.waypoints")
    dist, climb = _mission_dist_climb(waypoints)
    row["feat_mission_dist_m"] = dist
    row["feat_mission_climb_m"] = climb
    row["feat_mission_tortuosity"] = _tortuosity(waypoints)

    obstacles = flat_data.get("moving_obstacles")
    obstacles = obstacles if isinstance(obstacles, list) else []
    row["feat_obstacle_count"] = len(obstacles)
    speeds = []
    for o in obstacles:
        if isinstance(o, dict):
            vel = o.get("vel", [0, 0, 0])
            try:
                speeds.append(float(np.linalg.norm(np.array(vel, dtype=float))))
            except Exception:
                pass
    row["feat_obstacle_avg_speed"] = float(np.mean(speeds)) if speeds else 0.0

    geofence = flat_data.get("airspace.geofence")
    area = _polygon_area(geofence)
    row["feat_airspace_area_m2"] = area if area is not None else np.nan
    if area is None or area <= 0:
        row["feat_obstacle_density_per_km2"] = np.nan
    else:
        row["feat_obstacle_density_per_km2"] = (len(obstacles) / area) * 1e6

    weather_obj = flat_data.get("environment.weather")
    row["feat_weather_score"] = _weather_score(weather_obj)

    mass = row.get("uav.mass_kg", np.nan)
    power = row.get("uav.battery_model.hover_power_W", np.nan)
    if pd.notna(mass) and mass > 0 and pd.notna(power):
        row["feat_power_to_weight"] = power / mass
    else:
        row["feat_power_to_weight"] = np.nan

    # --- 2.6 Data Quality Features ---
    core_keys_for_dq = ["uav.mass_kg", "mission.waypoints", "environment.weather.wind_mps", "environment.gnss_jam_dbm"]
    core_present = sum([1 for k in core_keys_for_dq if flat_data.get(k) is not None])
    row["dq_core_present_pct"] = core_present / len(core_keys_for_dq)
    row["dq_weather_present"] = 1.0 if isinstance(weather_obj, dict) and len(weather_obj) > 0 else 0.0
    row["dq_uav_present"] = 1.0 if isinstance(flat_data.get("uav"), dict) else 0.0
    row["dq_mission_present"] = 1.0 if isinstance(flat_data.get("mission"), dict) else 0.0
    row["dq_comms_present"] = 1.0 if isinstance(flat_data.get("comms"), dict) else 0.0
    row["dq_sensors_present_pct"] = present_cnt / len(SENSOR_KEYS)

    # --- 2.7 Check Optional Fields ---
    optional_check_fields = list(BOOL_FIELDS_WITH_MISSING.keys()) + [f"has_{s}" for s in SENSOR_KEYS]
    for field in optional_check_fields:
        if field not in flat_data or flat_data[field] is None:
            missing_optional.append(field)

    # =========================================================================
    # STEP 3: Build DataFrame with Exact Columns
    # =========================================================================
    expected_input_columns = list(preprocessor.feature_names_in_)

    for col in expected_input_columns:
        if col not in row:
            row[col] = np.nan

    try:
        df = pd.DataFrame([row]).reindex(columns=expected_input_columns)
    except Exception as e:
        logger.error(f"❌ Failed to build DataFrame: {e}")
        return CanonicalizationResult(status="CANONICAL_ERROR", warnings=[str(e)])

    # =========================================================================
    # STEP 4: Preprocessing → Feature Vector (58,)
    # =========================================================================
    try:
        X_transformed = preprocessor.transform(df)
        feature_vector = X_transformed[0] if len(X_transformed.shape) == 2 else X_transformed
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        return CanonicalizationResult(status="CANONICAL_ERROR", warnings=[str(e)])

    # =========================================================================
    # STEP 5: Validate Feature Vector against Registry
    # =========================================================================
    if not feature_registry.validate_vector(feature_vector):
        logger.error("❌ Feature vector validation failed!")
        return CanonicalizationResult(
            status="CANONICAL_ERROR",
            warnings=["Feature vector dimension mismatch with FeatureRegistry"]
        )

    # =========================================================================
    # STEP 6: Build Warnings and Data Quality Report
    # =========================================================================
    for field in CORE_FIELDS:
        meta_warnings = feature_registry.get_warnings_for_values(field, row.get(field, 0))
        warnings.extend(meta_warnings)

    data_quality_report = {
        "dq_core_present_pct": row["dq_core_present_pct"],
        "dq_weather_present": row["dq_weather_present"],
        "dq_uav_present": row["dq_uav_present"],
        "dq_mission_present": row["dq_mission_present"],
        "dq_comms_present": row["dq_comms_present"],
        "dq_sensors_present_pct": row["dq_sensors_present_pct"],
        "missing_optional_fields": missing_optional,
        "feature_count": len(feature_vector),
    }

    logger.info(f"✅ Canonicalization complete. Feature vector shape: {feature_vector.shape}")
    return CanonicalizationResult(
        status="OK",
        df=df,
        feature_vector=feature_vector,
        data_quality_report=data_quality_report,
        missing_optional_fields=missing_optional,
        warnings=warnings
    )