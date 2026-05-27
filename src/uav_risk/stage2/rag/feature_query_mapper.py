"""
Feature Query Mapper - Dynamic Value-to-Query Translation
V3.1 FIX: map_scenario now properly applies SHAP boost and re-sorts
"""
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class FeatureQuery:
    """Generated query for a feature"""
    feature_name: str
    value: Any
    query_text: str
    risk_level: RiskLevel
    priority: int  # 1-10
    context: str

class FeatureQueryMapper:
    """
    Dynamically maps feature values to search queries.
    No hardcoded thresholds - reads from external config.
    """

    def __init__(self, config_module=None):
        self.config = config_module
        self._thresholds = self._load_thresholds()
        self._query_cache: Dict[str, str] = {}

    def _load_thresholds(self) -> Dict:
        """Load thresholds from config or external file"""
        if self.config and hasattr(self.config, "RISK_THRESHOLDS"):
            return self.config.RISK_THRESHOLDS

        return {
            "wind_speed_kt": {"warning": 15, "critical": 25, "emergency": 35},
            "flight_altitude_m": {"warning": 120, "critical": 400, "emergency": 500},
            "temperature_c": {"warning": 40, "critical": 50, "emergency": 60},
            "visibility_km": {"warning": 3, "critical": 1, "emergency": 0.5},
            "precipitation_mm": {"warning": 1, "critical": 5, "emergency": 10},
            "obstacle_proximity_m": {"warning": 50, "critical": 20, "emergency": 10},
            "battery_capacity_mah": {"warning": 20, "critical": 10, "emergency": 5},
            "communication_range_km": {"warning": 2, "critical": 1, "emergency": 0.5},
            "population_density": {"warning": 100, "critical": 1000, "emergency": 10000},
            "gps_quality": {"warning": 8, "critical": 5, "emergency": 3},
            "rc_link_quality": {"warning": 70, "critical": 50, "emergency": 30},
            "humidity_percent": {"warning": 80, "critical": 90, "emergency": 95},
        }

    def _get_risk_level(self, feature_name: str, value: float) -> RiskLevel:
        """Determine risk level based on thresholds"""
        if feature_name not in self._thresholds:
            return RiskLevel.LOW

        thresholds = self._thresholds[feature_name]

        if "emergency" in thresholds and value >= thresholds["emergency"]:
            return RiskLevel.EMERGENCY
        if "critical" in thresholds and value >= thresholds["critical"]:
            return RiskLevel.CRITICAL
        if "warning" in thresholds and value >= thresholds["warning"]:
            return RiskLevel.WARNING

        if "low_critical" in thresholds and value <= thresholds["low_critical"]:
            return RiskLevel.CRITICAL
        if "low_warning" in thresholds and value <= thresholds["low_warning"]:
            return RiskLevel.WARNING

        return RiskLevel.LOW

    def _build_query(self, feature_name: str, value: Any, 
                    risk_level: RiskLevel) -> str:
        """Build contextual query based on feature and risk"""

        base_query = ""
        if self.config and hasattr(self.config, "ALL_FEATURE_QUERIES"):
            base_query = self.config.ALL_FEATURE_QUERIES.get(feature_name, "")

        if not base_query:
            base_query = feature_name.replace("_", " ")

        if risk_level == RiskLevel.EMERGENCY:
            return f"{base_query} emergency critical severe danger immediate action required"
        elif risk_level == RiskLevel.CRITICAL:
            return f"{base_query} critical high risk danger safety limit exceeded"
        elif risk_level == RiskLevel.WARNING:
            return f"{base_query} warning caution elevated risk approaching limit"
        else:
            return f"{base_query} normal operation standard procedure"

    def _get_priority(self, feature_name: str, risk_level: RiskLevel) -> int:
        """Calculate query priority (1-10)"""
        base_priority = {
            RiskLevel.LOW: 3,
            RiskLevel.WARNING: 6,
            RiskLevel.CRITICAL: 9,
            RiskLevel.EMERGENCY: 10
        }.get(risk_level, 5)

        critical_params = ["flight_altitude_m", "wind_speed_kt", "obstacle_proximity_m", 
                          "battery_capacity_mah", "communication_range_km"]
        if feature_name in critical_params:
            base_priority = min(10, base_priority + 1)

        return base_priority

    def map_feature(self, feature_name: str, value: Any) -> FeatureQuery:
        """Map a single feature to an intelligent query"""
        try:
            numeric_value = float(value) if value is not None else 0.0
        except (ValueError, TypeError):
            numeric_value = 0.0

        risk_level = self._get_risk_level(feature_name, numeric_value)
        query_text = self._build_query(feature_name, value, risk_level)
        priority = self._get_priority(feature_name, risk_level)

        context = f"{feature_name}={value} (risk: {risk_level.value})"

        return FeatureQuery(
            feature_name=feature_name,
            value=value,
            query_text=query_text,
            risk_level=risk_level,
            priority=priority,
            context=context
        )

    def map_features(self, features: Dict[str, Any]) -> List[FeatureQuery]:
        """Map multiple features to queries, sorted by priority desc"""
        queries = []
        for feature_name, value in features.items():
            if value is not None:
                query = self.map_feature(feature_name, value)
                queries.append(query)

        queries.sort(key=lambda q: q.priority, reverse=True)
        return queries

    def map_scenario(self, core_features: Dict[str, Any], 
                    optional_features: Optional[Dict[str, Any]] = None,
                    shap_features: Optional[List[Tuple[str, float]]] = None) -> List[FeatureQuery]:
        """
        Map complete scenario to prioritized queries.

        V3.1 FIX: Properly applies SHAP boost and re-sorts.
        """
        all_features = dict(core_features)
        if optional_features:
            all_features.update(optional_features)

        # Get base queries sorted by risk priority
        queries = self.map_features(all_features)

        # Apply SHAP boost and re-sort
        if shap_features:
            shap_dict = {name: abs(val) for name, val in shap_features}

            for query in queries:
                if query.feature_name in shap_dict:
                    shap_boost = min(2, shap_dict[query.feature_name] * 10)
                    query.priority = min(10, query.priority + int(shap_boost))

            # FIX: Re-sort after applying SHAP boost
            queries.sort(key=lambda q: q.priority, reverse=True)

        logger.info(f"Mapped {len(queries)} features to queries "
                   f"(top priority: {queries[0].priority if queries else 0})")
        return queries

    def get_batch_queries(self, features: Dict[str, Any], 
                         batch_size: int = 10) -> List[List[FeatureQuery]]:
        """Split queries into batches for parallel processing"""
        queries = self.map_features(features)
        batches = []
        for i in range(0, len(queries), batch_size):
            batches.append(queries[i:i+batch_size])
        return batches