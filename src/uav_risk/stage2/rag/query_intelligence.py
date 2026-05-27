"""
Query Intelligence - Adaptive Retrieval, Learning, Scenario Analysis
V3.1 FIX: Uses .get() for all dict access to prevent KeyError
"""
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class QueryHistory:
    """Single query execution record"""
    query: str
    scenario_type: str
    used_hyde: bool
    top_score: float
    result_count: int
    latency_ms: float
    timestamp: str

    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class ScenarioProfile:
    """Learned profile for a scenario type"""
    scenario_type: str
    avg_top_score: float = 0.0
    hyde_success_rate: float = 0.0
    optimal_k: int = 60
    query_count: int = 0

    def update(self, top_score: float, used_hyde: bool):
        """Update profile with new result"""
        self.query_count += 1

        self.avg_top_score = (
            (self.avg_top_score * (self.query_count - 1) + top_score) / self.query_count
        )

        if used_hyde:
            success = 1.0 if top_score > 0.5 else 0.0
            self.hyde_success_rate = (
                (self.hyde_success_rate * (self.query_count - 1) + success) / self.query_count
            )

        if self.avg_top_score > 0.6:
            self.optimal_k = max(20, self.optimal_k - 5)
        elif self.avg_top_score < 0.3:
            self.optimal_k = min(120, self.optimal_k + 10)

class AdaptiveRRF:
    """Adaptive Reciprocal Rank Fusion"""

    def __init__(self, base_k: int = 60):
        self.base_k = base_k
        self.min_k = 20
        self.max_k = 120

    def calculate_k(self, corpus_size: int, 
                   scenario_profile: Optional[ScenarioProfile] = None) -> int:
        """Calculate optimal RRF k parameter"""
        if corpus_size < 1000:
            size_k = self.base_k - 20
        elif corpus_size < 10000:
            size_k = self.base_k
        elif corpus_size < 100000:
            size_k = self.base_k + 20
        else:
            size_k = self.base_k + 40

        if scenario_profile:
            size_k = int(0.7 * size_k + 0.3 * scenario_profile.optimal_k)

        return max(self.min_k, min(self.max_k, size_k))

    def fuse(self, dense_results: List[Tuple[str, float]], 
            sparse_results: List[Tuple[str, float]], 
            k: int,
            dense_weight: float = 0.6,
            sparse_weight: float = 0.4) -> List[Tuple[str, float]]:
        """Fuse dense and sparse results using RRF"""
        rrf_scores = defaultdict(float)

        for rank, (doc_id, score) in enumerate(dense_results):
            rrf_scores[doc_id] += dense_weight / (k + rank + 1)

        for rank, (doc_id, score) in enumerate(sparse_results):
            rrf_scores[doc_id] += sparse_weight / (k + rank + 1)

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return fused

class QueryIntelligence:
    """Central intelligence for query planning and learning"""

    def __init__(self, history_path: Optional[Path] = None, 
                 config_module=None):
        self.config = config_module
        self.history_path = history_path or Path("query_history.json")
        self.history: List[QueryHistory] = []
        self.scenario_profiles: Dict[str, ScenarioProfile] = {}
        self.adaptive_rrf = AdaptiveRRF()

        self._load_history()

    def _load_history(self):
        """Load query history from disk"""
        if self.history_path.exists():
            try:
                with open(self.history_path, "r") as f:
                    data = json.load(f)
                    self.history = [QueryHistory(**h) for h in data.get("history", [])]
                    profiles = data.get("profiles", {})
                    self.scenario_profiles = {
                        k: ScenarioProfile(**v) for k, v in profiles.items()
                    }
                logger.info(f"Loaded {len(self.history)} query history entries")
            except Exception as e:
                logger.warning(f"Failed to load history: {e}")

    def _save_history(self):
        """Save query history to disk"""
        data = {
            "history": [h.to_dict() for h in self.history[-1000:]],
            "profiles": {k: asdict(v) for k, v in self.scenario_profiles.items()},
            "last_updated": datetime.now().isoformat()
        }
        try:
            with open(self.history_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save history: {e}")

    def analyze_scenario(self, features: Dict[str, Any], 
                        free_text: Optional[str] = None) -> Dict[str, Any]:
        """Analyze scenario and determine retrieval strategy"""
        all_features = dict(features)

        scenario_type = self._classify_scenario(all_features, free_text)
        complexity = self._calculate_complexity(all_features, free_text)
        priority_features = self._extract_priority_features(all_features)
        risk_indicators = self._extract_risk_indicators(all_features)
        recommended_hyde = self._should_use_hyde(scenario_type, complexity, free_text)

        return {
            "scenario_type": scenario_type,
            "complexity": complexity,
            "recommended_hyde": recommended_hyde,
            "priority_features": priority_features,
            "risk_indicators": risk_indicators
        }

    def _classify_scenario(self, features: Dict[str, Any], 
                          free_text: Optional[str] = None) -> str:
        """Classify scenario into type - V3.1 FIX: Uses .get() everywhere"""

        # Check for emergency indicators (FIX: use .get() instead of direct access)
        if features.get("emergency_procedure_available") == True:
            return "emergency_response"
        if features.get("flight_termination_system") == True:
            return "emergency_response"
        if features.get("e_stop_functional") == True:
            return "emergency_response"

        # Check weather
        wind_speed = features.get("wind_speed_kt", 0)
        precipitation = features.get("precipitation_mm", 0)
        if wind_speed > 20 or precipitation > 5:
            return "adverse_weather"

        # Check regulatory
        if any(features.get(k) for k in ["airspace_class", "operational_authorization", "geo_fence_status"]):
            return "regulatory_compliance"

        # Check technical
        battery = features.get("battery_capacity_mah", 100)
        if battery < 20:
            return "technical_degradation"

        # Default based on operation type
        op_type = features.get("operation_type", "")
        op_type_str = str(op_type).lower()
        if "survey" in op_type_str:
            return "survey_mission"
        elif "delivery" in op_type_str:
            return "delivery_mission"
        elif "inspection" in op_type_str:
            return "inspection_mission"

        return "general_operation"

    def _calculate_complexity(self, features: Dict[str, Any], 
                           free_text: Optional[str] = None) -> float:
        """Calculate scenario complexity score (0-1)"""
        score = 0.0
        score += min(0.3, len(features) / 200)

        risk_features = ["population_density", "critical_infrastructure", "night_operation_approved"]
        risk_count = sum(1 for f in risk_features if features.get(f))
        score += risk_count * 0.15

        if free_text:
            score += min(0.2, len(free_text.split()) / 100)

        if "shap_variance" in features:
            score += min(0.2, features["shap_variance"])

        return min(1.0, score)

    def _extract_priority_features(self, features: Dict[str, Any]) -> List[str]:
        """Extract high-priority features"""
        priority = []

        critical_checks = {
            "wind_speed_kt": 25,
            "flight_altitude_m": 400,
            "obstacle_proximity_m": 20,
            "battery_capacity_mah": 10,
            "communication_range_km": 1,
            "gps_quality": 5,
            "rc_link_quality": 50,
        }

        for feat, threshold in critical_checks.items():
            val = features.get(feat)
            if val is not None:
                try:
                    if float(val) <= threshold:
                        priority.append(feat)
                except (ValueError, TypeError):
                    pass

        risk_flags = ["emergency_procedure_available", "flight_termination_system", 
                     "parachute_equipped", "system_redundancy_level"]
        for feat in risk_flags:
            if features.get(feat) == False:
                priority.append(feat)

        return priority

    def _extract_risk_indicators(self, features: Dict[str, Any]) -> List[str]:
        """Extract human-readable risk indicators"""
        indicators = []

        if features.get("wind_speed_kt", 0) > 20:
            indicators.append(f"High wind: {features['wind_speed_kt']} kt")
        if features.get("flight_altitude_m", 0) > 400:
            indicators.append(f"High altitude: {features['flight_altitude_m']} m")
        if features.get("obstacle_proximity_m", 999) < 30:
            indicators.append(f"Close obstacle: {features['obstacle_proximity_m']} m")
        if features.get("battery_capacity_mah", 100) < 15:
            indicators.append(f"Low battery: {features['battery_capacity_mah']}%")
        if features.get("population_density", 0) > 1000:
            indicators.append(f"Dense population: {features['population_density']}/km²")

        return indicators

    def _should_use_hyde(self, scenario_type: str, complexity: float, 
                        free_text: Optional[str] = None) -> bool:
        """Decide if HyDE should be used"""
        profile = self.scenario_profiles.get(scenario_type)

        if profile and profile.query_count > 5:
            if profile.hyde_success_rate > 0.6:
                return True
            if profile.hyde_success_rate < 0.3:
                return False

        if complexity > 0.5 and free_text:
            return True

        if scenario_type == "emergency_response":
            return True

        return complexity > 0.4

    def get_optimal_rrf_k(self, corpus_size: int, 
                         scenario_type: str) -> int:
        """Get optimal RRF k for scenario"""
        profile = self.scenario_profiles.get(scenario_type)
        return self.adaptive_rrf.calculate_k(corpus_size, profile)

    def record_query(self, query: str, scenario_type: str, 
                    used_hyde: bool, top_score: float, 
                    result_count: int, latency_ms: float):
        """Record query result for learning"""
        history = QueryHistory(
            query=query,
            scenario_type=scenario_type,
            used_hyde=used_hyde,
            top_score=top_score,
            result_count=result_count,
            latency_ms=latency_ms,
            timestamp=datetime.now().isoformat()
        )

        self.history.append(history)

        if scenario_type not in self.scenario_profiles:
            self.scenario_profiles[scenario_type] = ScenarioProfile(scenario_type=scenario_type)

        self.scenario_profiles[scenario_type].update(top_score, used_hyde)

        if len(self.history) % 10 == 0:
            self._save_history()

    def get_stats(self) -> Dict:
        """Get intelligence statistics"""
        return {
            "total_queries": len(self.history),
            "scenario_types": list(self.scenario_profiles.keys()),
            "profiles": {
                k: {
                    "query_count": v.query_count,
                    "avg_top_score": round(v.avg_top_score, 3),
                    "hyde_success_rate": round(v.hyde_success_rate, 3),
                    "optimal_k": v.optimal_k
                }
                for k, v in self.scenario_profiles.items()
            }
        }