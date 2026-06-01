"""
RAG V3 Configuration - Dynamic, Environment-aware, Hot-reloadable.
Production-ready with local offline model paths.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

from uav_risk.core.env import load_project_env

load_project_env()

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# Canonical Path Resolvers
# ═══════════════════════════════════════════════════════════
_STAGE2_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_DOCS_DIR = _STAGE2_DIR / "docs"
_DEFAULT_MODELS_DIR = _STAGE2_DIR / "knowledge" / "models"
_DEFAULT_INDEX_DIR = _STAGE2_DIR / "knowledge" / "vectdb"


def _resolve_path(value: Path | str) -> Path:
    return Path(value).expanduser().resolve()


def get_rag_base_dir() -> Path:
    # Base dir remains available for explicit compatibility overrides only.
    raw = os.getenv("UAV_RAG_BASE_DIR", str(_STAGE2_DIR / "knowledge"))
    return _resolve_path(raw)


def get_index_dir() -> Path:
    raw = os.getenv("UAV_RAG_INDEX_DIR")
    if raw:
        candidate = Path(raw).expanduser()
        if candidate.is_absolute():
            return _resolve_path(candidate)
        return _resolve_path(Path.cwd() / candidate)
    return _resolve_path(_DEFAULT_INDEX_DIR)


def get_dense_index_path() -> Path:
    return _resolve_path(get_index_dir() / "dense_index.faiss")


def get_sparse_index_path() -> Path:
    return _resolve_path(get_index_dir() / "sparse_index.pkl")


def get_dense_mapping_path() -> Path:
    return _resolve_path(get_index_dir() / "dense_mapping.json")


def get_index_metadata_path() -> Path:
    return _resolve_path(get_index_dir() / "metadata.json")


def get_docs_dir() -> Path:
    raw = os.getenv("UAV_RAG_DOCS_DIR")
    if raw:
        candidate = Path(raw).expanduser()
        if candidate.is_absolute():
            return _resolve_path(candidate)
        return _resolve_path(Path.cwd() / candidate)
    return _resolve_path(_DEFAULT_DOCS_DIR)


def get_models_dir() -> Path:
    raw = os.getenv("UAV_RAG_MODELS_DIR")
    if raw:
        candidate = Path(raw).expanduser()
        if candidate.is_absolute():
            return _resolve_path(candidate)
        return _resolve_path(Path.cwd() / candidate)
    return _resolve_path(_DEFAULT_MODELS_DIR)

# ═══════════════════════════════════════════════════════════
# Base Paths (مرنة عبر Environment)
# ═══════════════════════════════════════════════════════════
BASE_DIR = get_rag_base_dir()
INDEX_DIR = get_index_dir()
CACHE_DIR = _resolve_path(BASE_DIR / "cache")
LOG_DIR = _resolve_path(BASE_DIR / "logs")

# ═══════════════════════════════════════════════════════════
# Document & Model Paths (من البيئة أو القيم الافتراضية)
# ═══════════════════════════════════════════════════════════
DOCS_PATH = get_docs_dir()
_MODELS_DIR = get_models_dir()
EMBEDDING_PATH = _resolve_path(_MODELS_DIR / "embedding")
RERANKER_PATH = _resolve_path(_MODELS_DIR / "reranker")

# Ensure directories exist
for d in [INDEX_DIR, CACHE_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# Groq LLM Configuration
# ═══════════════════════════════════════════════════════════
@dataclass
class GroqLLMConfig:
    """Configuration for Groq LLM client"""
    api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("UAV_LLM_MODEL", "llama3-70b-8192"))
    temperature: float = 0.2
    max_tokens: int = 2000
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    def validate(self) -> bool:
        if not self.api_key:
            logger.error("GROQ_API_KEY not set!")
            return False
        return True

# ═══════════════════════════════════════════════════════════
# Dynamic Threshold Manager
# ═══════════════════════════════════════════════════════════
@dataclass
class DynamicThresholdManager:
    """Thresholds that adapt based on corpus statistics and query history"""

    base_min_score: float = 0.35
    corpus_size_factor: float = 0.02  # Adjust per 1000 docs
    query_history_window: int = 100

    # Runtime mutable thresholds
    min_relevance_score: float = field(default=0.35)
    hyde_trigger_threshold: float = field(default=0.45)
    rerank_cutoff: float = field(default=0.50)

    def __post_init__(self):
        self._history: List[float] = []
        self._last_updated = datetime.now()

    def update_from_corpus(self, corpus_size: int):
        """Adjust thresholds based on corpus size"""
        size_adjustment = (corpus_size / 1000) * self.corpus_size_factor
        self.min_relevance_score = min(0.55, self.base_min_score + size_adjustment)
        self.hyde_trigger_threshold = self.min_relevance_score + 0.10
        self.rerank_cutoff = self.min_relevance_score + 0.15
        logger.info(f"Thresholds updated: min={self.min_relevance_score:.3f}, "
                   f"hyde={self.hyde_trigger_threshold:.3f}, "
                   f"rerank={self.rerank_cutoff:.3f}")

    def record_query_result(self, top_score: float):
        """Learn from query results to refine thresholds"""
        self._history.append(top_score)
        if len(self._history) > self.query_history_window:
            self._history.pop(0)

        if len(self._history) >= 20:
            avg_top = sum(self._history) / len(self._history)
            # If average top score is low, lower threshold slightly
            if avg_top < self.min_relevance_score:
                self.min_relevance_score = max(0.25, avg_top - 0.05)

    def to_dict(self) -> Dict:
        return {
            "min_relevance_score": self.min_relevance_score,
            "hyde_trigger_threshold": self.hyde_trigger_threshold,
            "rerank_cutoff": self.rerank_cutoff,
            "history_size": len(self._history),
            "last_updated": self._last_updated.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DynamicThresholdManager":
        inst = cls(
            base_min_score=data.get("base_min_score", 0.35),
            min_relevance_score=data.get("min_relevance_score", 0.35),
            hyde_trigger_threshold=data.get("hyde_trigger_threshold", 0.45),
            rerank_cutoff=data.get("rerank_cutoff", 0.50)
        )
        inst._last_updated = datetime.fromisoformat(data.get("last_updated", datetime.now().isoformat()))
        return inst

# ═══════════════════════════════════════════════════════════
# Global Threshold Instance
# ═══════════════════════════════════════════════════════════
THRESHOLD_MANAGER = DynamicThresholdManager()

# ═══════════════════════════════════════════════════════════
# Core Settings
# ═══════════════════════════════════════════════════════════
class RAGConfig:
    """Centralized configuration with hot-reload support"""
    BASE_DIR = BASE_DIR
    INDEX_DIR = INDEX_DIR
    CACHE_DIR = CACHE_DIR
    LOG_DIR = LOG_DIR
    DOCS_PATH = DOCS_PATH
    DENSE_INDEX_PATH = get_dense_index_path()
    SPARSE_INDEX_PATH = get_sparse_index_path()
    DENSE_MAPPING_PATH = get_dense_mapping_path()
    INDEX_METADATA_PATH = get_index_metadata_path()


    # Model Settings
    EMBEDDING_MODEL = os.getenv("UAV_EMBED_MODEL", str(EMBEDDING_PATH))
    LLM_MODEL = os.getenv("UAV_LLM_MODEL", "llama3-70b-8192")

    # Retrieval Settings
    TOP_K_INITIAL = int(os.getenv("UAV_TOP_K", 20))
    TOP_K_FINAL = int(os.getenv("UAV_TOP_K_FINAL", 8))

    # Hybrid Settings
    DENSE_WEIGHT = float(os.getenv("UAV_DENSE_WEIGHT", 0.6))
    SPARSE_WEIGHT = float(os.getenv("UAV_SPARSE_WEIGHT", 0.4))

    # Reranker Settings
    USE_RERANKER = os.getenv("UAV_USE_RERANKER", "true").lower() == "true"
    RERANKER_MODEL = os.getenv("UAV_RERANKER_MODEL", str(RERANKER_PATH))
    RERANKER_TOP_K = int(os.getenv("UAV_RERANKER_TOP_K", 20))

    # Adaptive RRF
    RRF_BASE_K = int(os.getenv("UAV_RRF_K", 60))  # Base, adaptive override exists

    # Evidence Logging
    MAX_EVIDENCE_ENTRIES = int(os.getenv("UAV_MAX_EVIDENCE", 1000))
    EVIDENCE_ROTATION_SIZE = int(os.getenv("UAV_EVIDENCE_ROTATION", 5000))

    # Security
    FAISS_HMAC_SECRET = os.getenv("UAV_FAISS_SECRET", "")

    # Debug
    DEBUG_MODE = os.getenv("UAV_DEBUG", "false").lower() == "true"

    @classmethod
    def reload(cls):
        """Hot-reload configuration from environment"""
        for key, value in os.environ.items():
            if key.startswith("UAV_"):
                attr_name = key.replace("UAV_", "").lower()
                if hasattr(cls, attr_name):
                    current = getattr(cls, attr_name)
                    if isinstance(current, int):
                        setattr(cls, attr_name, int(value))
                    elif isinstance(current, float):
                        setattr(cls, attr_name, float(value))
                    elif isinstance(current, bool):
                        setattr(cls, attr_name, value.lower() == "true")
                    else:
                        setattr(cls, attr_name, value)
        logger.info("Configuration hot-reloaded from environment")

    @classmethod
    def to_dict(cls) -> Dict:
        return {
            k: v for k, v in cls.__dict__.items() 
            if not k.startswith("_") and not callable(v)
        }

# ═══════════════════════════════════════════════════════════
# Feature Query Definitions (68 Core + 130 Optional)
# ═══════════════════════════════════════════════════════════
CORE_FEATURE_QUERIES = {
    # Flight Dynamics
    "flight_altitude_m": "altitude flight operation UAV drone maximum ceiling AGL MSL",
    "wind_speed_kt": "wind speed knots UAV drone operation limit gust tolerance",
    "temperature_c": "temperature Celsius UAV drone battery motor performance limit",
    "visibility_km": "visibility kilometer UAV drone VLOS BVLOS operation requirement",
    "precipitation_mm": "precipitation rain mm UAV drone water resistance operation",
    "cloud_ceiling_m": "cloud ceiling meter UAV drone VLOS operation minimum",
    "humidity_percent": "humidity percentage UAV drone electronics moisture protection",
    "pressure_hpa": "atmospheric pressure hPa UAV drone altimeter calibration",

    # Aircraft Specs
    "uav_weight_kg": "UAV weight kg mass classification category regulation",
    "max_speed_ms": "maximum speed m/s UAV drone velocity limit regulation",
    "battery_capacity_mah": "battery capacity mAh UAV drone endurance flight time",
    "motor_count": "motor count UAV drone multirotor configuration redundancy",
    "wing_span_m": "wing span meter UAV drone fixed wing dimension",
    "payload_weight_kg": "payload weight kg UAV drone cargo delivery limit",
    "flight_time_min": "flight time minute UAV drone endurance battery limit",
    "communication_range_km": "communication range km UAV drone C2 link LOS",

    # Operational
    "operation_type": "operation type UAV drone commercial recreational survey inspection",
    "airspace_class": "airspace class UAV drone controlled uncontrolled restricted",
    "flight_mission": "flight mission UAV drone purpose objective task",
    "operator_certification": "operator certification UAV drone license remote pilot",
    "insurance_status": "insurance UAV drone liability coverage policy",
    "maintenance_status": "maintenance UAV drone inspection airworthiness condition",
    "software_version": "software firmware version UAV drone update patch",
    "hardware_version": "hardware version UAV drone model revision",

    # Risk Factors
    "population_density": "population density UAV drone overflight people crowd",
    "critical_infrastructure": "critical infrastructure UAV drone power plant airport hospital",
    "terrain_type": "terrain type UAV drone mountain urban forest water",
    "obstacle_proximity_m": "obstacle proximity meter UAV drone building tower distance",
    "emergency_landing_site_m": "emergency landing site meter UAV drone safe area",
    "gps_quality": "GPS quality UAV drone signal accuracy satellite count",
    "rc_link_quality": "RC link quality UAV drone control signal strength",
    "telemetry_link_quality": "telemetry link quality UAV drone data transmission",

    # Regulatory
    "regulatory_framework": "regulatory framework UAV drone regulation EASA FAA CAAC",
    "operational_authorization": "operational authorization UAV drone permit approval",
    "flight_plan_approved": "flight plan approved UAV drone ATC clearance",
    "notam_status": "NOTAM UAV drone notice to airmen restriction",
    "geo_fence_status": "geofence UAV drone restricted area no-fly zone",
    "remote_id_status": "remote ID UAV drone identification broadcast tracking",
    "conspicuity_requirement": "conspicuity UAV drone lighting marking visibility",
    "detect_avoid_capability": "detect and avoid UAV drone DAA sense avoid collision",

    # Environmental
    "magnetic_interference": "magnetic interference UAV drone compass calibration deviation",
    "rf_interference": "RF interference UAV drone signal jamming communication",
    "solar_activity": "solar activity UAV drone GPS interference geomagnetic storm",
    "bird_activity": "bird activity UAV drone wildlife strike risk",
    "dust_sand_conditions": "dust sand UAV drone engine intake filter protection",
    "icing_conditions": "icing UAV drone ice accumulation rotor wing performance",
    "turbulence_severity": "turbulence severity UAV drone gust load factor stability",
    "wind_shear_present": "wind shear UAV drone sudden wind change hazard",

    # Emergency
    "emergency_procedure_available": "emergency procedure UAV drone contingency plan",
    "parachute_equipped": "parachute UAV drone recovery system safety",
    "flight_termination_system": "flight termination system UAV drone FTS kill switch",
    "lost_link_procedure": "lost link procedure UAV drone fail-safe return home",
    "battery_fail_safe": "battery fail safe UAV drone low voltage return home",
    "geofence_breach_response": "geofence breach UAV drone response action violation",
    "system_redundancy_level": "system redundancy UAV drone backup fail-operational",
    "human_override_capability": "human override UAV drone manual control takeover",
}

OPTIONAL_FEATURE_QUERIES = {
    # Extended Environmental
    "uv_index": "UV index UAV drone plastic degradation sensor damage",
    "air_quality_index": "air quality AQI UAV drone pollution sensor accuracy",
    "pollen_concentration": "pollen concentration UAV drone filter clogging",
    "lightning_proximity_km": "lightning proximity km UAV drone thunderstorm safety",
    "fog_visibility_m": "fog visibility meter UAV drone IFR operation",
    "dust_devil_present": "dust devil UAV drone sudden vortex hazard",
    "sea_state": "sea state UAV drone maritime operation wave height",
    "tide_status": "tide status UAV drone coastal operation water level",

    # Extended Technical
    "propeller_type": "propeller type UAV drone fixed pitch variable pitch",
    "esc_rating_a": "ESC rating ampere UAV drone electronic speed controller",
    "imu_calibration_status": "IMU calibration UAV drone inertial measurement unit",
    "barometer_accuracy_m": "barometer accuracy meter UAV drone altitude precision",
    "compass_interference_percent": "compass interference percent UAV drone heading error",
    "vibration_level": "vibration level UAV drone gimbal camera image quality",
    "motor_temperature_c": "motor temperature Celsius UAV drone overheating protection",
    "esc_temperature_c": "ESC temperature Celsius UAV drone thermal protection",

    # Extended Operational
    "takeoff_surface": "takeoff surface UAV drone ground type platform ship",
    "landing_surface": "landing surface UAV drone ground type precision net",
    "launch_method": "launch method UAV drone hand launch catapult VTOL",
    "recovery_method": "recovery method UAV drone landing net parachute belly",
    "refueling_procedure": "refueling UAV drone battery swap hot swap charging",
    "transport_method": "transport method UAV drone case vehicle ship",
    "assembly_time_min": "assembly time minute UAV drone setup preparation",
    "pre_flight_check_status": "pre-flight check UAV drone inspection checklist",

    # Extended Mission
    "sensor_type": "sensor type UAV drone camera LiDAR thermal multispectral",
    "sensor_resolution_mp": "sensor resolution megapixel UAV drone camera quality",
    "gimbal_stabilization": "gimbal stabilization UAV drone camera vibration isolation",
    "data_link_encryption": "data link encryption UAV drone secure communication",
    "storage_capacity_gb": "storage capacity GB UAV drone onboard recording",
    "real_time_kinematic": "RTK UAV drone real-time kinematic GPS accuracy",
    "ppk_capability": "PPK UAV drone post-processed kinematic precision",
    "ground_control_station": "GCS UAV drone ground control station software",

    # Extended Regulatory
    "noise_level_db": "noise level dB UAV drone acoustic regulation limit",
    "privacy_compliance": "privacy compliance UAV drone GDPR data protection",
    "data_retention_policy": "data retention policy UAV drone storage deletion",
    "cross_border_operation": "cross border UAV drone international operation regulation",
    "night_operation_approved": "night operation UAV drone darkness lighting waiver",
    "beyond_vlos_approved": "BVLOS UAV drone beyond visual line sight approval",
    "over_people_approved": "over people UAV drone operation crowd waiver",
    "swarm_operation": "swarm UAV drone multiple coordinated formation",

    # Extended Safety
    "e_stop_functional": "emergency stop UAV drone e-stop kill switch",
    "low_battery_rtl": "return to launch RTL UAV drone low battery automatic",
    "geofence_enforced": "geofence enforced UAV drone boundary hard limit",
    "collision_avoidance_system": "collision avoidance UAV drone obstacle detection sensor",
    "terrain_following": "terrain following UAV drone AGL constant altitude",
    "auto_landing_capability": "auto landing UAV drone precision landing RTK",
    "weather_radar_onboard": "weather radar UAV drone onboard precipitation detection",
    "tcas_equipped": "TCAS UAV drone traffic collision avoidance system",

    # Extended Risk
    "cyber_security_score": "cyber security UAV drone hacking vulnerability protection",
    "supply_chain_integrity": "supply chain UAV drone component origin counterfeit",
    "operator_fatigue_score": "operator fatigue UAV drone remote pilot duty time",
    "training_currency": "training currency UAV drone recurrent qualification",
    "safety_management_system": "SMS UAV drone safety management system ISO",
    "just_culture_policy": "just culture UAV drone reporting non-punitive",
    "risk_assessment_updated": "risk assessment UAV drone SORA specific operation",
    "insurance_coverage_amount": "insurance coverage amount UAV drone liability million",
}

# Combine all
ALL_FEATURE_QUERIES = {**CORE_FEATURE_QUERIES, **OPTIONAL_FEATURE_QUERIES}

# ═══════════════════════════════════════════════════════════
# Risk Thresholds (External File Support)
# ═══════════════════════════════════════════════════════════
RISK_THRESHOLDS_PATH = BASE_DIR / "risk_thresholds.json"

DEFAULT_RISK_THRESHOLDS = {
    "wind_speed_kt": {"warning": 15, "critical": 25, "emergency": 35},
    "flight_altitude_m": {"warning": 120, "critical": 400, "emergency": 500},
    "temperature_c": {"warning": 40, "critical": 50, "emergency": 60, "low_warning": -10, "low_critical": -20},
    "visibility_km": {"warning": 3, "critical": 1, "emergency": 0.5},
    "precipitation_mm": {"warning": 1, "critical": 5, "emergency": 10},
    "obstacle_proximity_m": {"warning": 50, "critical": 20, "emergency": 10},
    "battery_capacity_mah": {"warning": 20, "critical": 10, "emergency": 5},  # % remaining
    "communication_range_km": {"warning": 2, "critical": 1, "emergency": 0.5},
    "population_density": {"warning": 100, "critical": 1000, "emergency": 10000},  # per km²
    "gps_quality": {"warning": 8, "critical": 5, "emergency": 3},  # satellite count
    "rc_link_quality": {"warning": 70, "critical": 50, "emergency": 30},  # % signal
    "humidity_percent": {"warning": 80, "critical": 90, "emergency": 95},
}

def load_risk_thresholds() -> Dict:
    """Load thresholds from external file with fallback to defaults"""
    if RISK_THRESHOLDS_PATH.exists():
        try:
            with open(RISK_THRESHOLDS_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load risk thresholds: {e}, using defaults")
    return DEFAULT_RISK_THRESHOLDS.copy()

def save_risk_thresholds(thresholds: Dict):
    """Save updated thresholds to external file"""
    with open(RISK_THRESHOLDS_PATH, "w") as f:
        json.dump(thresholds, f, indent=2)

RISK_THRESHOLDS = load_risk_thresholds()