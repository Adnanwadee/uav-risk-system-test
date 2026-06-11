"""
Module: uav_risk.ml.schemas
Purpose: Central authoritative data contracts and multi-class data structures for Stage-1 
         machine learning inference, fortified against multi-class SHAP blind-spots.
Dependencies: Strictly standalone (Zero domain dependencies).
Source References: LightGBM Production Standards, Lundberg & Lee (2017) SHAP Frameworks.
"""

from __future__ import annotations
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone


# =====================================================================
# Enums: Authoritative Flight Directives
# =====================================================================

class RiskClass(str, Enum):
    """
    Operational Risk categories generated explicitly by the Stage1 LightGBM classifier.
    Aligned with the 3 real labels extracted from the label encoder.
    """
    HIGH_RISK = "High Risk"
    MEDIUM_RISK = "Medium Risk"
    LOW_RISK = "Low Risk"
    
    @classmethod
    def from_string(cls, value: str) -> "RiskClass":
        """Convert a string value into its corresponding RiskClass enum safely."""
        mapping = {
            "High Risk": cls.HIGH_RISK,
            "Medium Risk": cls.MEDIUM_RISK,
            "Low Risk": cls.LOW_RISK,
        }
        return mapping.get(value, cls.MEDIUM_RISK)
    
    def to_decision(self) -> str:
        """Maps the internal ML risk class to a clear baseline operational directive."""
        mapping = {
            self.HIGH_RISK: "NO-GO",
            self.MEDIUM_RISK: "CONDITIONAL-GO",
            self.LOW_RISK: "GO",
        }
        return mapping.get(self, "NO-GO")


# =====================================================================
# Data Contracts (Schemas)
# =====================================================================

@dataclass
class FeatureImportance:
    """
    Represents the mathematical SHAP feature contribution for a specific scenario feature,
    properly decoded for multi-class semantic operational risk direction.
    """
    feature_name: str
    shap_value: float
    feature_value: Optional[float] = None
    description: Optional[str] = None
    direction: str = "unknown"
    rank: int = 0
    predicted_class: Optional[str] = None
    
    shap_values_all_classes: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
       
        if self.predicted_class == "Low Risk":
            self.direction = "decreases_risk" if self.shap_value > 0 else "increases_risk"
        elif self.predicted_class == "High Risk":
            self.direction = "increases_risk" if self.shap_value > 0 else "decreases_risk"
        else:
            self.direction = "stabilizes_medium_risk" if self.shap_value > 0 else "shifts_away_from_medium"

    def to_dict(self) -> Dict[str, Any]:
        """Convert data structure to standard primitive dictionary for JSON serialization."""
        return {
            "feature_name": self.feature_name,
            "shap_value": self.shap_value,
            "direction": self.direction,
            "feature_value": self.feature_value,
            "description": self.description,
            "rank": self.rank,
            "predicted_class": self.predicted_class,
            "shap_values_all_classes": self.shap_values_all_classes
        }


@dataclass
class MLResult:
    """
    Encapsulates the complete execution response from the Stage-1 machine learning layer,
    fortified against structural data disconnect anomalies and preserving immutable raw outputs.
    """
    risk_class: RiskClass
    risk_score: float
    confidence: float
    probabilities: Dict[str, float]
    top_features: List[FeatureImportance] = field(default_factory=list)
    drift_score: float = 0.0
    drift_detected: bool = False
    processing_time_ms: float = 0.0
    model_version: str = "unknown"
    feature_vector_hash: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    shap_expected_values: Optional[List[float]] = None
    
    raw_risk_class: Optional[str] = None
    raw_probabilities: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.raw_risk_class:
            self.raw_risk_class = self.risk_class.value
        if not self.raw_probabilities and self.probabilities:
            self.raw_probabilities = dict(self.probabilities)
            
        self.validate_bounds()

    def validate_bounds(self) -> None:
        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError(f"Constraint Violation: risk_score must be in [0, 1], got {self.risk_score}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Constraint Violation: confidence must be in [0, 1], got {self.confidence}")
        if self.probabilities:
            prob_sum = sum(self.probabilities.values())
            if abs(prob_sum - 1.0) > 0.0001:
                raise ValueError(f"Constraint Violation: Probabilities must sum to 1.0 with epsilon 1e-4, got {prob_sum}")

    def calibrate_probabilities(self, target_class: RiskClass) -> None:
    
        if self.risk_class == target_class or not self.probabilities:
            return
            
        old_class = self.risk_class.value
        new_class = target_class.value
        
        old_prob = self.probabilities.get(old_class, 0.0)
        new_prob = self.probabilities.get(new_class, 0.0)
        
        if old_prob > new_prob:
            self.probabilities[new_class] = old_prob
            self.probabilities[old_class] = new_prob
            
        self.risk_class = target_class
        self.confidence = self.probabilities[new_class]
        
        self.risk_score = calculate_risk_score(self.probabilities)
        self.validate_bounds()

    def to_dict(self) -> Dict[str, Any]:
        """Transforms the object into a fully serializable nested JSON format for DB logging."""
        return {
            "risk_class": self.risk_class.value,
            "risk_decision": self.risk_class.to_decision(),
            "risk_score": self.risk_score,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "top_features": [f.to_dict() for f in self.top_features],
            "drift_score": self.drift_score,
            "drift_detected": self.drift_detected,
            "processing_time_ms": self.processing_time_ms,
            "model_version": self.model_version,
            "feature_vector_hash": self.feature_vector_hash,
            "timestamp": self.timestamp,
            "raw_risk_class": self.raw_risk_class,
            "raw_probabilities": self.raw_probabilities
        }


@dataclass
class Stage1Bundle:
    model: Any
    preprocessor: Any
    label_encoder: Any
    shap_explainer: Any
    feature_names: List[str]
    feature_mapping: Dict[str, int]  
    class_names: List[str]
    training_stats: Dict[str, Any]
    policy_config: Dict[str, Any]
    model_metadata: Dict[str, Any]
    bundle_path: str

    
    @property
    def n_features(self) -> int:
        """Total features configured for input preprocessing."""
        return len(self.feature_names)
    
    @property
    def n_classes(self) -> int:
        """Total classification categories supported by the underlying model."""
        return len(self.class_names)
    
    def get_model_version(self) -> str:
        """Dynamic retrieval of the model execution iteration version from metadata."""
        return self.model_metadata.get("version", self.model_metadata.get("pipeline_version", "unknown"))


# =====================================================================
# Core Computational Helper Functions
# =====================================================================

def calculate_risk_score(probabilities: Dict[str, float]) -> float:
    """
    Executes a mathematical weighted aggregation over the prediction probabilities.
    
    Weights mapped strictly to real Artifact categories:
    - High Risk: 1.0
    - Medium Risk: 0.5
    - Low Risk: 0.0
    """
    weights = {
        "High Risk": 1.0,
        "Medium Risk": 0.5,
        "Low Risk": 0.0,
    }
    
    score = 0.0
    for class_name, prob in probabilities.items():
        score += prob * weights.get(class_name, 0.5)
        
    return round(score, 4)


def calculate_confidence(probabilities: Dict[str, float]) -> float:
    """Computes operational model confidence index as the maximal group assignment probability."""
    if not probabilities:
        return 0.0
    return round(max(probabilities.values()), 4)


def probabilities_to_dict(probabilities: Any, class_names: List[str]) -> Dict[str, float]:
    """
    Binds raw linear output probabilities to structural string category names cleanly.
    """
    if hasattr(probabilities, 'tolist'):
        probabilities = probabilities.tolist()
        
    if isinstance(probabilities, (list, tuple)) and len(probabilities) > 0:
        if isinstance(probabilities[0], (list, tuple)):
            probabilities = probabilities[0]
            
    if len(probabilities) != len(class_names):
        raise ValueError(f"Array mismatch: Expected {len(class_names)} probability elements, got {len(probabilities)}")
        
    return {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}


__all__ = [
    "RiskClass",
    "FeatureImportance",
    "MLResult",
    "Stage1Bundle",
    "calculate_risk_score",
    "calculate_confidence",
    "probabilities_to_dict",
]

# =====================================================================
# Architectural Registry Block:
# This file serves as the strict Foundational Interface Contract Layer for ML outcomes.
# This file depends on: None (Foundational Interface Contract Layer).
# Files depending on this file: src/uav_risk/ml/loader.py, src/uav_risk/ml/inference.py, src/uav_risk/ml/shap_explain.py
# =====================================================================
