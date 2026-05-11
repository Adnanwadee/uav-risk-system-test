"""
ML Module Schemas - Data models for Stage1 inference.

This module defines the core data structures used by the ML pipeline:
- RiskClass: Enum for risk categories
- MLResult: Complete inference result from the model
- FeatureImportance: Individual feature SHAP contribution
- ModelBundle: Container for loaded model artifacts

These schemas are used by:
- loader.py: Loading the model bundle
- inference.py: Returning inference results
- shap_explain.py: Explaining feature contributions
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone


# ============================================================
# Enums
# ============================================================

class RiskClass(str, Enum):
    """Risk categories from Stage1 LightGBM model."""
    HIGH_RISK = "High Risk"
    MEDIUM_RISK = "Medium Risk"
    LOW_RISK = "Low Risk"
    
    @classmethod
    def from_string(cls, value: str) -> "RiskClass":
        """Convert string to RiskClass enum."""
        mapping = {
            "High Risk": cls.HIGH_RISK,
            "Medium Risk": cls.MEDIUM_RISK,
            "Low Risk": cls.LOW_RISK,
        }
        return mapping.get(value, cls.MEDIUM_RISK)
    
    def to_decision(self) -> str:
        """Convert risk class to operational decision."""
        mapping = {
            self.HIGH_RISK: "NO-GO",
            self.MEDIUM_RISK: "CAUTION",
            self.LOW_RISK: "GO",
        }
        return mapping.get(self, "CAUTION")


# ============================================================
# Data Classes
# ============================================================

@dataclass
class FeatureImportance:
    """
    Individual feature importance from SHAP analysis.
    
    Attributes:
        feature_name: Name of the feature (from feature_mapping.json)
        shap_value: SHAP contribution for this feature (positive = increases risk)
        feature_value: Raw physical value of the feature
        description: Human-readable description of the feature
    """
    feature_name: str
    shap_value: float
    feature_value: Optional[float] = None
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "feature_name": self.feature_name,
            "shap_value": self.shap_value,
            "feature_value": self.feature_value,
            "description": self.description,
        }


@dataclass
class MLResult:
    """
    Complete inference result from Stage1 LightGBM model.
    
    Attributes:
        risk_class: Predicted risk category (High/Medium/Low)
        risk_score: Numeric score (0-1) derived from probabilities
        confidence: Model confidence in the prediction (0-1)
        probabilities: Raw probabilities for each class
        top_features: Most important features (by |SHAP|) with contributions
        model_version: Version of the model used
        mapping_version: Version of feature mapping used
        feature_vector_hash: Hash of the input feature vector (for audit)
        timestamp: UTC timestamp of inference
        shap_expected_values: Base SHAP values for each class
    """
    risk_class: RiskClass
    risk_score: float
    confidence: float
    probabilities: Dict[str, float]
    top_features: List[FeatureImportance]
    model_version: str = "unknown"
    mapping_version: str = "1.0"
    feature_vector_hash: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    shap_expected_values: Optional[List[float]] = None
    
    def __post_init__(self):
        """Validate that values are within expected ranges."""
        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError(f"risk_score must be in [0, 1], got {self.risk_score}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.probabilities:
            prob_sum = sum(self.probabilities.values())
            if abs(prob_sum - 1.0) > 0.01:
                raise ValueError(f"Probabilities must sum to ~1.0, got {prob_sum}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "risk_class": self.risk_class.value,
            "risk_decision": self.risk_class.to_decision(),
            "risk_score": self.risk_score,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "top_features": [f.to_dict() for f in self.top_features],
            "model_version": self.model_version,
            "mapping_version": self.mapping_version,
            "feature_vector_hash": self.feature_vector_hash,
            "timestamp": self.timestamp,
        }
    
    def get_operational_decision(self) -> str:
        """Get the operational decision (GO/CAUTION/NO-GO)."""
        return self.risk_class.to_decision()


@dataclass
class ModelBundle:
    """
    Container for loaded Stage1 model artifacts.
    
    Attributes:
        model: LightGBM classifier
        preprocessor: ColumnTransformer for feature preprocessing
        label_encoder: Encoder for converting class names to/from indices
        feature_names: List of 198 feature names in order
        class_names: List of class names ["High Risk", "Low Risk", "Medium Risk"]
        metadata: Additional model metadata (model_card.json content)
        feature_mapping: Feature definitions from feature_mapping.json
    """
    model: Any
    preprocessor: Any
    label_encoder: Any
    feature_names: List[str]
    class_names: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    feature_mapping: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_features(self) -> int:
        """Number of features expected by the model."""
        return len(self.feature_names)
    
    @property
    def n_classes(self) -> int:
        """Number of risk classes."""
        return len(self.class_names)
    
    def get_model_version(self) -> str:
        """
        Get model version from metadata dynamically.
        
        Returns:
            Model version string or 'unknown' if not found
        """
        return self.metadata.get("version", self.metadata.get("model_version", "unknown"))


# ============================================================
# Helper Functions
# ============================================================

def calculate_risk_score(probabilities: Dict[str, float]) -> float:
    """
    Calculate a normalized risk score from class probabilities.
    
    Weighted formula:
    - High Risk: weight 1.0
    - Medium Risk: weight 0.5
    - Low Risk: weight 0.0
    
    Args:
        probabilities: Dictionary mapping class name to probability
        
    Returns:
        Risk score between 0 (Low Risk) and 1 (High Risk)
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
    """
    Calculate confidence as the maximum probability.
    
    Args:
        probabilities: Dictionary mapping class name to probability
        
    Returns:
        Confidence score between 0 and 1
    """
    return round(max(probabilities.values()), 4)


def probabilities_to_dict(probabilities: Any, class_names: List[str]) -> Dict[str, float]:
    """
    Convert model probabilities (list or array) to dictionary.
    
    Handles various input formats:
    - numpy array
    - list of lists (batch prediction)
    - flat list
    
    Args:
        probabilities: List/array of probabilities from model.predict_proba()
        class_names: List of class names matching probabilities order
        
    Returns:
        Dictionary mapping class name to probability
        
    Raises:
        ValueError: If probabilities length doesn't match class_names
    """
    # Convert numpy array to list
    if hasattr(probabilities, 'tolist'):
        probabilities = probabilities.tolist()
    
    # Handle batch prediction (list of samples)
    if isinstance(probabilities, (list, tuple)):
        if len(probabilities) > 0 and isinstance(probabilities[0], (list, tuple)):
            probabilities = probabilities[0]
    
    # Validate length
    if len(probabilities) != len(class_names):
        raise ValueError(f"Expected {len(class_names)} probabilities, got {len(probabilities)}")
    
    return {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}