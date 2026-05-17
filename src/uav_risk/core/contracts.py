# src/uav_risk/core/contracts.py
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union, Annotated
from pydantic import BaseModel, ConfigDict, Field, model_validator
import math


# ───────────────────────────────────────────────────────────────
# Tier-Specific Payloads (Strict Contracts via Discriminator)
# ───────────────────────────────────────────────────────────────
class Tier1Input(BaseModel):
    """VLOS / Low-Risk Operations."""
    tier: Literal["1"] = "1"
    speed: Optional[float] = Field(None, ge=0.0, le=10.0)
    altitude: Optional[float] = Field(None, ge=0.0, le=120.0)
    distance: Optional[float] = Field(None, ge=0.0, le=500.0)
    visibility: Optional[float] = Field(None, ge=1000.0)
    wind: Optional[float] = Field(None, ge=0.0, le=10.0)
    flight_id: Optional[str] = None
    model_config = ConfigDict(extra="forbid")


class Tier2Input(BaseModel):
    """Standard / BVLOS-Prep Operations."""
    tier: Literal["2"] = "2"
    speed: Optional[float] = Field(None, ge=0.0, le=25.0)
    altitude: Optional[float] = Field(None, ge=0.0, le=400.0)
    distance: Optional[float] = Field(None, ge=0.0, le=5000.0)
    visibility: Optional[float] = Field(None, ge=500.0)
    wind: Optional[float] = Field(None, ge=0.0, le=12.0)
    flight_id: Optional[str] = None
    model_config = ConfigDict(extra="forbid")


class Tier3Input(BaseModel):
    """Complex / Mixed-Environment Operations."""
    tier: Literal["3"] = "3"
    speed: Optional[float] = Field(None, ge=0.0, le=44.7)
    altitude: Optional[float] = Field(None, ge=0.0, le=122.0)
    distance: Optional[float] = Field(None, ge=0.0)
    visibility: Optional[float] = Field(None, ge=0.0)
    wind: Optional[Union[float, Dict[str, Any]]] = Field(None)
    flight_id: Optional[str] = None
    model_config = ConfigDict(extra="forbid")


class Tier4Input(BaseModel):
    """High-Risk / Swarm / Critical Infrastructure."""
    tier: Literal["4"] = "4"
    speed: Optional[float] = Field(None, ge=0.0)
    altitude: Optional[float] = Field(None, ge=0.0)
    distance: Optional[float] = Field(None, ge=0.0)
    visibility: Optional[float] = Field(None, ge=0.0)
    wind: Optional[Union[float, Dict[str, Any]]] = Field(None)
    requires_vlos: Optional[bool] = Field(False)
    comms_redundancy: Optional[Literal["single", "dual", "triple"]] = Field("single")
    flight_id: Optional[str] = None
    model_config = ConfigDict(extra="forbid")


# Discriminated Union Type Alias
TierPayload = Annotated[
    Union[Tier1Input, Tier2Input, Tier3Input, Tier4Input],
    Field(discriminator="tier")
]


class FlightInput(BaseModel):
    """Master input model with strict tier discriminator and auto-fallback."""
    payload: TierPayload
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def auto_fallback_tier(cls, data: Any) -> Any:
        """If 'tier' is missing in payload dict, inject auto-detected tier before validation."""
        if isinstance(data, dict):
            payload = data.get("payload")
            if isinstance(payload, dict) and "tier" not in payload:
                payload["tier"] = cls._infer_tier(payload)
            data["payload"] = payload
        return data

    @classmethod
    def _infer_tier(cls, p: Dict[str, Any]) -> Literal["1", "2", "3", "4"]:
        speed = float(p.get("speed") or 0)
        alt = float(p.get("altitude") or 0)
        dist = float(p.get("distance") or 0)
        vis = float(p.get("visibility") or 9999)
        wind = p.get("wind")
        if isinstance(wind, dict):
            wind = float(wind.get("speed", 0))
        else:
            wind = float(wind or 0)

        if alt > 120 or speed > 25 or dist > 5000 or wind > 15 or vis < 500:
            return "4"
        if alt > 400 or speed > 44.7:
            return "3"
        return "2"

    # ─── Convenience Properties for Downstream Components ───
    @property
    def tier(self) -> Literal["1", "2", "3", "4"]:
        return self.payload.tier

    @property
    def tier_level(self) -> int:
        return int(self.payload.tier)

    @property
    def speed(self) -> Optional[float]: return getattr(self.payload, "speed", None)
    @property
    def altitude(self) -> Optional[float]: return getattr(self.payload, "altitude", None)
    @property
    def distance(self) -> Optional[float]: return getattr(self.payload, "distance", None)
    @property
    def visibility(self) -> Optional[float]: return getattr(self.payload, "visibility", None)
    @property
    def wind(self) -> Optional[Any]: return getattr(self.payload, "wind", None)
    @property
    def flight_id(self) -> Optional[str]: return getattr(self.payload, "flight_id", None)

    def validate_bounds(self) -> List[str]:
        issues: List[str] = []
        checks = [
            ("speed", 100.0, "unusually high (>100 m/s)"),
            ("altitude", 5000.0, "exceeds typical ceiling (>5000 m)"),
        ]
        for field, limit, msg in checks:
            val = getattr(self, field, None)
            if val is not None and float(val) > limit:
                issues.append(f"{field}: {msg}")
        return issues


# ───────────────────────────────────────────────────────────────
# Core Output Contracts
# ───────────────────────────────────────────────────────────────
class DataGapReport(BaseModel):
    missing_features: List[str] = Field(..., description="Names of missing features")
    applied_imputations: Dict[str, str] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")


class MLResult(BaseModel):
    risk_class: str = Field(..., description="Predicted risk class label")
    score: float = Field(..., description="Model score or logit value")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in [0,1]")
    shap_values: Dict[str, float] = Field(default_factory=dict, description="SHAP contributions")
    model_config = ConfigDict(extra="forbid")


class AgentState(BaseModel):
    flight_id: str = Field(..., description="Flight identifier")
    telemetry: Dict[str, Any] = Field(default_factory=dict)
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    iteration_count: int = Field(0, ge=0)
    context_pool: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")


class FinalReport(BaseModel):
    flight_id: str = Field(..., description="Flight identifier")
    final_decision: Literal["GO", "NO-GO"] = Field(...)
    merged_confidence: float = Field(..., ge=0.0, le=1.0)
    references: List[str] = Field(default_factory=list, description="Provenance URIs/IDs")
    model_config = ConfigDict(extra="forbid")