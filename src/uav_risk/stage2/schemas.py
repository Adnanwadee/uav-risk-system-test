from __future__ import annotations

from typing import Any, Dict, List, Literal
from pydantic import BaseModel, Field

# =========================
# Stage 2 – Request
# =========================
class Stage2Request(BaseModel):
    """
    Request schema for Stage-2 report generation.
    """
    scenario: Dict[str, Any] = Field(
        ...,
        description="Raw UAV scenario inputs (same format as Stage-1)",
    )

# =========================
# Rule Hit
# =========================
class RuleHit(BaseModel):
    rule_id: str
    severity: Literal["HARD", "SOFT"]
    message: str
    evidence: Dict[str, Any] = Field(default_factory=dict)

# =========================
# Rules Result
# =========================
class RulesResult(BaseModel):
    """
    Result of deterministic rules engine.
    computed must ALWAYS exist to avoid Pydantic validation errors.
    """
    hard_violations: List[RuleHit] = Field(default_factory=list)
    advisories: List[RuleHit] = Field(default_factory=list)
    computed: Dict[str, Any] = Field(default_factory=dict)

# =========================
# Evidence
# =========================
class EvidenceSnippet(BaseModel):
    source: str
    content: str
    citation: str | None = None

# =========================
# Data Quality
# =========================
class DataQualitySummary(BaseModel):
    present_count: int
    total_count: int
    completeness_ratio: float
    missing_keys: List[str] = Field(default_factory=list)

# =========================
# Stage 2 – Response
# =========================
class Stage2Response(BaseModel):
    status: Literal["OK", "ERROR"]
    decision: Literal["GO", "CAUTION", "NO_GO", "INSUFFICIENT_DATA"]

    facts: Dict[str, Any]

    report_md: str
    report_json: Dict[str, Any]

    evidence: List[EvidenceSnippet] = Field(default_factory=list)
    quality: Dict[str, Any] = Field(default_factory=dict)
