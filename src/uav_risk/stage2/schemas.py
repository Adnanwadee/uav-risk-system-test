from __future__ import annotations
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


# =========================
# Rule Hit
# =========================

class RuleHit(BaseModel):
    rule_id: str
    severity: Literal["HARD", "SOFT"]
    message: str
    evidence: Dict[str, Any] = Field(default_factory=dict)


# =========================
# Rules Result  ✅ FIXED
# =========================

class RulesResult(BaseModel):
    """
    Result of deterministic rules engine.
    """

    hard_violations: List[RuleHit] = Field(default_factory=list)
    advisories: List[RuleHit] = Field(default_factory=list)

    # 🔑 CRITICAL FIX
    computed: Dict[str, Any] = Field(default_factory=dict)
