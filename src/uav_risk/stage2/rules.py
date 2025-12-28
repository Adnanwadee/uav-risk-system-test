from __future__ import annotations

from typing import Any, Dict

from .schemas import RulesResult, RuleHit


def run_rules(stage1_facts: Dict[str, Any], inputs_snapshot: Dict[str, Any]) -> RulesResult:
    """
    Rules engine must represent explicit constraints and advisories.

    Product-grade principle:
    - DO NOT make the Stage-1 model a HARD veto via rules.
      The model is a signal handled by the Stage-2 policy layer.
    """
    res = RulesResult(hard_violations=[], advisories=[], computed={})

    risk_score = stage1_facts.get("risk_score")
    predicted = str(stage1_facts.get("predicted_class", "")).lower()
    confidence = float(stage1_facts.get("confidence", 0.0) or 0.0)

    # --------------------------------------------------
    # MODEL SIGNAL (ALWAYS SOFT)
    # --------------------------------------------------
    if predicted.startswith("high") and confidence >= 0.95:
        res.advisories.append(
            RuleHit(
                rule_id="SOFT_MODEL_HIGH_CONF",
                severity="SOFT",
                message="Model predicts High Risk with high confidence (signal only; not a hard veto).",
                evidence={"predicted_class": stage1_facts.get("predicted_class"), "confidence": confidence},
            )
        )

    if predicted.startswith("medium"):
        res.advisories.append(
            RuleHit(
                rule_id="SOFT_MODEL_MEDIUM",
                severity="SOFT",
                message="Model predicts Medium Risk; proceed with caution.",
                evidence={"predicted_class": stage1_facts.get("predicted_class"), "confidence": confidence},
            )
        )

    # --------------------------------------------------
    # WEATHER ADVISORIES (DOMAIN RULES)
    # --------------------------------------------------
    wind = inputs_snapshot.get("environment.weather.wind_mps")
    gust = inputs_snapshot.get("environment.weather.gust_mps")

    if isinstance(wind, (int, float)) and float(wind) >= 8.0:
        res.advisories.append(
            RuleHit(
                rule_id="SOFT_WIND_ELEVATED",
                severity="SOFT",
                message="Wind speed is elevated for typical small UAV operations.",
                evidence={"wind_mps": float(wind)},
            )
        )

    if isinstance(gust, (int, float)) and float(gust) >= 10.0:
        res.advisories.append(
            RuleHit(
                rule_id="SOFT_GUST_ELEVATED",
                severity="SOFT",
                message="Wind gusts are elevated; consider rescheduling or reducing exposure.",
                evidence={"gust_mps": float(gust)},
            )
        )

    # --------------------------------------------------
    # COMPUTED FIELDS (ALWAYS SET)
    # --------------------------------------------------
    res.computed["rule_count"] = len(res.hard_violations) + len(res.advisories)
    res.computed["risk_score"] = risk_score

    assert isinstance(res, RulesResult), "run_rules MUST return RulesResult"
    return res
