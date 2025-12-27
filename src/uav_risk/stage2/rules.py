from __future__ import annotations
from typing import Any, Dict

from .schemas import RulesResult, RuleHit


def run_rules(stage1_facts: Dict[str, Any], inputs_snapshot: Dict[str, Any]) -> RulesResult:
    res = RulesResult(
        hard_violations=[],
        advisories=[],
        computed={},)
    # computed موجود تلقائياً default_factory

    risk_score = stage1_facts.get("risk_score")
    predicted = str(stage1_facts.get("predicted_class", "")).lower()
    confidence = float(stage1_facts.get("confidence", 0.0) or 0.0)

    if predicted.startswith("high") and confidence >= 0.95:
        res.hard_violations.append(
            RuleHit(
                rule_id="HARD_MODEL_HIGH_CONF",
                severity="HARD",
                message="Model predicts High Risk with high confidence.",
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

    wind = inputs_snapshot.get("environment.weather.wind_mps")
    gust = inputs_snapshot.get("environment.weather.gust_mps")

    if isinstance(wind, (int, float)) and wind >= 8:
        res.advisories.append(
            RuleHit(
                rule_id="SOFT_WIND_ELEVATED",
                severity="SOFT",
                message="Wind speed is elevated for typical small UAV operations.",
                evidence={"wind_mps": wind},
            )
        )

    if isinstance(gust, (int, float)) and gust >= 10:
        res.advisories.append(
            RuleHit(
                rule_id="SOFT_GUST_ELEVATED",
                severity="SOFT",
                message="Wind gusts are elevated; consider rescheduling or reducing exposure.",
                evidence={"gust_mps": gust},
            )
        )


    # --------------------------------------------------
    # COMPUTED FIELDS (ALWAYS SET)
    # --------------------------------------------------
    res.computed["rule_count"] = (
        len(res.hard_violations) + len(res.advisories)
    )
    res.computed["risk_score"] = risk_score

    # 🔴 ASSERT قاتل – لو فشلت هنا نعرف فورًا
    assert isinstance(res, RulesResult), "run_rules MUST return RulesResult"

    return res
