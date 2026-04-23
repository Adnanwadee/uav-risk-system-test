# src/uav_risk/stage2/input_contract.py

from __future__ import annotations
from typing import List
from .schemas import UAVScenario, DataQualityProfile, TacticalDirective

class InputContractEngine:
    """
    Uncertainty-Aware Pipeline: Evaluates missing/degraded data 
    and generates structured Tactical Directives.
    """

    @classmethod
    def evaluate_scenario(cls, scenario: UAVScenario) -> DataQualityProfile:
        score = 1.0
        missing = []
        directives: List[TacticalDirective] = []

        # 1. Base Penalties for Important Fields
        if scenario.environment_weather_gust_mps is None:
            missing.append("gust_mps")
            score -= 0.15
            directives.append(TacticalDirective(
                rule_id="TAC_GUST_01",
                action="ASSUME_WORST_CASE",
                parameter="Wind Gusts",
                rationale="Gust data missing. Compute effective wind load assuming gusts are 1.5x sustained wind."
            ))

        if scenario.environment_weather_visibility_m is None:
            missing.append("visibility_m")
            score -= 0.15
            if scenario.mission_type == "VLOS":
                score -= 0.20 # Double penalty for VLOS operations
                directives.append(TacticalDirective(
                    rule_id="TAC_VIS_VLOS_01",
                    action="ENFORCE_VLOS",
                    parameter="Visibility",
                    rationale="Visibility missing in VLOS mission. High risk of regulatory breach."
                ))

        if scenario.uav_battery_model_hover_power_W is None:
            missing.append("hover_power_W")
            score -= 0.15
            directives.append(TacticalDirective(
                rule_id="TAC_PWR_01",
                action="LIMIT_MANEUVERS",
                parameter="Thrust",
                rationale="Cannot verify Thrust-to-Weight ratio. Avoid aggressive maneuvers."
            ))

        # 2. Exploit Extended Sensors & Degradation
        if scenario.extended_sensors:
            for sensor_name, sensor_data in scenario.extended_sensors.items():
                sensor_confidence = sensor_data.confidence or 1.0
                if sensor_confidence < 0.8:
                    penalty = (1.0 - sensor_confidence) * 0.15
                    score -= penalty
                    directives.append(TacticalDirective(
                        rule_id=f"TAC_SENS_{sensor_name.upper()}",
                        action="REQUEST_DATA",
                        parameter=sensor_name,
                        rationale=f"Sensor {sensor_name} is degraded (confidence {sensor_confidence:.2f})."
                    ))

        # 3. Safe Cascading Decay (Clamped) - Fix applied!
        if missing:
            decay_factor = min(len(missing) * 0.25, 0.6) # Max degradation is 60%
            score = max(0.0, score * (1.0 - decay_factor))

        score = max(0.0, round(score, 3))
        is_ml_reliable = score >= 0.75

        if not is_ml_reliable:
            directives.append(TacticalDirective(
                rule_id="TAC_SYS_ML_01",
                action="IGNORE_ML",
                parameter="XGBoost Tool",
                rationale="Data integrity below 75%. ML predictions are unsafe to use."
            ))

        return DataQualityProfile(
            confidence_score=score,
            missing_fields=missing,
            tactical_directives=directives,
            is_ml_reliable=is_ml_reliable
        )