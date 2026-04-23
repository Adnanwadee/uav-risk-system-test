# src/uav_risk/stage2/policies/deterministic_core.py

from __future__ import annotations
from typing import Optional, Dict, Any
from pydantic import BaseModel
from ..schemas import UAVScenario, MLResult
from .config import THRESHOLDS

class VetoResult(BaseModel):
    is_veto: bool
    reason: Optional[str] = None
    policy_id: Optional[str] = None

class DeterministicCore:
    """
    The Ultimate Safety Authority.
    Executes aviation-grade mathematical and regulatory hard constraints.
    """

    @staticmethod
    def _effective_wind_load(scenario: UAVScenario) -> float:
        """Engineering formula for effective wind load factoring in gusts."""
        sustained = scenario.environment_weather_wind_mps
        gust = scenario.environment_weather_gust_mps or (sustained * 1.5)
        return sustained + (0.5 * max(0, gust - sustained))

    @classmethod
    def _check_operational_context(cls, scenario: UAVScenario) -> VetoResult:
        """Contextual Safety Checks (Mission Type & Population)."""
        # BVLOS Wind Restriction
        if scenario.mission_type == "BVLOS":
            if scenario.environment_weather_wind_mps > THRESHOLDS.bvlos_max_wind_mps:
                return VetoResult(
                    is_veto=True, 
                    reason=f"BVLOS Restriction: Wind ({scenario.environment_weather_wind_mps}m/s) exceeds safe operational limit for BVLOS.", 
                    policy_id="CTX_BVLOS_WIND"
                )
        
        # Dense Population Risk
        if scenario.population_density == "DENSE" and scenario.airspace_altitude_agl_m > THRESHOLDS.dense_pop_max_alt_m:
            return VetoResult(
                is_veto=True, 
                reason="Ground Risk Violation: High altitude flight over dense population requires specific mitigation.", 
                policy_id="CTX_GROUND_RISK"
            )
            
        return VetoResult(is_veto=False)

    @classmethod
    def _check_flight_capability(cls, scenario: UAVScenario) -> VetoResult:
        eff_wind = cls._effective_wind_load(scenario)
        max_speed = scenario.uav_max_speed_mps
        
        if eff_wind >= max_speed:
            return VetoResult(
                is_veto=True,
                reason=f"Physics Violation: Effective wind load ({eff_wind:.1f} m/s) exceeds UAV max speed ({max_speed} m/s).",
                policy_id="PHYS_WIND_01"
            )
        
        power = scenario.uav_battery_model_hover_power_W
        mass = scenario.uav_mass_kg
        if power is not None:
            power_per_kg = power / mass
            if power_per_kg < THRESHOLDS.min_power_density_wkg:
                return VetoResult(
                    is_veto=True,
                    reason=f"Insufficient Thrust: Power density ({power_per_kg:.1f} W/kg) is below safe flight margin.",
                    policy_id="PHYS_PWR_02"
                )
        return VetoResult(is_veto=False)

    @classmethod
    def _check_regulatory_and_airspace(cls, scenario: UAVScenario) -> VetoResult:
        if scenario.airspace_altitude_agl_m > THRESHOLDS.max_altitude_agl_m:
            return VetoResult(
                is_veto=True, 
                reason=f"Regulatory Violation: Altitude ({scenario.airspace_altitude_agl_m}m) exceeds max AGL limit.", 
                policy_id="REG_ALT_01"
            )
        return VetoResult(is_veto=False)

    @classmethod
    def _check_system_integrity(cls, scenario: UAVScenario) -> VetoResult:
        if scenario.comms_uplink_status in ["DEGRADED", "LOST"]:
            return VetoResult(
                is_veto=True, 
                reason=f"Critical Comms Failure: Uplink status is {scenario.comms_uplink_status}.", 
                policy_id="COMMS_LOL_01"
            )
            
        if scenario.environment_gnss_jam_dbm > THRESHOLDS.critical_jamming_dbm:
            return VetoResult(
                is_veto=True, 
                reason="Severe GNSS Jamming: High risk of Fly-away.", 
                policy_id="NAV_INT_01"
            )
        return VetoResult(is_veto=False)

    @classmethod
    def pre_flight_veto_check(cls, scenario: UAVScenario) -> VetoResult:
        """The Master Aviation Check Sequence."""
        checks = [
            cls._check_operational_context,
            cls._check_flight_capability,
            cls._check_regulatory_and_airspace,
            cls._check_system_integrity
        ]
        for check in checks:
            res = check(scenario)
            if res.is_veto:
                return res
        return VetoResult(is_veto=False)

    @classmethod
    def post_flight_veto_check(cls, agent_decision: str, scenario: UAVScenario, tool_outputs: Dict[str, Any]) -> VetoResult:
        """The Ultimate Fail-safe against AI and ML failures."""
        pre = cls.pre_flight_veto_check(scenario)
        if pre.is_veto:
            return VetoResult(is_veto=True, reason=f"SAFETY OVERRIDE: Hard rule violated: {pre.reason}", policy_id="SYS_VETO_POST")
        
        if agent_decision == "GO":
            ml_json = tool_outputs.get("ml_prediction")
            if ml_json:
                # Assuming the tool returns a JSON string or dict. We check risk score.
                # Simplified check logic for the example:
                if isinstance(ml_json, dict) and ml_json.get("risk_score", 0.0) > 0.85:
                    return VetoResult(
                        is_veto=True, 
                        reason="ML-Physics Conflict: High ML Risk Score contradicts GO decision.", 
                        policy_id="ML_VETO_POST"
                    )
                
        return VetoResult(is_veto=False)