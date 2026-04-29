"""
Physics Agent — Causal Physics Guardian (V13.0 - Dynamic Physics)
================================================================
Role: Executes high-fidelity aerodynamic and energy simulations.
Update: Fully decoupled from static limits. The agent now extracts
real-time variables (mass, thrust, hover_power) directly from the user payload.

Author: Stage 2 — ACE System
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass
from typing import Optional, List
import numpy as np

# استيراد العقود البرمجية المحدثة
from uav_risk.stage2.schemas import RuntimeFlightData, PhysicsRiskReport
from pydantic import BaseModel

logger = logging.getLogger("PhysicsAgent")

class DronePhysicalSpec(BaseModel):
    """
    [تحديث هندسي]: هذه أصبحت مجرد قيم احتياطية (Fallbacks) ولن يتم فرضها
    إذا كانت بيانات المستخدم الحية تحتوي على هذه القيم.
    """
    mass_kg: float = 1.3
    max_thrust_n: float = 45.0
    rotor_area_m2: float = 0.25
    drag_coefficient: float = 0.8
    frontal_area_m2: float = 0.05
    max_wind_tolerance_ms: float = 12.0
    battery_capacity_wh: float = 50.0
    hover_power_w: float = 220.0
    structural_load_limit_n: float = 100.0


class PhysicsAgent:
    def __init__(self, spec: Optional[DronePhysicalSpec] = None):
        self.spec = spec or DronePhysicalSpec()

    async def analyze(self, data: RuntimeFlightData) -> PhysicsRiskReport:
        t_start = time.perf_counter()
        warnings = []

        current_mass = data.mass_kg
        current_thrust = data.max_thrust_n
        current_hover_power = data.hover_power_w
        
        # قراءة نوع الطائرة الذي تم حقنه
        uav_type = getattr(data, 'uav_type', 'quadrotor')

        logger.info(f"REAL Physics Payload -> Mass: {current_mass}kg | Thrust: {current_thrust}N | Type: {uav_type}")

        # ---------------------------------------------------------
        # 2. الحسابات الفيزيائية السببية الذكية (Smart Causal Math)
        # ---------------------------------------------------------
        gravity_force = current_mass * 9.81
        
        # [الذكاء 1]: الحد الهيكلي أصبح ديناميكياً (الطائرة تتحمل 3 أضعاف وزنها G-Force)
        dynamic_structural_limit = gravity_force * 3.0 
        
        # [الذكاء 2]: معامل السحب يتغير حسب نوع الطائرة (الجناح الثابت يقطع الهواء أسهل)
        active_drag_coeff = 0.3 if "fixed" in uav_type else self.spec.drag_coefficient

        if current_thrust <= 0 or gravity_force <= 0:
            tw_ratio = 0.0
            thrust_margin = -1.0
            warnings.append("CRITICAL: Invalid mass or thrust inputs. Drone cannot fly.")
        else:
            tw_ratio = current_thrust / gravity_force
            thrust_margin = (current_thrust - gravity_force) / current_thrust

        wind_force = 0.5 * 1.225 * (data.wind_speed_ms**2) * active_drag_coeff * self.spec.frontal_area_m2
        structural_ratio = (gravity_force + wind_force) / dynamic_structural_limit

        mc_samples = 500
        wind_samples = np.random.normal(data.wind_speed_ms, 2.0, mc_samples)
        failures = np.sum(wind_samples > self.spec.max_wind_tolerance_ms)
        failure_prob = float(failures / mc_samples)

        flight_time_hours = data.estimated_flight_time_min / 60.0
        required_energy_wh = current_hover_power * flight_time_hours
        available_energy_wh = self.spec.battery_capacity_wh * (data.battery_level_pct / 100.0)
        
        if available_energy_wh <= 0:
            battery_margin = -100.0
        else:
            battery_margin = ((available_energy_wh - (required_energy_wh / 0.8)) / available_energy_wh) * 100.0

        if tw_ratio < 1.5: warnings.append(f"LOW_THRUST_TO_WEIGHT: Ratio is {tw_ratio:.2f}. Limited maneuverability.")
        if battery_margin < 15: warnings.append(f"CRITICAL_ENERGY_RESERVE: Margin is {battery_margin:.1f}%. High risk of exhaustion.")
        if failure_prob > 0.2: warnings.append(f"WIND_GUST_RISK: {failure_prob:.1%} probability of exceeding tolerance.")
        if structural_ratio > 0.8: warnings.append(f"STRUCTURAL_STRESS: Frame load at {structural_ratio:.1%} of dynamic limit.")

        risk_level = "LOW"
        go_no_go = "GO"
        
        if failure_prob > 0.5 or battery_margin < 0 or tw_ratio < 1.1 or structural_ratio >= 1.0:
            risk_level = "CRITICAL"
            go_no_go = "NO-GO"
        elif failure_prob > 0.2 or battery_margin < 20 or tw_ratio < 1.4 or structural_ratio > 0.8:
            risk_level = "MODERATE"
            go_no_go = "CAUTION"

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        return PhysicsRiskReport(
            go_no_go=go_no_go, risk_level=risk_level, mc_failure_probability=failure_prob,
            mc_confidence_interval=(max(0.0, failure_prob - 0.05), min(1.0, failure_prob + 0.05)),
            mc_samples=mc_samples, thrust_margin_ratio=float(thrust_margin), battery_margin_pct=float(battery_margin),
            structural_load_ratio=float(structural_ratio), wind_tolerance_ratio=data.wind_speed_ms / self.spec.max_wind_tolerance_ms,
            projected_risk_level=risk_level, projected_failure_probability=failure_prob, warnings=warnings,
            equations_used=["DynamicStructuralLimit", "TypeAwareDrag", "EnergyDrainWh", "MonteCarloSafetyScan"],
            calculation_time_ms=elapsed_ms
        )