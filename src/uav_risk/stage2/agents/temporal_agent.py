"""
Temporal Agent — Predictive Safety via Kalman Filtering
=========================================================
Version 4 — Aviation Grade Hardening (NaN Shields):
  FIX-T1: Tautology in wind projection warning replaced with meaningful comparison.
  FIX-T2: Scipy removed as dependency. t-distribution computed via Hill (1970).
  FIX-T3: OU process variance now uses exact steady-state formula.
  [NEW] FIX-V4: Runtime NaN Guards injected to prevent Kalman Filter Matrix poisoning 
                and logical black holes in safety comparisons.

Author: Stage 2 — ACE System
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Data Contracts
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SensorReading:
    """A single timestamped sensor measurement."""
    timestamp_s: float
    wind_speed_ms: float
    battery_pct: float
    altitude_m: float


@dataclass
class TemporalStateEstimate:
    """Kalman Filter output: optimal state estimate + uncertainty."""
    wind_speed_ms: float
    wind_speed_variance: float
    wind_trend_ms_per_min: float

    battery_pct: float
    battery_variance: float
    battery_drain_rate_pct_per_min: float

    wind_increasing: bool
    battery_draining_fast: bool

    horizon_min: float
    projected_wind_ms: float
    projected_battery_pct: float

    wind_trend_p_value: float
    battery_trend_p_value: float

    temporal_warnings: list[str] = field(default_factory=list)
    estimation_time_ms: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Self-contained t-distribution p-value (no scipy dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _regularized_incomplete_beta(x: float, a: float, b: float, max_iter: int = 300) -> float:
    """
    Regularized incomplete beta function I_x(a, b) via series expansion.
    """
    if not (math.isfinite(x) and math.isfinite(a) and math.isfinite(b)):
        return 1.0 # Fail-safe fallback for NaN inputs
        
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularized_incomplete_beta(1.0 - x, b, a, max_iter)

    log_beta_ab = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - log_beta_ab) / a

    total = 1.0
    term = 1.0
    for m in range(1, max_iter):
        term *= x * (a + b + m - 1.0) / (a + m)
        total += term
        if abs(term) < 1e-12:
            break

    return float(np.clip(front * total, 0.0, 1.0))


def t_distribution_p_value(t_stat: float, df: int) -> float:
    """
    Two-tailed p-value for Student's t-test.
    """
    if not math.isfinite(t_stat) or df < 1:
        return 1.0   # Undefined / NaN input — conservative

    if t_stat == 0.0:
        return 1.0

    x = float(df) / (float(df) + t_stat ** 2)
    return _regularized_incomplete_beta(x, df / 2.0, 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 1D Kalman Filter
# ─────────────────────────────────────────────────────────────────────────────

class KalmanFilter1D:
    """Standard 1D Kalman Filter. State: x = [value, rate]ᵀ"""

    def __init__(
        self,
        initial_value: float,
        measurement_noise_var: float,
        process_noise_var: float,
        initial_variance: float = 1.0,
    ):
        self.x = np.array([[initial_value], [0.0]])
        self.P = np.array([[initial_variance, 0.0],
                           [0.0,              initial_variance]])
        self.H = np.array([[1.0, 0.0]])
        self.R = np.array([[measurement_noise_var]])
        self.Q_base = process_noise_var

    def predict(self, dt: float) -> None:
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = np.array([[self.Q_base * dt ** 2, self.Q_base * dt],
                      [self.Q_base * dt,       self.Q_base]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, measurement: float) -> None:
        z = np.array([[measurement]])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        
        # 🚨 [Kalman Singularity Guard] 🚨
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Handle singular matrix to prevent crash
            S_inv = np.zeros_like(S)
            
        K = self.P @ self.H.T @ S_inv
        self.x = self.x + K @ y
        I = np.eye(2)
        IKH = I - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T

    def step(self, measurement: float, dt: float) -> tuple[float, float, float]:
        self.predict(dt)
        self.update(measurement)
        return float(self.x[0, 0]), float(self.x[1, 0]), float(self.P[0, 0])

    def project(
        self,
        horizon_seconds: float,
        mean_reversion_rate: float = 0.0,
    ) -> tuple[float, float]:
        if mean_reversion_rate > 0:
            theta = mean_reversion_rate
            T = horizon_seconds
            decay = math.exp(-theta * T)
            rate_contribution = self.x[1, 0] * (1.0 - decay) / theta
        else:
            rate_contribution = self.x[1, 0] * horizon_seconds

        x_proj_value = self.x[0, 0] + rate_contribution

        if mean_reversion_rate > 0:
            theta = mean_reversion_rate
            T = horizon_seconds
            sigma2_0 = float(self.P[0, 0])
            sigma2_inf = self.Q_base / (2.0 * theta)
            decay2 = math.exp(-2.0 * theta * T)
            proj_var = sigma2_inf + (sigma2_0 - sigma2_inf) * decay2
            proj_var = max(proj_var, sigma2_inf)
        else:
            F = np.array([[1.0, horizon_seconds], [0.0, 1.0]])
            Q = np.array([[self.Q_base * horizon_seconds ** 2, self.Q_base * horizon_seconds],
                          [self.Q_base * horizon_seconds,       self.Q_base]])
            P_proj = F @ self.P @ F.T + Q
            proj_var = float(P_proj[0, 0])

        return x_proj_value, proj_var


# ─────────────────────────────────────────────────────────────────────────────
# Trend Significance Test
# ─────────────────────────────────────────────────────────────────────────────

def compute_trend_significance(values: list[float], times: list[float]) -> tuple[float, float]:
    if len(values) < 3:
        return 0.0, 1.0

    n = len(values)
    t = np.array(times, dtype=float)
    y = np.array(values, dtype=float)

    t_mean, y_mean = np.mean(t), np.mean(y)
    Stt = float(np.sum((t - t_mean) ** 2))

    if Stt < 1e-10:
        return 0.0, 1.0

    slope = float(np.sum((t - t_mean) * (y - y_mean)) / Stt)
    intercept = y_mean - slope * t_mean

    y_pred = slope * t + intercept
    residuals = y - y_pred
    s2 = float(np.sum(residuals ** 2) / (n - 2)) if n > 2 else 1.0

    se_slope = math.sqrt(s2 / Stt) if Stt > 0 else 1.0

    if se_slope < 1e-10:
        return slope, 0.0 

    t_stat = slope / se_slope
    p_value = t_distribution_p_value(t_stat, df=n - 2)

    return slope, p_value


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Agent
# ─────────────────────────────────────────────────────────────────────────────

class TemporalAgent:
    WIND_SENSOR_NOISE_VAR = 0.25
    BATTERY_SENSOR_NOISE_VAR = 0.10
    WIND_PROCESS_NOISE = 0.01
    BATTERY_PROCESS_NOISE = 0.005

    FAST_DRAIN_THRESHOLD_PCT_PER_MIN = 2.0
    WIND_INCREASE_THRESHOLD_MS_PER_MIN = 0.5
    WIND_PROJECTION_INCREASE_RATIO = 1.20

    def __init__(self, projection_horizon_min: float = 5.0):
        self.horizon_min = projection_horizon_min
        self.horizon_s = projection_horizon_min * 60.0

        self._wind_kf: Optional[KalmanFilter1D] = None
        self._battery_kf: Optional[KalmanFilter1D] = None
        self._wind_history: list[float] = []
        self._battery_history: list[float] = []
        self._time_history: list[float] = []
        self._last_timestamp: Optional[float] = None
        self.history_window_s: float = 120.0 

    def reset(self) -> None:
        self._wind_kf = None
        self._battery_kf = None
        self._wind_history.clear()
        self._battery_history.clear()
        self._time_history.clear()
        self._last_timestamp = None

    def _create_fatal_nan_report(self, reason: str, elapsed_ms: float) -> TemporalStateEstimate:
        """
        [Aviation-Grade]: يولد تقرير رفض زمني حتمي في حال تلوث البيانات بـ NaN.
        قيم متعمدة (بطارية 0% ورياح 999) لضمان الفشل القطعي دون تمرير الـ NaN لـ ConsensusAgent.
        """
        return TemporalStateEstimate(
            wind_speed_ms=999.0, 
            wind_speed_variance=999.0,
            wind_trend_ms_per_min=99.0,
            battery_pct=0.0, 
            battery_variance=999.0,
            battery_drain_rate_pct_per_min=99.0,
            wind_increasing=True,
            battery_draining_fast=True,
            horizon_min=self.horizon_min,
            projected_wind_ms=999.0,
            projected_battery_pct=0.0,
            wind_trend_p_value=0.0,
            battery_trend_p_value=0.0,
            temporal_warnings=[f"FATAL VETO (Temporal): {reason}"],
            estimation_time_ms=elapsed_ms
        )

    def process_reading(self, reading: SensorReading) -> TemporalStateEstimate:
        t_start = time.perf_counter()

        # 🚨 [حارس وقت التشغيل 1 - Input NaN Guard] 🚨
        # يمنع تسمم مصفوفات فلتر كالمان (NaN Poisoning) منذ اللحظة الأولى.
        if math.isnan(reading.wind_speed_ms) or math.isnan(reading.battery_pct):
            return self._create_fatal_nan_report(
                "NaN detected in incoming sensor telemetry.", 
                (time.perf_counter() - t_start) * 1000
            )

        if self._wind_kf is None:
            self._wind_kf = KalmanFilter1D(
                initial_value=reading.wind_speed_ms,
                measurement_noise_var=self.WIND_SENSOR_NOISE_VAR,
                process_noise_var=self.WIND_PROCESS_NOISE,
                initial_variance=2.0,
            )
            self._battery_kf = KalmanFilter1D(
                initial_value=reading.battery_pct,
                measurement_noise_var=self.BATTERY_SENSOR_NOISE_VAR,
                process_noise_var=self.BATTERY_PROCESS_NOISE,
                initial_variance=1.0,
            )
            self._last_timestamp = reading.timestamp_s
            dt = 0.0
        else:
            dt = reading.timestamp_s - self._last_timestamp
            self._last_timestamp = reading.timestamp_s

        if dt > 0:
            wind_est, wind_rate, wind_var = self._wind_kf.step(reading.wind_speed_ms, dt)
            batt_est, batt_rate, batt_var = self._battery_kf.step(reading.battery_pct, dt)
        else:
            wind_est  = float(self._wind_kf.x[0, 0])
            wind_rate = float(self._wind_kf.x[1, 0])
            wind_var  = float(self._wind_kf.P[0, 0])
            batt_est  = float(self._battery_kf.x[0, 0])
            batt_rate = float(self._battery_kf.x[1, 0])
            batt_var  = float(self._battery_kf.P[0, 0])

        # 🚨 [حارس وقت التشغيل 2 - KF Matrix Corruption Guard] 🚨
        if math.isnan(wind_est) or math.isnan(batt_est):
            return self._create_fatal_nan_report(
                "Kalman Filter matrices collapsed into NaN.", 
                (time.perf_counter() - t_start) * 1000
            )

        wind_rate_per_min = wind_rate * 60.0
        batt_rate_per_min = batt_rate * 60.0

        self._wind_history.append(wind_est)
        self._battery_history.append(batt_est)
        self._time_history.append(reading.timestamp_s)

        cutoff_time = reading.timestamp_s - self.history_window_s
        while self._time_history and self._time_history[0] < cutoff_time:
            self._wind_history.pop(0)
            self._battery_history.pop(0)
            self._time_history.pop(0)

        wind_slope, wind_p = compute_trend_significance(self._wind_history, self._time_history)
        batt_slope, batt_p = compute_trend_significance(self._battery_history, self._time_history)

        wind_slope_per_min = wind_slope * 60.0
        batt_slope_per_min = batt_slope * 60.0

        proj_wind, proj_wind_var = self._wind_kf.project(
            self.horizon_s, mean_reversion_rate=0.02
        )
        proj_batt, proj_batt_var = self._battery_kf.project(
            self.horizon_s, mean_reversion_rate=0.0
        )

        proj_wind = max(0.0, proj_wind)
        proj_batt = float(np.clip(proj_batt, 0.0, 100.0))

        wind_increasing = (
            wind_slope_per_min > self.WIND_INCREASE_THRESHOLD_MS_PER_MIN
            and wind_p < 0.05
        )
        battery_draining_fast = (
            batt_slope_per_min < -self.FAST_DRAIN_THRESHOLD_PCT_PER_MIN
            and batt_p < 0.05
        )

        warnings = []

        if wind_increasing:
            warnings.append(
                f"TREND↑ Wind increasing at {wind_slope_per_min:.2f} m/s/min "
                f"(p={wind_p:.3f}). Projected in {self.horizon_min:.0f}min: {proj_wind:.1f} m/s"
            )

        if battery_draining_fast:
            warnings.append(
                f"TREND↓ Battery draining at {abs(batt_slope_per_min):.2f}%/min "
                f"(p={batt_p:.3f}). Projected in {self.horizon_min:.0f}min: {proj_batt:.1f}%"
            )

        if proj_wind > wind_est * self.WIND_PROJECTION_INCREASE_RATIO and proj_wind > 5.0:
            warnings.append(
                f"PROJECTION: Wind expected to increase from {wind_est:.1f} → "
                f"{proj_wind:.1f} m/s (+{(proj_wind/wind_est - 1)*100:.0f}%) "
                f"in {self.horizon_min:.0f} min  [95% CI ±{math.sqrt(proj_wind_var)*1.645:.1f} m/s]"
            )

        if proj_batt < 20.0:
            warnings.append(
                f"PROJECTION: Battery projected at {proj_batt:.1f}% in "
                f"{self.horizon_min:.0f} min — below 20% emergency threshold"
            )

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        return TemporalStateEstimate(
            wind_speed_ms=wind_est,
            wind_speed_variance=wind_var,
            wind_trend_ms_per_min=wind_slope_per_min,
            battery_pct=batt_est,
            battery_variance=batt_var,
            battery_drain_rate_pct_per_min=batt_rate_per_min,
            wind_increasing=wind_increasing,
            battery_draining_fast=battery_draining_fast,
            horizon_min=self.horizon_min,
            projected_wind_ms=proj_wind,
            projected_battery_pct=proj_batt,
            wind_trend_p_value=wind_p,
            battery_trend_p_value=batt_p,
            temporal_warnings=warnings,
            estimation_time_ms=elapsed_ms,
        )

    def process_batch(self, readings: list[SensorReading]) -> TemporalStateEstimate:
        self.reset()
        result = None
        for reading in readings:
            result = self.process_reading(reading)
        return result