from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RiskDriver:
    driver_id: str          # stable id e.g., "WEATHER_WIND"
    title: str              # human readable
    value: Any
    severity: str           # LOW / MEDIUM / HIGH / UNKNOWN
    rationale: str          # why it matters
    domains: List[str]      # knowledge domains to search


def _sev_from_bool_default_med(flag: Optional[bool]) -> str:
    """
    Product-grade default:
    - None => UNKNOWN
    - False => LOW
    - True  => MEDIUM (NOT HIGH by default)
    Rationale: boolean flags indicate presence, not magnitude. HIGH should be reserved
    for quantified exceedances or explicit "loss of function" states.
    """
    if flag is None:
        return "UNKNOWN"
    return "MEDIUM" if flag else "LOW"


def build_risk_context(
    inputs_snapshot: Dict[str, Any],
    stage1_facts: Dict[str, Any],
    rules: Dict[str, Any],
    input_contract: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a comprehensive risk context from structured Stage-2 inputs.
    PURE function: no side effects, no decisions.

    Product-grade design notes:
    - Quantified hazards (wind/gust/jamming) can be HIGH using numeric thresholds.
    - Boolean hazards (multipath, em_interference) default to MEDIUM when True.
    - Model prediction is informational; policy layer decides how to use it.
    """

    snap = inputs_snapshot or {}
    s1 = stage1_facts or {}
    contract = input_contract or {}
    rules_dict = rules or {}

    drivers: List[RiskDriver] = []

    # ==========================================================
    # Weather (quantified)
    # ==========================================================
    wind = snap.get("environment.weather.wind_mps")
    gust = snap.get("environment.weather.gust_mps")
    vis = snap.get("environment.weather.visibility")

    if wind is not None:
        try:
            w = float(wind)
            sev = "HIGH" if w >= 12 else ("MEDIUM" if w >= 6 else "LOW")
            drivers.append(RiskDriver(
                driver_id="WEATHER_WIND",
                title="Weather: wind speed",
                value=w,
                severity=sev,
                rationale="Elevated winds reduce controllability and increase flight-path deviations.",
                domains=["weather", "regulations", "company_sop"]
            ))
        except Exception:
            drivers.append(RiskDriver(
                driver_id="WEATHER_WIND",
                title="Weather: wind speed",
                value=wind,
                severity="UNKNOWN",
                rationale="Wind speed value is non-numeric or malformed; cannot assess severity reliably.",
                domains=["weather", "regulations", "company_sop"]
            ))

    if gust is not None:
        try:
            g = float(gust)
            sev = "HIGH" if g >= 15 else ("MEDIUM" if g >= 10 else "LOW")
            drivers.append(RiskDriver(
                driver_id="WEATHER_GUST",
                title="Weather: wind gusts",
                value=g,
                severity=sev,
                rationale="Wind gusts introduce instability and transient load exceedances.",
                domains=["weather", "regulations", "company_sop"]
            ))
        except Exception:
            drivers.append(RiskDriver(
                driver_id="WEATHER_GUST",
                title="Weather: wind gusts",
                value=gust,
                severity="UNKNOWN",
                rationale="Gust value is non-numeric or malformed; cannot assess severity reliably.",
                domains=["weather", "regulations", "company_sop"]
            ))

    if vis is not None:
        drivers.append(RiskDriver(
            driver_id="WEATHER_VISIBILITY",
            title="Weather: visibility",
            value=vis,
            severity="UNKNOWN",
            rationale="Visibility affects VLOS compliance and obstacle detection capability.",
            domains=["weather", "regulations"]
        ))

    # ==========================================================
    # GNSS / EM (mixed: quantified + boolean)
    # ==========================================================
    jam = snap.get("environment.gnss_jam_dbm")
    multipath = snap.get("environment.gnss_multipath")
    em = snap.get("environment.em_interference")

    if jam is not None:
        try:
            j = float(jam)
            # NOTE: this thresholding is project-specific; adjust if you have a validated mapping.
            # Here we treat stronger (closer to 0) as worse.
            sev = "HIGH" if j >= -70 else ("MEDIUM" if j >= -85 else "LOW")
            drivers.append(RiskDriver(
                driver_id="GNSS_JAMMING",
                title="GNSS: jamming level",
                value=j,
                severity=sev,
                rationale="GNSS interference degrades navigation accuracy and integrity monitoring.",
                domains=["safety", "regulations", "uav_manual"]
            ))
        except Exception:
            drivers.append(RiskDriver(
                driver_id="GNSS_JAMMING",
                title="GNSS: jamming level",
                value=jam,
                severity="UNKNOWN",
                rationale="Jamming level value is non-numeric or malformed; cannot assess severity reliably.",
                domains=["safety", "regulations", "uav_manual"]
            ))

    if multipath is not None:
        drivers.append(RiskDriver(
            driver_id="GNSS_MULTIPATH",
            title="GNSS: multipath",
            value=multipath,
            severity=_sev_from_bool_default_med(multipath),
            rationale="Multipath can bias position estimates in urban or reflective environments; severity depends on magnitude and mitigation (sensor fusion / RAIM).",
            domains=["safety", "uav_manual", "regulations"]
        ))

    if em is not None:
        drivers.append(RiskDriver(
            driver_id="EM_INTERFERENCE",
            title="Electromagnetic interference",
            value=em,
            severity=_sev_from_bool_default_med(em),
            rationale="EM interference may affect GNSS, comms, and onboard electronics; severity depends on intensity and shielding/mitigation.",
            domains=["safety", "company_sop", "regulations"]
        ))

    # ==========================================================
    # DAA / Airspace (mostly policy/regulatory context)
    # ==========================================================
    sep = snap.get("daa.sep_threshold_m")
    ttc = snap.get("daa.ttc_threshold_s")
    alt = snap.get("airspace.altitude_agl_max_m")

    if sep is not None:
        drivers.append(RiskDriver(
            driver_id="DAA_SEPARATION",
            title="DAA: separation threshold",
            value=sep,
            severity="UNKNOWN",
            rationale="Separation thresholds drive collision-avoidance margins; severity depends on airspace density and DAA performance validation.",
            domains=["regulations", "company_sop"]
        ))

    if ttc is not None:
        drivers.append(RiskDriver(
            driver_id="DAA_TTC",
            title="DAA: time-to-collision threshold",
            value=ttc,
            severity="UNKNOWN",
            rationale="Low TTC thresholds reduce available reaction time; severity depends on detection range and closure rates.",
            domains=["regulations", "company_sop"]
        ))

    if alt is not None:
        drivers.append(RiskDriver(
            driver_id="AIRSPACE_ALTITUDE",
            title="Airspace: maximum AGL altitude",
            value=alt,
            severity="UNKNOWN",
            rationale="Altitude constraints affect regulatory exposure and risk profile.",
            domains=["regulations", "company_sop"]
        ))

    # ==========================================================
    # Communications
    # ==========================================================
    uplink = snap.get("comms.uplink_ok")
    downlink = snap.get("comms.downlink_ok")

    if uplink is not None:
        # If uplink_ok is False => MEDIUM by default (loss of C2 may be HIGH depending on CONOPS)
        sev = "MEDIUM" if (uplink is False) else "LOW"
        drivers.append(RiskDriver(
            driver_id="COMMS_UPLINK",
            title="Communications: uplink status",
            value=uplink,
            severity=sev,
            rationale="Loss of uplink impacts command and control continuity.",
            domains=["company_sop", "regulations"]
        ))

    if downlink is not None:
        sev = "MEDIUM" if (downlink is False) else "LOW"
        drivers.append(RiskDriver(
            driver_id="COMMS_DOWNLINK",
            title="Communications: downlink status",
            value=downlink,
            severity=sev,
            rationale="Loss of downlink reduces telemetry and situational awareness.",
            domains=["company_sop", "regulations"]
        ))

    # ==========================================================
    # Model output (informational only)
    # ==========================================================
    pred = s1.get("predicted_class")
    conf = s1.get("confidence")

    if pred is not None:
        drivers.append(RiskDriver(
            driver_id="MODEL_PREDICTION",
            title="Model risk assessment (Stage-1)",
            value={"predicted_class": pred, "confidence": conf},
            severity="HIGH" if str(pred).lower().startswith("high") else "UNKNOWN",
            rationale="Model indicates latent risk patterns learned from historical data.",
            domains=["company_sop"]
        ))

    # ==========================================================
    # Missing safety inputs
    # ==========================================================
    missing_safety = contract.get("missing_safety_keys") or []
    if missing_safety:
        drivers.append(RiskDriver(
            driver_id="MISSING_SAFETY_INPUTS",
            title="Missing safety-critical inputs",
            value=missing_safety,
            severity="MEDIUM",
            rationale="Missing safety inputs reduce enforceability of operational constraints and degrade decision reliability.",
            domains=["company_sop", "regulations"]
        ))

    # ==========================================================
    # Rules fired (meta-driver)
    # ==========================================================
    hard = rules_dict.get("hard_violations") or []
    adv = rules_dict.get("advisories") or []

    if hard or adv:
        drivers.append(RiskDriver(
            driver_id="RULES_TRIGGERED",
            title="Triggered safety rules",
            value={"hard": hard, "advisories": adv},
            severity="HIGH" if hard else "MEDIUM",
            rationale="Triggered rules indicate explicit operational or regulatory concerns.",
            domains=["company_sop", "regulations"]
        ))

    return {
        "risk_drivers": drivers,
        "input_contract": contract,
    }
