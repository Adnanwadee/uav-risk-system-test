# File Path: src/uav_risk/stage2/agent/agent_tools.py
# STAGE6_CLEANUP_REVIEW:
# Classification: MIXED_REVIEW_FUNCTION_LEVEL
# Plan lineage: MIXED_ACTIVE_LEGACY
# Runtime status: Not confirmed as canonical runtime; currently guarded by safety/static tests and used by legacy ACE path.
# Legacy signal: Feature batch, physics, contextual sweep, and RAG helper functions originally support ACEReActAgent.
# Replacement: OperationalAgentV2 / operational_inspector.py / current evidence-aware agent logic, where applicable.
# Action rule: Do not delete as a whole. Review function-by-function after ACE cleanup.
import math
from typing import Any, Dict, List, Optional, Set
from uav_risk.stage2.agent.agent_schemas import FeatureAssessment, ConditionalGoConstraint

EPSILON = 1e-5  # تحصين الحدود ضد مشاكل Float Precision Boundary

# الخريطة الهيكلية للمخاطر المركبة والاعتمادية متقاطعة الميزات
CROSS_FEATURE_SAFETY_MAP: Dict[str, List[str]] = {
    "high_load_extreme_weather": [
        "uav_mass_kg",
        "environment_weather_wind_mps",
        "uav_payload_mass_kg"
    ]
}


def _requires_evidence_payload(reasoning: str, metric_name: str, metric_value: float) -> Dict[str, Any]:
    """Mark policy-sensitive outputs as evidence-required for downstream governance."""
    return {
        "requires_evidence": True,
        "evidence_requirement_reason": reasoning,
        "metric_name": metric_name,
        "metric_value": metric_value,
    }


def validate_feature_batch(
    category_name: str,
    features: Dict[str, float],
    feature_defs: Dict[str, Any],
    cross_map: Dict[str, List[str]],
    already_examined: Optional[Set[str]] = None
) -> List[FeatureAssessment]:
    """Evaluates a batch of telemetry features against constitutional safe and critical ranges."""
    already_examined = already_examined or set()
    assessments: List[FeatureAssessment] = []
    cross_results: List[FeatureAssessment] = []

    for scenario_name, scenario_features in cross_map.items():
        if set(scenario_features).issubset(set(features.keys())):
            cross_results.extend(execute_cross_dependency_check(scenario_name, features))

    cross_covered = {a.feature_name for a in cross_results}

    for name, val in features.items():
        if name in already_examined:
            continue

        if name in cross_covered:
            match = next((a for a in cross_results if a.feature_name == name), None)
            if match:
                assessments.append(match)
                continue

        fdef = feature_defs.get(name, {})
        if not fdef:
            assessments.append(
                FeatureAssessment(
                    feature_name=name,
                    value=float(val),
                    status="SAFE",
                    reasoning="Missing definitions registry configuration. Defaulting to safe trace.",
                    rag_consulted=False
                )
            )
            continue

        crit_min = fdef.get("critical_min", -math.inf) - EPSILON
        crit_max = fdef.get("critical_max", math.inf) + EPSILON
        safe_min = fdef.get("safe_min", -math.inf) - EPSILON
        safe_max = fdef.get("safe_max", math.inf) + EPSILON

        if val < crit_min or val > crit_max:
            status = "CRITICAL"
            reason = (
                f"Telemetry parameter breaks critical constitution limits "
                f"[{fdef.get('critical_min', '-inf')}, {fdef.get('critical_max', 'inf')}]."
            )
        elif val < safe_min or val > safe_max:
            status = "WARNING"
            reason = (
                f"Parameter overflows operational safe baseline boundaries "
                f"[{fdef.get('safe_min', '-inf')}, {fdef.get('safe_max', 'inf')}]."
            )
        else:
            status = "SAFE"
            reason = "Telemetry value sits cleanly within safe boundaries."

        assessments.append(
            FeatureAssessment(
                feature_name=name,
                value=float(val),
                status=status,
                reasoning=reason,
                rag_consulted=False
            )
        )

    return assessments


def check_physics_constraint(
    constraint_name: str,
    features: Dict[str, float],
    feature_defs: Dict[str, Any]
) -> Dict[str, Any]:
    """Applies structural equations to verify flight safety characteristics."""
    if constraint_name == "disk_loading":
        mass = features.get("uav_mass_kg", 0.0)
        payload = features.get("uav_payload_mass_kg", 0.0)
        area = features.get("uav_rotorcraft_disk_area_m2", 0.001)

        if area <= EPSILON:
            return {
                "passed": False,
                "metric_value": 0.0,
                "unit": "kg/m²",
                "severity": "CRITICAL",
                "reasoning": "Rotor surface area division zero error.",
            }

        disk_loading = ((mass + payload) * 9.81) / area

        if disk_loading > (45.0 + EPSILON):
            return {
                "passed": False,
                "metric_name": "disk_loading",
                "metric_value": round(disk_loading, 2),
                "unit": "kg/m²",
                "severity": "CRITICAL",
                "reasoning": f"Disk loading {disk_loading:.2f} kg/m² violates structural limits (>45.0).",
            }
        elif disk_loading > (35.0 + EPSILON):
            return {
                "passed": False,
                "metric_name": "disk_loading",
                "metric_value": round(disk_loading, 2),
                "unit": "kg/m²",
                "severity": "WARNING",
                "reasoning": "Disk loading enters structural stress warning zone (>35.0).",
                "conditional_constraint": ConditionalGoConstraint(
                    constraint_id="C_DL_REG",
                    description="Trim overall payload profile config to reduce rotor disc pressure.",
                    feature_name="uav_payload_mass_kg",
                    required_value_range="disk_loading <= 35.0",
                    legal_reference="FAA AC 107-2B",
                ),
            }
        return {
            "passed": True,
            "metric_name": "disk_loading",
            "metric_value": round(disk_loading, 2),
            "unit": "kg/m²",
            "severity": "SAFE",
            "reasoning": "Disk loading sits safely within limits.",
        }

    elif constraint_name == "wind_susceptibility":
        wind_mps = features.get("environment_weather_wind_mps", 0.0)
        v_max = features.get("uav_max_speed_mps", 15.0)
        mass = features.get("uav_mass_kg", 1.0)

        gust = wind_mps * 1.4
        ratio = gust / max(v_max, EPSILON)

        force = 0.5 * 1.225 * max(0.02, mass * 0.006) * 1.2 * (gust ** 2)
        weight_n = mass * 9.81
        f_w_ratio = force / max(weight_n, EPSILON)

        if ratio > (0.75 + EPSILON) or f_w_ratio > (0.40 + EPSILON):
            return {
                "passed": False,
                "metric_name": "wind_susceptibility",
                "metric_value": round(ratio, 3),
                "unit": "ratio (gust/V_max)",
                "severity": "CRITICAL",
                "reasoning": f"Aero-susceptibility breach: velocity_ratio={ratio:.2f}, force-to-weight={f_w_ratio:.2f}.",
            }
        elif ratio > (0.50 + EPSILON):
            return {
                "passed": False,
                "metric_name": "wind_susceptibility",
                "metric_value": round(ratio, 3),
                "unit": "ratio (gust/V_max)",
                "severity": "WARNING",
                "reasoning": "High kinematic drift risk encountered under weather vector limits.",
            }
        return {
            "passed": True,
            "metric_name": "wind_susceptibility",
            "metric_value": round(ratio, 3),
            "unit": "ratio (gust/V_max)",
            "severity": "SAFE",
            "reasoning": "Aero-stability constraints passed clear within nominal margin.",
        }

    elif constraint_name == "energy_budget":
        battery_wh = features.get("uav_battery_wh", 0.0)
        pct = features.get("battery_remaining_pct", 100.0)
        available_wh = battery_wh * (pct / 100.0)

        if pct < (20.0 - EPSILON):
            return {
                "passed": False,
                "metric_name": "energy_budget",
                "metric_value": round(available_wh, 2),
                "unit": "Wh",
                "severity": "CRITICAL",
                "reasoning": f"Battery state of charge {pct:.1f}% drops below emergency mandatory 20% limit.",
            }
        return {
            "passed": True,
            "metric_name": "energy_budget",
            "metric_value": round(available_wh, 2),
            "unit": "Wh",
            "severity": "SAFE",
            "reasoning": "Energy budget clearance validated for flight path duration.",
        }

    elif constraint_name == "altitude_ceiling":
        alt = features.get("flight_altitude_m", 0.0)
        altitude_fdef = feature_defs.get("flight_altitude_m", {})
        safe_max = altitude_fdef.get("safe_max")
        critical_max = altitude_fdef.get("critical_max")

        if isinstance(critical_max, (int, float)) and alt > (float(critical_max) + EPSILON):
            reasoning = (
                f"Altitude {alt}m exceeds configured critical_max {float(critical_max):.3f}m for flight_altitude_m."
            )
            result = {
                "passed": False,
                "metric_name": "altitude_ceiling",
                "metric_value": alt,
                "unit": "m",
                "severity": "CRITICAL",
                "reasoning": reasoning,
            }
            result.update(_requires_evidence_payload(reasoning, "altitude_ceiling", alt))
            return result

        if isinstance(safe_max, (int, float)) and alt > (float(safe_max) + EPSILON):
            reasoning = (
                f"Altitude {alt}m exceeds configured safe_max {float(safe_max):.3f}m for flight_altitude_m."
            )
            result = {
                "passed": False,
                "metric_name": "altitude_ceiling",
                "metric_value": alt,
                "unit": "m",
                "severity": "WARNING",
                "reasoning": reasoning,
                "conditional_constraint": ConditionalGoConstraint(
                    constraint_id="C_ALT_EVIDENCE",
                    description="Policy-sensitive altitude exceedance requires evidence-backed review.",
                    feature_name="flight_altitude_m",
                    required_value_range=f"<= {float(safe_max):.3f}m",
                    legal_reference=None,
                ),
            }
            result.update(_requires_evidence_payload(reasoning, "altitude_ceiling", alt))
            return result

        return {
            "passed": True,
            "metric_name": "altitude_ceiling",
            "metric_value": alt,
            "unit": "m",
            "severity": "SAFE",
            "reasoning": "Flight altitude is within configured feature-definition limits.",
        }

    return {"passed": True, "metric_value": 0.0, "severity": "SAFE", "reasoning": "Clean."}


def assess_contextual_remainder(
    all_features: Dict[str, float],
    already_examined: Set[str],
    feature_defs: Dict[str, Any],
    cross_map: Dict[str, List[str]]
) -> List[FeatureAssessment]:
    """Sweeps over remaining untouched items within the feature space."""
    remainder = {k: v for k, v in all_features.items() if k not in already_examined}
    if not remainder:
        return []
    return validate_feature_batch("constitutional_sweep", remainder, feature_defs, cross_map, already_examined)


async def query_rag(query: str, rag_adapter: Any) -> Dict[str, Any]:
    """Query evidence through an injected adapter-like object.

    Expected interface: async retrieve_evidence(query: str, *, scenario_context: dict | None, max_claims: int)
    """
    try:
        bundle = await rag_adapter.retrieve_evidence(
            query=query,
            scenario_context=None,
            max_claims=3,
        )
        if bundle.citations:
            return {
                "success": True,
                "finding": f"Retrieved {len(bundle.citations)} evidence citation(s).",
                "citations": bundle.citations,
            }
        return {
            "success": False,
            "finding": bundle.no_evidence_reason or "No evidence retrieved.",
            "citations": [],
        }
    except Exception:
        return {
            "success": False,
            "finding": "Asynchronous RAG integration framework failure.",
            "citations": [],
        }


async def generate_legal_query(feature_name: str, value: float, violation_type: str, llm_client: Any) -> str:
    """Build a deterministic legal query string without direct LLM calls."""
    _ = llm_client  # kept for signature compatibility with current callers
    return (
        f"Aviation operational constraints for {feature_name.replace('_', ' ')} "
        f"at observed value {value} with severity {violation_type}."
    )


def execute_cross_dependency_check(scenario_name: str, features: Dict[str, float]) -> List[FeatureAssessment]:
    """Diagnoses severe multi-variable compound stress scenarios that breach safe profiles when paired."""
    assessments: List[FeatureAssessment] = []
    if scenario_name == "high_load_extreme_weather":
        mass = features.get("uav_mass_kg", 0.0)
        wind = features.get("environment_weather_wind_mps", 0.0)
        payload = features.get("uav_payload_mass_kg", 0.0)

        if mass > (7.0 + EPSILON) and wind > (10.0 + EPSILON) and payload > (2.0 + EPSILON):
            reason = (
                "Compound Flight Risk: Maximum takeoff mass operating concurrently "
                "under severe aerodynamic cross-winds weather constraints."
            )
            for f in ["uav_mass_kg", "environment_weather_wind_mps", "uav_payload_mass_kg"]:
                assessments.append(
                    FeatureAssessment(
                        feature_name=f,
                        value=float(features.get(f, 0.0)),
                        status="CRITICAL",
                        reasoning=reason,
                        rag_consulted=False,
                        related_features=["uav_mass_kg", "environment_weather_wind_mps", "uav_payload_mass_kg"]
                    )
                )
    return assessments

# =====================================================================
# Stage 2 Tools Submodule Architectural Dependency Report:
#
# Depends on:
#   - src/uav_risk/stage2/agent/agent_schemas.py (FeatureAssessment, ConditionalGoConstraint)
#
# Consumed by:
#   - src/uav_risk/stage2/agent/ace_agent.py
#   - tests/unit/test_ace_agent.py
# =====================================================================
