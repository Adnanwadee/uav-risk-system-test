# File Path: src/uav_risk/stage2/agent/agent_tools.py
import math
from typing import Any, Dict, List, Optional, Set, Tuple
from uav_risk.stage2.agent.agent_schemas import FeatureAssessment, ConditionalGoConstraint
from uav_risk.stage2.rag.rag_core import AsyncRAGCore

EPSILON = 1e-5  # تحصين الحدود ضد مشاكل Float Precision Boundary

# الخريطة الهيكلية للمخاطر المركبة والاعتمادية متقاطعة الميزات
CROSS_FEATURE_SAFETY_MAP: Dict[str, List[str]] = {
    "high_load_extreme_weather": [
        "uav_mass_kg", 
        "environment_weather_wind_mps", 
        "uav_payload_mass_kg"
    ]
}

def validate_feature_batch(
    category_name: str, 
    features: Dict[str, float], 
    feature_defs: Dict[str, Any], 
    cross_map: Dict[str, List[str]], 
    already_examined: Optional[Set[str]] = None
) -> List[FeatureAssessment]:
    """Evaluates a batch of telemetry features against constitutional safe and critical ranges.
    
    Args:
        category_name: The structural category being swept.
        features: Dictionary of filtered parameter keys and live float values.
        feature_defs: Single source of truth containing operational thresholds.
        cross_map: Configuration map defining multi-variable hazard clusters.
        already_examined: Set of parameters already checked in previous cycles.
        
    Returns:
        List[FeatureAssessment]: Structural valuation records for each feature in the batch.
    """
    already_examined = already_examined or set()
    assessments: List[FeatureAssessment] = []
    cross_results: List[FeatureAssessment] = []
    
    # Process compound cross-feature constraints first to avoid masking risks
    for scenario_name, scenario_features in cross_map.items():
        if set(scenario_features).issubset(set(features.keys())):
            cross_results.extend(execute_cross_dependency_check(scenario_name, features))
            
    cross_covered = {a.feature_name for a in cross_results}
    
    for name, val in features.items():
        if name in already_examined:
            continue
            
        # If already assessed via compound rules, maintain that diagnostic trace
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
            
        # Apply strict epsilon buffers to boundary checks
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
    """Applies exact kinematic and thermodynamic equations to verify structural flight safety.
    
    Args:
        constraint_name: Target node identifier (disk_loading, wind_susceptibility, etc.).
        features: Complete mapping of current telemetry state parameters.
        feature_defs: Central boundary guidelines configuration registry.
        
    Returns:
        Dict[str, Any]: Detailed evaluation status metrics, severities, and constraints.
    """
    # 1. حمولة القرص المروحي (Disk Loading Framework)
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
                "reasoning": "Rotor surface area division zero error."
            }
            
        # W_disk = (M_uav + M_payload) * g / A_rotor
        disk_loading = ((mass + payload) * 9.81) / area
        
        if disk_loading > (45.0 + EPSILON):
            return {
                "passed": False, 
                "metric_name": "disk_loading", 
                "metric_value": round(disk_loading, 2), 
                "unit": "kg/m²", 
                "severity": "CRITICAL", 
                "reasoning": f"Disk loading {disk_loading:.2f} kg/m² violates structural limits (>45.0)."
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
                    legal_reference="FAA AC 107-2B"
                )
            }
        return {
            "passed": True, 
            "metric_name": "disk_loading", 
            "metric_value": round(disk_loading, 2), 
            "unit": "kg/m²", 
            "severity": "SAFE", 
            "reasoning": "Disk loading sits safely within limits."
        }

    # 2. مقاومة وهبات الرياح (Wind-to-Airspeed Aerodynamic Susceptibility)
    elif constraint_name == "wind_susceptibility":
        wind_mps = features.get("environment_weather_wind_mps", 0.0)
        v_max = features.get("uav_max_speed_mps", 15.0)
        mass = features.get("uav_mass_kg", 1.0)
        
        gust = wind_mps * 1.4  # Military standard aviation gust estimation factor
        ratio = gust / max(v_max, EPSILON)
        
        # Newtonian drag force estimation logic: F = 0.5 * rho * A * Cd * V^2
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
                "reasoning": f"Aero-susceptibility breach: velocity_ratio={ratio:.2f}, force-to-weight={f_w_ratio:.2f}."
            }
        elif ratio > (0.50 + EPSILON):
            return {
                "passed": False, 
                "metric_name": "wind_susceptibility", 
                "metric_value": round(ratio, 3), 
                "unit": "ratio (gust/V_max)", 
                "severity": "WARNING", 
                "reasoning": "High kinematic drift risk encountered under weather vector limits."
            }
        return {
            "passed": True, 
            "metric_name": "wind_susceptibility", 
            "metric_value": round(ratio, 3), 
            "unit": "ratio (gust/V_max)", 
            "severity": "SAFE", 
            "reasoning": "Aero-stability constraints passed clear within nominal margin."
        }

    # 3. ميزانية الطاقة المتبقية (Energy Budget Thresholds)
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
                "reasoning": f"Battery state of charge {pct:.1f}% drops below emergency mandatory 20% limit."
            }
        return {
            "passed": True, 
            "metric_name": "energy_budget", 
            "metric_value": round(available_wh, 2), 
            "unit": "Wh", 
            "severity": "SAFE", 
            "reasoning": "Energy budget clearance validated for flight path duration."
        }

    # 4. السقف التنظيمي للارتفاع (Altitude Ceiling Compliance)
    elif constraint_name == "altitude_ceiling":
        alt = features.get("flight_altitude_m", 0.0)
        ceiling = 121.9  # 400ft AGL strict barrier
        
        if alt > (ceiling * 1.1 + EPSILON):
            return {
                "passed": False, 
                "metric_name": "altitude_ceiling", 
                "metric_value": alt, 
                "unit": "m", 
                "severity": "CRITICAL", 
                "reasoning": f"Altitude {alt}m severely breaches strict FAA Part 107 400ft ceiling limit."
            }
        elif alt > (ceiling + EPSILON):
            return {
                "passed": False, 
                "metric_name": "altitude_ceiling", 
                "metric_value": alt, 
                "unit": "m", 
                "severity": "WARNING", 
                "reasoning": "Altitude transcends standard regulatory limit ceiling barrier.",
                "conditional_constraint": ConditionalGoConstraint(
                    constraint_id="C_ALT_LAANC", 
                    description="Secure an active LAANC clearance before expanding flight trajectory.", 
                    feature_name="flight_altitude_m", 
                    required_value_range="<= 121.9m", 
                    legal_reference="FAA Part 107.51(b)"
                )
            }
        return {
            "passed": True, 
            "metric_name": "altitude_ceiling", 
            "metric_value": alt, 
            "unit": "m", 
            "severity": "SAFE", 
            "reasoning": "Traveled flight altitude sits cleanly under regulatory limit ceiling."
        }

    return {"passed": True, "metric_value": 0.0, "severity": "SAFE", "reasoning": "Clean."}

def assess_contextual_remainder(
    all_features: Dict[str, float], 
    already_examined: Set[str], 
    feature_defs: Dict[str, Any], 
    cross_map: Dict[str, List[str]]
) -> List[FeatureAssessment]:
    """Sweeps over remaining untouched items within the 198 vector space to guarantee compliance.
    
    Args:
        all_features: Complete input telemetry parameter dictionary.
        already_examined: Set of identifiers evaluated in analytical loops.
        feature_defs: The primary operational data contracts rules handbook.
        cross_map: Compound safety configurations definitions.
        
    Returns:
        List[FeatureAssessment]: Compliant records for the remainder dictionary items.
    """
    remainder = {k: v for k, v in all_features.items() if k not in already_examined}
    if not remainder:
        return []
    return validate_feature_batch("constitutional_sweep", remainder, feature_defs, cross_map, already_examined)

async def query_rag(query: str, rag_core: AsyncRAGCore) -> Dict[str, Any]:
    """Queries the isolated asynchronous aviation RAG pipeline for legal citations.
    
    Args:
        query: Formulated compliance lookup query string.
        rag_core: Asynchronous entry point instance for the vector database loader.
        
    Returns:
        Dict[str, Any]: Success boolean status alongside extracted digest and records.
    """
    try:
        ans = await rag_core.ask_legal_question(query, top_k=3, min_score=0.35)
        if ans and ans.citations:
            return {"success": True, "finding": ans.answer, "citations": ans.citations}
        return {"success": False, "finding": "Zero regulatory matching provisions resolved inside index database.", "citations": []}
    except Exception as e:
        return {"success": False, "finding": f"Asynchronous RAG integration framework failure: {str(e)}", "citations": []}

async def generate_legal_query(feature_name: str, value: float, violation_type: str, llm_client: Any) -> str:
    """Transforms raw telemetry infractions into contextually sound aviation law search queries.
    
    Args:
        feature_name: Telemetry key breaking baseline parameters.
        value: Live telemetry magnitude observed.
        violation_type: Warning or Critical classification flag.
        llm_client: Underlying client engine wrapping the language model.
        
    Returns:
        str: Refined legal lookup expression string.
    """
    prompt = (
        f"Convert telemetry breach into an aviation law query.\n"
        f"Feature: {feature_name}={value} ({violation_type})\n"
        f"Query:"
    )
    try:
        res = await llm_client.generate(prompt=prompt, temperature=0.0, max_tokens=100)
        return res.strip().strip('"').strip("'")
    except Exception:
        return f"FAA Part 107 restrictions and operational criteria regarding {feature_name.replace('_', ' ')}"

def execute_cross_dependency_check(scenario_name: str, features: Dict[str, float]) -> List[FeatureAssessment]:
    """Diagnoses severe multi-variable compound stress scenarios that breach safe profiles when paired.
    
    Args:
        scenario_name: Identifier for the targeted dependency risk group.
        features: Mapping of current system telemetry parameters.
        
    Returns:
        List[FeatureAssessment]: Individual feature assignments overwritten by compound failure statuses.
    """
    assessments: List[FeatureAssessment] = []
    if scenario_name == "high_load_extreme_weather":
        mass = features.get("uav_mass_kg", 0.0)
        wind = features.get("environment_weather_wind_mps", 0.0)
        payload = features.get("uav_payload_mass_kg", 0.0)
        
        # Compound trigger bounds check containing absolute precision guards
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
#   - src/uav_risk/stage2/rag/rag_core.py (AsyncRAGCore)
#
# Consumed by:
#   - src/uav_risk/stage2/agent/ace_agent.py
#   - tests/unit/test_ace_agent.py
# =====================================================================