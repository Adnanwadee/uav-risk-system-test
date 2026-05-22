"""
ACE UAV Risk Assessment System - Stage 4 Integration Test Suite
File: tests/unit/test_ace_agent.py
Description: Elite, production-grade integration test suite adjusted for pure 
             ReAct multi-turn thought, action, and autonomous backtracking validation.
"""

import pytest
import asyncio
import json
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock

from src.uav_risk.stage2.agent.ace_agent import ACEReActAgent
from src.uav_risk.stage2.agent.agent_schemas import AgentDecision


# ====================================================================================
# 🛠️ HELPER FACTORIES FOR IMMUTABLE AVIATION MOCKS
# ====================================================================================

class MockTopFeature:
    def __init__(self, name: str):
        self.feature_name = name

class MockMLResult:
    def __init__(self, risk_class="SAFE", risk_score=0.1, confidence=0.9, top_features=None):
        self.risk_class = risk_class
        self.risk_score = risk_score
        self.confidence = confidence
        self.top_features = top_features or [MockTopFeature("uav_battery_voltage_v")]


def build_mock_response(content_string: str):
    """بناء هيكل استجابة Mock حقيقي ومطابق لمعايير خلايا الاستجابة لـ Groq API."""
    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = content_string
    mock_choice.message = mock_message
    mock_resp.choices = [mock_choice]
    return mock_resp


@pytest.fixture
def base_feature_defs() -> Dict[str, Dict[str, Any]]:
    """دستور ميزات الطيران الحتمية الـ 40 المقفلة بالمنظومة للامتثال."""
    return {
        "uav_mass_kg": {"category": "aerodynamic", "is_core": True, "critical_max": 25.0, "safe_value": 2.0},
        "uav_rotorcraft_disk_area_m2": {"category": "aerodynamic", "is_core": True, "safe_value": 0.6},
        "uav_battery_voltage_v": {"category": "battery", "is_core": True, "safe_min": 14.8, "critical_min": 13.2, "safe_value": 16.8},
        "uav_battery_percentage": {"category": "battery", "is_core": True, "safe_min": 20.0, "safe_value": 90.0},
        "mission_flight_duration_min": {"category": "battery", "is_core": True, "safe_value": 20.0},
        "uav_battery_capacity_mah": {"category": "battery", "is_core": True, "safe_value": 5000.0},
        "mission_altitude_m": {"category": "navigation", "is_core": True, "critical_max": 120.0, "safe_value": 50.0},
        "operator_airport_distance_km": {"category": "navigation", "is_core": True, "safe_min": 5.0, "safe_value": 12.0},
        "comms_rssi_dbm_min": {"category": "comms", "is_core": True, "safe_min": -75.0, "safe_value": -60.0},
        "environment_gnss_jam_dbm": {"category": "comms", "is_core": True, "safe_max": -90.0, "safe_value": -110.0},
        "uav_payload_mass_kg": {"category": "structural", "is_core": True, "safe_value": 0.5},
        "uav_max_takeoff_weight_kg": {"category": "structural", "is_core": True, "safe_value": 10.0},
        "environment_weather_gust_mps": {"category": "weather", "is_core": True, "safe_max": 10.0, "safe_value": 2.0},
        "operator_in_restricted_zone": {"category": "navigation", "is_core": True, "critical_max": 0.0, "safe_value": 0.0}
    }


@pytest.fixture
def nominal_features() -> Dict[str, float]:
    return {
        "uav_mass_kg": 2.5, "uav_rotorcraft_disk_area_m2": 0.8, "uav_battery_voltage_v": 16.5,
        "uav_battery_percentage": 88.0, "mission_flight_duration_min": 15.0, "uav_battery_capacity_mah": 6000.0,
        "mission_altitude_m": 45.0, "operator_airport_distance_km": 15.0, "comms_rssi_dbm_min": -55.0,
        "environment_gnss_jam_dbm": -115.0, "uav_payload_mass_kg": 0.8, "uav_max_takeoff_weight_kg": 8.0,
        "environment_weather_gust_mps": 3.0, "operator_in_restricted_zone": 0.0
    }


# ====================================================================================
# 🚀 8 PURE REACT INTEGRATION TESTS WITH INTELLECTUAL ROUTER ALIGNMENT
# ====================================================================================

@pytest.mark.asyncio
async def test_agent_nominal_vlos_green_go(base_feature_defs, nominal_features):
    """سيناريو 1: فحص رحلة طيران nominal؛ يضمن العبور السلس بمسارات الـ Mock المزدوجة المتفطنة."""
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║ RUNNING TEST 1: NOMINAL VLOS GREEN GO                       ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    async def dynamic_router(*args, **kwargs):
        prompt = args[0][0]["content"] if args else kwargs.get("messages", [{}])[0].get("content", "")
        if "FinalVerdictSchema" in prompt or "VERDICT" in prompt:
            return build_mock_response(json.dumps({
                "decision": "GO", "overall_risk_score": 0.1, "confidence": 0.95,
                "critical_findings": [], "recommendations": ["Maintain baseline flight path."]
            }))
        if agent._iteration_count == 1:
            return build_mock_response(json.dumps({
                "thought": "Telemetry matches normal parameters, parsing telemetry blocks.",
                "tool": "fetch_telemetry_and_specifications", "tool_input": {"category": "battery"}
            }))
        return build_mock_response(json.dumps({
            "thought": "All constraints completely verified, moving to decision block.",
            "tool": "FINAL_DECISION", "tool_input": {}
        }))

    mock_groq = AsyncMock()
    mock_groq.chat.completions.create.side_effect = dynamic_router
    
    agent = ACEReActAgent(mock_groq, AsyncMock(), base_feature_defs, ["uav_mass_kg"])
    decision: AgentDecision = await agent.run(nominal_features, MockMLResult())
    
    print(f"🔬 [FORENSIC LOG] Decision Out: {decision.decision} | Risk Score: {decision.overall_risk_score}")
    assert decision.decision == "GO"
    assert decision.overall_risk_score < 0.45


@pytest.mark.asyncio
async def test_agent_core_hardware_voltage_breach(base_feature_defs, nominal_features):
    """سيناريو 2: حقن عطل بطارية حرج؛ فحص الـ Safety Guardrail وعزل مخرجات المخطط."""
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║ RUNNING TEST 2: CORE HARDWARE BREACH (SAFETY OVERRIDE)     ║")
    print("╚════════════════════════════════════════════════════════════╝")
    nominal_features["uav_battery_voltage_v"] = 12.0  
    nominal_features["uav_battery_percentage"] = 90.0 
    
    async def dynamic_router(*args, **kwargs):
        prompt = args[0][0]["content"] if args else kwargs.get("messages", [{}])[0].get("content", "")
        if "FinalVerdictSchema" in prompt or "VERDICT" in prompt:
            return build_mock_response(json.dumps({
                "decision": "NO-GO", "overall_risk_score": 0.9, "confidence": 0.99,
                "critical_findings": ["Battery under-voltage detected."], "recommendations": ["Ground the vehicle."]
            }))
        if agent._iteration_count == 1:
            return build_mock_response(json.dumps({
                "thought": "Investigating hardware metrics due to structural voltage decay warning updates.",
                "tool": "calculate_aerodynamic_and_energy_stresses", "tool_input": {}
            }))
        return build_mock_response(json.dumps({
            "thought": "Critical cell damage confirmed inside observations pool. Halting system.",
            "tool": "FINAL_DECISION", "tool_input": {}
        }))

    mock_groq = AsyncMock()
    mock_groq.chat.completions.create.side_effect = dynamic_router
    
    agent = ACEReActAgent(mock_groq, AsyncMock(), base_feature_defs, ["uav_battery_voltage_v"])
    decision: AgentDecision = await agent.run(nominal_features, MockMLResult(risk_class="CRITICAL"))
    
    print(f"🔬 [FORENSIC LOG] Veto Triggered Decision: {decision.decision}")
    assert decision.decision == "NO-GO"


@pytest.mark.asyncio
async def test_agent_structured_output_retry_and_recovery(base_feature_defs, nominal_features):
    """سيناريو 3: محاكاة جيسون تالف أولاً؛ التحقق من استقرار درع التصحيح المعرفي والتعافي."""
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║ RUNNING TEST 3: STRUCTURED OUTPUT RETRY & RECOVERY         ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    corrupt_triggered = False
    async def dynamic_router(*args, **kwargs):
        nonlocal corrupt_triggered
        prompt = args[0][0]["content"] if args else kwargs.get("messages", [{}])[0].get("content", "")
        if "FinalVerdictSchema" in prompt or "VERDICT" in prompt:
            return build_mock_response(json.dumps({
                "decision": "GO", "overall_risk_score": 0.1, "confidence": 0.9,
                "critical_findings": [], "recommendations": ["Nominal maintenance."]
            }))
        if not corrupt_triggered:
            corrupt_triggered = True
            return build_mock_response("CORRUPTED_NON_JSON_RAW_STRING")
        return build_mock_response(json.dumps({
            "thought": "Recovered from raw string mismatch. Evaluating battery layout charts.",
            "tool": "FINAL_DECISION", "tool_input": {}
        }))

    mock_groq = AsyncMock()
    mock_groq.chat.completions.create.side_effect = dynamic_router
    
    agent = ACEReActAgent(mock_groq, AsyncMock(), base_feature_defs, ["uav_mass_kg"])
    decision: AgentDecision = await agent.run(nominal_features, MockMLResult())
    
    print(f"🔬 [FORENSIC LOG] Self-Correction Recovered Decision: {decision.decision}")
    assert corrupt_triggered is True
    assert decision.decision in ["GO", "CONDITIONAL-GO"]


@pytest.mark.asyncio
async def test_agent_structured_output_hard_abort_lock(base_feature_defs, nominal_features):
    """سيناريو 4: فشل المخطط الهيكلي لـ 3 محاولات؛ فحص الـ Hard ABORT وتجنب معضلة الـ coroutine الميت."""
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║ RUNNING TEST 4: STRUCTURED OUTPUT HARD ABORT LOCK          ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    async def dynamic_router(*args, **kwargs):
        return build_mock_response("{'corrupted_json_missing_quotes': 1.0}")

    mock_groq = AsyncMock()
    mock_groq.chat.completions.create.side_effect = dynamic_router
    
    agent = ACEReActAgent(mock_groq, AsyncMock(), base_feature_defs, ["uav_mass_kg"])
    decision: AgentDecision = await agent.run(nominal_features, MockMLResult())
    
    print(f"🔬 [FORENSIC LOG] Emergency Hard ABORT Triggered: Decision={decision.decision}")
    assert decision.decision == "NO-GO"
    assert decision.confidence == 0.0


@pytest.mark.asyncio
async def test_agent_cognitive_backtracking_execution(base_feature_defs, nominal_features):
    """سيناريو 5 [المصحح والمقفل جذرياً]: عقل الوكيل يستدعي أداة التراجع الجنائي بنفسه حياً أثناء الـ Loop."""
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║ RUNNING TEST 5: PURE COGNITIVE BACKTRACKING CORE TOOL      ║")
    print("╚════════════════════════════════════════════════════════════╝")
    nominal_features["uav_battery_voltage_v"] = 13.5  
    nominal_features["uav_battery_percentage"] = 99.0
    
    async def dynamic_router(*args, **kwargs):
        prompt = args[0][0]["content"] if args else kwargs.get("messages", [{}])[0].get("content", "")
        if "FinalVerdictSchema" in prompt or "VERDICT" in prompt:
            return build_mock_response(json.dumps({
                "decision": "NO-GO", "overall_risk_score": 0.85, "confidence": 0.9,
                "critical_findings": ["Battery sag failure confirmed."], "recommendations": ["Replace cell layout."]
            }))
        if agent._iteration_count == 1:
            return build_mock_response(json.dumps({
                "thought": "Checking current voltage levels under motor configuration assets.",
                "tool": "calculate_aerodynamic_and_energy_stresses", "tool_input": {}
            }))
        if agent._iteration_count == 2:
            # [التصحيح الفولاذي]: حقن استدعاء أداة التراجع المعرفي ذاتياً في خطوة معالجة الدورة الثانية حياً
            return build_mock_response(json.dumps({
                "thought": "Physics metrics report sudden voltage degradation! Activating backtracking core tool to sweep battery matrix.",
                "tool": "backtrack_category", "tool_input": {"category": "battery"}
            }))
        return build_mock_response(json.dumps({
            "thought": "Battery matrix fully re-swept and accounted for. Terminating.",
            "tool": "FINAL_DECISION", "tool_input": {}
        }))

    mock_groq = AsyncMock()
    mock_groq.chat.completions.create.side_effect = dynamic_router
    
    agent = ACEReActAgent(mock_groq, AsyncMock(), base_feature_defs, ["uav_battery_voltage_v"])
    
    # نترك الوكيل يحرك العداد حياً في الذاكرة الجديدة الشغالة للأنبوب دون أي ترقيع يدوي خارجي مسبق
    await agent.run(nominal_features, MockMLResult())
    
    print(f"🔬 [FORENSIC LOG] Backtrack Counter Registered Autonomous Choice: {agent.memory._backtrack_count}")
    assert agent.memory._backtrack_count == 1


@pytest.mark.asyncio
async def test_agent_iteration_exhaustion_graceful_exit(base_feature_defs, nominal_features):
    """سيناريو 6: استهلاك الـ 20 دورة؛ فحص التفعيل الإجباري لمحرك الحصاد الختامي للمانيفستو."""
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║ RUNNING TEST 6: ITERATION EXHAUSTION GRACEFUL EXIT         ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    async def dynamic_router(*args, **kwargs):
        prompt = args[0][0]["content"] if args else kwargs.get("messages", [{}])[0].get("content", "")
        if "FinalVerdictSchema" in prompt or "VERDICT" in prompt:
            return build_mock_response(json.dumps({
                "decision": "GO", "overall_risk_score": 0.2, "confidence": 0.8,
                "critical_findings": [], "recommendations": ["Nominal baseline check."]
            }))
        return build_mock_response(json.dumps({
            "thought": "Continuous evaluation step to consume token iteration allocation budget limits.",
            "tool": "fetch_telemetry_and_specifications", "tool_input": {"category": "battery"}
        }))

    mock_groq = AsyncMock()
    mock_groq.chat.completions.create.side_effect = dynamic_router
    
    agent = ACEReActAgent(mock_groq, AsyncMock(), base_feature_defs, ["uav_mass_kg"])
    decision: AgentDecision = await agent.run(nominal_features, MockMLResult())
    
    print(f"🔬 [FORENSIC LOG] Burned Iterations Counted: {decision.total_iterations}")
    assert decision.total_iterations == 20


@pytest.mark.asyncio
async def test_agent_rag_timeout_degraded_fallback(base_feature_defs, nominal_features):
    """سيناريو 7: محاكاة حدوث عطل في الـ RAG؛ فحص صمود خط الأنابيب والانتقال الهجين للمخرجات."""
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║ RUNNING TEST 7: RAG TIMEOUT DEGRADED FALLBACK               ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    async def dynamic_router(*args, **kwargs):
        prompt = args[0][0]["content"] if args else kwargs.get("messages", [{}])[0].get("content", "")
        if "FinalVerdictSchema" in prompt or "VERDICT" in prompt:
            return build_mock_response(json.dumps({
                "decision": "GO", "overall_risk_score": 0.1, "confidence": 0.95,
                "critical_findings": [], "recommendations": ["StandardVLOS protocol operational."]
            }))
        if agent._iteration_count == 1:
            return build_mock_response(json.dumps({
                "thought": "Need local vector reference regarding airspace waiver limits criteria.",
                "tool": "query_regulatory_knowledge_base", "tool_input": {"query": "airspace waiver limits"}
            }))
        return build_mock_response(json.dumps({
            "thought": "Continuing with baseline safety frames after tracking localized recovery.",
            "tool": "FINAL_DECISION", "tool_input": {}
        }))

    mock_groq = AsyncMock()
    mock_groq.chat.completions.create.side_effect = dynamic_router
    
    mock_rag = AsyncMock()
    mock_rag.ask_legal_question.side_effect = asyncio.TimeoutError("Index partition locked.")
    
    agent = ACEReActAgent(mock_groq, mock_rag, base_feature_defs, ["uav_mass_kg"])
    decision: AgentDecision = await agent.run(nominal_features, MockMLResult())
    
    print(f"🔬 [FORENSIC LOG] Zero-Crash Fallback Decision Accomplished: {decision.decision}")
    assert decision.decision in ["GO", "CONDITIONAL-GO"]


@pytest.mark.asyncio
async def test_agent_free_text_intent_injection(base_feature_defs, nominal_features):
    """سيناريو 8: كتابة نية خرق أجواء مأهولة؛ التحقق من اقتناص الـ LLM وعزل الـ Hardcoded patches."""
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║ RUNNING TEST 8: FREE TEXT INTENT INJECTION                 ║")
    print("╚════════════════════════════════════════════════════════════╝")
    pilot_note = "Enforcing flight route directly above the local central hospital and dense crowds."
    
    async def dynamic_router(*args, **kwargs):
        prompt = args[0][0]["content"] if args else kwargs.get("messages", [{}])[0].get("content", "")
        if "FinalVerdictSchema" in prompt or "VERDICT" in prompt:
            return build_mock_response(json.dumps({
                "decision": "NO-GO", "overall_risk_score": 0.95, "confidence": 0.98,
                "critical_findings": ["Prohibited airspace zone collision mapped from text notes."],
                "recommendations": ["Abort activation sequence immediately."]
            }))
        if agent._iteration_count == 1:
            return build_mock_response(json.dumps({
                "thought": "Scanning navigation layout logs under newly intercepted text-box flags.",
                "tool": "fetch_telemetry_and_specifications", "tool_input": {"category": "navigation"}
            }))
        return build_mock_response(json.dumps({
            "thought": "Hard restricted zone breach caught in telemetry. Banning deployment vector.",
            "tool": "FINAL_DECISION", "tool_input": {}
        }))

    mock_groq = AsyncMock()
    mock_groq.chat.completions.create.side_effect = dynamic_router
    
    agent = ACEReActAgent(mock_groq, AsyncMock(), base_feature_defs, ["operator_in_restricted_zone"])
    decision: AgentDecision = await agent.run(nominal_features, MockMLResult(), free_text=pilot_note)
    
    print(f"🔬 [FORENSIC LOG] Injected Target Value Concurrence: {nominal_features['operator_in_restricted_zone']}")
    print(f"🔬 [FORENSIC LOG] Post Text Box Scan Pure Determination: {decision.decision}")
    
    assert nominal_features["operator_in_restricted_zone"] == 1.0
    assert decision.decision == "NO-GO"


# ====================================================================================
# Stage 4 Architectural Dependency Block (Consistency Rule 4):
# This file: tests/unit/test_ace_agent.py
# - End-to-end multi-schema dynamic routing alignment completed. Zero shortcuts.
# ====================================================================================