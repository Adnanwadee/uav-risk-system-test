# File Path: tests/unit/test_ace_agent.py
import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock
from src.uav_risk.ml.schemas import MLResult
from src.uav_risk.ml.inference import FeatureImportance
from src.uav_risk.stage2.agent.ace_agent import ACEReActAgent
from src.uav_risk.stage2.agent.fallback import StaticFallbackAssessor
from src.uav_risk.stage2.agent.agent_schemas import AgentDecision, FeatureAssessment
from src.uav_risk.stage2.rag.schemas import LegalCitation, LegalAnswer
from src.uav_risk.stage2.rag.rag_core import AsyncRAGCore

@pytest.fixture
def base_testing_environment():
    """Provides a rigorous baseline feature registry map and mock ML inference output results."""
    defs = {
        "uav_mass_kg": {"critical_min": 0.5, "critical_max": 25.0, "safe_min": 1.0, "safe_max": 7.0, "is_core": True},
        "operator_in_restricted_zone": {"critical_min": 0.0, "critical_max": 0.0, "safe_min": 0.0, "safe_max": 0.0, "is_core": True},
        "environment_weather_wind_speed_ms": {"critical_min": 0.0, "critical_max": 20.0, "safe_min": 0.0, "safe_max": 12.0, "is_core": True},
        "payload_mass_kg": {"critical_min": 0.0, "critical_max": 10.0, "safe_min": 0.0, "safe_max": 2.0, "is_core": False},
        "uav_battery_wh": {"critical_min": 50.0, "critical_max": 2000.0, "safe_min": 100.0, "safe_max": 1500.0, "is_core": True},
        "battery_remaining_pct": {"critical_min": 20.0, "critical_max": 100.0, "safe_min": 30.0, "safe_max": 100.0, "is_core": True},
        "flight_altitude_m": {"critical_min": 0.0, "critical_max": 150.0, "safe_min": 0.0, "safe_max": 121.9, "is_core": True}
    }
    ml_res = MLResult(
        risk_score=0.15,
        risk_class="LOW",
        confidence=0.94,
        probabilities={"LOW": 0.94, "MEDIUM": 0.06},
        top_features=[FeatureImportance("uav_mass_kg", 0.01, "neutral", 2.0, "mass", 1)],
        drift_score=0.0,
        drift_detected=False,
        processing_time_ms=1.0,
        model_version="v4"
    )
    return defs, ml_res

@pytest.mark.asyncio
async def test_agent_perfect_flight_scenario(base_testing_environment):
    """Scenario 1: Nominal parameters, clean telemetry, low ML risk score -> Mandatory GO decision."""
    llm, rag, defs, ml_res = AsyncMock(), AsyncMock(spec=AsyncRAGCore), base_testing_environment[0], base_testing_environment[1]
    llm.generate.return_value = '{"thought": "Nominal profile detected.", "action": "FINAL_SYNTHESIS", "tool_input": {}}'
    
    agent = ACEReActAgent(llm, rag, defs)
    v_feats = {
        "uav_mass_kg": 2.0, "operator_in_restricted_zone": 0.0, "environment_weather_wind_speed_ms": 4.0,
        "payload_mass_kg": 0.5, "uav_battery_wh": 200.0, "battery_remaining_pct": 95.0, "flight_altitude_m": 40.0,
        "uav_rotorcraft_disk_area_m2": 1.0, "uav_max_speed_ms": 25.0
    }
    
    decision = await agent.run(v_feats, ml_res)
    assert decision.decision == "GO"
    assert decision.fallback_degraded_mode is False
    assert decision.overall_risk_score == 0.15

@pytest.mark.asyncio
async def test_agent_strict_core_lock_override(base_testing_environment):
    """Scenario 2: Critical parameter breached -> Hardware interlock triggers NO-GO, overriding LLM."""
    llm, rag, defs, ml_res = AsyncMock(), AsyncMock(spec=AsyncRAGCore), base_testing_environment[0], base_testing_environment[1]
    llm.generate.return_value = '{"thought": "Attempting path clearance overrides.", "action": "FINAL_SYNTHESIS", "tool_input": {}}'
    
    agent = ACEReActAgent(llm, rag, defs)
    v_feats = {
        "uav_mass_kg": 2.0, "operator_in_restricted_zone": 1.0, "environment_weather_wind_speed_ms": 4.0,
        "payload_mass_kg": 0.5, "uav_battery_wh": 200.0, "battery_remaining_pct": 95.0, "flight_altitude_m": 40.0,
        "uav_rotorcraft_disk_area_m2": 1.0, "uav_max_speed_ms": 25.0
    }
    
    decision = await agent.run(v_feats, ml_res)
    assert decision.decision == "NO-GO"
    assert any("Sovereign Safety Core Lock Triggered" in f for f in decision.critical_findings)

@pytest.mark.asyncio
async def test_agent_conditional_go_scenario(base_testing_environment):
    """Scenario 3: Altitude warning zone infraction -> CONDITIONAL-GO decision with preserved constraints."""
    llm, rag, defs, ml_res = AsyncMock(), AsyncMock(spec=AsyncRAGCore), base_testing_environment[0], base_testing_environment[1]
    llm.generate.return_value = '{"thought": "Evaluating alert metrics.", "action": "FINAL_SYNTHESIS", "tool_input": {}}'
    
    agent = ACEReActAgent(llm, rag, defs)
    v_feats = {
        "uav_mass_kg": 2.0, "operator_in_restricted_zone": 0.0, "environment_weather_wind_speed_ms": 4.0,
        "payload_mass_kg": 0.5, "uav_battery_wh": 200.0, "battery_remaining_pct": 95.0, "flight_altitude_m": 130.0,
        "uav_rotorcraft_disk_area_m2": 1.0, "uav_max_speed_ms": 25.0
    }
    
    decision = await agent.run(v_feats, ml_res)
    assert decision.decision == "CONDITIONAL-GO"
    assert len(decision.conditional_constraints) >= 1
    assert decision.conditional_constraints[0].constraint_id == "C_ALT_LAANC"

@pytest.mark.asyncio
async def test_agent_dynamic_rag_invocation(base_testing_environment):
    """Scenario 4: Valid tool call path execution -> Verifies real-time asynchronous RAG retrieval loop integration."""
    llm, rag, defs, ml_res = AsyncMock(), AsyncMock(spec=AsyncRAGCore), base_testing_environment[0], base_testing_environment[1]
    llm.generate.side_effect = [
        '{"thought": "Querying compliance metrics.", "action": "query_rag", "tool_input": {"query": "airspace class rules"}}',
        '{"thought": "Synthesis path verified.", "action": "FINAL_SYNTHESIS", "tool_input": {}}'
    ]
    
    mock_answer = LegalAnswer(
        query="airspace class rules",  # حقن الاستعلام لحماية عقد الفئة
        answer="Restricted flight corridor directives mapped.",
        citations=[LegalCitation("FAA Code", "FAA.pdf", "107.51")],
        confidence_score=0.95,
        rag_available=True
    )
    rag.ask_legal_question.return_value = mock_answer
    
    agent = ACEReActAgent(llm, rag, defs)
    v_feats = {
        "uav_mass_kg": 2.0, "operator_in_restricted_zone": 0.0, "environment_weather_wind_speed_ms": 4.0,
        "payload_mass_kg": 0.5, "uav_battery_wh": 200.0, "battery_remaining_pct": 95.0, "flight_altitude_m": 40.0,
        "uav_rotorcraft_disk_area_m2": 1.0, "uav_max_speed_ms": 25.0
    }
    
    decision = await agent.run(v_feats, ml_res)
    assert rag.ask_legal_question.called is True
    assert len(decision.rag_queries_made) == 1

@pytest.mark.asyncio
async def test_agent_terminal_json_fallback(base_testing_environment):
    """Scenario 5: Corrupted unparseable text format strings returned -> Recovers cleanly via rule matrix fallback."""
    llm, rag, defs, ml_res = AsyncMock(), AsyncMock(spec=AsyncRAGCore), base_testing_environment[0], base_testing_environment[1]
    llm.generate.return_value = "CRITICAL ERROR: Malformed unformatted raw text message block bypassing schema structures."
    
    agent = ACEReActAgent(llm, rag, defs)
    v_feats = {
        "uav_mass_kg": 2.0, "operator_in_restricted_zone": 0.0, "environment_weather_wind_speed_ms": 4.0,
        "payload_mass_kg": 0.5, "uav_battery_wh": 200.0, "battery_remaining_pct": 95.0, "flight_altitude_m": 40.0,
        "uav_rotorcraft_disk_area_m2": 1.0, "uav_max_speed_ms": 25.0
    }
    
    decision = await agent.run(v_feats, ml_res)
    assert decision.decision == "NO-GO"
    assert decision.fallback_degraded_mode is True

@pytest.mark.asyncio
async def test_agent_full_198_features_scenario(base_testing_environment):
    """Scenario 6: Wide full vector sweep scanning -> Detects hidden downstream infrastructure infractions safely."""
    llm, rag, defs, ml_res = AsyncMock(), AsyncMock(spec=AsyncRAGCore), base_testing_environment[0], base_testing_environment[1]
    llm.generate.return_value = '{"thought": "Scanning complete parameter space.", "action": "assess_contextual_remainder", "tool_input": {}}'
    
    full_validated_features = {f"feature_{i:03d}": 10.0 for i in range(198)}
    full_validated_features["flight_altitude_m"] = 290.0
    full_validated_features["uav_rotorcraft_disk_area_m2"] = 1.0
    full_validated_features["uav_max_speed_ms"] = 25.0
    
    agent = ACEReActAgent(llm, rag, defs)
    decision = await agent.run(full_validated_features, ml_res)
    assert decision.decision == "NO-GO"

@pytest.mark.asyncio
async def test_agent_free_text_hazard_extraction(base_testing_environment):
    """Scenario 7: Ground log notes containing Arabic telemetry hazards -> Decoded natively to risk profiles."""
    llm, rag, defs, ml_res = AsyncMock(), AsyncMock(spec=AsyncRAGCore), base_testing_environment[0], base_testing_environment[1]
    llm.generate.side_effect = [
        '{"hazard_detected": true, "critical_findings": ["Flight trajectory penetrates hospital infrastructure boundaries."], "rag_queries": ["FAA civilian overflight rules"]}',
        '{"thought": "Wrapping operation records.", "action": "FINAL_SYNTHESIS", "tool_input": {}}'
    ]
    
    mock_answer = LegalAnswer(
        query="FAA civilian overflight rules",  # حقن الاستعلام لحماية عقد الفئة
        answer="Strict overflight protection clause matched.",
        citations=[],
        confidence_score=0.88,
        rag_available=True
    )
    rag.ask_legal_question.return_value = mock_answer
    
    agent = ACEReActAgent(llm, rag, defs)
    v_feats = {
        "uav_mass_kg": 2.0, "operator_in_restricted_zone": 0.0, "environment_weather_wind_speed_ms": 4.0,
        "payload_mass_kg": 0.5, "uav_battery_wh": 200.0, "battery_remaining_pct": 95.0, "flight_altitude_m": 40.0,
        "uav_rotorcraft_disk_area_m2": 1.0, "uav_max_speed_ms": 25.0
    }
    
    decision = await agent.run(v_feats, ml_res, free_text="تحليق استكشافي طارئ ومباشر فوق تجمع سكني مزدحم للغاية")
    assert len(decision.critical_findings) >= 1
    assert any("Free-Text Risk Flag" in f for f in decision.critical_findings)

@pytest.mark.asyncio
async def test_agent_circuit_breaker_tripping_behavior(base_testing_environment):
    """Scenario 8: Remote endpoint cluster crashes -> Circuit breaker trips, activating local static fallback assessor."""
    llm, rag, defs, ml_res = AsyncMock(), AsyncMock(spec=AsyncRAGCore), base_testing_environment[0], base_testing_environment[1]
    llm.generate.side_effect = Exception("Cloud Engine API Cluster Connection Severed completely - Node Timeout Failure.")
    
    agent = ACEReActAgent(llm, rag, defs)
    v_feats = {
        "uav_mass_kg": 2.0, "operator_in_restricted_zone": 0.0, "environment_weather_wind_speed_ms": 4.0,
        "payload_mass_kg": 0.5, "uav_battery_wh": 200.0, "battery_remaining_pct": 95.0, "flight_altitude_m": 40.0,
        "uav_rotorcraft_disk_area_m2": 1.0, "uav_max_speed_ms": 25.0
    }
    
    decision = await agent.run(v_feats, ml_res)
    assert decision.decision == "NO-GO"
    assert decision.fallback_degraded_mode is True
    assert agent.cb.state == "OPEN"

@pytest.mark.asyncio
async def test_agent_float_precision_boundary_stability(base_testing_environment):
    """Scenario 9: Edge value parameter boundaries -> Epsilon margins guard against floating-point resolution noise."""
    llm, rag, defs, ml_res = AsyncMock(), AsyncMock(spec=AsyncRAGCore), base_testing_environment[0], base_testing_environment[1]
    llm.generate.return_value = '{"thought": "Evaluating edge boundary tolerance metrics.", "action": "FINAL_SYNTHESIS", "tool_input": {}}'
    
    agent = ACEReActAgent(llm, rag, defs)
    v_feats = {
        "uav_mass_kg": 2.0, "operator_in_restricted_zone": 0.0, "environment_weather_wind_speed_ms": 4.0,
        "payload_mass_kg": 0.5, "uav_battery_wh": 200.0, "battery_remaining_pct": 90.0, 
        "flight_altitude_m": 121.9,
        "uav_rotorcraft_disk_area_m2": 1.0, "uav_max_speed_ms": 25.0
    }
    
    decision = await agent.run(v_feats, ml_res)
    assert decision.fallback_degraded_mode is False
    assert decision.decision == "GO"

@pytest.mark.asyncio
async def test_agent_network_latency_timeout_fallback(base_testing_environment):
    """Scenario 10: Sluggish cloud processing channels -> Exceeds 5.0s firewall SLA, falling back gracefully."""
    llm, rag, defs, ml_res = AsyncMock(), AsyncMock(spec=AsyncRAGCore), base_testing_environment[0], base_testing_environment[1]
    
    async def hung_generate_simulation(*args, **kwargs):
        await asyncio.sleep(7.0)
        return '{"thought": "Late response execution payload.", "action": "FINAL_SYNTHESIS", "tool_input": {}}'
        
    llm.generate = hung_generate_simulation
    agent = ACEReActAgent(llm, rag, defs)
    v_feats = {
        "uav_mass_kg": 2.0, "operator_in_restricted_zone": 0.0, "environment_weather_wind_speed_ms": 4.0,
        "payload_mass_kg": 0.5, "uav_battery_wh": 200.0, "battery_remaining_pct": 95.0, "flight_altitude_m": 40.0,
        "uav_rotorcraft_disk_area_m2": 1.0, "uav_max_speed_ms": 25.0
    }
    
    decision = await agent.run(v_feats, ml_res)
    assert decision.decision == "NO-GO"
    assert decision.fallback_degraded_mode is True

# =====================================================================
# Stage 2 Agent Testing Submodule Architectural Dependency Report:
#
# Depends on:
#   - src/uav_risk/ml/schemas.py (MLResult)
#   - src/uav_risk/stage2/agent/ace_agent.py (ACEReActAgent)
#   - src/uav_risk/stage2/agent/agent_schemas.py (AgentDecision)
#   - src/uav_risk/stage2/rag/schemas.py (LegalCitation, LegalAnswer)
# =====================================================================