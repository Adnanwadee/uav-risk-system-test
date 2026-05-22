# File Path: scripts/simulate_agent_live.py
import os
import sys
import json
import asyncio
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import AsyncGroq
from src.uav_risk.ml.schemas import MLResult
from src.uav_risk.ml.inference import FeatureImportance
from src.uav_risk.stage2.agent.ace_agent import ACEReActAgent
from src.uav_risk.stage2.rag.rag_core import AsyncRAGCore

load_dotenv()

class LiveGroqPlatformClient:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("🚨 GROQ_API_KEY is missing from .env")
        self.client = AsyncGroq(api_key=api_key)
        self.model_name = "llama-3.3-70b-versatile"

    async def generate(self, prompt: str, system: str = "", temperature: float = 0.0, max_tokens: int = 1024, response_format: Optional[Dict[str, Any]] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        completion = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format
        )
        return str(completion.choices[0].message.content)

async def execute_live_cognitive_simulation() -> None:
    print("=" * 90)
    print("🛫 RUNNING ACE LIVE COGNITIVE INTERACTIVE SIMULATION")
    print("=" * 90)
    
    api_key = os.getenv("GROQ_API_KEY", "")
    llm_client = LiveGroqPlatformClient(api_key=api_key)
    
    rag_core = AsyncRAGCore()
    if hasattr(rag_core, "_initialize_retriever"):
        await rag_core._initialize_retriever()

    feature_defs = {
        "uav_mass_kg": {"critical_min": 0.5, "critical_max": 25.0, "safe_min": 1.0, "safe_max": 7.0, "is_core": True},
        "operator_in_restricted_zone": {"critical_min": 0.0, "critical_max": 0.0, "safe_min": 0.0, "safe_max": 0.0, "is_core": True},
        "environment_weather_wind_speed_ms": {"critical_min": 0.0, "critical_max": 20.0, "safe_min": 0.0, "safe_max": 12.0, "is_core": True},
        "payload_mass_kg": {"critical_min": 0.0, "critical_max": 10.0, "safe_min": 0.0, "safe_max": 2.0, "is_core": False},
        "uav_battery_wh": {"critical_min": 50.0, "critical_max": 2000.0, "safe_min": 100.0, "safe_max": 1500.0, "is_core": True},
        "battery_remaining_pct": {"critical_min": 20.0, "critical_max": 100.0, "safe_min": 30.0, "safe_max": 100.0, "is_core": True},
        "flight_altitude_m": {"critical_min": 0.0, "critical_max": 150.0, "safe_min": 0.0, "safe_max": 121.9, "is_core": True}
    }

    live_telemetry = {
        "uav_mass_kg": 4.5, "operator_in_restricted_zone": 0.0, "environment_weather_wind_speed_ms": 4.0,
        "payload_mass_kg": 1.2, "uav_battery_wh": 500.0, "battery_remaining_pct": 88.0, "flight_altitude_m": 125.5,
        "uav_rotorcraft_disk_area_m2": 1.0, "uav_max_speed_ms": 22.0
    }

    mock_ml_result = MLResult(
        risk_score=0.28, risk_class="LOW", confidence=0.91, probabilities={"LOW": 0.85, "MEDIUM": 0.15},
        top_features=[FeatureImportance("flight_altitude_m", 0.12, "positive", 125.5, "navigation", 1)],
        drift_score=0.0, drift_detected=False, processing_time_ms=1.5, model_version="v4.5-live"
    )

    agent = ACEReActAgent(llm_client=llm_client, rag_core=rag_core, feature_defs=feature_defs)
    
    print("\n🧠 Sending telemetry stream to Live Agent Loop...")
    decision = await agent.run(validated_features=live_telemetry, ml_result=mock_ml_result)
    
    print("\n" + "=" * 90)
    print("📊 LIVE AGENT COGNITIVE REPORT")
    print("=" * 90)
    print(f"🎖️ SOVEREIGN DECISION  : {decision.decision}")
    print(f"📈 RISK SCORE           : {decision.overall_risk_score}")
    print(f"🎯 CONFIDENCE           : {decision.confidence * 100:.2f}%")
    print(f"🔄 TOTAL ITERATIONS    : {decision.total_iterations}")
    print("-" * 90)

    print("\n📝 STEP-BY-STEP REASONING CHAIN:")
    for step in decision.reasoning_chain:
        print(f"\n[Cycle {step.step_number}]")
        print(f"  🤔 Thought   : {step.thought}")
        print(f"  🛠️ Tool Used : {step.action}")
        print(f"  👁️ Observation: {step.observation}")
        print("-" * 50)

    if decision.conditional_constraints:
        print("\n⚠️ MANDATORY CONDITIONAL CONSTRAINTS:")
        for idx, c in enumerate(decision.conditional_constraints, 1):
            print(f"  {idx}. [{c.constraint_id}] {c.description} -> Range: {c.required_value_range}")

    print("\n📋 RECOMMENDATIONS:")
    for rec in decision.recommendations:
        print(f"  ✔ {rec}")
    print("=" * 90)

if __name__ == "__main__":
    asyncio.run(execute_live_cognitive_simulation())