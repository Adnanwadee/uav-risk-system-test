"""
ACE UAV Risk Assessment System - Stage 4 (Pure ReAct Agent Engine)
File: src/uav_risk/stage2/agent/ace_agent.py
Description: Production-grade ReAct agent engine with exact import links,
             strict structured outputs, and integrated cognitive final synthesis.
"""

import json
import time
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import structlog

from src.uav_risk.stage2.agent.agent_schemas import AgentDecision, ReasoningStep, ToolCall, FeatureAssessment
from src.uav_risk.stage2.agent.agent_memory import AgentMemory
from src.uav_risk.stage2.agent.agent_tools import (
    fetch_telemetry_and_specifications,
    calculate_aerodynamic_and_energy_stresses,
    query_regulatory_knowledge_base,
    execute_unexamined_manifest_harvester,
    backtrack_category
)

logger = structlog.get_logger()


class CognitiveFeatureEvaluation(BaseModel):
    feature_name: str = Field(..., description="The standardized feature key identifier.")
    assigned_status: str = Field(..., description="The status evaluated by LLM. Must be: SAFE | WARNING | CRITICAL.")
    analytical_reasoning: str = Field(..., description="Aviation engineering justification for this status assignment.")
    action_required: Optional[str] = Field(None, description="Required mitigation checklist for the operator.")


class AgentActionSchema(BaseModel):
    thought: str = Field(..., description="Metacognitive analysis evaluating working memory records.")
    tool: str = Field(..., description="The tool selection. Must be exactly one of: 'fetch_telemetry_and_specifications' | 'calculate_aerodynamic_and_energy_stresses' | 'query_regulatory_knowledge_base' | 'backtrack_category' | 'FINAL_DECISION'.")
    tool_input: Dict[str, Any] = Field(default_factory=dict, description="Parameters matching tool signatures.")
    cognitive_evaluations: Optional[List[CognitiveFeatureEvaluation]] = Field(None, description="Dynamic updates to working memory based on raw tool data.")


class FinalVerdictSchema(BaseModel):
    decision: str = Field(..., description="The absolute sovereign determination. Must be exactly: GO | NO-GO | CONDITIONAL-GO")
    overall_risk_score: float = Field(..., description="Calculated composite risk index between 0.0 and 1.0.")
    confidence: float = Field(..., description="Structured certainty score regarding this flight verdict between 0.0 and 1.0.")
    critical_findings: List[str] = Field(default_factory=list, description="Summary statement of core violations.")
    recommendations: List[str] = Field(default_factory=list, description="Actionable checklists for secondary tracks.")


class ACEReActAgent:
    MAX_ITERATIONS = 20
    MAX_RAG_QUERIES = 8

    def __init__(self, groq_client: Any, rag_core: Any, feature_defs: Dict[str, Dict[str, Any]], core_feature_names: List[str]):
        self.client = groq_client  
        self.rag_core = rag_core
        self.feature_defs = feature_defs
        self.core_feature_names = core_feature_names
        self.memory = AgentMemory(list(feature_defs.keys()))
        self._iteration_count = 0
        self._rag_counter = 0

        logger.info("pure_cognitive_agent_instantiated", max_budget=self.MAX_ITERATIONS)

    async def run(self, validated_features: Dict[str, float], ml_result: Any, free_text: Optional[str] = None) -> AgentDecision:
        start_time = time.perf_counter()
        self._iteration_count = 0
        self._rag_counter = 0
        self.memory = AgentMemory(list(self.feature_defs.keys()))

        logger.info("pure_agent_pipeline_ignited", machine_learning_class=ml_result.risk_class)

        top_shap = [f.feature_name for f in getattr(ml_result, "top_features", []) if hasattr(f, "feature_name")]
        self.memory.reprioritize_with_shap(top_shap_features=top_shap, core_features=self.core_feature_names)

        if free_text:
            await self._process_free_text(free_text, validated_features)

        try:
            while self._iteration_count < self.MAX_ITERATIONS and self.memory.pending_features:
                self._iteration_count += 1
                
                action_pack = await self._inference_pure_react_step(ml_result)
                
                tool_name = action_pack.get("tool")
                tool_input = action_pack.get("tool_input", {})
                thought = action_pack.get("thought", "PROCESSING")
                cognitive_evals = action_pack.get("cognitive_evaluations")

                if cognitive_evals:
                    for ev in cognitive_evals:
                        assessment_record = FeatureAssessment(
                            feature_name=ev["feature_name"],
                            value=validated_features.get(ev["feature_name"], 0.0),
                            status=ev["assigned_status"].upper(),
                            reasoning=ev["analytical_reasoning"],
                            rag_consulted=False,
                            action_required=ev.get("action_required")
                        )
                        self.memory.mark_feature_examined(assessment_record)

                if tool_name == "FINAL_DECISION":
                    logger.info("agent_voluntarily_concluded_reasoning", iteration=self._iteration_count)
                    break
                    
                observation, tool_call_record = await self._act(tool_name, tool_input, validated_features)
                
                self.memory.add_reasoning_step(ReasoningStep(
                    step_number=self._iteration_count,
                    thought=thought,
                    action=json.dumps({"tool": tool_name, "tool_input": tool_input}),
                    tool_call=tool_call_record,
                    observation=observation,
                    features_examined=list(self.memory.examined_features.keys())
                ))

        except Exception as e:
            if "STRIKE_3_COLLAPSE" in str(e):
                elapsed = (time.perf_counter() - start_time) * 1000.0
                return self._generate_hard_abort_decision(elapsed)
            raise e

        execute_unexamined_manifest_harvester(self.memory, validated_features, self.feature_defs)
        elapsed_total = (time.perf_counter() - start_time) * 1000.0
        
        return await self._synthesize_decision_via_llm(ml_result, elapsed_total)

    async def _process_free_text(self, free_text: str, validated_features: Dict[str, float]) -> None:
        normalized_text = free_text.lower()
        if "hospital" in normalized_text or "crowd" in normalized_text or "populated" in normalized_text:
            validated_features["operator_in_restricted_zone"] = 1.0
            logger.warn("safety_flag_injected_from_free_text", target_feature="operator_in_restricted_zone")

    async def _inference_pure_react_step(self, ml_result: Any) -> Dict[str, Any]:
        summary = self.memory.build_context_summary()
        system_prompt = (
            f"You are the supreme cognitive UAV safety inspector. State Summary: {summary}.\n"
            f"Machine Learning Tree Risk Level: {ml_result.risk_class}.\n"
            f"Respond strictly via valid JSON matching this schema:\n"
            f"{AgentActionSchema.model_json_schema()}"
        )
        return await self._execute_structured_groq_call(system_prompt, AgentActionSchema)

    async def _synthesize_decision_via_llm(self, ml_result: Any, elapsed_ms: float) -> AgentDecision:
        history_summary = self.memory.build_context_summary()
        examined_manifest = [
            {"feature": f.feature_name, "status": f.status, "reasoning": f.reasoning}
            for f in self.memory.examined_features.values()
        ]
        synthesis_prompt = (
            f"URGENT AVIATION VERDICT REQUIRED. Logs: {history_summary}.\n"
            f"Manifest Dump: {json.dumps(examined_manifest)}.\n"
            f"Respond strictly via valid JSON matching this schema:\n"
            f"{FinalVerdictSchema.model_json_schema()}"
        )
        verdict_pack = await self._execute_structured_groq_call(synthesis_prompt, FinalVerdictSchema)
        
        assigned_decision = verdict_pack.get("decision", "NO-GO").upper()
        calculated_risk = float(verdict_pack.get("overall_risk_score", 1.0))
        confidence = float(verdict_pack.get("confidence", 0.0))
        critical_findings = verdict_pack.get("critical_findings", [])
        recommendations = verdict_pack.get("recommendations", [])

        memory_has_criticals = any(f.status == "CRITICAL" for f in self.memory.examined_features.values())
        if (memory_has_criticals or calculated_risk >= 0.7 or ml_result.risk_class == "CRITICAL") and assigned_decision == "GO":
            assigned_decision = "NO-GO"
            critical_findings.append("CRITICAL_SAFETY_GUARDRAIL_OVERRIDE: Telemetry anomalies inside memory forced a hard ground lock veto.")

        citations_pool = []
        for cached_ans in self.memory.rag_cache.values():
            if hasattr(cached_ans, "citations") and cached_ans.citations:
                citations_pool.extend(cached_ans.citations)

        return AgentDecision(
            decision=assigned_decision,
            overall_risk_score=max(calculated_risk, ml_result.risk_score, self.memory.get_overall_risk_so_far()),
            confidence=confidence,
            reasoning_chain=list(self.memory.reasoning_steps),
            feature_assessments=list(self.memory.examined_features.values()),
            critical_findings=critical_findings,
            recommendations=recommendations,
            legal_citations=citations_pool,
            rag_queries_made=list(self.memory.rag_queries_history),
            memory_snapshots=[self.memory.get_snapshot()],
            total_iterations=self._iteration_count,
            processing_time_ms=elapsed_ms
        )

    async def _execute_structured_groq_call(self, prompt_content: str, schema_class: Any) -> Dict[str, Any]:
        attempts = 0
        last_error = ""
        while attempts < 3:
            attempts += 1
            try:
                messages = [{"role": "user", "content": prompt_content}]
                if last_error:
                    messages.append({"role": "assistant", "content": f"Schema Repair Target Log: {last_error}"})

                completion = await self.client.chat.completions.create(
                    messages=messages,
                    model="llama3-70b-8192",
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                raw_json = completion.choices[0].message.content
                parsed = json.loads(raw_json)
                validated = schema_class(**parsed)
                return validated.model_dump()
            except Exception as e:
                last_error = str(e)
                logger.error("structured_output_parse_retry", attempt=attempts, schema=schema_class.__name__, error=str(e))
                if attempts >= 3:
                    raise RuntimeError("STRIKE_3_COLLAPSE")
        return {}

    async def _act(self, tool_name: str, tool_input: Dict[str, Any], validated_features: Dict[str, float]) -> tuple[str, ToolCall]:
        start = time.perf_counter()
        try:
            if tool_name == "fetch_telemetry_and_specifications":
                cat = tool_input.get("category", "battery")
                facts = fetch_telemetry_and_specifications(cat, validated_features, self.feature_defs, self.memory)
                obs = f"Factual Telemetry for Category '{cat}': {json.dumps(facts)}"
            elif tool_name == "calculate_aerodynamic_and_energy_stresses":
                report = calculate_aerodynamic_and_energy_stresses(validated_features, self.feature_defs)
                obs = f"Aviation Mechanical Physics Report: {json.dumps(report)}"
            elif tool_name == "query_regulatory_knowledge_base" and self._rag_counter < self.MAX_RAG_QUERIES:
                self._rag_counter += 1
                q = tool_input.get("query", "airspace limits")
                ans = await query_regulatory_knowledge_base(q, self.rag_core, self.memory)
                obs = f"Regulatory offline documentation chunks retrieved: {ans.answer}. Absolute Citations: {len(ans.citations)}"
            elif tool_name == "backtrack_category":
                if self.memory.can_backtrack():
                    self.memory.increment_backtrack()
                    cat = tool_input.get("category", "battery")
                    facts = backtrack_category(cat, validated_features, self.feature_defs, self.memory)
                    obs = f"Cognitive Backtracking Tool Triggered: Category '{cat}' has been entirely reopened. Re-fetched facts: {json.dumps(facts)}"
                else:
                    obs = "Backtracking tool rejected: Hard session limit budget completely exhausted."
            else:
                obs = f"Execution rejection: Tool '{tool_name}' missing or argument layout incompatible."

            elapsed = (time.perf_counter() - start) * 1000.0
            record = ToolCall(tool_name=tool_name, tool_input=tool_input, tool_output=obs, execution_time_ms=elapsed, success=True)
            return obs, record
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000.0
            obs = f"Degraded Tool Recovery: Command failed. Error trace: {str(e)}"
            record = ToolCall(tool_name=tool_name, tool_input=tool_input, tool_output=obs, execution_time_ms=elapsed, success=False, error_message=str(e))
            return obs, record

    def _generate_hard_abort_decision(self, elapsed_ms: float) -> AgentDecision:
        return AgentDecision(
            decision="NO-GO", overall_risk_score=1.0, confidence=0.0, reasoning_chain=[],
            feature_assessments=list(self.memory.examined_features.values()),
            critical_findings=["CRITICAL_AGENT_STRUCTURED_OUTPUT_COLLAPSE: LLM systematically broke valid schema structures after 3 attempts."],
            recommendations=["Enforce absolute ground lock.", "Manual inspector verification mandatory."],
            legal_citations=[], rag_queries_made=[], memory_snapshots=[self.memory.get_snapshot()], total_iterations=1, processing_time_ms=elapsed_ms
        )

# ====================================================================================
# Stage 4 Architectural Dependency Block (Consistency Rule 4):
# - Depends on: agent_schemas.py, agent_memory.py, agent_tools.py
# ====================================================================================