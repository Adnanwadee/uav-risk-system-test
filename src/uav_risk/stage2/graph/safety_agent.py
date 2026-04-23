# src/uav_risk/stage2/graph/safety_agent.py
from __future__ import annotations
from typing import Dict, Any, Literal
import json
import re
import logging
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from ..schemas import AgentState, FinalDecision
from ..input_contract import InputContractEngine
from ..policies.deterministic_core import DeterministicCore
from ..tools.toolbox import AGENT_TOOLS

logger = logging.getLogger(__name__)

# ==========================================
# 1. LLM Setup
# ==========================================
LLM = ChatGroq(model="llama3-70b-8192", temperature=0.0)
LLM_STRUCTURED = LLM.with_structured_output(FinalDecision)
MAX_REFLECTION_ITERATIONS = 2

# ==========================================
# 2. Graph Nodes
# ==========================================

def assess_quality_node(state: AgentState) -> Dict[str, Any]:
    quality_profile = InputContractEngine.evaluate_scenario(state["scenario"])
    return {
        "data_quality_profile": quality_profile,
        "confidence_score": quality_profile.confidence_score,
        "iteration_count": 0,
        "reasoning_chain": []
    }

def pre_flight_check_node(state: AgentState) -> Dict[str, Any]:
    veto_res = DeterministicCore.pre_flight_veto_check(state["scenario"])
    return {"deterministic_veto_pre": veto_res.reason if veto_res.is_veto else None}

def reasoning_node(state: AgentState) -> Dict[str, Any]:
    profile = state["data_quality_profile"]
    
    directives_text = "\n".join([f"- [{d.rule_id}] {d.rationale} -> ACTION: {d.action}" for d in profile.tactical_directives])
    
    sys_prompt = f"""You are a Safety-Critical Aviation Agent.
    Your task is to analyze UAV flight risks and issue a GO, NO_GO, or CAUTION decision.
    
    ⚠️ MANDATORY TACTICAL DIRECTIVES (STRICTLY OBEY):
    {directives_text if directives_text else "Data integrity is optimal. Proceed with standard analysis."}
    
    RULES:
    1. Base decisions on physics, weather dynamics, and aviation regulations.
    2. Use 'search_aviation_regulations' for legal thresholds.
    3. If 'IGNORE_ML' is in directives, DO NOT call the ML tool.
    4. Provide explicit Chain-of-Thought before concluding.
    """
    
    messages = [SystemMessage(content=sys_prompt)] + state["messages"]
    
    # Dynamic tool binding based on data reliability
    available_tools = AGENT_TOOLS if profile.is_ml_reliable else [t for t in AGENT_TOOLS if t.name != "get_ml_risk_prediction"]
    llm_bound = LLM.bind_tools(available_tools)
    
    response = llm_bound.invoke(messages)
    return {"messages": [response]}

def self_correction_node(state: AgentState) -> Dict[str, Any]:
    last_msg = state["messages"][-1]
    
    if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
        # Verify citation integrity
        citations = re.findall(r'\[E\d+\]', last_msg.content)
        if citations and not state.get("retrieved_regulations"):
            correction = HumanMessage(
                content="⚠️ SYSTEM ALERT: You cited regulations [E...] without querying the database. "
                        "Use the search tool to retrieve actual legal text, or remove fake citations."
            )
            return {
                "messages": [correction],
                "iteration_count": state["iteration_count"] + 1
            }
            
    return {"iteration_count": state["iteration_count"] + 1}

def post_flight_check_node(state: AgentState) -> Dict[str, Any]:
    # Safe extraction of tool outputs
    tool_outputs = {}
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage) and msg.name == "get_ml_risk_prediction":
            try:
                content = msg.content
                tool_outputs["ml_prediction"] = json.loads(content) if isinstance(content, str) else content
            except Exception as e:
                logger.warning(f"Failed to parse ML tool output: {e}")

    # Robust draft decision extraction
    last_ai = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
    content = last_ai.content.lower() if last_ai else ""
    if "no_go" in content or "no-go" in content or "reject" in content:
        draft = "NO_GO"
    elif "caution" in content or "advisory" in content:
        draft = "CAUTION"
    else:
        draft = "GO"

    post_veto = DeterministicCore.post_flight_veto_check(draft, state["scenario"], tool_outputs)
    return {"deterministic_veto_post": post_veto.reason if post_veto.is_veto else None}

def format_output_node(state: AgentState) -> Dict[str, Any]:
    if state.get("deterministic_veto_pre"):
        return {
            "agent_decision": "NO_GO",
            "reasoning_chain": [f"PHYSICS_VETO: {state['deterministic_veto_pre']}"]
        }
    if state.get("deterministic_veto_post"):
        return {
            "agent_decision": "NO_GO",
            "reasoning_chain": [f"AI_OVERRIDE_VETO: {state['deterministic_veto_post']}"]
        }

    # Fallback for structured output failure
    try:
        final_decision: FinalDecision = LLM_STRUCTURED.invoke(state["messages"])
        chain = (state.get("reasoning_chain") or []) + [final_decision.justification]
        return {
            "agent_decision": final_decision.decision,
            "reasoning_chain": chain
        }
    except Exception as e:
        logger.error(f"Structured output failed: {e}. Falling back to cautious decision.")
        return {
            "agent_decision": "CAUTION",
            "reasoning_chain": [f"LLM_FORMAT_FAILURE: Fallback to CAUTION due to {e}"]
        }

# ==========================================
# 3. Routing Logic
# ==========================================

def route_pre_flight(state: AgentState) -> Literal["format_output", "reasoning"]:
    return "format_output" if state.get("deterministic_veto_pre") else "reasoning"

def route_reasoning(state: AgentState) -> Literal["tools", "self_correction"]:
    last = state["messages"][-1]
    return "tools" if last.tool_calls else "self_correction"

def route_self_correction(state: AgentState) -> Literal["reasoning", "post_flight_check"]:
    last = state["messages"][-1]
    if isinstance(last, HumanMessage) and state["iteration_count"] < MAX_REFLECTION_ITERATIONS:
        return "reasoning"
    return "post_flight_check"

# ==========================================
# 4. Graph Compilation
# ==========================================

workflow = StateGraph(AgentState)
workflow.add_node("assess_quality", assess_quality_node)
workflow.add_node("pre_flight_check", pre_flight_check_node)
workflow.add_node("reasoning", reasoning_node)
workflow.add_node("tools", ToolNode(AGENT_TOOLS))
workflow.add_node("self_correction", self_correction_node)
workflow.add_node("post_flight_check", post_flight_check_node)
workflow.add_node("format_output", format_output_node)

workflow.set_entry_point("assess_quality")
workflow.add_edge("assess_quality", "pre_flight_check")
workflow.add_conditional_edges("pre_flight_check", route_pre_flight)
workflow.add_conditional_edges("reasoning", route_reasoning)
workflow.add_edge("tools", "reasoning")
workflow.add_conditional_edges("self_correction", route_self_correction)
workflow.add_edge("post_flight_check", "format_output")
workflow.add_edge("format_output", END)

# Production Compilation with Audit Checkpointing
checkpointer = MemorySaver()  # استبدل بـ PostgresSaver أو RedisSaver في الإنتاج
safety_agent_app = workflow.compile(checkpointer=checkpointer)