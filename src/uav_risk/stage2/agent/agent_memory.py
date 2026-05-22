# File Path: src/uav_risk/stage2/agent/agent_memory.py
import hashlib
import time
from typing import Any, Dict, List, Optional, Set
from src.uav_risk.stage2.agent.agent_schemas import (
    ConditionalGoConstraint,
    FeatureAssessment,
    ReasoningStep,
    ToolCall,
)

class AgentMemory:
    """إدارة الذاكرة العاملة السلبية المحدثة؛ تمنع التضخم التربيعي للرموز وتستخرج القيود حياً."""
    
    def __init__(self, all_feature_names: List[str], dynamic_dependency_graph: Dict[str, List[str]]):
        """Initializes the short-term sliding context and execution registries.
        
        Args:
            all_feature_names: Complete list of the 198 feature vector keys.
            dynamic_dependency_graph: Maps features to downstream physics or RAG targets.
        """
        self.examined_features: Dict[str, FeatureAssessment] = {}
        self.critical_findings: List[str] = []
        self.reasoning_steps: List[ReasoningStep] = []
        self.rag_queries_made: List[str] = []
        self.legal_citations: List[Any] = []
        self.tool_history: List[ToolCall] = []
        self.conditional_constraints: List[ConditionalGoConstraint] = []
        
        self._all_feature_names: Set[str] = set(all_feature_names)
        self._backtrack_registry: Dict[str, int] = {}
        self._executed_physics_checks: Set[str] = set()
        self._pending_rag_queries: List[str] = []
        self._dependency_graph: Dict[str, List[str]] = dynamic_dependency_graph

    def record_feature_assessment(self, assessment: FeatureAssessment) -> None:
        """Stores a telemetry assessment and automatically updates pending analytical targets."""
        self.examined_features[assessment.feature_name] = assessment
        if assessment.status in ("CRITICAL", "WARNING"):
            related_actions = self._dependency_graph.get(assessment.feature_name, [])
            for action in related_actions:
                if "query_rag:" in action:
                    topic = action.split("query_rag:")[1]
                    query = f"{assessment.feature_name}={assessment.value}: criteria for {topic}"
                    if query not in self._pending_rag_queries:
                        self._pending_rag_queries.append(query)

    def record_conditional_constraint(self, constraint: ConditionalGoConstraint) -> None:
        """حقن وحفظ قيود الموافقة المشروطة المستخرجة حياً من الفحوصات الفيزيائية."""
        if constraint not in self.conditional_constraints:
            self.conditional_constraints.append(constraint)

    def record_tool_call(self, call: ToolCall) -> None:
        """Tracks tool metrics and increments the backtrack registry upon repetition loop signatures."""
        if self._is_duplicate_call(call):
            self._backtrack_registry[call.tool_name] = self._backtrack_registry.get(call.tool_name, 0) + 1
        else:
            self._backtrack_registry[call.tool_name] = 0
            
        self.tool_history.append(call)
        if "check_physics_constraint" in call.tool_name:
            constraint = call.tool_input.get("constraint_name", "")
            if constraint:
                self._executed_physics_checks.add(str(constraint))

    def record_rag_query(self, query: str, citations: List[Any]) -> None:
        """Logs completed regulatory searches and clears matched predictive paths."""
        self.rag_queries_made.append(query)
        self.legal_citations.extend(citations)
        # Prune the pending queue matching the evaluated topic fragment
        query_topic = query.split(":")[-1].strip() if ":" in query else query
        self._pending_rag_queries = [
            q for q in self._pending_rag_queries if not q.endswith(query_topic)
        ]

    def get_backtrack_count(self, tool_name: str) -> int:
        """Returns consecutive calls made to the same tool with identical parameters."""
        return self._backtrack_registry.get(tool_name, 0)

    def get_unexamined_features(self) -> Set[str]:
        """Calculates the remainder of the 198 feature space that lacks constitutional evaluation."""
        return self._all_feature_names - set(self.examined_features.keys())

    def is_physics_check_done(self, constraint_name: str) -> bool:
        """Verifies if an essential physical node check has passed the gateway pipeline."""
        return constraint_name in self._executed_physics_checks

    def build_rich_context_for_llm(
        self, 
        validated_features: Dict[str, float], 
        feature_defs: Dict[str, Any], 
        priority_features: List[str]
    ) -> str:
        """Generates a compressed sliding token summary optimizing context footprint.
        
        Maintains awareness by printing strict historical telemetry constraints,
        active anomalies, and only the last 2 reasoning cycles.
        """
        lines = ["=== HIGH-PRIORITY FEATURES (SHAP-Ranked Real-Values) ==="]
        for fname in priority_features[:10]:
            val = validated_features.get(fname)
            if val is None:
                continue
            fdef = feature_defs.get(fname, {})
            examined = fname in self.examined_features
            status = self.examined_features[fname].status if examined else "NOT_CHECKED"
            lines.append(
                f"  - {fname}: VALUE={val:.4f} | "
                f"SAFE_RANGE=[{fdef.get('safe_min', 'N/A')}, {fdef.get('safe_max', 'N/A')}] | "
                f"STATUS={status}"
            )
            
        if self.critical_findings:
            lines.append("\n=== ACTIVE CRITICAL FINDINGS ===")
            for finding in self.critical_findings[-5:]:
                lines.append(f"  🚨 {finding}")
                
        if self.conditional_constraints:
            lines.append("\n=== ACTIVE CONDITIONAL GO CONSTRAINTS ===")
            for c in self.conditional_constraints:
                lines.append(f"  ⚠️ {c.constraint_id}: {c.description} (Target Range: {c.required_value_range})")
                
        if self._pending_rag_queries:
            lines.append("\n=== SUGGESTED NEXT RAG QUERIES ===")
            for q in self._pending_rag_queries[:3]:
                lines.append(f"  → {q}")
                
        if self.reasoning_steps:
            lines.append("\n=== SLIDING REASONING HISTORY (Last 2 Cycles) ===")
            for step in self.reasoning_steps[-2:]:
                lines.append(
                    f"  Step {step.step_number}: Thought: {step.thought[:80]}... | "
                    f"Action: {step.action} -> Obs: {step.observation[:120]}..."
                )
                
        lines.append(
            f"\n=== PROGRESS: Examined {len(self.examined_features)}/{len(self._all_feature_names)} | "
            f"RAG Calls: {len(self.rag_queries_made)}"
        )
        return "\n".join(lines)

    def _is_duplicate_call(self, current_call: ToolCall) -> bool:
        """Determines if the exact tool invocation payload matches its immediate history signature."""
        if not self.tool_history:
            return False
        prev = [c for c in self.tool_history if c.tool_name == current_call.tool_name]
        if not prev:
            return False
        
        h1 = hashlib.md5(str(sorted(prev[-1].tool_input.items())).encode()).hexdigest()
        h2 = hashlib.md5(str(sorted(current_call.tool_input.items())).encode()).hexdigest()
        return h1 == h2


class DynamicCacheManager:
    """مدير الكاش غير التزامني للعمليات الحتمية المدمجة واستعلامات RAG المتشابهة."""
    
    def __init__(self) -> None:
        self._physics_cache: Dict[str, Dict[str, Any]] = {}
        self._rag_cache: Dict[str, Dict[str, Any]] = {}

    def get_physics(self, key: str) -> Optional[Dict[str, Any]]:
        return self._physics_cache.get(key)

    def set_physics(self, key: str, value: Dict[str, Any]) -> None:
        self._physics_cache[key] = value

    def get_rag(self, query: str) -> Optional[Dict[str, Any]]:
        return self._rag_cache.get(query)

    def set_rag(self, query: str, value: Dict[str, Any]) -> None:
        self._rag_cache[query] = value

# =====================================================================
# Stage 2 Memory Submodule Architectural Dependency Report:
#
# Depends on:
#   - src/uav_risk/stage2/agent/agent_schemas.py (ConditionalGoConstraint, FeatureAssessment, ReasoningStep, ToolCall)
#
# Consumed by:
#   - src/uav_risk/stage2/agent/ace_agent.py
#   - tests/unit/test_ace_agent.py
# =====================================================================