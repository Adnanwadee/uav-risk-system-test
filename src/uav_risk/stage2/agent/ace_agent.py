# File Path: src/uav_risk/stage2/agent/ace_agent.py
import json
import time
import asyncio
from typing import Dict, Any, List, Optional, Literal, Set
from uav_risk.ml.schemas import MLResult
from uav_risk.stage2.rag.rag_core import AsyncRAGCore
from uav_risk.stage2.agent.agent_schemas import (
    AgentDecision, 
    ReasoningStep, 
    ToolCall, 
    FeatureAssessment, 
    LoopAction, 
    ConditionalGoConstraint
)
from uav_risk.stage2.agent.agent_memory import AgentMemory, DynamicCacheManager
from uav_risk.stage2.agent.agent_tools import (
    validate_feature_batch, 
    check_physics_constraint, 
    assess_contextual_remainder,
    query_rag, 
    generate_legal_query, 
    CROSS_FEATURE_SAFETY_MAP
)
from uav_risk.stage2.agent.fallback import StaticFallbackAssessor
import structlog
from uav_risk.utils.json_sanitize import strict_aviation_json_sanitizer

logger = structlog.get_logger(__name__)


class AsyncTokenBucketLimiter:
    """Meters requests using a token bucket algorithm to prevent API quota breaches."""
    def __init__(self, rpm: int = 60):
        self.rate = rpm / 60.0
        self.capacity = rpm
        self.tokens = float(rpm)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.last_update) * self.rate)
                self.last_update = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                await asyncio.sleep(0.1)


class AsyncCircuitBreaker:
    """Circuit breaker pattern isolating failing cloud endpoints and initializing fallbacks."""
    def __init__(self, failure_threshold: int = 3, recovery_time_sec: int = 10):
        self.threshold = failure_threshold
        self.recovery_time = recovery_time_sec
        self.state: Literal["CLOSED", "OPEN", "HALF-OPEN"] = "CLOSED"
        self.failures = 0
        self.last_failure_time = 0.0

    def record_success(self) -> None:
        self.failures = 0
        self.state = "CLOSED"

    def record_failure(self) -> bool:
        self.failures += 1
        self.last_failure_time = time.monotonic()
        if self.failures >= self.threshold:
            self.state = "OPEN"
            return True
        return False

    def allow_execution(self) -> bool:
        if self.state == "OPEN":
            if time.monotonic() - self.last_failure_time > self.recovery_time:
                self.state = "HALF-OPEN"
                return True
            return False
        return True


class ACEReActAgent:
    """Sovereign cognitive orchestration engine analyzing multi-variable safety and compliance maps."""
    def __init__(self, llm_client: Any, rag_core: AsyncRAGCore, feature_defs: Dict[str, Any], config_json: Optional[str] = None):
        self.llm = llm_client
        self.rag_core = rag_core
        self.feature_defs = feature_defs
        
        cfg = json.loads(config_json) if config_json else {}
        self.physics_graph = cfg.get("PHYSICS_DEPENDENCY_GRAPH", {
            "uav_mass_kg": ["check_physics_constraint:disk_loading", "check_physics_constraint:wind_susceptibility"],
            "uav_battery_wh": ["check_physics_constraint:energy_budget"],
            "flight_altitude_m": ["check_physics_constraint:altitude_ceiling"]
        })
        self.limiter = AsyncTokenBucketLimiter(rpm=cfg.get("LLM_RPM_LIMIT", 60))
        self.cb = AsyncCircuitBreaker()
        self.cache = DynamicCacheManager()
        
        self.MAX_ITERATIONS = 12
        self.MAX_JSON_RETRIES = 3
        self.MAX_BACKTRACKS = 3
        self.MAX_STUCK_BEFORE_FALLBACK = 5

    def _get_features_by_category(self, features: Dict[str, float], category: str) -> Dict[str, float]:
        prefixes = {
            "battery": ["uav_battery_", "battery_remaining_pct", "uav_power_"],
            "aerodynamic": ["uav_mass_kg", "uav_payload_mass_kg", "uav_wingspan_m", "uav_max_speed_mps"],
            "navigation": ["gps_satellites_count", "flight_altitude_m", "flight_distance_m", "home_distance_m"],
            "weather": ["environment_weather_wind_mps", "environment_temperature_c", "environment_humidity_pct"],
            "comms": ["comms_rssi_dbm_min", "environment_gnss_jam_dbm", "rc_signal_strength_pct"],
            "operator": ["operator_experience_flights", "pilot_license_status", "license_level"],
            "airspace": ["airspace_class_restricted", "airport_distance_km", "operator_in_restricted_zone"],
            "mission": ["flight_duration_min", "operation_type_bvlos", "is_night_flight"]
        }
        target_prefixes = prefixes.get(category, [])
        return {k: v for k, v in features.items() if any(k.startswith(p) for p in target_prefixes)}

    async def run(self, validated_features: Dict[str, float], ml_result: MLResult, free_text: Optional[str] = None) -> AgentDecision:
        start_time = time.perf_counter()
        memory = AgentMemory(list(validated_features.keys()), self.physics_graph)
        priority_features = [f.feature_name for f in ml_result.top_features[:10]]
        
        if free_text and free_text.strip():
            if self._is_safe_free_text(free_text):
                await self._process_free_text_bilingual(free_text, memory)
            else:
                logger.warning("free_text_rejected_by_guard", reason="unsafe content detected")
            
        for constraint in ["disk_loading", "wind_susceptibility", "energy_budget", "altitude_ceiling"]:
            cache_key = f"{constraint}_{hash(frozenset(validated_features.items()))}"
            cached_res = self.cache.get_physics(cache_key)
            
            if cached_res:
                res = cached_res
            else:
                res = check_physics_constraint(constraint, validated_features, self.feature_defs)
                self.cache.set_physics(cache_key, res)
                
            if "conditional_constraint" in res and res["conditional_constraint"]:
                memory.record_conditional_constraint(res["conditional_constraint"])
                
            if not res.get("passed", True):
                feat_assessment = FeatureAssessment(
                    feature_name=f"physics_{constraint}", 
                    value=float(res.get("metric_value", 0.0)), 
                    status=res.get("severity", "WARNING"), 
                    reasoning=res.get("reasoning", ""), 
                    rag_consulted=False
                )
                memory.record_feature_assessment(feat_assessment)
                if res.get("severity") == "CRITICAL":
                    memory.critical_findings.append(f"Physical Core Lock Breach [{constraint}]: {res.get('reasoning')}")

        iteration = 0
        stuck_counter = 0
        signature_history = []
        
        while iteration < self.MAX_ITERATIONS:
            iteration += 1
            if not self.cb.allow_execution():
                return StaticFallbackAssessor.assess_safely(validated_features, self.feature_defs, "Circuit Breaker Tripped - External Cloud AI Layer Offline.")
                
            rich_context = memory.build_rich_context_for_llm(validated_features, self.feature_defs, priority_features)
            loop_action = await self._think_and_decide_action_with_retry(rich_context, iteration)
            
            if not loop_action:
                return StaticFallbackAssessor.assess_safely(validated_features, self.feature_defs, f"Terminal structured JSON schema retry exhaustion at cycle {iteration}")
                
            # ✅ التعديل الذكي: التتبع المكامل للتوقيع (اسم الأداة + معاملاتها) لمنع تعطل فحص الفئات المتباينة
            current_signature = f"{loop_action.action}:{json.dumps(loop_action.tool_input, sort_keys=True)}"
            
            if signature_history.count(current_signature) >= self.MAX_BACKTRACKS:
                stuck_counter += 1
                if stuck_counter >= self.MAX_STUCK_BEFORE_FALLBACK:
                    return StaticFallbackAssessor.assess_safely(validated_features, self.feature_defs, f"Deadlock safety breach on identical repetitive execution loop: {current_signature}")
                loop_action = LoopAction(thought="Identical tool parameters loop signature detected. Shifting constraint space.", action="assess_contextual_remainder", tool_input={})
                current_signature = f"assess_contextual_remainder:{json.dumps({})}"

            signature_history.append(current_signature)
            obs_str, examined_list = await self._execute_tool_wrapper(loop_action, validated_features, memory)
            
            tool_call_record = ToolCall(loop_action.action, loop_action.tool_input, obs_str, 1.0, True)
            memory.record_tool_call(tool_call_record)
            memory.reasoning_steps.append(ReasoningStep(iteration, loop_action.thought, loop_action.action, tool_call_record, obs_str, examined_list))
            
            if loop_action.action == "FINAL_SYNTHESIS":
                break
                
        return self._synthesize_sovereign_decision(memory, validated_features, ml_result, iteration, start_time)

    async def _think_and_decide_action_with_retry(self, context: str, iteration: int, retry=0, err="") -> Optional[LoopAction]:
        if retry >= self.MAX_JSON_RETRIES:
            return None
        prompt = f"=== ITERATION {iteration} ===\n{context}\n"
        if err:
            prompt += f"\nCORRECTION REQUIRED: Previous response failed schema contract parsing. Error: {err}. Re-output valid JSON mapping object matching structural keys."
        
        try:
            await self.limiter.acquire()
            # مهلة زمنية مريحة لمنع الاختناق أثناء استدعاء السحابة
            # Use a conservative SLA timeout for LLM responses; if exceeded,
            # escalate to a safe fallback immediately (tests assert 5s behavior).
            raw_response = await asyncio.wait_for(self._call_llm_structured(prompt), timeout=5.0)
            
            # ✅ تطهير فوري وحتمي من كتل الـ Markdown لمنع كسر هيكلية الـ Parser
            clean_raw = raw_response.strip()
            if clean_raw.startswith("```json"):
                clean_raw = clean_raw[7:]
            if clean_raw.endswith("```"):
                clean_raw = clean_raw[:-3]
            clean_raw = clean_raw.strip()
            
            parsed = json.loads(clean_raw)
            self.cb.record_success()
            
            # ✅ معالجة الحالات المتباينة للأحرف (case-insensitive keys) لمنع الـ KeyErrors القاتلة
            thought = parsed.get("thought", parsed.get("Thought", parsed.get("THOUGHT", "Executing core reactive evaluation step.")))
            action = parsed.get("action", parsed.get("Action", parsed.get("ACTION", "assess_contextual_remainder")))
            tool_input = parsed.get("tool_input", parsed.get("Tool_Input", parsed.get("TOOL_INPUT", {})))
            if not isinstance(tool_input, dict):
                tool_input = {}
                
            return LoopAction(thought=str(thought), action=str(action), tool_input=tool_input)
            
        except Exception as e:
            logger.error("agent_llm_chain_call_failed_retrying", cycle=iteration, retry=retry, error=str(e))
            # If we timed out waiting for the LLM, treat this as an immediate
            # failure that should trigger fallback (tests expect this behavior).
            if isinstance(e, asyncio.TimeoutError):
                self.cb.record_failure()
                return None
            if self.cb.record_failure():
                return None
            # Exponential backoff for transient errors
            await asyncio.sleep(2.0 * (retry + 1))
            return await self._think_and_decide_action_with_retry(context, iteration, retry + 1, str(e))

    async def _execute_tool_wrapper(self, action: LoopAction, features: Dict[str, float], memory: AgentMemory) -> tuple[str, List[str]]:
        name = action.action
        t_input = action.tool_input
        
        if name == "validate_feature_batch":
            cat = t_input.get("category_name", "battery")
            batch = self._get_features_by_category(features, cat)
            res = validate_feature_batch(cat, batch, self.feature_defs, CROSS_FEATURE_SAFETY_MAP, set(memory.examined_features.keys()))
            for a in res:
                memory.record_feature_assessment(a)
                if a.status == "CRITICAL":
                    memory.critical_findings.append(f"[{cat}] {a.feature_name}: {a.reasoning}")
            return f"Processed {len(res)} metrics within categorical alignment batch '{cat}'.", [a.feature_name for a in res]
            
        elif name == "query_rag":
            q = t_input.get("query", "FAA operational rules drone")
            cached = self.cache.get_rag(q)
            if cached:
                memory.record_rag_query(q, cached["citations"])
                return str(cached["obs"]), []
            res = await query_rag(q, self.rag_core)
            memory.record_rag_query(q, res.get("citations", []))
            obs = f"Regulatory law lookup complete. Citations linked: {len(res.get('citations', []))}. Digest: {res['finding'][:120]}"
            self.cache.set_rag(q, {"citations": res.get("citations", []), "obs": obs})
            return obs, []
            
        elif name == "assess_contextual_remainder":
            res = assess_contextual_remainder(features, set(memory.examined_features.keys()), self.feature_defs, CROSS_FEATURE_SAFETY_MAP)
            for a in res:
                memory.record_feature_assessment(a)
                if a.status == "CRITICAL":
                    memory.critical_findings.append(f"[Contextual Sweep Core Anomaly] {a.feature_name}: {a.reasoning}")
            return f"Constitutional matrix sweep processed over remainder space. Anomaly pins: {len(res)}", [a.feature_name for a in res]
            
        elif name == "generate_legal_query":
            q = await generate_legal_query(t_input.get("feature_name", ""), float(t_input.get("value", 0.0)), t_input.get("violation_type", "WARNING"), self.llm)
            res = await query_rag(q, self.rag_core)
            memory.record_rag_query(q, res.get("citations", []))
            return f"Expansion structural query built: '{q}'. Citations pulled: {len(res.get('citations', []))}", []

        return "Milestone cleared.", []

    async def _process_free_text_bilingual(self, text: str, memory: AgentMemory) -> None:
        prompt = f"Analyze ground station logs for operational drone hazards. Map Arabic flight notes natively into exact risk vectors.\nLog Text: '''{text}'''"
        try:
            raw = await self._call_llm_structured(prompt)
            parsed = json.loads(raw)
            parsed = strict_aviation_json_sanitizer(parsed)
            if parsed.get("hazard_detected"):
                for find in parsed.get("critical_findings", []):
                    memory.critical_findings.append(f"[Free-Text Risk Flag] {str(find)[:400]}")
                for q in parsed.get("rag_queries", []):
                    # sanitize query string
                    q_s = str(q)[:1000]
                    res = await query_rag(q_s, self.rag_core)
                    memory.record_rag_query(q_s, res.get("citations", []))
        except Exception:
            pass

    def _is_safe_free_text(self, text: str) -> bool:
        t = text.lower()
        # reject URLs, code fences, long injections
        if "http://" in t or "https://" in t:
            return False
        if "```" in t or "<script" in t or "eval(" in t or "{{" in t:
            return False
        if len(t) > 5000:
            return False
        return True

    def _synthesize_sovereign_decision(self, memory: AgentMemory, features: Dict[str, float], ml_result: MLResult, iterations: int, start_time: float) -> AgentDecision:
        critical_findings = memory.critical_findings.copy()
        recommendations = []
        
        for name, a in memory.examined_features.items():
            if a.status == "CRITICAL":
                critical_findings.append(f"Parametric Safety Failure Fracture: {name}={a.value} ({a.status}). reasoning: {a.reasoning}")
                recommendations.append(f"Immediate terminal component maintenance required for parameter: {name}")

        ml_lock = ml_result.risk_score > 0.70
        airspace_lock = features.get("operator_in_restricted_zone", 0.0) == 1.0
        altitude_val = features.get("flight_altitude_m", 0.0)
        altitude_warning_lock = altitude_val > 121.9
        
        if ml_lock or airspace_lock or len(critical_findings) > 0:
            final_decision = "NO-GO"
            risk = max(ml_result.risk_score, 0.95)
            if ml_lock:
                critical_findings.append(f"Sovereign Safety Core Lock Triggered: ML inference risk boundaries broken ({ml_result.risk_score:.3f} > 0.70).")
            if airspace_lock:
                critical_findings.append("Sovereign Safety Core Lock Triggered: Flight presence inside forbidden restricted national airspace zone.")
        elif altitude_warning_lock:
            final_decision = "CONDITIONAL-GO"
            risk = min(ml_result.risk_score + 0.12, 0.68)
            recommendations.append("SOVEREIGN OVERRIDE: Flight path transcends standard operational altitude ceiling. CONDITIONAL-GO enforced pending active LAANC approval verification.")
        else:
            warnings = [v for k, v in memory.examined_features.items() if v.status == "WARNING"]
            if warnings or memory.conditional_constraints:
                final_decision = "CONDITIONAL-GO"
                risk = min(ml_result.risk_score + 0.12, 0.68)
                for w in warnings[:4]:
                    memory.record_conditional_constraint(
                        ConditionalGoConstraint(
                            constraint_id=f"RESOLVE_{w.feature_name.upper()}", 
                            description=f"Resolve alert anomaly profile. Context: {w.reasoning[:80]}", 
                            feature_name=w.feature_name, 
                            required_value_range="Restore safe limits", 
                            legal_reference="Aviation Standard Logbook Operational Code"
                        )
                    )
                recommendations.append("CONDITIONAL-GO: Dynamic flight path cleared conditional upon full telemetry compliance mapping of conditional constraints.")
            else:
                final_decision = "GO"
                risk = ml_result.risk_score
                recommendations.append("Aviation baseline tracking system running smoothly within nominal safety limits.")

        coverage = len(memory.examined_features) / max(len(features), 1.0)
        confidence = ml_result.confidence * (0.7 + 0.3 * coverage)
        
        return AgentDecision(
            decision=final_decision,
            overall_risk_score=round(risk, 4),
            confidence=round(confidence, 4),
            reasoning_chain=memory.reasoning_steps,
            feature_assessments=memory.examined_features,
            critical_findings=critical_findings,
            recommendations=recommendations,
            legal_citations=memory.legal_citations,
            rag_queries_made=memory.rag_queries_made,
            total_iterations=iterations,
            processing_time_ms=round((time.perf_counter() - start_time) * 1000.0, 2),
            fallback_degraded_mode=False,
            conditional_constraints=memory.conditional_constraints,
            agent_version="v4.5.0-production",
            prompt_hash="sha256_framework_lock_gate6"
        )

    async def _call_llm_structured(self, prompt: str) -> str:
        """فرض التثبيت الأنطولوجي الصارم والقالب الهيكلي لتوجيه حتمية الـ JSON حياً."""
        injection_prompt = (
            "CRITICAL SYSTEM MANDATE:\n"
            "You are a critical aviation ReAct agent loop node. You MUST return a valid JSON object matching the keys below. "
            "Do not output markdown code block fences (like ```json ... ```), do not add commentary outside the raw JSON text block.\n"
            "REQUIRED STRUCTURE:\n"
            '{\n  "thought": "Your analytical logical reasoner tracking description here",\n  "action": "validate_feature_batch",\n  "tool_input": {"category_name": "battery"}\n}\n\n'
            f"INPUT CONTEXT AND POOL POINTERS:\n{prompt}"
        )
        return await self.llm.generate(injection_prompt)


# =====================================================================
# Stage 2 Agent Submodule Architectural Dependency Report:
#
# Depends on:
#   - src/uav_risk/ml/schemas.py (MLResult Structural Link)
#   - src/uav_risk/stage2/rag/rag_core.py (AsyncRAGCore Engine Link)
#   - src/uav_risk/stage2/agent/agent_schemas.py (AgentDecision Contracts Lock)
#
# Consumed by:
#   - src/uav_risk/stage2/pipeline.py (Master Subsystem Coordination Core)
# =====================================================================