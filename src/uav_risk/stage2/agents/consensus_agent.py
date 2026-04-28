"""
Consensus Agent — The Authoritative Decision Council (V13.0 - ML Demoted & Integrated)
======================================================================================
Role: Synthesize outputs from Physics, Temporal, Legal, and (now) ML agents 
      into a single, calibrated, legally defensible flight decision.

Architectural Upgrades:
 - ML Demotion: Stage 1 ML is no longer a Tier-0 gateway dictator. It is now a 10% weighted consultant.
 - 4-Agent Deliberation: Math engine upgraded to handle 4 voting entities seamlessly.
 - Dynamic Telemetry Parsing: Extracts ML scores directly from the live telemetry dict.

Author: Stage 2 — ACE System
"""

from __future__ import annotations

import math
import time
import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TYPE_CHECKING, Dict, Any

# ── Safe Type Hinting (No Circular Imports) ──
if TYPE_CHECKING:
    from .legal_agent import LegalRiskReport

from .physics_agent import PhysicsRiskReport
from .temporal_agent import TemporalStateEstimate

logger = logging.getLogger("ConsensusAgent")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration & Tunable Parameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConsensusConfig:
    # Agent Deliberation Weights (Total = 1.0)
    # [تحديث هندسي]: تقليص سلطة الـ ML لصالح الفيزياء والقانون
    weight_physics: float = 0.40
    weight_legal: float = 0.30
    weight_temporal: float = 0.20
    weight_ml: float = 0.10  # ML is now a consultant, not a dictator!
    
    # Temporal Heuristic Scores
    temporal_wind_trend_penalty: float = 0.35
    temporal_battery_drain_penalty: float = 0.35
    temporal_critical_battery_penalty: float = 0.20
    temporal_high_wind_penalty: float = 0.15
    
    # Thresholds
    entropy_hitl_threshold: float = 0.70
    ccs_caution_lower: float = 0.20
    ccs_caution_upper: float = 0.50
    ccs_hitl_lower: float = 0.35
    ccs_hitl_upper: float = 0.65

# ─────────────────────────────────────────────────────────────────────────────
# Shared Decision Vocabulary
# ─────────────────────────────────────────────────────────────────────────────

class FinalDecision(str, Enum):
    GO      = "GO"       
    CAUTION = "CAUTION"  
    NO_GO   = "NO-GO"    

# ─────────────────────────────────────────────────────────────────────────────
# Data Contracts
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentVote:
    """Normalized representation of a single agent's position."""
    agent_name: str            
    raw_decision: str          
    normalized_risk_score: float 
    weight: float              
    warnings: list[str]        
    evidence_quality: float    

@dataclass
class DeliberationMetrics:
    physics_nrs: float
    temporal_nrs: float
    legal_nrs: float
    ml_nrs: float # تم إضافة الـ ML

    weighted_risk_score: float       
    effective_weights: dict          

    raw_entropy: float               
    max_entropy: float               
    normalized_entropy: float        

    calibrated_confidence_score: float

    decision_method: str             
    hitl_triggered: bool             
    hitl_reason: Optional[str]       

@dataclass
class ConsensusReport:
    final_decision: FinalDecision
    decision_confidence: str          
    mission_authorized: bool          

    calibrated_confidence_score: float   
    risk_summary: str                    

    physics_decision: str
    physics_nrs: float
    physics_warnings: list[str]

    temporal_decision: str
    temporal_nrs: float
    temporal_warnings: list[str]

    legal_decision: str
    legal_nrs: float
    legal_violations: list[str]
    
    # بيانات الـ ML في التقرير النهائي
    ml_decision: str
    ml_nrs: float
    ml_warnings: list[str]

    metrics: DeliberationMetrics
    hitl_required: bool
    hitl_reason: Optional[str]

    all_warnings: list[str]
    required_mitigations: list[str]
    disqualifying_conditions: list[str]  

    deliberation_steps: list[str]   
    total_time_ms: float
    agent_times_ms: dict            

# ─────────────────────────────────────────────────────────────────────────────
# Vote Translation Layer
# ─────────────────────────────────────────────────────────────────────────────

_COMPLIANCE_TO_RISK_SCORE: dict[str, float] = {
    "COMPLIANT":     0.00,
    "RESTRICTED":    0.40,   
    "NON_COMPLIANT": 1.00,
    "UNCERTAIN":     0.70,   
}

_LEGAL_DECISION_MAP: dict[str, str] = {
    "GO":      "GO",
    "CAUTION": "CAUTION",
    "NO-GO":   "NO-GO",
    "NO_GO":   "NO-GO",
}

def _normalize_decision(d: str) -> str:
    return _LEGAL_DECISION_MAP.get(str(d).strip().upper().replace("_", "-"), "CAUTION")

def translate_physics_vote(report: PhysicsRiskReport, cfg: ConsensusConfig) -> AgentVote:
    all_warnings = list(report.warnings)
    if report.projected_risk_level in ("HIGH", "CRITICAL"):
        all_warnings.append(
            f"PROJECTED: Physics risk will reach {report.projected_risk_level} "
            f"(P={report.projected_failure_probability:.1%}) within forecast horizon."
        )

    return AgentVote(
        agent_name="physics",
        raw_decision=report.go_no_go,
        normalized_risk_score=report.mc_failure_probability,
        weight=cfg.weight_physics,
        warnings=all_warnings,
        evidence_quality=1.0,
    )

def translate_temporal_vote(estimate: TemporalStateEstimate, cfg: ConsensusConfig) -> AgentVote:
    nrs = 0.0

    if estimate.wind_increasing: nrs += cfg.temporal_wind_trend_penalty
    if estimate.battery_draining_fast: nrs += cfg.temporal_battery_drain_penalty
    if estimate.projected_battery_pct < 20.0: nrs += cfg.temporal_critical_battery_penalty
    if estimate.projected_wind_ms > 9.6: nrs += cfg.temporal_high_wind_penalty

    nrs = min(nrs, 1.0)
    decision = "GO" if nrs < 0.15 else "CAUTION" if nrs < 0.45 else "NO-GO"

    quality = 1.0 if (estimate.wind_trend_p_value < 0.05 or estimate.battery_trend_p_value < 0.05) else 0.65

    return AgentVote(
        agent_name="temporal",
        raw_decision=decision,
        normalized_risk_score=nrs,
        weight=cfg.weight_temporal,
        warnings=list(estimate.temporal_warnings),
        evidence_quality=quality,
    )

def translate_legal_vote(legal_report: 'LegalRiskReport', cfg: ConsensusConfig) -> AgentVote:
    status_str = str(getattr(legal_report.compliance_status, 'value', legal_report.compliance_status)).upper()
    nrs = _COMPLIANCE_TO_RISK_SCORE.get(status_str, 0.70)
    quality = {"COMPLIANT": 1.00, "NON_COMPLIANT": 1.00, "RESTRICTED": 0.85, "UNCERTAIN": 0.50}.get(status_str, 0.50)
    legal_go_no_go = str(getattr(legal_report.go_no_go, 'value', legal_report.go_no_go))
    
    warnings = [f"LEGAL: {v}" for v in legal_report.critical_violations] + \
               [f"MITIGATION REQUIRED: {m}" for m in legal_report.required_mitigations]

    return AgentVote(
        agent_name="legal",
        raw_decision=_normalize_decision(legal_go_no_go),
        normalized_risk_score=nrs,
        weight=cfg.weight_legal,
        warnings=warnings,
        evidence_quality=quality,
    )

def translate_ml_vote(ml_score: float, ml_class: str, cfg: ConsensusConfig) -> AgentVote:
    """[تحديث هندسي]: يترجم نتيجة الإحصاء الأعمى للـ ML إلى صوت يشارك في المجلس بنسبة 10%."""
    nrs = max(0.0, min(1.0, float(ml_score)))
    decision = "NO-GO" if nrs >= 0.85 else "CAUTION" if nrs >= 0.5 else "GO"
    
    # الجودة الإثباتية منخفضة قليلاً لأن الـ ML إحصائي وليس سببي (Statistical not Causal)
    return AgentVote(
        agent_name="ml",
        raw_decision=decision,
        normalized_risk_score=nrs,
        weight=cfg.weight_ml,
        warnings=[f"ML Model flagged statistical risk score: {nrs:.2f}"] if nrs > 0.4 else [],
        evidence_quality=0.85, 
    )

# ─────────────────────────────────────────────────────────────────────────────
# Hard-Stop Conditions
# ─────────────────────────────────────────────────────────────────────────────

def _check_hard_stops(physics: PhysicsRiskReport, temporal: TemporalStateEstimate, legal: 'LegalRiskReport') -> list[str]:
    stops = []
    if physics.wind_tolerance_ratio > 1.0: stops.append(f"H1-WIND: Wind at {physics.wind_tolerance_ratio:.1%} of rated tolerance.")
    if physics.battery_margin_pct < 0: stops.append(f"H2-BATTERY: Mission deficit of {abs(physics.battery_margin_pct):.1f} min.")
    if physics.structural_load_ratio > 1.0: stops.append(f"H3-STRUCTURE: Frame limit exceeded ({physics.structural_load_ratio:.1%}).")
        
    legal_status = str(getattr(legal.compliance_status, 'value', legal.compliance_status)).upper()
    if legal_status == "NON_COMPLIANT": stops.append("H4-LEGAL: Flight is strictly NON_COMPLIANT.")
    if physics.mc_failure_probability > 0.50: stops.append(f"H5-PHYSICS: Critical Monte Carlo Risk ({physics.mc_failure_probability:.1%}).")
    return stops

# ─────────────────────────────────────────────────────────────────────────────
# Deliberation Engine
# ─────────────────────────────────────────────────────────────────────────────

def _compute_deliberation(votes: list[AgentVote], steps: list[str], cfg: ConsensusConfig) -> DeliberationMetrics:
    adjusted = {v.agent_name: v.weight * v.evidence_quality for v in votes}
    total_adj = sum(adjusted.values())
    
    if total_adj <= 0.001:
        steps.append("[WARNING] Total adjusted weight near zero. Defaulting to equal distribution.")
        eff_weights = {name: 1.0 / len(votes) for name in adjusted.keys()}
    else:
        eff_weights = {name: w / total_adj for name, w in adjusted.items()}

    steps.append("[STEP 1] Adjusted Weights: " + ", ".join(f"{v.agent_name}→{eff_weights[v.agent_name]:.3f}" for v in votes))
    vote_by_name = {v.agent_name: v for v in votes}

    weighted_nrs = sum(eff_weights[v.agent_name] * v.normalized_risk_score for v in votes)
    steps.append(f"[STEP 2] Weighted Risk Score (NRS) across all 4 agents = {weighted_nrs:.4f}")

    decisions = [_normalize_decision(v.raw_decision) for v in votes]
    counts = Counter(decisions)
    probs = [c / len(decisions) for c in counts.values()]
    H_raw = -sum(p * math.log(p) for p in probs if p > 0)
    H_max = math.log(3)                
    H_norm = H_raw / H_max if H_max > 0 else 0.0

    steps.append(f"[STEP 3] Semantic Entropy: H_norm={H_norm:.4f}")

    CCS = (1.0 - H_norm) * weighted_nrs + H_norm * 0.5
    steps.append(f"[STEP 4] CCS = {CCS:.4f}")

    hitl = False
    hitl_reason = None
    if H_norm > cfg.entropy_hitl_threshold:
        hitl = True
        hitl_reason = f"High Entropy (H={H_norm:.2f}). Agents have irreconcilable disagreement."
    elif cfg.ccs_hitl_lower <= CCS <= cfg.ccs_hitl_upper:
        hitl = True
        hitl_reason = f"CCS ({CCS:.2f}) falls in uncertainty zone. Cannot confirm safety."

    return DeliberationMetrics(
        physics_nrs=vote_by_name["physics"].normalized_risk_score,
        temporal_nrs=vote_by_name["temporal"].normalized_risk_score,
        legal_nrs=vote_by_name["legal"].normalized_risk_score,
        ml_nrs=vote_by_name["ml"].normalized_risk_score,
        weighted_risk_score=weighted_nrs,
        effective_weights=eff_weights,
        raw_entropy=H_raw,
        max_entropy=H_max,
        normalized_entropy=H_norm,
        calibrated_confidence_score=CCS,
        decision_method="",  
        hitl_triggered=hitl,
        hitl_reason=hitl_reason,
    )

def _synthesize_decision(metrics: DeliberationMetrics, hard_stops: list[str], votes: list[AgentVote], steps: list[str], cfg: ConsensusConfig) -> tuple[FinalDecision, str]:
    if hard_stops:
        steps.append("[DECISION-P0] HARD STOP TRIGGERED → NO-GO")
        return FinalDecision.NO_GO, "hard_stop"

    decisions = [_normalize_decision(v.raw_decision) for v in votes]
    if len(set(decisions)) == 1:
        mapping = {"GO": FinalDecision.GO, "CAUTION": FinalDecision.CAUTION, "NO-GO": FinalDecision.NO_GO}
        steps.append(f"[DECISION-P1] Unanimous → {mapping[decisions[0]].value}")
        return mapping[decisions[0]], "unanimous"

    if metrics.normalized_entropy > cfg.entropy_hitl_threshold:
        steps.append("[DECISION-P2] High entropy → CAUTION")
        return FinalDecision.CAUTION, "entropy_override"

    if metrics.calibrated_confidence_score < cfg.ccs_caution_lower:
        base, method = FinalDecision.GO, "ccs_go"
    elif metrics.calibrated_confidence_score < cfg.ccs_caution_upper:
        base, method = FinalDecision.CAUTION, "ccs_caution"
    else:
        base, method = FinalDecision.NO_GO, "ccs_no_go"

    phys_norm = _normalize_decision(next(v for v in votes if v.agent_name == "physics").raw_decision)
    if phys_norm == "NO-GO" and base in (FinalDecision.GO, FinalDecision.CAUTION):
        steps.append("[DECISION-P4] Physics Veto: Escalated to NO-GO (Gravity overrides all).")
        return FinalDecision.NO_GO, f"{method}+physics_veto(no-go)"
    
    if phys_norm == "CAUTION" and base == FinalDecision.GO:
        steps.append("[DECISION-P4] Physics Veto: Downgraded GO to CAUTION.")
        return FinalDecision.CAUTION, f"{method}+physics_veto(caution)"

    return base, method

class ConsensusAgent:
    def __init__(self, config: Optional[ConsensusConfig] = None):
        self.config = config or ConsensusConfig()

    def deliberate(
        self,
        physics: PhysicsRiskReport,
        temporal: TemporalStateEstimate,
        legal: 'LegalRiskReport',
        telemetry: Optional[Dict[str, Any]] = None # تم إضافة الـ Telemetry لدعم الـ ML
    ) -> ConsensusReport:
        
        t_start = time.perf_counter()
        steps: list[str] = []

        # استخراج بيانات الـ ML من القاموس الحي
        ml_score = 0.0
        ml_class = "UNKNOWN"
        if telemetry:
            ml_score = float(telemetry.get("stage1_ml_risk_score", 0.0))
            ml_class = telemetry.get("stage1_ml_predicted_class", "UNKNOWN")

        phys_vote = translate_physics_vote(physics, self.config)
        temp_vote = translate_temporal_vote(temporal, self.config)
        legal_vote = translate_legal_vote(legal, self.config)
        ml_vote = translate_ml_vote(ml_score, ml_class, self.config)
        
        # [التحديث المعماري]: الطاولة الآن تتسع لـ 4 وكلاء
        all_votes = [phys_vote, temp_vote, legal_vote, ml_vote]

        hard_stops = _check_hard_stops(physics, temporal, legal)
        metrics = _compute_deliberation(all_votes, steps, self.config)
        
        final_decision, decision_method = _synthesize_decision(metrics, hard_stops, all_votes, steps, self.config)
        metrics.decision_method = decision_method

        CCS, H_norm = metrics.calibrated_confidence_score, metrics.normalized_entropy
        conf_label = "LOW" if H_norm > 0.5 or 0.35 <= CCS <= 0.65 else ("HIGH" if CCS < 0.20 or CCS > 0.75 else "MODERATE")
        
        summary = ""
        if hard_stops: summary = f"MISSION DENIED: {len(hard_stops)} hard limit(s) exceeded."
        elif final_decision == FinalDecision.GO: summary = f"All systems nominal. Mission authorized (Safety Certainty: {(1-CCS)*100:.0f}%)."
        elif final_decision == FinalDecision.CAUTION: summary = f"Marginal conditions (CCS={CCS:.2f}) or Entropy ({H_norm:.2f}). Human review required."
        else: summary = f"MISSION DENIED: Operating envelope exceeded (CCS={CCS:.2f})."

        all_warnings = [f"[PHYSICS] {w}" for w in phys_vote.warnings] + \
                       [f"[TEMPORAL] {w}" for w in temp_vote.warnings] + \
                       [f"[LEGAL] {w}" for w in legal_vote.warnings] + \
                       [f"[ML_ENGINE] {w}" for w in ml_vote.warnings]

        return ConsensusReport(
            final_decision=final_decision,
            decision_confidence=conf_label,
            mission_authorized=(final_decision == FinalDecision.GO),
            calibrated_confidence_score=CCS,
            risk_summary=summary,
            
            physics_decision=physics.go_no_go, physics_nrs=phys_vote.normalized_risk_score, physics_warnings=list(physics.warnings),
            temporal_decision=temp_vote.raw_decision, temporal_nrs=temp_vote.normalized_risk_score, temporal_warnings=list(temporal.temporal_warnings),
            legal_decision=str(getattr(legal.go_no_go, 'value', legal.go_no_go)), legal_nrs=legal_vote.normalized_risk_score, legal_violations=list(legal.critical_violations),
            
            ml_decision=ml_vote.raw_decision, ml_nrs=ml_vote.normalized_risk_score, ml_warnings=list(ml_vote.warnings),
            
            metrics=metrics, hitl_required=metrics.hitl_triggered, hitl_reason=metrics.hitl_reason,
            all_warnings=all_warnings, required_mitigations=list(legal.required_mitigations) if legal.required_mitigations else [],
            disqualifying_conditions=hard_stops, deliberation_steps=steps, total_time_ms=(time.perf_counter() - t_start) * 1000,
            agent_times_ms={"physics": getattr(physics, 'calculation_time_ms', 0.0), "temporal": getattr(temporal, 'estimation_time_ms', 0.0), "legal": getattr(legal, 'execution_time_ms', 0.0)}
        )