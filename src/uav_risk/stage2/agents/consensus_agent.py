"""
Consensus Agent — The Authoritative Decision Council (V6 - Production Grade)
=============================================================================
Role: Synthesize outputs from three specialist agents into a single,
      calibrated, legally defensible flight authorization decision.

Architectural Upgrades in V6:
 - Shifted from false P(failure) to Normalized Risk Score (NRS) to solve semantic conflation.
 - Implemented Double-Edged Physics Veto (can downgrade GO to CAUTION).
 - Added Zero-Division Guards for weight normalization.
 - Centralized Heuristics via ConsensusConfig.
 - Resolved Circular Imports via TYPE_CHECKING.

Author: Stage 2 — ACE System
Standard: ISO/IEC 25010 (Reliability), DO-178C (Audit Trail)
"""

from __future__ import annotations

import math
import time
import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TYPE_CHECKING

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
    # Agent Deliberation Weights
    weight_physics: float = 0.50
    weight_legal: float = 0.30
    weight_temporal: float = 0.20
    
    # Temporal Heuristic Scores (Converted to Risk Indices)
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
    normalized_risk_score: float # Replaced false P(failure) with mathematically sound NRS [0,1]
    weight: float              
    warnings: list[str]        
    evidence_quality: float    

@dataclass
class DeliberationMetrics:
    physics_nrs: float
    temporal_nrs: float
    legal_nrs: float

    weighted_risk_score: float       
    effective_weights: dict          

    raw_entropy: float               
    max_entropy: float               
    normalized_entropy: float        

    calibrated_confidence_score: float  # CCS ∈ [0, 1]

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
        normalized_risk_score=report.mc_failure_probability, # Physics outputs true probability, mapped to NRS
        weight=cfg.weight_physics,
        warnings=all_warnings,
        evidence_quality=1.0,
    )

def translate_temporal_vote(estimate: TemporalStateEstimate, cfg: ConsensusConfig) -> AgentVote:
    nrs = 0.0

    if estimate.wind_increasing:
        nrs += cfg.temporal_wind_trend_penalty
    if estimate.battery_draining_fast:
        nrs += cfg.temporal_battery_drain_penalty
    if estimate.projected_battery_pct < 20.0:
        nrs += cfg.temporal_critical_battery_penalty
    if estimate.projected_wind_ms > 9.6:   
        nrs += cfg.temporal_high_wind_penalty

    nrs = min(nrs, 1.0)

    if nrs < 0.15: decision = "GO"
    elif nrs < 0.45: decision = "CAUTION"
    else: decision = "NO-GO"

    any_significant = (estimate.wind_trend_p_value < 0.05 or estimate.battery_trend_p_value < 0.05)
    quality = 1.0 if any_significant else 0.65

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

    quality_map = {
        "COMPLIANT":     1.00,
        "NON_COMPLIANT": 1.00,
        "RESTRICTED":    0.85,
        "UNCERTAIN":     0.50,
    }
    quality = quality_map.get(status_str, 0.50)

    legal_go_no_go = str(getattr(legal_report.go_no_go, 'value', legal_report.go_no_go))
    legal_warnings = [f"LEGAL: {v}" for v in legal_report.critical_violations]
    legal_warnings += [f"MITIGATION REQUIRED: {m}" for m in legal_report.required_mitigations]

    return AgentVote(
        agent_name="legal",
        raw_decision=_normalize_decision(legal_go_no_go),
        normalized_risk_score=nrs,
        weight=cfg.weight_legal,
        warnings=legal_warnings,
        evidence_quality=quality,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Hard-Stop Conditions
# ─────────────────────────────────────────────────────────────────────────────

def _check_hard_stops(
    physics: PhysicsRiskReport,
    temporal: TemporalStateEstimate,
    legal: 'LegalRiskReport',
) -> list[str]:
    stops = []
    
    if physics.wind_tolerance_ratio > 1.0:
        stops.append(f"H1-WIND: Wind at {physics.wind_tolerance_ratio:.1%} of rated tolerance.")
    if physics.battery_margin_pct < 0:
        stops.append(f"H2-BATTERY: Mission deficit of {abs(physics.battery_margin_pct):.1f} min.")
    if physics.structural_load_ratio > 1.0:
        stops.append(f"H3-STRUCTURE: Frame limit exceeded ({physics.structural_load_ratio:.1%}).")
        
    legal_status = str(getattr(legal.compliance_status, 'value', legal.compliance_status)).upper()
    if legal_status == "NON_COMPLIANT":
        stops.append("H4-LEGAL: Flight is strictly NON_COMPLIANT.")
        
    # Clarified distinction: structural/thrust failure in MC vs simple battery abort
    if physics.mc_failure_probability > 0.50:
        stops.append(f"H5-PHYSICS: Critical Monte Carlo Risk ({physics.mc_failure_probability:.1%}).")

    return stops

# ─────────────────────────────────────────────────────────────────────────────
# Deliberation Engine — Core Mathematics
# ─────────────────────────────────────────────────────────────────────────────

def _compute_deliberation(
    votes: list[AgentVote],
    steps: list[str],
    cfg: ConsensusConfig
) -> DeliberationMetrics:
    
    # ── Step 1: Evidence-quality-adjusted weights (With Zero-Division Guard) ──
    adjusted    = {v.agent_name: v.weight * v.evidence_quality for v in votes}
    total_adj   = sum(adjusted.values())
    
    if total_adj <= 0.001:  # Guard against mathematical collapse
        steps.append("[WARNING] Total adjusted weight near zero. Defaulting to equal distribution.")
        eff_weights = {name: 1.0 / len(votes) for name in adjusted.keys()}
    else:
        eff_weights = {name: w / total_adj for name, w in adjusted.items()}

    steps.append(
        f"[STEP 1] Adjusted Weights: " + 
        ", ".join(f"{v.agent_name}→{eff_weights[v.agent_name]:.3f}" for v in votes)
    )

    vote_by_name = {v.agent_name: v for v in votes}
    nrs_physics  = vote_by_name["physics"].normalized_risk_score
    nrs_temporal = vote_by_name["temporal"].normalized_risk_score
    nrs_legal    = vote_by_name["legal"].normalized_risk_score

    # ── Step 2: Weighted Normalized Risk Score (Replaces invalid P_w) ──
    weighted_nrs = sum(eff_weights[v.agent_name] * v.normalized_risk_score for v in votes)

    steps.append(
        f"[STEP 2] Weighted Risk Score (NRS): "
        f"{eff_weights['physics']:.2f}×{nrs_physics:.2f} + "
        f"{eff_weights['temporal']:.2f}×{nrs_temporal:.2f} + "
        f"{eff_weights['legal']:.2f}×{nrs_legal:.2f} = {weighted_nrs:.4f}"
    )

    # ── Step 3: Semantic Entropy ──
    decisions = [_normalize_decision(v.raw_decision) for v in votes]
    counts = Counter(decisions)
    n = len(decisions)
    probs = [c / n for c in counts.values()]
    H_raw = -sum(p * math.log(p) for p in probs if p > 0)
    H_max = math.log(3)                
    H_norm = H_raw / H_max

    steps.append(f"[STEP 3] Semantic Entropy: H_norm={H_norm:.4f}")

    # ── Step 4: Calibrated Confidence Score ──
    CCS = (1.0 - H_norm) * weighted_nrs + H_norm * 0.5
    steps.append(f"[STEP 4] CCS = {CCS:.4f}")

    # ── HITL trigger logic ──
    hitl = False
    hitl_reason = None

    if H_norm > cfg.entropy_hitl_threshold:
        hitl = True
        hitl_reason = f"High Entropy (H={H_norm:.2f}). Agents have irreconcilable disagreement."
    elif cfg.ccs_hitl_lower <= CCS <= cfg.ccs_hitl_upper:
        hitl = True
        hitl_reason = f"CCS ({CCS:.2f}) falls in uncertainty zone. Cannot confirm safety."

    return DeliberationMetrics(
        physics_nrs=nrs_physics,
        temporal_nrs=nrs_temporal,
        legal_nrs=nrs_legal,
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

# ─────────────────────────────────────────────────────────────────────────────
# Final Decision Synthesis
# ─────────────────────────────────────────────────────────────────────────────

def _synthesize_decision(
    metrics: DeliberationMetrics,
    hard_stops: list[str],
    votes: list[AgentVote],
    steps: list[str],
    cfg: ConsensusConfig
) -> tuple[FinalDecision, str]:
    
    CCS  = metrics.calibrated_confidence_score
    H    = metrics.normalized_entropy
    decisions = [_normalize_decision(v.raw_decision) for v in votes]

    # P0: Hard stops
    if hard_stops:
        steps.append("[DECISION-P0] HARD STOP TRIGGERED → NO-GO")
        return FinalDecision.NO_GO, "hard_stop"

    # P1: Unanimous
    if len(set(decisions)) == 1:
        mapping = {"GO": FinalDecision.GO, "CAUTION": FinalDecision.CAUTION, "NO-GO": FinalDecision.NO_GO}
        steps.append(f"[DECISION-P1] Unanimous → {mapping[decisions[0]].value}")
        return mapping[decisions[0]], "unanimous"

    # P2: Entropy override
    if H > cfg.entropy_hitl_threshold:
        steps.append("[DECISION-P2] High entropy → CAUTION")
        return FinalDecision.CAUTION, "entropy_override"

    # P3: CCS Base Classification
    if CCS < cfg.ccs_caution_lower:
        base = FinalDecision.GO
        method = "ccs_go"
    elif CCS < cfg.ccs_caution_upper:
        base = FinalDecision.CAUTION
        method = "ccs_caution"
    else:
        base = FinalDecision.NO_GO
        method = "ccs_no_go"

    # P4: Double-Edged Physics Veto (Crucial Safety Fix)
    physics_vote = next(v for v in votes if v.agent_name == "physics")
    phys_norm = _normalize_decision(physics_vote.raw_decision)

    if phys_norm == "NO-GO" and base in (FinalDecision.GO, FinalDecision.CAUTION):
        steps.append("[DECISION-P4] Physics Veto: Escalated to NO-GO (Gravity overrides all).")
        return FinalDecision.NO_GO, f"{method}+physics_veto(no-go)"
    
    if phys_norm == "CAUTION" and base == FinalDecision.GO:
        steps.append("[DECISION-P4] Physics Veto: Downgraded GO to CAUTION due to physical margins.")
        return FinalDecision.CAUTION, f"{method}+physics_veto(caution)"

    return base, method

# ─────────────────────────────────────────────────────────────────────────────
# Presentation & Wrappers
# ─────────────────────────────────────────────────────────────────────────────

def _confidence_label(CCS: float, H_norm: float) -> str:
    if H_norm > 0.5 or 0.35 <= CCS <= 0.65: return "LOW"
    if CCS < 0.20 or CCS > 0.75: return "HIGH"
    return "MODERATE"

def _risk_summary(decision: FinalDecision, CCS: float, H_norm: float, hard_stops: list[str]) -> str:
    if hard_stops:
        return f"MISSION DENIED: {len(hard_stops)} hard limit(s) exceeded."
    if decision == FinalDecision.GO:
        return f"All systems nominal. Mission authorized (Safety Certainty: {(1-CCS)*100:.0f}%)."
    if decision == FinalDecision.CAUTION:
        if H_norm > 0.5:
            return f"Agents disagree (Entropy={H_norm:.2f}). Human oversight mandated."
        return f"Marginal conditions detected (CCS={CCS:.2f}). Human review required."
    return f"MISSION DENIED: Operating envelope exceeded (CCS={CCS:.2f})."

class ConsensusAgent:
    def __init__(self, config: Optional[ConsensusConfig] = None):
        self.config = config or ConsensusConfig()

    def deliberate(
        self,
        physics: PhysicsRiskReport,
        temporal: TemporalStateEstimate,
        legal: 'LegalRiskReport',
    ) -> ConsensusReport:
        
        t_start = time.perf_counter()
        steps: list[str] = []

        phys_vote = translate_physics_vote(physics, self.config)
        temp_vote = translate_temporal_vote(temporal, self.config)
        legal_vote = translate_legal_vote(legal, self.config)
        all_votes = [phys_vote, temp_vote, legal_vote]

        hard_stops = _check_hard_stops(physics, temporal, legal)
        metrics = _compute_deliberation(all_votes, steps, self.config)
        
        final_decision, decision_method = _synthesize_decision(
            metrics, hard_stops, all_votes, steps, self.config
        )
        metrics.decision_method = decision_method

        confidence_label = _confidence_label(metrics.calibrated_confidence_score, metrics.normalized_entropy)
        summary = _risk_summary(final_decision, metrics.calibrated_confidence_score, metrics.normalized_entropy, hard_stops)

        all_warnings = [f"[PHYSICS] {w}" for w in phys_vote.warnings] + \
                       [f"[TEMPORAL] {w}" for w in temp_vote.warnings] + \
                       [f"[LEGAL] {w}" for w in legal_vote.warnings]

        total_ms = (time.perf_counter() - t_start) * 1000

        return ConsensusReport(
            final_decision=final_decision,
            decision_confidence=confidence_label,
            mission_authorized=(final_decision == FinalDecision.GO),
            calibrated_confidence_score=metrics.calibrated_confidence_score,
            risk_summary=summary,
            physics_decision=physics.go_no_go,
            physics_nrs=phys_vote.normalized_risk_score,
            physics_warnings=list(physics.warnings),
            temporal_decision=temp_vote.raw_decision,
            temporal_nrs=temp_vote.normalized_risk_score,
            temporal_warnings=list(temporal.temporal_warnings),
            legal_decision=str(getattr(legal.go_no_go, 'value', legal.go_no_go)),
            legal_nrs=legal_vote.normalized_risk_score,
            legal_violations=list(legal.critical_violations),
            metrics=metrics,
            hitl_required=metrics.hitl_triggered,
            hitl_reason=metrics.hitl_reason,
            all_warnings=all_warnings,
            required_mitigations=list(legal.required_mitigations) if legal.required_mitigations else [],
            disqualifying_conditions=hard_stops,
            deliberation_steps=steps,
            total_time_ms=total_ms,
            agent_times_ms={
                "physics": getattr(physics, 'calculation_time_ms', 0.0),
                "temporal": getattr(temporal, 'estimation_time_ms', 0.0),
                "legal": getattr(legal, 'execution_time_ms', 0.0),
            },
        )