from __future__ import annotations

from dataclasses import dataclass

from uav_risk.stage2.contracts import (
    AgentEvidenceReference,
    AgentFinding,
    AgentFindingSeverity,
    AgentFindingType,
    AgentRecommendation,
    DecisionConfidenceLevel,
    DecisionEngineResult,
    DecisionPolicyConfig,
    DecisionStageContribution,
    DecisionStageName,
    EvidenceBundle,
    EvidenceSupportStatus,
    FinalDecision,
    Stage2AssessmentInput,
    Stage2AssessmentResult,
    Stage2Status,
)

_DEFAULT_POLICY = DecisionPolicyConfig()

DEFAULT_STAGE_WEIGHTS: dict[str, float] = {
    DecisionStageName.CORE.value: 0.0,
    DecisionStageName.ML.value: _DEFAULT_POLICY.ml_weight,
    DecisionStageName.SHAP.value: _DEFAULT_POLICY.shap_weight,
    DecisionStageName.RAG.value: _DEFAULT_POLICY.rag_weight,
    DecisionStageName.AGENT.value: _DEFAULT_POLICY.agent_weight,
    DecisionStageName.SCENARIO_PROFILE.value: _DEFAULT_POLICY.scenario_profile_weight,
    DecisionStageName.LLM.value: _DEFAULT_POLICY.llm_weight,
}

_GO_THRESHOLD = _DEFAULT_POLICY.go_threshold
_NO_GO_THRESHOLD = _DEFAULT_POLICY.no_go_threshold

_CONFIDENCE_MARGIN = 0.12

_SEVERITY_SCORE = {
    AgentFindingSeverity.INFO: 0.05,
    AgentFindingSeverity.LOW: 0.15,
    AgentFindingSeverity.MEDIUM: 0.40,
    AgentFindingSeverity.HIGH: 0.70,
    AgentFindingSeverity.CRITICAL: 0.95,
}

_HIGH_CONCERN_SHAP_TOKENS = (
    "airspace",
    "no_fly",
    "restricted",
    "comms",
    "fault",
    "failure",
    "swarm",
)

_SHAP_TOPIC_TOKENS = {
    "weather": ("weather", "wind", "gust", "turbulence", "thermal"),
    "airspace": ("airspace", "no_fly", "restricted", "altitude", "agl", "ceiling"),
    "c2": ("comms", "uplink", "downlink", "c2", "link", "telemetry"),
    "payload": ("payload", "mass", "weight", "loading"),
    "swarm": ("swarm", "multi_uas", "formation"),
    "ground_risk": ("ground_risk", "population", "adjacent_area", "operational_volume", "traffic", "obstacle", "landing"),
    "energy": ("battery", "reserve", "endurance", "fuel", "energy"),
    "faults": ("fault", "failure", "degraded", "emergency"),
    "vlos": ("vlos", "visual_line_of_sight", "line_of_sight", "los"),
}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "blocked", "veto", "no_go"}
    return False


def _metadata_string(stage2_input: Stage2AssessmentInput, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = stage2_input.metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _contribution(
    *,
    stage: DecisionStageName,
    weight: float,
    contribution: float,
    signal: str,
    summary: str,
    reasons: list[str] | None = None,
    limitations: list[str] | None = None,
    metadata: dict[str, str | int | float | bool | None] | None = None,
) -> DecisionStageContribution:
    return DecisionStageContribution(
        stage=stage,
        weight=weight,
        contribution=_clamp(contribution),
        signal=signal,
        summary=summary,
        reasons=reasons or [],
        limitations=limitations or [],
        metadata=metadata or {},
    )


@dataclass(frozen=True)
class _StageEvaluation:
    contribution: DecisionStageContribution
    reasons: list[str]
    limitations: list[str]
    blocking_reasons: list[str]
    required_actions: list[str]
    evidence_refs: list[AgentEvidenceReference]


class WeightedDecisionEngine:
    def __init__(
        self,
        *,
        policy: DecisionPolicyConfig | None = None,
        stage_weights: dict[str, float] | None = None,
    ) -> None:
        self.policy = policy or DecisionPolicyConfig()
        weights = {
            DecisionStageName.CORE.value: 0.0,
            DecisionStageName.ML.value: self.policy.ml_weight,
            DecisionStageName.SHAP.value: self.policy.shap_weight,
            DecisionStageName.RAG.value: self.policy.rag_weight,
            DecisionStageName.AGENT.value: self.policy.agent_weight,
            DecisionStageName.SCENARIO_PROFILE.value: self.policy.scenario_profile_weight,
            DecisionStageName.LLM.value: self.policy.llm_weight,
        }
        if stage_weights:
            weights.update(stage_weights)
        self.stage_weights = weights

    def evaluate(
        self,
        stage2_input: Stage2AssessmentInput,
        stage2_result: Stage2AssessmentResult,
    ) -> DecisionEngineResult:
        evaluations = [
            self._evaluate_core(stage2_input),
            self._evaluate_ml(stage2_input),
            self._evaluate_shap(stage2_input),
            self._evaluate_rag(stage2_result.evidence_bundles),
            self._evaluate_agent(stage2_result),
            self._evaluate_scenario_profile(stage2_result),
            self._evaluate_llm_placeholder(),
        ]

        contributions = [item.contribution for item in evaluations]
        weighted_score = sum(
            item.contribution.contribution * item.contribution.weight
            for item in evaluations
            if item.contribution.stage != DecisionStageName.CORE
        )
        score = _clamp(weighted_score)

        reasons: list[str] = []
        limitations: list[str] = []
        blocking_reasons: list[str] = []
        required_actions: list[str] = []
        evidence_refs: list[AgentEvidenceReference] = []
        for item in evaluations:
            reasons.extend(item.reasons)
            limitations.extend(item.limitations)
            blocking_reasons.extend(item.blocking_reasons)
            required_actions.extend(item.required_actions)
            evidence_refs.extend(item.evidence_refs)

        has_hard_block = bool(blocking_reasons)
        has_insufficient = any(
            bundle.support_status == EvidenceSupportStatus.INSUFFICIENT_EVIDENCE
            for bundle in stage2_result.evidence_bundles
        )
        has_critical_finding = bool(
            stage2_result.agent_result
            and any(f.severity == AgentFindingSeverity.CRITICAL for f in stage2_result.agent_result.findings)
        )
        elevated_ml_signal = any(
            token in stage2_input.ml.predicted_class.lower() for token in ("medium", "high")
        )

        if has_hard_block:
            final_decision = FinalDecision.NO_GO
            score = 1.0
        elif score >= self.policy.no_go_threshold:
            final_decision = FinalDecision.NO_GO
        elif score >= self.policy.go_threshold or has_insufficient or has_critical_finding or elevated_ml_signal:
            final_decision = FinalDecision.CAUTION
        else:
            final_decision = FinalDecision.GO

        confidence = self._confidence_level(
            score=score,
            hard_block=has_hard_block,
            has_insufficient=has_insufficient,
            has_errors=bool(stage2_result.errors) or stage2_result.status in {Stage2Status.DEGRADED, Stage2Status.FAILED},
            limitations=limitations,
        )

        decision_reasons = self._decision_reasons(final_decision, score, reasons, has_insufficient)

        return DecisionEngineResult(
            final_decision=final_decision,
            decision_score=round(score, 4),
            confidence_level=confidence,
            stage_weights=dict(self.stage_weights),
            stage_contributions=contributions,
            decision_reasons=decision_reasons,
            blocking_reasons=blocking_reasons,
            required_actions=self._dedupe(required_actions),
            limitations=self._dedupe(limitations),
            evidence_refs=self._dedupe_refs(evidence_refs),
            metadata={
                "engine_version": "decision_engine_v1",
                "policy_name": self.policy.policy_name,
                "policy_version": self.policy.policy_version,
                "weight_rationale_ml": self.policy.weight_rationales.get("ml"),
                "weight_rationale_shap": self.policy.weight_rationales.get("shap"),
                "weight_rationale_rag": self.policy.weight_rationales.get("rag"),
                "weight_rationale_agent": self.policy.weight_rationales.get("agent"),
                "weight_rationale_scenario_profile": self.policy.weight_rationales.get("scenario_profile"),
                "weight_rationale_llm": self.policy.weight_rationales.get("llm"),
                "go_threshold": self.policy.go_threshold,
                "no_go_threshold": self.policy.no_go_threshold,
                "score_meaning": "higher means higher operational concern",
            },
        )

    def _weight(self, stage: DecisionStageName) -> float:
        return float(self.stage_weights.get(stage.value, 0.0))

    def _evaluate_core(self, stage2_input: Stage2AssessmentInput) -> _StageEvaluation:
        blocked = any(
            _truthy(stage2_input.metadata.get(key))
            for key in ("core_hard_veto", "hard_veto", "blocked")
        )
        reason = _metadata_string(
            stage2_input,
            ("core_blocking_reason", "blocked_reason", "hard_veto_reason"),
        ) or "Core hard veto / blocked state is present."
        contribution = 1.0 if blocked else 0.0
        return _StageEvaluation(
            contribution=_contribution(
                stage=DecisionStageName.CORE,
                weight=self._weight(DecisionStageName.CORE),
                contribution=contribution,
                signal="blocked" if blocked else "clear",
                summary=reason if blocked else "No Stage2 hard-veto metadata was provided.",
                reasons=[reason] if blocked else ["Core hard-veto override was not active."],
                metadata={"override_gate": True, "blocked": blocked},
            ),
            reasons=[reason] if blocked else [],
            limitations=[],
            blocking_reasons=[reason] if blocked else [],
            required_actions=["Do not launch until the blocking Core condition is resolved."] if blocked else [],
            evidence_refs=[],
        )

    def _evaluate_ml(self, stage2_input: Stage2AssessmentInput) -> _StageEvaluation:
        label = stage2_input.ml.predicted_class.strip()
        lower = label.lower()
        if "high" in lower:
            base = 0.85
        elif "medium" in lower:
            base = 0.55
        elif "low" in lower:
            base = 0.20
        else:
            base = 0.45

        probabilities = dict(stage2_input.ml.probabilities)
        top_probability = 0.0
        margin = None
        limitations: list[str] = []
        if probabilities:
            ordered = sorted(probabilities.values(), reverse=True)
            top_probability = float(ordered[0])
            if len(ordered) > 1:
                margin = float(ordered[0] - ordered[1])
            score = 0.7 * base + 0.3 * top_probability
            if margin is not None and margin < 0.15:
                score = min(1.0, score + 0.08)
                limitations.append("ML probability distribution is close; avoid overconfident interpretation.")
        else:
            score = base
            limitations.append("ML probabilities were not provided.")

        reasons = [f"ML predicted class is '{label}'."]
        if probabilities:
            reasons.append(f"Top ML probability is {top_probability:.3f}.")
        return _StageEvaluation(
            contribution=_contribution(
                stage=DecisionStageName.ML,
                weight=self._weight(DecisionStageName.ML),
                contribution=score,
                signal=label,
                summary="ML risk signal contributes to operational concern, but is not legal authority.",
                reasons=reasons,
                limitations=limitations,
                metadata={
                    "predicted_class": label,
                    "top_probability": round(top_probability, 4),
                    "probability_margin": round(margin, 4) if margin is not None else None,
                },
            ),
            reasons=reasons,
            limitations=limitations,
            blocking_reasons=[],
            required_actions=[],
            evidence_refs=[],
        )

    def _evaluate_shap(self, stage2_input: Stage2AssessmentInput) -> _StageEvaluation:
        features = list(stage2_input.ml.shap_top_features)
        limitations: list[str] = []
        if not features:
            limitations.append("No SHAP features provided.")
            score = 0.05
            topics: set[str] = set()
            high_concern = False
        else:
            names = [item.feature for item in features]
            topics = self._topics_from_names(names)
            score = min(0.45, 0.10 + 0.05 * len(topics))
            joined = " ".join(names).lower()
            high_concern = any(token in joined for token in _HIGH_CONCERN_SHAP_TOKENS)
            if high_concern:
                score = min(0.65, score + 0.10)

        reasons = [
            "SHAP is used as model explanation context only, not as causal proof.",
            f"SHAP topic count: {len(topics)}.",
        ]
        return _StageEvaluation(
            contribution=_contribution(
                stage=DecisionStageName.SHAP,
                weight=self._weight(DecisionStageName.SHAP),
                contribution=score,
                signal="topics_present" if topics else "no_topics",
                summary="SHAP topics add explanatory context and uncertainty cues.",
                reasons=reasons,
                limitations=limitations,
                metadata={
                    "topic_count": len(topics),
                    "topics": ",".join(sorted(topics)),
                    "high_concern_topic_present": high_concern,
                },
            ),
            reasons=reasons,
            limitations=limitations,
            blocking_reasons=[],
            required_actions=[],
            evidence_refs=[],
        )

    def _evaluate_rag(self, evidence_bundles: list[EvidenceBundle]) -> _StageEvaluation:
        limitations: list[str] = []
        reasons: list[str] = []
        if not evidence_bundles:
            score = 0.70
            signal = "missing"
            limitations.append("No RAG evidence bundles were available.")
        else:
            insufficient = [b for b in evidence_bundles if b.support_status == EvidenceSupportStatus.INSUFFICIENT_EVIDENCE]
            partial = [b for b in evidence_bundles if b.support_status == EvidenceSupportStatus.PARTIALLY_SUPPORTED]
            supported = [b for b in evidence_bundles if b.support_status == EvidenceSupportStatus.SUPPORTED]
            citation_count = sum(len(b.citations) for b in evidence_bundles)
            if insufficient:
                score = 0.65
                signal = "insufficient_evidence"
                limitations.extend(
                    b.no_evidence_reason or f"Insufficient evidence for query '{b.query}'."
                    for b in insufficient
                )
            elif not citation_count:
                score = 0.70
                signal = "no_citations"
                limitations.append("RAG evidence bundles did not include citations.")
            elif supported and not partial and len(supported) == len(evidence_bundles):
                score = 0.20
                signal = "supported"
            else:
                score = 0.45
                signal = "mixed_support"
            reasons.append(
                f"RAG evidence bundles: {len(evidence_bundles)}; citations: {citation_count}."
            )

        refs = self._refs_from_bundles(evidence_bundles)
        return _StageEvaluation(
            contribution=_contribution(
                stage=DecisionStageName.RAG,
                weight=self._weight(DecisionStageName.RAG),
                contribution=score,
                signal=signal,
                summary="RAG evidence affects uncertainty and traceability, not automatic approval.",
                reasons=reasons or ["RAG evidence was evaluated."],
                limitations=limitations,
                metadata={"bundle_count": len(evidence_bundles), "evidence_ref_count": len(refs)},
            ),
            reasons=reasons,
            limitations=limitations,
            blocking_reasons=[],
            required_actions=[],
            evidence_refs=refs,
        )

    def _evaluate_agent(self, stage2_result: Stage2AssessmentResult) -> _StageEvaluation:
        agent = stage2_result.agent_result
        limitations: list[str] = []
        required_actions: list[str] = []
        refs: list[AgentEvidenceReference] = []
        if agent is None:
            score = 0.70
            signal = "missing"
            limitations.append("Agent result is not available.")
            findings: list[AgentFinding] = []
        else:
            findings = list(agent.findings)
            severity_score = max((_SEVERITY_SCORE.get(f.severity, 0.05) for f in findings), default=0.20)
            score = severity_score
            if any(f.finding_type in {AgentFindingType.OPERATIONAL_UNCERTAINTY, AgentFindingType.LIMITATION} for f in findings):
                score = min(1.0, score + 0.08)
            if agent.recommendation == AgentRecommendation.NO_GO:
                score = max(score, 0.80)
            elif agent.recommendation == AgentRecommendation.CAUTION:
                score = max(score, 0.45)
            signal = agent.recommendation.value
            for action in agent.action_items:
                if action.summary.strip():
                    required_actions.append(action.summary.strip())
                refs.extend(action.evidence_references)
            for finding in findings:
                refs.extend(finding.evidence_references)

        reasons = [f"Agent finding count: {len(findings)}."]
        return _StageEvaluation(
            contribution=_contribution(
                stage=DecisionStageName.AGENT,
                weight=self._weight(DecisionStageName.AGENT),
                contribution=score,
                signal=signal,
                summary="Agent findings and provisional recommendation contribute operational judgment.",
                reasons=reasons,
                limitations=limitations,
                metadata={"finding_count": len(findings), "action_count": len(required_actions)},
            ),
            reasons=reasons,
            limitations=limitations,
            blocking_reasons=[],
            required_actions=required_actions,
            evidence_refs=refs,
        )

    def _evaluate_scenario_profile(self, stage2_result: Stage2AssessmentResult) -> _StageEvaluation:
        agent = stage2_result.agent_result
        relevant: list[AgentFinding] = []
        if agent is not None:
            for finding in agent.findings:
                meta = finding.metadata if isinstance(finding.metadata, dict) else {}
                if any(
                    bool(str(meta.get(key, "")).strip())
                    for key in ("topic", "related_scenario_fields", "related_profile_fields")
                ) or str(meta.get("support_status", "")) in {"scenario_derived", "model_explanation"}:
                    relevant.append(finding)

        if relevant:
            score = max((_SEVERITY_SCORE.get(f.severity, 0.05) for f in relevant), default=0.10)
            signal = "concerns_present"
            reasons = [f"Scenario/profile concern findings: {len(relevant)}."]
        else:
            score = 0.10
            signal = "neutral"
            reasons = ["No explicit scenario/profile inspector concerns were found."]

        return _StageEvaluation(
            contribution=_contribution(
                stage=DecisionStageName.SCENARIO_PROFILE,
                weight=self._weight(DecisionStageName.SCENARIO_PROFILE),
                contribution=score,
                signal=signal,
                summary="Scenario/profile inspector context contributes operational concern signals.",
                reasons=reasons,
                metadata={"finding_count": len(relevant)},
            ),
            reasons=reasons,
            limitations=[],
            blocking_reasons=[],
            required_actions=[],
            evidence_refs=[],
        )

    def _evaluate_llm_placeholder(self) -> _StageEvaluation:
        return _StageEvaluation(
            contribution=_contribution(
                stage=DecisionStageName.LLM,
                weight=self._weight(DecisionStageName.LLM),
                contribution=0.0,
                signal="not_configured",
                summary="LLM synthesis not evaluated in this patch.",
                reasons=["LLM contribution is a no-op placeholder for future synthesis consistency checks."],
                metadata={"llm_called": False},
            ),
            reasons=[],
            limitations=[],
            blocking_reasons=[],
            required_actions=[],
            evidence_refs=[],
        )

    def _confidence_level(
        self,
        *,
        score: float,
        hard_block: bool,
        has_insufficient: bool,
        has_errors: bool,
        limitations: list[str],
    ) -> DecisionConfidenceLevel:
        if hard_block or has_insufficient or has_errors or len(limitations) >= 3:
            return DecisionConfidenceLevel.LOW
        distance = min(abs(score - self.policy.go_threshold), abs(score - self.policy.no_go_threshold))
        if distance >= _CONFIDENCE_MARGIN and not limitations:
            return DecisionConfidenceLevel.HIGH
        return DecisionConfidenceLevel.MEDIUM

    def _decision_reasons(
        self,
        decision: FinalDecision,
        score: float,
        reasons: list[str],
        has_insufficient: bool,
    ) -> list[str]:
        out = [f"Weighted decision score is {score:.3f}; final decision is {decision.value}."]
        if has_insufficient:
            out.append("Insufficient evidence prevents overconfident GO.")
        out.extend(reasons[:8])
        return self._dedupe(out)

    def _topics_from_names(self, names: list[str]) -> set[str]:
        topics: set[str] = set()
        for raw in names:
            lower = raw.lower()
            for topic, tokens in _SHAP_TOPIC_TOKENS.items():
                if any(token in lower for token in tokens):
                    topics.add(topic)
        return topics

    def _refs_from_bundles(self, bundles: list[EvidenceBundle]) -> list[AgentEvidenceReference]:
        refs: list[AgentEvidenceReference] = []
        for bundle in bundles:
            for claim in bundle.claims:
                if claim.citations:
                    refs.append(
                        AgentEvidenceReference(
                            claim_id=claim.claim_id,
                            citation_ids=[citation.citation_id for citation in claim.citations],
                            summary=claim.claim,
                        )
                    )
            if not bundle.claims and bundle.citations:
                refs.append(
                    AgentEvidenceReference(
                        claim_id=f"bundle:{bundle.bundle_id}",
                        citation_ids=[citation.citation_id for citation in bundle.citations],
                        summary=f"Evidence bundle for query '{bundle.query}'.",
                    )
                )
        return refs

    def _dedupe(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for value in values:
            item = value.strip()
            if not item or item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    def _dedupe_refs(self, refs: list[AgentEvidenceReference]) -> list[AgentEvidenceReference]:
        seen: set[tuple[str, tuple[str, ...]]] = set()
        out: list[AgentEvidenceReference] = []
        for ref in refs:
            key = (ref.claim_id, tuple(ref.citation_ids))
            if key in seen:
                continue
            seen.add(key)
            out.append(ref)
        return out


def evaluate_stage2_decision(
    stage2_input: Stage2AssessmentInput,
    stage2_result: Stage2AssessmentResult,
    *,
    policy: DecisionPolicyConfig | None = None,
) -> DecisionEngineResult:
    return WeightedDecisionEngine(policy=policy).evaluate(stage2_input, stage2_result)
