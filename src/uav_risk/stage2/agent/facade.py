from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from uav_risk.stage2.contracts import (
    AgentActionItem,
    AgentEvidenceReference,
    AgentFinding,
    AgentFindingSeverity,
    AgentFindingType,
    AgentInput,
    AgentRecommendation,
    AgentResult,
    EvidenceBundle,
    EvidenceSupportStatus,
    PublicReasoningTrace,
    Stage2Error,
    Stage2Status,
)


class AgentResultFacade:
    """Public-safe adapter from legacy agent outputs to canonical AgentResult."""

    def __init__(self, agent: Any | None = None) -> None:
        self._agent = agent

    async def run(self, agent_input: AgentInput) -> AgentResult:
        if self._agent is None:
            return self._degraded_result(
                message="Agent is not configured.",
                code="agent_not_configured",
                evidence_bundles=agent_input.evidence_bundles,
            )

        method = None
        for name in ("run", "assess", "evaluate"):
            candidate = getattr(self._agent, name, None)
            if callable(candidate):
                method = candidate
                break

        if method is None:
            return self._degraded_result(
                message="Agent method is not available.",
                code="agent_method_missing",
                evidence_bundles=agent_input.evidence_bundles,
            )

        try:
            result = method(agent_input)
            if hasattr(result, "__await__"):
                result = await result
            return self.normalize_result(
                result,
                agent_input=agent_input,
                evidence_bundles=agent_input.evidence_bundles,
            )
        except Exception:
            return self._degraded_result(
                message="Agent execution failed.",
                code="agent_execution_failed",
                evidence_bundles=agent_input.evidence_bundles,
            )

    def normalize_result(
        self,
        raw_result: Any,
        *,
        agent_input: AgentInput | None = None,
        evidence_bundles: list[EvidenceBundle] | None = None,
    ) -> AgentResult:
        mapping = self._to_mapping(raw_result)
        bundles = list(evidence_bundles or [])

        recommendation = self._normalize_recommendation(
            self._safe_get(mapping, ("recommendation", "decision", "final_decision"), default=None)
        )
        status = self._normalize_status(self._safe_get(mapping, ("status",), default=None), recommendation)
        confidence = self._normalize_confidence(
            self._safe_get(mapping, ("confidence", "overall_confidence"), default=None),
            recommendation,
        )

        findings = self._extract_findings(mapping)
        action_items = self._extract_action_items(mapping)

        trace = PublicReasoningTrace(
            observations=self._to_str_list(self._safe_get(mapping, ("observations", "critical_findings"), default=[])),
            checks_performed=self._to_str_list(self._safe_get(mapping, ("checks_performed",), default=[])),
            evidence_consulted=self._extract_evidence_refs(mapping),
            conflicts=self._to_str_list(self._safe_get(mapping, ("conflicts",), default=[])),
            limitations=self._to_str_list(self._safe_get(mapping, ("limitations",), default=[])),
        )

        if recommendation in {
            AgentRecommendation.GO,
            AgentRecommendation.CAUTION,
            AgentRecommendation.NO_GO,
        } and status == Stage2Status.COMPLETED and not findings:
            findings.append(
                AgentFinding(
                    finding_id="auto_finding_1",
                    finding_type=AgentFindingType.TOOL_CHECK,
                    severity=AgentFindingSeverity.INFO,
                    summary="Agent produced a recommendation without structured findings.",
                    requires_evidence=False,
                    evidence_references=[],
                    metadata={},
                )
            )

        has_insufficient_bundle = any(
            bundle.support_status == EvidenceSupportStatus.INSUFFICIENT_EVIDENCE
            for bundle in bundles
        )
        clearly_structural_no_go = recommendation == AgentRecommendation.NO_GO and any(
            finding.finding_type == AgentFindingType.STRUCTURAL for finding in findings
        )

        if has_insufficient_bundle and not clearly_structural_no_go:
            recommendation = AgentRecommendation.INSUFFICIENT_EVIDENCE
            if confidence > 0.5:
                confidence = 0.5

        errors = self._extract_errors(mapping)

        metadata = {}
        raw_meta = self._safe_get(mapping, ("metadata",), default={})
        if isinstance(raw_meta, dict):
            for key, value in raw_meta.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    metadata[str(key)] = value

        return AgentResult(
            status=status,
            recommendation=recommendation,
            confidence=confidence,
            findings=findings,
            action_items=action_items,
            reasoning_trace=trace,
            evidence_bundles=bundles,
            errors=errors,
            metadata=metadata,
        )

    def _degraded_result(self, *, message: str, code: str, evidence_bundles: list[EvidenceBundle]) -> AgentResult:
        return AgentResult(
            status=Stage2Status.DEGRADED,
            recommendation=AgentRecommendation.DEGRADED,
            confidence=0.0,
            findings=[
                AgentFinding(
                    finding_id="agent_degraded",
                    finding_type=AgentFindingType.LIMITATION,
                    severity=AgentFindingSeverity.HIGH,
                    summary=message,
                    evidence_references=[],
                    requires_evidence=False,
                    metadata={},
                )
            ],
            action_items=[],
            reasoning_trace=PublicReasoningTrace(
                observations=[message],
                checks_performed=[],
                evidence_consulted=[],
                conflicts=[],
                limitations=[message],
            ),
            evidence_bundles=list(evidence_bundles),
            errors=[Stage2Error(code=code, message=message, details={})],
            metadata={},
        )

    def _safe_get(self, obj: Any, names: tuple[str, ...], default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            for name in names:
                if name in obj:
                    return obj[name]
            return default
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
        return default

    def _to_mapping(self, obj: Any) -> dict[str, Any]:
        if isinstance(obj, dict):
            return obj
        if is_dataclass(obj):
            return asdict(obj)
        dump = getattr(obj, "model_dump", None)
        if callable(dump):
            value = dump()
            if isinstance(value, dict):
                return value
        if hasattr(obj, "__dict__"):
            return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
        return {}

    def _normalize_recommendation(self, value: Any) -> AgentRecommendation:
        if value is None:
            return AgentRecommendation.DEGRADED
        normalized = str(value).strip().lower().replace("-", "_")
        mapping = {
            "go": AgentRecommendation.GO,
            "approve": AgentRecommendation.GO,
            "approved": AgentRecommendation.GO,
            "caution": AgentRecommendation.CAUTION,
            "conditional_go": AgentRecommendation.CAUTION,
            "conditional": AgentRecommendation.CAUTION,
            "no_go": AgentRecommendation.NO_GO,
            "reject": AgentRecommendation.NO_GO,
            "rejected": AgentRecommendation.NO_GO,
            "insufficient_evidence": AgentRecommendation.INSUFFICIENT_EVIDENCE,
            "no_evidence": AgentRecommendation.INSUFFICIENT_EVIDENCE,
        }
        return mapping.get(normalized, AgentRecommendation.DEGRADED)

    def _normalize_status(self, value: Any, recommendation: AgentRecommendation) -> Stage2Status:
        if isinstance(value, Stage2Status):
            return value
        if value is not None:
            text = str(value).strip().lower()
            for status in Stage2Status:
                if text == status.value:
                    return status
        if recommendation in {
            AgentRecommendation.GO,
            AgentRecommendation.CAUTION,
            AgentRecommendation.NO_GO,
            AgentRecommendation.INSUFFICIENT_EVIDENCE,
        }:
            return Stage2Status.COMPLETED
        return Stage2Status.DEGRADED

    def _normalize_confidence(self, value: Any, recommendation: AgentRecommendation) -> float:
        if value is None:
            default = 0.0 if recommendation in {AgentRecommendation.DEGRADED, AgentRecommendation.INSUFFICIENT_EVIDENCE} else 0.5
            return default
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = 0.0 if recommendation in {AgentRecommendation.DEGRADED, AgentRecommendation.INSUFFICIENT_EVIDENCE} else 0.5
        return max(0.0, min(1.0, confidence))

    def _extract_findings(self, mapping: dict[str, Any]) -> list[AgentFinding]:
        findings: list[AgentFinding] = []
        raw_findings = self._safe_get(mapping, ("findings",), default=[])
        if isinstance(raw_findings, list):
            for idx, item in enumerate(raw_findings):
                item_map = self._to_mapping(item)
                if not item_map:
                    continue
                refs = self._extract_evidence_refs(item_map)
                finding_type = self._normalize_finding_type(self._safe_get(item_map, ("finding_type", "type"), default="tool_check"))
                requires_evidence = bool(self._safe_get(item_map, ("requires_evidence",), default=False))
                if finding_type == AgentFindingType.EVIDENCE_BACKED and not refs:
                    finding_type = AgentFindingType.OPERATIONAL_UNCERTAINTY
                    requires_evidence = False
                if requires_evidence and not refs:
                    requires_evidence = False
                summary = str(self._safe_get(item_map, ("summary", "reasoning", "message"), default="")).strip()
                if not summary:
                    continue
                findings.append(
                    AgentFinding(
                        finding_id=str(self._safe_get(item_map, ("finding_id", "id"), default=f"finding_{idx+1}")).strip(),
                        finding_type=finding_type,
                        severity=self._normalize_severity(self._safe_get(item_map, ("severity",), default="info")),
                        summary=summary,
                        evidence_references=refs,
                        requires_evidence=requires_evidence,
                        metadata={},
                    )
                )
        if not findings:
            critical = self._to_str_list(self._safe_get(mapping, ("critical_findings",), default=[]))
            for idx, text in enumerate(critical):
                if not text:
                    continue
                findings.append(
                    AgentFinding(
                        finding_id=f"critical_{idx+1}",
                        finding_type=AgentFindingType.STRUCTURAL,
                        severity=AgentFindingSeverity.HIGH,
                        summary=text,
                        evidence_references=[],
                        requires_evidence=False,
                        metadata={},
                    )
                )
        return findings

    def _extract_action_items(self, mapping: dict[str, Any]) -> list[AgentActionItem]:
        actions: list[AgentActionItem] = []
        raw_actions = self._safe_get(mapping, ("action_items", "recommendations"), default=[])
        if isinstance(raw_actions, list):
            for idx, item in enumerate(raw_actions):
                if isinstance(item, str):
                    summary = item.strip()
                    if not summary:
                        continue
                    actions.append(
                        AgentActionItem(
                            action_id=f"action_{idx+1}",
                            summary=summary,
                            priority=AgentFindingSeverity.MEDIUM,
                            evidence_references=[],
                            metadata={},
                        )
                    )
                else:
                    item_map = self._to_mapping(item)
                    summary = str(self._safe_get(item_map, ("summary", "action"), default="")).strip()
                    if not summary:
                        continue
                    actions.append(
                        AgentActionItem(
                            action_id=str(self._safe_get(item_map, ("action_id", "id"), default=f"action_{idx+1}")).strip(),
                            summary=summary,
                            priority=self._normalize_severity(self._safe_get(item_map, ("priority", "severity"), default="medium")),
                            evidence_references=self._extract_evidence_refs(item_map),
                            metadata={},
                        )
                    )
        return actions

    def _extract_evidence_refs(self, mapping: dict[str, Any]) -> list[AgentEvidenceReference]:
        refs: list[AgentEvidenceReference] = []
        raw = self._safe_get(mapping, ("evidence_consulted", "evidence_references"), default=[])
        if isinstance(raw, list):
            for idx, item in enumerate(raw):
                item_map = self._to_mapping(item)
                claim_id = str(self._safe_get(item_map, ("claim_id",), default="")).strip()
                summary = str(self._safe_get(item_map, ("summary",), default="")).strip()
                cids = self._safe_get(item_map, ("citation_ids",), default=[])
                if not isinstance(cids, list):
                    cids = []
                cids = [str(v).strip() for v in cids if str(v).strip()]
                if not claim_id or not summary:
                    continue
                refs.append(
                    AgentEvidenceReference(
                        claim_id=claim_id,
                        citation_ids=cids,
                        summary=summary,
                    )
                )
        return refs

    def _to_str_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(v).strip() for v in value if str(v).strip()]

    def _extract_errors(self, mapping: dict[str, Any]) -> list[Stage2Error]:
        errs: list[Stage2Error] = []
        raw = self._safe_get(mapping, ("errors",), default=[])
        if isinstance(raw, list):
            for item in raw:
                item_map = self._to_mapping(item)
                code = str(self._safe_get(item_map, ("code",), default="")).strip()
                message = str(self._safe_get(item_map, ("message",), default="")).strip()
                if code and message:
                    errs.append(Stage2Error(code=code, message=message, details={}))
        return errs

    def _normalize_finding_type(self, value: Any) -> AgentFindingType:
        text = str(value).strip().lower()
        for ft in AgentFindingType:
            if text == ft.value:
                return ft
        return AgentFindingType.TOOL_CHECK

    def _normalize_severity(self, value: Any) -> AgentFindingSeverity:
        text = str(value).strip().lower()
        for sev in AgentFindingSeverity:
            if text == sev.value:
                return sev
        return AgentFindingSeverity.INFO
