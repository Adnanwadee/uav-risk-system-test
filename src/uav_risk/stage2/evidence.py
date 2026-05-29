# File Path: src/uav_risk/stage2/evidence.py
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
import time
from typing import Dict, Any, List, Optional
import structlog

# الاستيراد المطلق للعقود والأنواع لضمان حتمية اتساق النظام الكلي
from uav_risk.core.data_validator import ValidationResult
from uav_risk.ml.schemas import MLResult
from uav_risk.stage2.agent.agent_schemas import AgentDecision

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class AuditEvidencePack:
    """الحزمة المتكاملة للأدلة الجنائية الرقمية لإثبات سلامة ومبررات قرارات الطيران الحرجة."""
    flight_id: str
    decision: str
    overall_confidence: float
    raw_snapshot: dict               # البيانات الأصلية المطهّرة والمقصوصة عند حد 50KB
    user_provided_features: dict     # الميزات الفيزيائية التي قدمها المستخدم مباشرة
    imputed_features: list[str]      # الميزات التي تم اشتقاقها أو ملؤها تلقائياً بالمعادلات
    ml_result_snapshot: dict         # لقطة هيكلية كاملة من نتائج وتنبؤات نموذج LightGBM
    shap_top_features: list[dict]    # أهم ميزات التأثير الرياضي مستخرجة من قيم SHAP
    agent_decision_snapshot: dict    # لقطة تفكير وحالة موديول الوكيل الذكي ReAct
    legal_citations: list[dict]      # الاستشهادات والبنود التشريعية الموثقة (FAA / SORA)
    processing_time_ms: float
    system_version: str
    created_at: str


class EvidenceBuilder:
    """محرك بناء وتطهير حزم الأدلة الرقمية لضمان الامتثال وعدم التلاعب خلف الكواليس."""

    @staticmethod
    def build_final_pack(
        flight_id: str,
        validated_features: dict[str, float],
        validation_result: ValidationResult,
        ml_result: MLResult,
        agent_decision: AgentDecision,
        raw_telemetry: dict
    ) -> AuditEvidencePack:
        """
        يجمع وينسق مدخلات ومخرجات كافة بوابات العقل البرمجي لتوليد الوثيقة الجنائية الموحدة.
        """
        logger.info("evidence_pack_assembly_initiated", flight_id=flight_id)
        start_time = time.perf_counter()

        # 1. مطابقة وتطهير حجم ملف البيانات الحية للامتثال لحدود الـ 50KB لمنع تضخم الذاكرة
        raw_snapshot = EvidenceBuilder._sanitize_and_cap_telemetry(raw_telemetry, max_kb=50)

        # 2. استخراج وفصل الميزات المقدمة حياً من سجلات الفحص
        user_provided_features = {}
        imputed_features = []

        if hasattr(validation_result, 'validation_records') and validation_result.validation_records:
            for record in validation_result.validation_records:
                if record.status == "PROVIDED":
                    user_provided_features[record.feature_name] = record.final_value
                elif record.status in ["IMPUTED", "CORRECTED", "DERIVED"]:
                    imputed_features.append(record.feature_name)
        else:
            # مسار بديل مرن لحماية تشغيل الطيران في بيئات الاختبار
            user_provided_features = validated_features
            logger.warning("validation_records_absent_using_validated_features_fallback", flight_id=flight_id)

        # 3. بناء لقطة الـ ML ومعالجة دوال التحويل الهيكلي
        if hasattr(ml_result, 'to_dict') and callable(ml_result.to_dict):
            ml_snapshot = ml_result.to_dict()
        else:
            ml_snapshot = asdict(ml_result)

        # 4. تجميع وفك قيم تأثير ميزات الـ SHAP العشرة الأولى
        shap_top_features = []
        if hasattr(ml_result, 'top_features') and ml_result.top_features:
            for feat in ml_result.top_features[:10]:
                if hasattr(feat, 'to_dict') and callable(feat.to_dict):
                    shap_top_features.append(feat.to_dict())
                else:
                    shap_top_features.append(asdict(feat))

        # 5. صياغة اللقطة الدلالية الحاكمة لتفكير وقرارات الوكيل الذكي
        agent_snapshot = {
            "decision": agent_decision.decision,
            "risk_score": agent_decision.overall_risk_score,
            "features_examined": len(agent_decision.feature_assessments) if agent_decision.feature_assessments else 0,
            "critical_findings": agent_decision.critical_findings,
            "rag_queries": agent_decision.rag_queries_made
        }

        # 6. استخراج الاستشهادات القانونية الموثقة وتفريغ القواميس
        citations = []
        if agent_decision.legal_citations:
            for cit in agent_decision.legal_citations:
                if hasattr(cit, '__dict__'):
                    citations.append(cit.__dict__)
                else:
                    citations.append(asdict(cit))

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        
        pack = AuditEvidencePack(
            flight_id=flight_id,
            decision=agent_decision.decision,
            overall_confidence=float(agent_decision.confidence),
            raw_snapshot=raw_snapshot,
            user_provided_features=user_provided_features,
            imputed_features=imputed_features,
            ml_result_snapshot=ml_snapshot,
            shap_top_features=shap_top_features,
            agent_decision_snapshot=agent_snapshot,
            legal_citations=citations,
            processing_time_ms=elapsed_ms,
            system_version="ACE-v4.5.0-Production",
            created_at=datetime.now(timezone.utc).isoformat()
        )

        logger.info("evidence_pack_assembly_complete_success", flight_id=flight_id, elapsed_time_ms=elapsed_ms)
        return pack

    @staticmethod
    def _sanitize_and_cap_telemetry(raw_data: dict, max_kb: int = 50) -> dict:
        """يفحص حجم قاموس التيليميتري؛ ويقوم بقصه بشكل آمن في حال تجاوزه الحدود الفيزيائية للمنظومة."""
        try:
            serialized = json.dumps(raw_data, ensure_ascii=False)
            if len(serialized.encode('utf-8')) > max_kb * 1024:
                logger.warning("telemetry_payload_exceeded_size_limit_truncating", allowed_kb=max_kb)
                return {
                    "audit_alert_note": f"Raw payload truncated automatically. Size violated strict {max_kb}KB certification ceiling.",
                    "truncated_at_timestamp": datetime.now(timezone.utc).isoformat(),
                    "partial_snapshot_keys": list(raw_data.keys())[:15],
                    "truncated_data_stub": {k: str(v)[:150] for k, v in list(raw_data.items())[:10]}
                }
            return raw_data
        except Exception as err:
            logger.error("telemetry_json_serialization_failed", error=str(err))
            return {"error": f"Failed to secure raw digital telemetry packet mapping: {str(err)}"}


# =====================================================================
# Stage 2 Evidence Submodule Architectural Dependency Report:
#
# Depends on:
#   - src/uav_risk/core/data_validator.py (ValidationResult, FeatureValidationRecord)
#   - src/uav_risk/ml/schemas.py (MLResult)
#   - src/uav_risk/stage2/agent/agent_schemas.py (AgentDecision)
#
# Consumed by:
#   - src/uav_risk/stage2/llm/report_writer.py
#   - src/uav_risk/stage2/pipeline.py
# =====================================================================
