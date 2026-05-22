"""
ACE UAV Risk Assessment System - Stage 4 (Agent Memory Engine)
File: src/uav_risk/stage2/agent/agent_memory.py
Description: Production-grade, hyper-focused session working memory for tracking 
             198 features without leakage, caching RAG metrics, and locking backtracking loops.
"""

from typing import Dict, List, Any, Optional
import structlog
from src.uav_risk.stage2.agent.agent_schemas import FeatureAssessment, ReasoningStep
from src.uav_risk.stage2.rag.schemas import LegalAnswer

# إعداد السجلات المنظمة القياسية للامتثال الجوي
logger = structlog.get_logger()


class AgentMemory:
    """
    محرك الذاكرة الشغالة (Working Memory Engine) لوكيل الـ ReAct الموحد.
    يضمن التغطية الشاملة لـ 198 ميزة كاملة، إدارة الكاش التنظيمي، والتحكم الفولاذي بميزانية التراجع.
    """

    def __init__(self, all_feature_names: List[str]):
        """
        تهيئة الذاكرة وحشو مصفوفة التتبع الحتمية بكافة المتغيرات الهندسية.
        
        Args:
            all_feature_names (List[str]): القائمة الكاملة لأسماء الـ 198 ميزة المستخرجة من الدستور.
        """
        # الحاوية المركزية لجميع التقييمات التشغيلية التي تمت حياً لمنع التكرار الحسابي
        self.examined_features: Dict[str, FeatureAssessment] = {}
        
        # سجل السقوط الجنائي الحتمي؛ يقتنص أي خرق حرج للـ Core Hardware/Regulations
        self.critical_findings: List[str] = []
        
        # ذاكرة الكاش التنظيمية المحلية لمنع استنزاف نافذة الكلمات والـ API Rate Limits للـ RAG
        self.rag_cache: Dict[str, LegalAnswer] = {}
        
        # السلسلة التاريخية المكتملة لخطوات تفكير وحركات الوكيل (Reasoning Logs)
        self.reasoning_steps: List[ReasoningStep] = []
        
        # مصفوفة الانتظار الحتمية الحامية؛ تضمن عدم إسقاط أو نسيان أي ميزة من الـ 198
        self.pending_features: List[str] = sorted(list(all_feature_names))
        
        # إجمالي المستودع الهندسي المربوط بالسيستم للتحقق من عدم حدوث أي تسرب دلالي
        self._total_features_manifest: List[str] = list(self.pending_features)
        
        # الموجز النصي عالي الكثافة المحدث تلقائياً لتغذية عقل النموذج في الدورات التالية
        self.context_summary: str = ""
        
        # السجل التاريخي لكافة الكويريات التشريعية الموجهة للفهارس المتجهية
        self.rag_queries_history: List[str] = []
        
        # صمام الأمان الميكانيكي المضاف: عداد التحكم الصارم لمنع استنزاف ميزانية الدورات
        self._backtrack_count: int = 0

        logger.info(
            "agent_memory_initialized",
            total_features_registered=len(self._total_features_manifest),
            backtrack_limit=2,
            initial_state="CLEAN"
        )

    def reprioritize_with_shap(self, top_shap_features: List[str], core_features: List[str]) -> None:
        """
        إعادة ترتيب مصفوفة الانتظار (Queue Reprioritization) بناءً على الأوزان الإحصائية والدستور.
        تضمن البدء الفوري بالفحص الفكري لأعلى 10 ميزات تأثيراً حسب SHAP، تليها الميزات الحتمية.
        
        Args:
            top_shap_features (List[str]): ميزات شجرة القرار الأعلى وزناً القادمة من Stage-1 ML Result.
            core_features (List[str]): قائمة الـ 40 ميزة الحتمية المعرفة بكتلة النظام المركزية.
        """
        # 1. فرز وتصفية ميزات SHAP المتواجدة فعلياً بالدستور والمتبقية في الانتظار
        shap_ordered = [f for f in top_shap_features if f in self.pending_features]
        
        # 2. فرز وتصفية ميزات الـ Core الحتمية المتبقية والتي لم تذكر في أعلى الـ SHAP
        core_ordered = [f for f in core_features if f in self.pending_features and f not in shap_ordered]
        
        # 3. تجميع الميزات الثانوية المتبقية (158 ميزة محاكاة إحصائية) في ذيل المصفوفة
        secondary_ordered = [f for f in self.pending_features if f not in shap_ordered and f not in core_ordered]
        
        # 4. التحديث الحتمي لمصفوفة الانتظار الحية
        self.pending_features = shap_ordered + core_ordered + secondary_ordered
        
        logger.info(
            "agent_memory_queue_reprioritized",
            shap_count=len(shap_ordered),
            core_count=len(core_ordered),
            secondary_count=len(secondary_ordered),
            total_queue_len=len(self.pending_features)
        )

    def mark_feature_examined(self, assessment: FeatureAssessment) -> None:
        """
        تسجيل الكائن التشغيلي للميزة وسحب اسمها فوراً وبشكل قطعي من قائمة الانتظار الحتمية.
        إذا ثبت كسر المتغير للحدود الحرجة، يتم قفله داخل سجل التحقيق الجنائي كـ CRITICAL.
        """
        name = assessment.feature_name
        self.examined_features[name] = assessment
        
        # معالجة الخروقات الحرجة وعزلها الفوري لحماية منطق القرار السيادي
        if assessment.status == "CRITICAL":
            log_entry = f"Feature [{name}] (Value: {assessment.value}) CRITICAL - Reason: {assessment.reasoning}"
            if log_entry not in self.critical_findings:
                self.critical_findings.append(log_entry)
                logger.warning("critical_aviation_anomaly_logged", feature=name, value=assessment.value)
        
        # السحب الفوري القطعي من مصفوفة الانتظار لمنع حدوث "العمى المعرفي" أو إهمال أي ميزة
        if name in self.pending_features:
            self.pending_features.remove(name)

    def is_feature_examined(self, feature_name: str) -> bool:
        """التحقق السريع مما إذا كانت الميزة خضعت للفحص والتقييم الفكري مسبقاً."""
        return feature_name in self.examined_features

    def get_unexamined_by_category(self, category: str, feature_defs: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        استخراج وتجميع الميزات التابعة لقطاع دلالي معين والتي لا تزال معلقة في طابور الانتظار.
        تُستدعى أوتوماتيكياً بواسطة أداة الجرف المجمع (Batch Validation Tool) لحفظ سياق الكلمات.
        """
        return [
            name for name in self.pending_features
            if feature_defs.get(name, {}).get("category") == category
        ]

    def add_reasoning_step(self, step: ReasoningStep) -> None:
        """أرشفة خطوة الـ ReAct loop الحية بداخل مسار الأثر الاستدلالي للامتثال الجوي."""
        self.reasoning_steps.append(step)

    def cache_rag_result(self, query: str, result: LegalAnswer) -> None:
        """تخزين مخرجات استعلام الـ RAG محلياً لمنع التكرار الحسابي وحفظ الـ Rate Limits للـ API."""
        self.rag_cache[query] = result
        if query not in self.rag_queries_history:
            self.rag_queries_history.append(query)

    def get_cached_rag(self, query: str) -> Optional[LegalAnswer]:
        """استرجاع الإجابة التشريعية المخزنة فوراً عند رصد تطابق دلالي للكويري (Cache Hit)."""
        return self.rag_cache.get(query)

    def can_backtrack(self) -> bool:
        """
        حارس الميزانية: فحص منطقي صارم يقفل بروتوكول التراجع المعرفي عند استهلاك محاولتين.
        يضمن الالتزام الحتمي بميزانية الدورات الـ 20 المتاحة للرحلة كاملة.
        """
        return self._backtrack_count < 2

    def increment_backtrack(self) -> None:
        """زيادة عداد التراجع بمقدار دورة واحدة عند تفعيل درع التراجع المعرفي المجمع لقطاع فيزيائي."""
        self._backtrack_count += 1
        logger.info(
            "cognitive_backtracking_incremented",
            current_backtrack_count=self._backtrack_count,
            budget_status="LOCKED" if self._backtrack_count >= 2 else "AVAILABLE"
        )

    def build_context_summary(self, total_features_count: int = 198) -> str:
        """
        صياغة الـ Context Summary عالي الكثافة الدلالية لحقنه في الموجه التالي للـ LLM.
        يلخص بدقة حجم الإنجاز، النواقص، رصيد التراجع المتبقي، وأحدث الخروقات المكتشفة حياً.
        """
        examined_count = len(self.examined_features)
        critical_count = sum(1 for f in self.examined_features.values() if f.status == "CRITICAL")
        warning_count = sum(1 for f in self.examined_features.values() if f.status == "WARNING")
        safe_count = sum(1 for f in self.examined_features.values() if f.status == "SAFE")
        
        self.context_summary = (
            f"Progress: {examined_count}/{total_features_count} verified. "
            f"Queue size: {len(self.pending_features)} pending evaluation. "
            f"Current Matrix: {critical_count} CRITICAL breaches, {warning_count} WARNINGS, {safe_count} SAFE nodes. "
            f"Backtrack Counter: {self._backtrack_count}/2 burned. "
            f"Active Core Anomalies: {self.critical_findings[-3:] if self.critical_findings else 'None'}."
        )
        return self.context_summary

    def get_snapshot(self) -> Dict[str, Any]:
        """إنتاج لقطة هيكلية فورية (Factual Snapshot) لحالة الجلسة الحية لحقنها بكتل الأدلة الجنائية."""
        return {
            "examined_count": len(self.examined_features),
            "pending_count": len(self.pending_features),
            "backtrack_count": self._backtrack_count,
            "critical_findings_manifest": list(self.critical_findings),
            "rag_queries_total": len(self.rag_queries_history),
            "cache_size": len(self.rag_cache)
        }

    def get_overall_risk_so_far(self) -> float:
        """
        الحساب الرياضي الاستباقي لمعدل خطر الرحلة الجوية بناءً على الاختراقات المخزنة بالذاكرة.
        يتم عزل الأوزان وتقييد النتيجة نهائياً عند السقف الآمن 1.0.
        """
        critical_count = sum(1 for f in self.examined_features.values() if f.status == "CRITICAL")
        warning_count = sum(1 for f in self.examined_features.values() if f.status == "WARNING")
        
        composite_risk = (critical_count * 0.3) + (warning_count * 0.1)
        return min(1.0, composite_risk)

    def get_statistics(self) -> Dict[str, Any]:
        """مصفوفة العدادات القياسية للوحة التحكم ومحرك كتابة التقارير الختامية."""
        return {
            "total_examined": len(self.examined_features),
            "critical_count": sum(1 for f in self.examined_features.values() if f.status == "CRITICAL"),
            "warning_count": sum(1 for f in self.examined_features.values() if f.status == "WARNING"),
            "safe_count": sum(1 for f in self.examined_features.values() if f.status == "SAFE"),
            "rag_queries_executed": len(self.rag_queries_history),
            "backtrack_count": self._backtrack_count
        }


# ====================================================================================
# Stage 4 Architectural Dependency Block (Consistency Rule 4):
#
# This file: src/uav_risk/stage2/agent/agent_memory.py
# - Depends on:
#   1. src/uav_risk/stage2/agent/agent_schemas.py (FeatureAssessment, ReasoningStep)
#   2. src/uav_risk/stage2/rag/schemas.py (LegalAnswer)
# - Is consumed by:
#   1. src/uav_risk/stage2/agent/agent_tools.py (Physics & RAG Tactical Tools)
#   2. src/uav_risk/stage2/agent/ace_agent.py (Core ReAct Controller)
#   3. tests/unit/test_ace_agent.py (8-Gate Verification Suite)
#
# All class interfaces, data models, and signatures are tightly locked.
# ====================================================================================