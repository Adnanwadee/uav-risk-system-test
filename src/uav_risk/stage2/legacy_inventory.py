from __future__ import annotations

from enum import Enum
from pydantic import BaseModel

 
class Stage2FileStatus(str, Enum):
    ACTIVE_V2 = "active_v2"
    COMPATIBILITY_BRIDGE = "compatibility_bridge"
    LEGACY_DO_NOT_USE = "legacy_do_not_use"
    CANDIDATE_FOR_LATER_REMOVAL = "candidate_for_later_removal"
    HEAVY_RUNTIME_RESOURCE = "heavy_runtime_resource"
    UNKNOWN = "unknown"




class Stage2FileInventoryItem(BaseModel):
    path: str
    status: Stage2FileStatus
    reason: str
    replacement: str | None
    safe_to_import: bool


def get_stage2_legacy_inventory() -> list[Stage2FileInventoryItem]:
    """Curated Stage2 inventory for migration planning.

    This function intentionally avoids importing heavy runtime modules or walking
    large filesystem trees.
    """
    items: list[Stage2FileInventoryItem] = [
        Stage2FileInventoryItem(
            path="src/uav_risk/stage2/contracts.py",
            status=Stage2FileStatus.ACTIVE_V2,
            reason="Canonical Stage2 typed contracts.",
            replacement=None,
            safe_to_import=True,
        ),
        Stage2FileInventoryItem(
            path="src/uav_risk/stage2/pipeline_v2.py",
            status=Stage2FileStatus.ACTIVE_V2,
            reason="V2 orchestrator boundary aligned to canonical contracts.",
            replacement=None,
            safe_to_import=True,
        ),
        Stage2FileInventoryItem(
            path="src/uav_risk/stage2/reporting.py",
            status=Stage2FileStatus.ACTIVE_V2,
            reason="Deterministic report renderer without LLM dependency.",
            replacement=None,
            safe_to_import=True,
        ),
        Stage2FileInventoryItem(
            path="src/uav_risk/stage2/rag/adapter.py",
            status=Stage2FileStatus.ACTIVE_V2,
            reason="Strict RAG evidence adapter producing canonical EvidenceBundle.",
            replacement=None,
            safe_to_import=True,
        ),
        Stage2FileInventoryItem(
            path="src/uav_risk/stage2/rag/runtime_diagnostics.py",
            status=Stage2FileStatus.ACTIVE_V2,
            reason="Runtime diagnostics and provenance guard for RAG infrastructure.",
            replacement=None,
            safe_to_import=True,
        ),
        Stage2FileInventoryItem(
            path="src/uav_risk/stage2/rag/quality.py",
            status=Stage2FileStatus.ACTIVE_V2,
            reason="Opt-in RAG runtime quality harness.",
            replacement=None,
            safe_to_import=True,
        ),
        Stage2FileInventoryItem(
            path="src/uav_risk/stage2/agent/facade.py",
            status=Stage2FileStatus.ACTIVE_V2,
            reason="Public-safe normalization bridge for legacy agent outputs.",
            replacement=None,
            safe_to_import=True,
        ),
        Stage2FileInventoryItem(
            path="src/uav_risk/stage2/agent/operational_agent.py",
            status=Stage2FileStatus.ACTIVE_V2,
            reason="Deterministic evidence-seeking OperationalAgentV2.",
            replacement=None,
            safe_to_import=True,
        ),
        Stage2FileInventoryItem(
            path="src/uav_risk/stage2/rag/schemas.py",
            status=Stage2FileStatus.COMPATIBILITY_BRIDGE,
            reason="Compatibility schema exports for mixed legacy/v2 imports.",
            replacement="src/uav_risk/stage2/contracts.py",
            safe_to_import=True,
        ),
        Stage2FileInventoryItem(
            path="src/uav_risk/stage2/agent/agent_schemas.py",
            status=Stage2FileStatus.COMPATIBILITY_BRIDGE,
            reason="Legacy dataclasses plus aliases to canonical contracts.",
            replacement="src/uav_risk/stage2/contracts.py",
            safe_to_import=True,
        ),
        Stage2FileInventoryItem(
            path="src/uav_risk/stage2/pipeline.py",
            status=Stage2FileStatus.LEGACY_DO_NOT_USE,
            reason="Legacy orchestrator coupled to old core/router/LLM stack.",
            replacement="src/uav_risk/stage2/pipeline_v2.py",
            safe_to_import=False,
        ),
        Stage2FileInventoryItem(
            path="src/uav_risk/stage2/agent/ace_agent.py",
            status=Stage2FileStatus.LEGACY_DO_NOT_USE,
            reason="Legacy ReAct loop with stale rag_core/groq dependencies.",
            replacement="src/uav_risk/stage2/agent/operational_agent.py",
            safe_to_import=False,
        ),
        Stage2FileInventoryItem(
            path="src/uav_risk/stage2/agent/agent_memory.py",
            status=Stage2FileStatus.CANDIDATE_FOR_LATER_REMOVAL,
            reason="Primarily tied to legacy chain-based agent behavior.",
            replacement="src/uav_risk/stage2/agent/operational_agent.py",
            safe_to_import=False,
        ),
        Stage2FileInventoryItem(
            path="src/uav_risk/stage2/llm/report_writer.py",
            status=Stage2FileStatus.CANDIDATE_FOR_LATER_REMOVAL,
            reason="Legacy LLM-driven reporting; replaced by deterministic reporting.py.",
            replacement="src/uav_risk/stage2/reporting.py",
            safe_to_import=False,
        ),
        Stage2FileInventoryItem(
            path="src/uav_risk/stage2/knowledge/models/*",
            status=Stage2FileStatus.HEAVY_RUNTIME_RESOURCE,
            reason="Local embedding/reranker runtime resources.",
            replacement=None,
            safe_to_import=False,
        ),
        Stage2FileInventoryItem(
            path="src/uav_risk/stage2/knowledge/vectdb/*",
            status=Stage2FileStatus.HEAVY_RUNTIME_RESOURCE,
            reason="Canonical vector database/index resources.",
            replacement=None,
            safe_to_import=False,
        ),
        Stage2FileInventoryItem(
            path="src/uav_risk/stage2/rag/rag_core_v3.py",
            status=Stage2FileStatus.ACTIVE_V2,
            reason="Primary async retrieval orchestrator used by runtime adapter build path.",
            replacement=None,
            safe_to_import=True,
        ),
        Stage2FileInventoryItem(
            path="src/uav_risk/stage2/rag/hybrid_retriever.py",
            status=Stage2FileStatus.ACTIVE_V2,
            reason="Hybrid dense/sparse retrieval and source-aware scoring engine.",
            replacement=None,
            safe_to_import=True,
        ),
        Stage2FileInventoryItem(
            path="src/uav_risk/stage2/rag/build_index.py",
            status=Stage2FileStatus.ACTIVE_V2,
            reason="Canonical index build and repair utilities.",
            replacement=None,
            safe_to_import=True,
        ),
        Stage2FileInventoryItem(
            path="src/uav_risk/stage2/rag/faiss_security.py",
            status=Stage2FileStatus.ACTIVE_V2,
            reason="Index signature/integrity validation for runtime and diagnostics.",
            replacement=None,
            safe_to_import=True,
        ),
        Stage2FileInventoryItem(
            path="src/uav_risk/stage2/rag/query_intelligence.py",
            status=Stage2FileStatus.ACTIVE_V2,
            reason="Query planning/rrf utilities used by rag_core_v3.",
            replacement=None,
            safe_to_import=True,
        ),
        Stage2FileInventoryItem(
            path="src/uav_risk/stage2/rag/feature_query_mapper.py",
            status=Stage2FileStatus.ACTIVE_V2,
            reason="Feature-to-query mapper used by rag_core_v3 scenario search.",
            replacement=None,
            safe_to_import=True,
        ),
        Stage2FileInventoryItem(
            path="src/uav_risk/stage2/rag/evidence_logger.py",
            status=Stage2FileStatus.ACTIVE_V2,
            reason="Evidence logging utility used by rag_core_v3 runtime path.",
            replacement=None,
            safe_to_import=True,
        ),
        Stage2FileInventoryItem(
            path="src/uav_risk/stage2/rag/sparse_index_builder.py",
            status=Stage2FileStatus.CANDIDATE_FOR_LATER_REMOVAL,
            reason="Standalone sparse builder not required for canonical runtime with prebuilt sparse_index.pkl.",
            replacement="src/uav_risk/stage2/rag/build_index.py",
            safe_to_import=True,
        ),
        Stage2FileInventoryItem(
            path="src/uav_risk/stage2/rag/groq_llm.py",
            status=Stage2FileStatus.CANDIDATE_FOR_LATER_REMOVAL,
            reason="Optional future LLM component; not in current evidence retrieval path.",
            replacement=None,
            safe_to_import=False,
        ),
        Stage2FileInventoryItem(
            path="src/uav_risk/stage2/rag/hyde_pipeline.py",
            status=Stage2FileStatus.CANDIDATE_FOR_LATER_REMOVAL,
            reason="Optional future query expansion component; not evidence source in current path.",
            replacement=None,
            safe_to_import=False,
        ),
        Stage2FileInventoryItem(
            path="src/uav_risk/stage2/rag/prompts_v3.py",
            status=Stage2FileStatus.CANDIDATE_FOR_LATER_REMOVAL,
            reason="Prompt templates for optional future LLM path; not active in evidence retrieval.",
            replacement=None,
            safe_to_import=False,
        ),
        Stage2FileInventoryItem(
            path="src/uav_risk/stage2/docs/*",
            status=Stage2FileStatus.HEAVY_RUNTIME_RESOURCE,
            reason="Evidence corpus files consumed by retrieval runtime.",
            replacement=None,
            safe_to_import=False,
        ),
    ]
    return items
