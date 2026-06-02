from __future__ import annotations

import importlib

from uav_risk.stage2.contracts import EvidenceCitation


def test_legal_citation_import_from_rag_schemas_works() -> None:
    from uav_risk.stage2.rag.schemas import LegalCitation

    assert LegalCitation is EvidenceCitation


def test_stage2_rag_adapter_import_from_package_works() -> None:
    from uav_risk.stage2.rag import Stage2RAGAdapter

    assert Stage2RAGAdapter.__name__ == "Stage2RAGAdapter"


def test_importing_rag_package_is_lightweight_no_heavy_symbols_loaded() -> None:
    """Importing the RAG package namespace must not newly load heavy runtime modules.

    This test uses a before/after module delta because prior tests in the same
    pytest process may already have imported config helpers. The safety contract
    is about import side effects caused by importing uav_risk.stage2.rag itself.
    """

    before = set(importlib.sys.modules)

    import uav_risk.stage2.rag as rag_package

    after = set(importlib.sys.modules)
    newly_loaded = after - before

    assert rag_package.__name__ == "uav_risk.stage2.rag"

    forbidden_new_modules = {
        "uav_risk.stage2.rag.config_v3",
        "uav_risk.stage2.rag.rag_core_v3",
        "uav_risk.stage2.rag.hybrid_retriever",
        "uav_risk.stage2.rag.faiss_security",
        "faiss",
        "sentence_transformers",
        "onnxruntime",
        "torch",
        "transformers",
    }

    assert forbidden_new_modules.isdisjoint(newly_loaded)

def test_rag_schemas_reexports_do_not_break_existing_schema_imports() -> None:
    from uav_risk.stage2.rag.schemas import (
        DocumentChunk,
        EvidenceBundle,
        EvidenceClaim,
        EvidenceCitation,
        EvidenceOrigin,
        EvidenceSourceType,
        EvidenceSupportStatus,
        EvidenceUse,
        RAGResponse,
        ScenarioFeatures,
        SearchResult,
    )

    assert DocumentChunk.__name__ == "DocumentChunk"
    assert SearchResult.__name__ == "SearchResult"
    assert ScenarioFeatures.__name__ == "ScenarioFeatures"
    assert RAGResponse.__name__ == "RAGResponse"
    assert EvidenceCitation is not None
    assert EvidenceClaim is not None
    assert EvidenceBundle is not None
    assert EvidenceSourceType is not None
    assert EvidenceSupportStatus is not None
    assert EvidenceUse is not None
    assert EvidenceOrigin is not None
