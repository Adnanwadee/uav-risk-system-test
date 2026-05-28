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
    rag_pkg = importlib.import_module("uav_risk.stage2.rag")
    assert hasattr(rag_pkg, "Stage2RAGAdapter")

    # Heavy modules should not be imported as a side-effect.
    assert "uav_risk.stage2.rag.config_v3" not in importlib.sys.modules
    assert "uav_risk.stage2.rag.hybrid_retriever" not in importlib.sys.modules
    assert "uav_risk.stage2.rag.rag_core_v3" not in importlib.sys.modules


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
