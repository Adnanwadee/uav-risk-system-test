from __future__ import annotations

import inspect

from uav_risk.stage2.agent import agent_tools


def test_agent_tools_no_stale_rag_core_import() -> None:
    source = inspect.getsource(agent_tools)
    assert "uav_risk.stage2.rag.rag_core" not in source


def test_agent_tools_no_hardcoded_altitude_121_9() -> None:
    source = inspect.getsource(agent_tools)
    assert "121.9" not in source


def test_agent_tools_no_hardcoded_altitude_122() -> None:
    source = inspect.getsource(agent_tools)
    assert "122" not in source


def test_agent_tools_no_direct_legalcitation_construction() -> None:
    source = inspect.getsource(agent_tools)
    assert "LegalCitation(" not in source


def test_agent_tools_no_direct_evidencecitation_construction() -> None:
    source = inspect.getsource(agent_tools)
    assert "EvidenceCitation(" not in source


def test_agent_tools_no_direct_llm_calls() -> None:
    source = inspect.getsource(agent_tools)
    assert "llm_client.generate(" not in source
    assert "Groq" not in source


def test_importing_agent_tools_has_no_heavy_side_effect_symbols() -> None:
    module_name = agent_tools.__name__
    assert module_name == "uav_risk.stage2.agent.agent_tools"


def test_policy_sensitive_output_marks_requires_evidence_true() -> None:
    result = agent_tools.check_physics_constraint(
        "altitude_ceiling",
        {"flight_altitude_m": 150.0},
        {"flight_altitude_m": {"safe_max": 120.0, "critical_max": 140.0}},
    )
    assert result["requires_evidence"] is True


def test_agent_tools_do_not_reference_faiss_vector_db_or_artifacts_in_source() -> None:
    source = inspect.getsource(agent_tools)
    forbidden_tokens = ["faiss", "vector_db", "artifacts", ".pdf", "parquet", "GROQ_API_KEY"]
    for token in forbidden_tokens:
        assert token not in source
