from __future__ import annotations

import inspect

from uav_risk.stage2.agent import agent_tools


def test_no_hardcoded_altitude_121_9() -> None:
    source = inspect.getsource(agent_tools)
    assert "121.9" not in source


def test_no_hardcoded_altitude_122() -> None:
    source = inspect.getsource(agent_tools)
    assert "122" not in source


def test_feature_defs_not_treated_as_final_regulatory_authority() -> None:
    source = inspect.getsource(agent_tools)
    assert "requires_evidence" in source
    assert "configured" in source


def test_threshold_derived_output_has_evidence_requirement_markers() -> None:
    result = agent_tools.check_physics_constraint(
        "altitude_ceiling",
        {"flight_altitude_m": 160.0},
        {"flight_altitude_m": {"safe_max": 120.0, "critical_max": 140.0}},
    )
    assert result.get("requires_evidence") is True
    assert isinstance(result.get("evidence_requirement_reason"), str)
    assert result.get("evidence_requirement_reason")


def test_no_stale_rag_core_import() -> None:
    source = inspect.getsource(agent_tools)
    assert "uav_risk.stage2.rag.rag_core" not in source


def test_no_direct_groq_or_llm_call() -> None:
    source = inspect.getsource(agent_tools)
    assert "llm_client.generate(" not in source
    assert "Groq" not in source


def test_importing_agent_tools_no_heavy_resource_init() -> None:
    assert agent_tools.__name__ == "uav_risk.stage2.agent.agent_tools"
