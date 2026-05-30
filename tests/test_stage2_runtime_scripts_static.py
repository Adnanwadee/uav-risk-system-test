from __future__ import annotations

from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_rag_diagnostic_script_exists_and_has_run_quality_flag() -> None:
    src = _read("scripts/run_stage2_rag_diagnostic.py")
    assert "--run-quality" in src


def test_pipeline_smoke_script_exists_and_has_use_real_rag_flag() -> None:
    src = _read("scripts/run_stage2_pipeline_v2_smoke.py")
    assert "--use-real-rag" in src


def test_rebuild_and_validate_scripts_exist() -> None:
    assert Path("scripts/rebuild_stage2_rag_index.py").exists()
    assert Path("scripts/validate_stage2_rag_index.py").exists()


def test_rebuild_script_supports_force_flag() -> None:
    src = _read("scripts/rebuild_stage2_rag_index.py")
    assert "--force" in src


def test_scripts_do_not_import_groq_or_llm_clients() -> None:
    src = (
        _read("scripts/run_stage2_rag_diagnostic.py").lower()
        + _read("scripts/run_stage2_pipeline_v2_smoke.py").lower()
        + _read("scripts/rebuild_stage2_rag_index.py").lower()
        + _read("scripts/validate_stage2_rag_index.py").lower()
    )
    assert "groq" not in src
    assert "report_writer" not in src


def test_scripts_do_not_import_core_ml_api_feature_generation_paths() -> None:
    src = (
        _read("scripts/run_stage2_rag_diagnostic.py")
        + _read("scripts/run_stage2_pipeline_v2_smoke.py")
        + _read("scripts/rebuild_stage2_rag_index.py")
        + _read("scripts/validate_stage2_rag_index.py")
    )
    assert "uav_risk.core" not in src
    assert "uav_risk.ml.loader" not in src
    assert "uav_risk.api" not in src


def test_scripts_do_not_reference_masterflightpayload_or_featurerouter() -> None:
    src = _read("scripts/run_stage2_rag_diagnostic.py") + _read("scripts/run_stage2_pipeline_v2_smoke.py")
    assert "MasterFlightPayload" not in src
    assert "FeatureRouter" not in src
    assert "generate_all_features_map" not in src


def test_scripts_include_index_provenance_and_quality_flags() -> None:
    src = _read("scripts/run_stage2_rag_diagnostic.py") + _read("scripts/run_stage2_pipeline_v2_smoke.py")
    assert "index_provenance" in src or "index_provenance_status" in src
    assert "quality_is_proven" in src
    assert "rag_quality_is_proven" in src
    assert "scenario_evidence_complete" in src
    assert "scenario_evidence_status" in src
    assert "evidence_bundle_details" in src
    assert "no_evidence_reason" in src
    assert "retrieval_usable" in src


def test_validate_script_includes_expected_fields() -> None:
    src = _read("scripts/validate_stage2_rag_index.py")
    assert "provenance_status" in src
    assert "dense_index_loadable" in src
    assert "faiss_ntotal" in src
    assert "dense_mapping_count" in src


def test_validate_script_uses_runtime_provenance_helper() -> None:
    src = _read("scripts/validate_stage2_rag_index.py")
    assert "inspect_rag_index_provenance" in src


def test_rebuild_script_defaults_to_canonical_build_not_legacy_repair() -> None:
    src = _read("scripts/rebuild_stage2_rag_index.py")
    assert "build_rag_index(force=args.force)" in src
    assert "--repair-from-existing" in src


def test_backend_trace_validation_script_exists_and_emits_phase_contract() -> None:
    src = _read("scripts/run_backend_trace_validation.py")
    assert "phase_count" in src
    assert "1. API input / request contract" in src
    assert "15. API response contract" in src
