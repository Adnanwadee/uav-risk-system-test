import json
import joblib
from uav_risk.ml.feature_defs import get_all_feature_names, get_core_features


def test_artifact_and_bundle_feature_list_match():
    with open('artifacts/stage1_feature_mapping.json', 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    art_list = mapping.get('feature_names') if isinstance(mapping, dict) else mapping

    bundle = joblib.load('artifacts/stage1_production_bundle.pkl')
    bundle_list = bundle.get('feature_names')

    assert isinstance(art_list, list) and isinstance(bundle_list, list), "Artifact or bundle missing feature list"
    assert len(art_list) == len(bundle_list), f"Length mismatch artifact={len(art_list)} bundle={len(bundle_list)}"
    assert art_list == bundle_list, "Feature ordering or names differ between artifact and bundle"


def test_code_registry_includes_artifact_and_core_features():
    art_list = get_all_feature_names()
    core = get_core_features()

    missing_cores = [c for c in core if c not in art_list]
    assert not missing_cores, f"Core features missing from artifact mapping: {missing_cores}"
