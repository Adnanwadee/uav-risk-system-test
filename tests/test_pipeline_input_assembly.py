import pytest
import numpy as np
from uav_risk.ml.loader import assemble_feature_vector_from_dict, Stage1Bundle
from uav_risk.ml.feature_defs import get_all_feature_names


def make_fake_bundle():
    names = get_all_feature_names()
    # minimal fake bundle
    class B:
        pass
    b = B()
    b.feature_names = names
    b.feature_mapping = {n: i for i, n in enumerate(names)}
    b.model = None
    b.preprocessor = None
    b.training_stats = {}
    b.policy_config = {}
    b.model_metadata = {}
    b.shap_explainer = None
    b.bundle_path = "test"
    b.label_encoder = None
    b.class_names = ["Low Risk","Medium Risk","High Risk"]
    return b


def test_assemble_feature_vector_happy_path():
    b = make_fake_bundle()
    # provide safe values for all features (core features explicitly provided)
    from uav_risk.ml.feature_defs import get_safe_value, get_core_features
    input_map = {name: get_safe_value(name) for name in b.feature_names}
    # ensure core features are present as provided values
    for c in get_core_features():
        input_map[c] = get_safe_value(c)
    vec, meta = assemble_feature_vector_from_dict(input_map, b)
    assert isinstance(vec, np.ndarray)
    assert vec.shape[0] == len(b.feature_names)
    assert len(meta["provided"]) >= 40


def test_assemble_feature_vector_missing_core_raises():
    b = make_fake_bundle()
    from uav_risk.ml.feature_defs import get_safe_value
    input_map = {name: get_safe_value(name) for name in b.feature_names}
    # remove a core feature
    core_feat = b.feature_names[0]
    input_map.pop(core_feat, None)
    with pytest.raises(Exception):
        assemble_feature_vector_from_dict(input_map, b)
