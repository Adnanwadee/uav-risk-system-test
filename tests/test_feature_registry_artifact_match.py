import pytest
from uav_risk.ml import feature_defs


def test_feature_registry_matches_artifact():
    ok, msg = feature_defs.validate_feature_registry_against_artifact(strict=True)
    assert ok, msg
