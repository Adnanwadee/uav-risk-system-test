from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uav_risk.ml.loader import assemble_raw_feature_vector, load_stage1_bundle
from uav_risk.ml.raw_schema import RAW_FEATURE_NAMES


GOLDEN_ROWS = [
    pytest.param(980, "Low Risk", 0.95, id="low-risk-row-980"),
    pytest.param(16, "Medium Risk", 0.80, id="medium-risk-row-16"),
    pytest.param(4, "High Risk", 0.95, id="high-risk-row-4"),
]


@pytest.fixture(scope="module")
def bundle():
    return load_stage1_bundle("artifacts")


@pytest.fixture(scope="module")
def raw_training_frame() -> pd.DataFrame:
    return pd.read_parquet("artifacts/uav_stage1_clean.parquet")


def _predict_raw_row(bundle, row: pd.Series) -> tuple[str, dict[str, float], np.ndarray]:
    raw_names = list(bundle.preprocessor.feature_names_in_)
    raw_frame = pd.DataFrame([row[raw_names].to_dict()], columns=raw_names)
    processed = bundle.preprocessor.transform(raw_frame)
    processed_frame = pd.DataFrame(processed, columns=bundle.feature_names)
    probabilities_raw = bundle.model.predict_proba(processed_frame)[0]
    probabilities = {name: float(probabilities_raw[idx]) for idx, name in enumerate(bundle.class_names)}
    predicted_label = bundle.class_names[int(np.argmax(probabilities_raw))]
    return predicted_label, probabilities, processed


@pytest.mark.parametrize(("row_index", "expected_label", "min_confidence"), GOLDEN_ROWS)
def test_golden_raw_parquet_rows_predict_expected_classes(
    bundle,
    raw_training_frame,
    row_index: int,
    expected_label: str,
    min_confidence: float,
    capsys,
):
    row = raw_training_frame.iloc[row_index]
    predicted_label, probabilities, processed = _predict_raw_row(bundle, row)

    with capsys.disabled():
        print(
            "golden row "
            f"{row_index}: true={row['label_risk_category']} predicted={predicted_label} probabilities={probabilities}"
        )

    assert processed.shape == (1, 198)
    assert row["label_risk_category"] == expected_label
    assert predicted_label == expected_label
    assert abs(sum(probabilities.values()) - 1.0) < 1e-6
    assert probabilities[expected_label] >= min_confidence


def test_synthetic_low_like_fixture_deviation_report_against_golden_low(bundle, raw_training_frame, capsys):
    from test_raw_end_to_end_ml import low_like_scenario, valid_profile

    low_row_index = 980
    raw_vector, metadata = assemble_raw_feature_vector(valid_profile(), low_like_scenario(), bundle=bundle)
    synthetic_map = metadata["raw_feature_map"]

    assert raw_vector.shape == (197,)
    assert metadata["raw_feature_names"] == list(RAW_FEATURE_NAMES)

    golden_row = raw_training_frame.iloc[low_row_index]
    raw_names = list(bundle.preprocessor.feature_names_in_)
    numeric_raw_names = [
        name
        for name in raw_names
        if pd.api.types.is_numeric_dtype(raw_training_frame[name])
        and name in synthetic_map
        and np.isfinite(float(synthetic_map[name]))
    ]

    deviations = []
    for name in numeric_raw_names:
        series = raw_training_frame[name].astype(float)
        std = float(series.std(ddof=0))
        if std == 0.0 or not np.isfinite(std):
            continue
        synthetic_value = float(synthetic_map[name])
        golden_value = float(golden_row[name])
        z_delta = abs(synthetic_value - golden_value) / std
        deviations.append((z_delta, name, synthetic_value, golden_value))

    top_deviations = sorted(deviations, reverse=True)[:12]
    with capsys.disabled():
        print("synthetic low-like vs golden Low row 980 top numeric deviations:")
        for z_delta, name, synthetic_value, golden_value in top_deviations:
            print(f"{name}: synthetic={synthetic_value:.6g} golden={golden_value:.6g} abs_z_delta={z_delta:.3f}")

    assert top_deviations
