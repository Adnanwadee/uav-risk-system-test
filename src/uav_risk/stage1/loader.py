from __future__ import annotations

# ============================================================
# CRITICAL FIX: inject missing function into __main__ namespace
# ============================================================
import __main__

def to_string_safe(x):
    """
    Must match EXACTLY the function used in Stage-1 preprocessing
    """
    try:
        return x.astype(str)
    except Exception:
        return x

# Inject into __main__ so pickle can find it
__main__.to_string_safe = to_string_safe  # 🔴 THIS IS THE KEY LINE


from dataclasses import dataclass
from pathlib import Path
import json
import joblib


@dataclass(frozen=True)
class Stage1Artifacts:
    policy: dict
    preprocessor: object
    reg_model: object
    clf_model: object
    clf_calibrator: object
    label_encoder: object


def load_stage1_artifacts(artifacts_dir: str | Path = "artifacts") -> Stage1Artifacts:
    d = Path(artifacts_dir)

    policy = json.loads((d / "stage1_policy_config_v2.json").read_text(encoding="utf-8"))

    # These pickles REQUIRE __main__.to_string_safe
    preprocessor = joblib.load(d / "uav_stage1_preprocessor_v2.pkl")
    reg_model = joblib.load(d / "xgb_reg_stage1_v2.pkl")
    clf_model = joblib.load(d / "xgb_clf_stage1_v2.pkl")
    clf_calibrator = joblib.load(d / "clf_calibrator_stage1_v2.pkl")
    label_encoder = joblib.load(d / "label_encoder_stage1_v2.pkl")

    return Stage1Artifacts(
        policy=policy,
        preprocessor=preprocessor,
        reg_model=reg_model,
        clf_model=clf_model,
        clf_calibrator=clf_calibrator,
        label_encoder=label_encoder,
    )
