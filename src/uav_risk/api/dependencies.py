from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
import os

from fastapi import Request

from uav_risk.core.config import get_settings
from uav_risk.ml.loader import load_stage1_bundle
from uav_risk.ml.schemas import Stage1Bundle


def _artifact_dir() -> str:
    return get_settings().UAV_ARTIFACTS_DIR


@lru_cache(maxsize=1)
def load_cached_stage1_bundle() -> Stage1Bundle:
    return load_stage1_bundle(_artifact_dir())


def get_stage1_bundle(request: Request) -> Stage1Bundle:
    bundle: Any = getattr(request.app.state, "stage1_bundle", None)
    if bundle is None:
        bundle = load_cached_stage1_bundle()
        request.app.state.stage1_bundle = bundle
    return bundle


def get_profile_storage_root() -> Path:
    return Path(os.getenv("UAV_PROFILE_STORAGE_DIR", "data/profiles"))
