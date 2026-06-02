"""
Project environment loading utilities.

This module loads local .env files early without printing or logging secret
values. Runtime modules that rely on os.getenv should call load_project_env()
before reading environment variables.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EnvFileStatus(BaseModel):
    """Safe status for one attempted environment file load."""

    path: str = Field(..., description="Absolute environment file path.")
    role: str = Field(..., description="Environment file role.")
    exists: bool = Field(..., description="Whether the file exists.")
    loaded: bool = Field(..., description="Whether python-dotenv loaded the file.")


class EnvLoadReport(BaseModel):
    """Safe non-secret summary of environment loading."""

    project_root: str = Field(..., description="Resolved repository root.")
    loaded_any: bool = Field(..., description="Whether any .env file loaded.")
    files: list[EnvFileStatus] = Field(default_factory=list)


def get_project_root() -> Path:
    """
    Resolve the repository root from this source file.

    Returns:
        Absolute repository root path.
    """

    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def load_project_env(*, override: bool = False) -> EnvLoadReport:
    """
    Load project environment files once per Python process.

    Loading order:
    1. Repository-root .env
    2. Legacy src/uav_risk/stage2/.env fallback, if present

    Existing process environment variables win by default.

    Args:
        override: Whether .env values override existing os.environ values.

    Returns:
        Safe environment loading report without secret values.
    """

    project_root = get_project_root()
    candidates: list[tuple[str, Path]] = [
        ("root", project_root / ".env"),
        ("legacy_stage2", project_root / "src" / "uav_risk" / "stage2" / ".env"),
    ]

    statuses: list[EnvFileStatus] = []

    for role, path in candidates:
        exists = path.is_file()
        loaded = bool(load_dotenv(dotenv_path=path, override=override)) if exists else False
        statuses.append(
            EnvFileStatus(
                path=str(path),
                role=role,
                exists=exists,
                loaded=loaded,
            )
        )

    report = EnvLoadReport(
        project_root=str(project_root),
        loaded_any=any(status.loaded for status in statuses),
        files=statuses,
    )

    logger.debug(
        "Project environment load completed.",
        extra={
            "project_root": report.project_root,
            "loaded_any": report.loaded_any,
            "env_files": [status.model_dump() for status in report.files],
        },
    )

    return report
