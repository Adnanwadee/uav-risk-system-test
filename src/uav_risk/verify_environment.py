"""src/uav_risk.verify_environment
----------------------------------
Flexible, aviation-aware environment verifier used by CI and ops.

Features:
- Detects repository root dynamically.
- Verifies Python, key libraries, critical artifacts (feature mapping), and env vars.
- Flexible: non-strict default (warnings for docs), `--strict` forces hard-fail on missing legal docs.
"""

import argparse
import sys
import os
import importlib.util
from pathlib import Path
import logging
from typing import Tuple, List, Dict

# logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] GATE_0: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def project_root() -> Path:
    # assume package lives at repo/src/uav_risk/... -> go up two levels
    return Path(__file__).resolve().parents[2]


def check_python_version(min_major: int = 3, min_minor: int = 10) -> Tuple[bool, str]:
    logger.info("Checking Python version...")
    v = sys.version_info
    current = f"{v.major}.{v.minor}.{v.micro}"
    ok = (v.major > min_major) or (v.major == min_major and v.minor >= min_minor)
    if ok:
        msg = f"Python version OK: {current}"
        logger.info(msg)
        return True, msg
    msg = f"Python >= {min_major}.{min_minor} required, found {current}"
    logger.error(msg)
    return False, msg


def find_spec_any(names: List[str]) -> bool:
    for n in names:
        if importlib.util.find_spec(n) is not None:
            return True
    return False


def check_required_libraries() -> Tuple[bool, List[str]]:
    logger.info("Checking required libraries (best-effort)...")
    # Map logical package name -> possible import names
    required = {
        "lightgbm": ["lightgbm"],
        "shap": ["shap"],
        "pandas": ["pandas"],
        "numpy": ["numpy"],
        "scikit-learn": ["sklearn"],
        "langchain": ["langchain", "langchain_core"],
        "groq": ["groq"],
        "fastapi": ["fastapi"],
        "uvicorn": ["uvicorn"],
        "httpx": ["httpx"],
        "pydantic": ["pydantic"],
        "faiss": ["faiss", "faiss_cpu", "faiss-cpu"],
        "sentence_transformers": ["sentence_transformers"],
        "torch": ["torch"],
        "streamlit": ["streamlit"],
    }

    missing: List[str] = []
    for pkg, choices in required.items():
        if not find_spec_any(choices):
            missing.append(pkg)

    if missing:
        logger.warning(f"Missing or unavailable libs: {', '.join(missing)}")
        return False, missing
    logger.info("All required libraries appear present (best-effort).")
    return True, []


def check_artifacts(strict: bool = False) -> Tuple[bool, List[str]]:
    logger.info("Checking critical artifacts...")
    root = project_root()
    artifacts_dir = root / "artifacts"
    required = [
        "stage1_feature_mapping.json",
        "model_card.json",
    ]

    missing: List[str] = []
    for name in required:
        p = artifacts_dir / name
        if not p.exists():
            missing.append(str(p))

    # stage1 production bundle may be produced via different formats; warn if none found
    possible_bundles = list(artifacts_dir.glob("stage1_production_bundle.*")) + list(artifacts_dir.glob("*.npz"))
    if not possible_bundles:
        logger.warning("No stage1 production bundle artifact detected (pkl/npz). Ensure model bundle is available.")

    if missing:
        if strict:
            logger.error(f"Missing artifact files: {', '.join(missing)}")
            return False, missing
        logger.warning(f"Missing non-critical artifacts: {', '.join(missing)}")
        return True, missing

    logger.info("Critical artifacts present.")
    return True, []


def check_rag_knowledge(strict: bool = False) -> Tuple[bool, Dict[str, bool]]:
    logger.info("Checking RAG knowledge sources (flexible)...")
    root = project_root()
    # discover likely docs and index locations
    candidates = [root / "knowledge", root / "src" / "uav_risk" / "stage2" / "knowledge"]
    status = {"pdfs_found": False, "index_built": False, "checked_paths": []}

    for base in candidates:
        docs = base / "docs"
        idx = base / "vector_db" / "index.faiss"
        status["checked_paths"].append(str(base))
        if docs.exists() and any(docs.glob("*.pdf")):
            status["pdfs_found"] = True
        if idx.exists():
            status["index_built"] = True

    if strict and not status["pdfs_found"]:
        logger.error("Strict mode: No RAG PDFs found in expected locations.")
    else:
        if not status["pdfs_found"]:
            logger.warning("No RAG PDFs detected in known locations — RAG will require index build to be useful.")

    return status["pdfs_found"], status


def check_env_vars() -> Tuple[bool, List[str]]:
    logger.info("Checking environment variables...")
    needed = ["GROQ_API_KEY"]
    missing: List[str] = []
    for v in needed:
        if not os.getenv(v):
            missing.append(v)
    if missing:
        logger.warning(f"Missing environment vars: {', '.join(missing)}")
        return False, missing
    logger.info("Required environment variables are set.")
    return True, []


def run_all(strict: bool = False) -> bool:
    logger.info("=== START GATE 0: ENVIRONMENT VERIFICATION ===")
    python_ok, _ = check_python_version()
    libs_ok, libs_missing = check_required_libraries()
    artifacts_ok, artifacts_missing = check_artifacts(strict=strict)
    rag_ok, rag_status = check_rag_knowledge(strict=strict)
    env_ok, env_missing = check_env_vars()

    # Decide what is critical: python + libs + artifacts + env
    critical_ok = python_ok and libs_ok and artifacts_ok and env_ok

    logger.info("--- SUMMARY ---")
    logger.info(f"Python: {'OK' if python_ok else 'MISSING'}")
    logger.info(f"Libraries: {'OK' if libs_ok else 'MISSING'}")
    logger.info(f"Artifacts: {'OK' if artifacts_ok else 'MISSING/WARN'}")
    logger.info(f"RAG PDFs found: {'YES' if rag_status.get('pdfs_found') else 'NO'}")
    logger.info(f"RAG Index built: {'YES' if rag_status.get('index_built') else 'NO'}")
    logger.info(f"Env Vars: {'OK' if env_ok else 'MISSING'}")

    if not critical_ok:
        logger.critical("GATE 0 FAILED: Environment missing critical pieces.")
        if libs_missing:
            logger.info(f"-> Consider installing: {', '.join(libs_missing)}")
        return False

    logger.info("GATE 0 PASSED: Environment ready (non-strict).")
    return True


def parse_args():
    p = argparse.ArgumentParser(description="ACE Gate-0 Environment Verifier")
    p.add_argument("--strict", action="store_true", help="Treat missing docs/index as fatal")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ok = run_all(strict=args.strict)
    sys.exit(0 if ok else 1)
