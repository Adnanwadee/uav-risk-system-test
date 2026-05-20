"""
Environment Verification Script (Gate 0)
Ensures all requirements (Python, libraries, artifacts, env vars) are met before starting the ACE system.
"""

import sys
import os
import importlib.util
from pathlib import Path
import logging
from typing import Tuple, List, Dict

# إعداد لوجر بسيط خاص بسكربت التحقق
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] GATE_0: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def check_python_version() -> Tuple[bool, str]:
    """
    Verifies that the Python version is 3.10 or higher.
    Returns: Tuple of (is_valid: bool, message: str)
    """
    logger.info("Checking Python version...")
    version = sys.version_info
    current_version = f"{version.major}.{version.minor}.{version.micro}"
    
    if version >= (3, 10):
        msg = f"Python version is valid: {current_version}"
        logger.info(msg)
        return True, msg
    else:
        msg = f"Python 3.10+ required, found {current_version}"
        logger.error(msg)
        return False, msg

def check_required_libraries() -> Tuple[bool, List[str]]:
    """
    Verifies that all critical third-party libraries are installed.
    Returns: Tuple of (all_installed: bool, missing_libs: List[str])
    """
    logger.info("Checking required libraries...")
    required_libs = [
        "lightgbm", "shap", "pandas", "numpy", "sklearn",
        "langchain_core", "groq", "fastapi", "uvicorn", 
        "httpx", "pydantic", "faiss", "sentence_transformers", 
        "torch", "streamlit", "plotly", "structlog"
    ]
    
    missing_libs: List[str] = []
    
    for lib in required_libs:
        # بعض المكتبات لها اسم استيراد مختلف عن اسم الحزمة
        import_name = "sklearn" if lib == "scikit-learn" else lib
        import_name = "faiss" if lib == "faiss-cpu" else import_name
        
        if importlib.util.find_spec(import_name) is None:
            missing_libs.append(lib)
            
    if not missing_libs:
        logger.info("All required libraries are installed.")
        return True, missing_libs
    else:
        logger.error(f"Missing libraries: {', '.join(missing_libs)}")
        return False, missing_libs

def check_artifact_files() -> Tuple[bool, List[str]]:
    """
    Verifies the existence of Stage-1 ML artifacts.
    Returns: Tuple of (all_exist: bool, missing_files: List[str])
    """
    logger.info("Checking ML artifacts...")
    artifacts_dir = Path("artifacts")
    required_files = [
        "stage1_production_bundle.pkl",
        "stage1_feature_mapping.json",
        "model_card.json"
    ]
    
    missing_files: List[str] = []
    
    for file_name in required_files:
        file_path = artifacts_dir / file_name
        if not file_path.exists():
            missing_files.append(str(file_path))
            
    if not missing_files:
        logger.info("All ML artifacts are present.")
        return True, missing_files
    else:
        logger.error(f"Missing ML artifacts: {', '.join(missing_files)}")
        return False, missing_files

def check_knowledge_files() -> Tuple[bool, Dict[str, bool]]:
    """
    Verifies the existence of RAG knowledge documents and index.
    Returns: Tuple of (is_valid: bool, status: dict)
    """
    logger.info("Checking RAG knowledge files...")
    docs_dir = Path("src/uav_risk/stage2/knowledge/docs")
    index_path = Path("src/uav_risk/stage2/knowledge/vector_db_backup_20260512_005519/index.faiss") # الاعتماد على المسار الموجود حالياً
    
    status = {
        "pdfs_found": False,
        "index_built": False
    }
    
    # فحص ملفات الـ PDF
    if docs_dir.exists() and any(docs_dir.glob("*.pdf")):
        status["pdfs_found"] = True
        logger.info("RAG source documents (PDFs) found.")
    else:
        logger.warning(f"No PDFs found in {docs_dir}. RAG will not have source data to build index.")
        
    # فحص الـ Index
    if index_path.exists():
        status["index_built"] = True
        logger.info("FAISS index found.")
    else:
        logger.warning(f"FAISS index not found at {index_path}. It must be built before full operations.")
        
    # نعتبر الفحص ناجحاً مبدئياً إذا وجدت ملفات PDF (الاندكس يمكن بناؤه لاحقاً)
    is_valid = status["pdfs_found"]
    return is_valid, status

def check_env_vars() -> Tuple[bool, List[str]]:
    """
    Verifies the existence of required environment variables.
    Returns: Tuple of (all_exist: bool, missing_vars: List[str])
    """
    logger.info("Checking environment variables...")
    required_vars = ["GROQ_API_KEY"]
    missing_vars: List[str] = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
            
    if not missing_vars:
        logger.info("All required environment variables are set.")
        return True, missing_vars
    else:
        logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
        return False, missing_vars

def run_all_checks() -> bool:
    """
    Executes all environment checks sequentially.
    Returns True only if all critical checks pass.
    """
    logger.info("=== STARTING GATE 0 ENVIRONMENT VERIFICATION ===")
    
    python_ok, _ = check_python_version()
    libs_ok, libs_missing = check_required_libraries()
    artifacts_ok, artifacts_missing = check_artifact_files()
    knowledge_ok, knowledge_status = check_knowledge_files()
    env_ok, env_missing = check_env_vars()
    
    all_passed = all([python_ok, libs_ok, artifacts_ok, env_ok])
    
    logger.info("=== VERIFICATION SUMMARY ===")
    logger.info(f"Python Version: {'PASS ✅' if python_ok else 'FAIL ❌'}")
    logger.info(f"Libraries: {'PASS ✅' if libs_ok else 'FAIL ❌'}")
    logger.info(f"ML Artifacts: {'PASS ✅' if artifacts_ok else 'FAIL ❌'}")
    logger.info(f"RAG Knowledge: {'PASS ✅ (PDFs exist)' if knowledge_ok else 'WARNING ⚠️ (No PDFs)'}")
    logger.info(f"Index Built: {'YES ✅' if knowledge_status['index_built'] else 'NO ⚠️'}")
    logger.info(f"Env Variables: {'PASS ✅' if env_ok else 'FAIL ❌'}")
    
    if not all_passed:
        logger.critical("🚨 GATE 0 FAILED: The environment is not ready for execution.")
        if not libs_ok:
            logger.info(f"-> Run: pip install {' '.join(libs_missing)}")
        if not artifacts_ok:
            logger.info("-> Ensure stage1 artifacts are placed in the 'artifacts/' directory.")
        if not env_ok:
            logger.info("-> Set missing environment variables (e.g., export GROQ_API_KEY='your-key').")
        return False
        
    logger.info("🚀 GATE 0 PASSED: Environment is verified and ready.")
    return True

if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)

"""
Dependencies:
- Depends on: Standard library, pathlib, OS environment, filesystem (artifacts folder).
- Depended on by: This is a standalone script meant to be run manually or by CI/CD pipelines before starting `main.py` or `app.py`.
"""