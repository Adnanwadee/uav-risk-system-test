"""
ML Model Loader - Load Stage1 production bundle and artifacts.

This module handles loading of:
- stage1_production_bundle.pkl (master bundle with model, preprocessor, encoder)
- stage1_feature_mapping.json (feature names and order)
- model_card.json (metadata for versioning and documentation)

SECURITY NOTE: joblib.load() and pickle can execute arbitrary code.
Only load bundles from trusted sources (signed URLs, internal registry, versioned artifacts).
"""

import json
import logging
import hashlib
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List

import joblib

from .schemas import ModelBundle

# Configure logging
logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Raised when model loading fails."""
    pass


def _verify_file_hash(file_path: str, expected_hash: Optional[str]) -> bool:
    """
    Verify file integrity using SHA256 hash.
    
    Args:
        file_path: Path to the file to verify
        expected_hash: Expected SHA256 hash (if None, verification is skipped)
    
    Returns:
        True if hash matches or no hash provided, False otherwise
    """
    if not expected_hash:
        return True  # Skip if no hash provided
    
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    
    actual_hash = sha256.hexdigest()
    if actual_hash != expected_hash:
        logger.error(f"Hash mismatch for {file_path}: expected {expected_hash}, got {actual_hash}")
        return False
    logger.debug(f"Hash verification passed for {file_path}")
    return True


def load_stage1_bundle(
    bundle_path: str,
    feature_mapping_path: Optional[str] = None,
    model_card_path: Optional[str] = None,
    expected_feature_count: int = 198,
    verify_hash: bool = True
) -> ModelBundle:
    """
    Load the Stage1 production bundle and associated artifacts.
    
    Args:
        bundle_path: Path to stage1_production_bundle.pkl
        feature_mapping_path: Optional path to stage1_feature_mapping.json
        model_card_path: Optional path to model_card.json
        expected_feature_count: Expected number of features (default: 198)
        verify_hash: Whether to verify SHA256 hash if available in model_card
    
    Returns:
        ModelBundle object with all artifacts
    
    Raises:
        ModelLoadError: If loading fails or required keys are missing
        FileNotFoundError: If bundle file does not exist
    """
    logger.info(f"Loading Stage1 bundle from: {bundle_path}")
    
    # 1. Check file existence and security
    if not Path(bundle_path).exists():
        raise FileNotFoundError(f"Bundle file not found: {bundle_path}")
    
    if not bundle_path.endswith(('.pkl', '.joblib')):
        logger.warning(f"Unusual file extension for joblib: {bundle_path}")
    
    logger.info("⚠️  Security Note: Ensure bundle source is trusted (joblib can execute arbitrary code)")
    
    # 2. Load model card first (for hash verification)
    model_card = {}
    if model_card_path and Path(model_card_path).exists():
        try:
            with open(model_card_path, 'r') as f:
                model_card = json.load(f)
            logger.info(f"✅ Model card loaded: version {model_card.get('version', 'unknown')}")
        except Exception as e:
            logger.warning(f"Could not load model card: {e}")
    
    # 3. Verify file hash if requested and hash is available
    if verify_hash and model_card.get("bundle_sha256"):
        if not _verify_file_hash(bundle_path, model_card["bundle_sha256"]):
            raise ModelLoadError("Bundle integrity check failed (hash mismatch)")
    
    # 4. Load master bundle
    try:
        bundle = joblib.load(bundle_path)
        logger.info("✅ Master bundle loaded successfully")
    except Exception as e:
        raise ModelLoadError(f"Failed to load bundle from {bundle_path}: {e}")
    
    # 5. Validate bundle structure
    required_keys = ['model', 'preprocessor', 'label_encoder', 'feature_names', 'class_names']
    missing_keys = [key for key in required_keys if key not in bundle]
    
    if missing_keys:
        # Try to find alternative names
        available_keys = list(bundle.keys())
        logger.debug(f"Bundle keys: {available_keys}")
        
        alt_mappings = {
            'model': ['model', 'classifier', 'estimator', 'lightgbm_model'],
            'preprocessor': ['preprocessor', 'pipeline', 'preprocessing_pipeline', 'column_transformer'],
            'label_encoder': ['label_encoder', 'encoder', 'le', 'label_encoder_'],
            'feature_names': ['feature_names', 'features', 'feature_names_', 'input_features'],
            'class_names': ['class_names', 'classes', 'target_names', 'label_classes']
        }
        
        for key in missing_keys:
            for alt in alt_mappings.get(key, []):
                if alt in bundle:
                    bundle[key] = bundle[alt]
                    logger.info(f"Using alternative '{alt}' for '{key}'")
                    break
        
        # Re-check after mapping
        missing_keys = [key for key in required_keys if key not in bundle]
        if missing_keys:
            raise ModelLoadError(f"Bundle missing required keys: {missing_keys}. Available: {available_keys}")
    
    # 6. Load feature mapping (if provided)
    feature_mapping = {}
    if feature_mapping_path and Path(feature_mapping_path).exists():
        try:
            with open(feature_mapping_path, 'r') as f:
                feature_mapping = json.load(f)
            logger.info(f"✅ Feature mapping loaded: {len(feature_mapping.get('feature_names', []))} features")
        except Exception as e:
            logger.warning(f"Could not load feature mapping: {e}")
    
    # 7. Build ModelBundle
    model_bundle = ModelBundle(
        model=bundle['model'],
        preprocessor=bundle['preprocessor'],
        label_encoder=bundle['label_encoder'],
        feature_names=bundle['feature_names'],
        class_names=bundle['class_names'],
        metadata=model_card,
        feature_mapping=feature_mapping
    )
    
    # 8. Validate bundle
    validate_bundle(model_bundle, expected_feature_count=expected_feature_count)
    
    return model_bundle


def load_stage1_bundle_from_artifacts(
    artifacts_dir: str = "artifacts",
    expected_feature_count: int = 198,
    verify_hash: bool = True
) -> ModelBundle:
    """
    Convenience function to load bundle from default artifacts directory.
    
    Args:
        artifacts_dir: Path to artifacts directory (relative or absolute)
        expected_feature_count: Expected number of features (default: 198)
        verify_hash: Whether to verify SHA256 hash if available
    
    Returns:
        ModelBundle object
    """
    base_path = Path(artifacts_dir)
    
    bundle_path = base_path / "stage1_production_bundle.pkl"
    feature_mapping_path = base_path / "stage1_feature_mapping.json"
    model_card_path = base_path / "model_card.json"
    
    # Check if bundle exists
    if not bundle_path.exists():
        raise ModelLoadError(f"Bundle not found at {bundle_path}")
    
    return load_stage1_bundle(
        bundle_path=str(bundle_path),
        feature_mapping_path=str(feature_mapping_path) if feature_mapping_path.exists() else None,
        model_card_path=str(model_card_path) if model_card_path.exists() else None,
        expected_feature_count=expected_feature_count,
        verify_hash=verify_hash
    )


def validate_bundle(bundle: ModelBundle, expected_feature_count: int = 198) -> bool:
    """
    Validate that the loaded bundle is ready for inference.
    
    Args:
        bundle: ModelBundle object to validate
        expected_feature_count: Expected number of features (default: 198)
    
    Returns:
        True if valid, raises exception otherwise
    
    Raises:
        ModelLoadError: If validation fails
    """
    # 1. Check model
    if bundle.model is None:
        raise ModelLoadError("Model is None")
    
    # 2. Check preprocessor
    if bundle.preprocessor is None:
        raise ModelLoadError("Preprocessor is None")
    
    # 3. Check label_encoder
    if bundle.label_encoder is None:
        raise ModelLoadError("Label encoder is None")
    
    # 4. Check feature_names
    if not bundle.feature_names:
        raise ModelLoadError("Feature names list is empty")
    
    # 5. Check feature count
    if len(bundle.feature_names) != expected_feature_count:
        logger.warning(
            f"Feature count mismatch: expected {expected_feature_count}, "
            f"got {len(bundle.feature_names)}. This may cause inference errors."
        )
    
    # 6. Check model type (optional but recommended)
    model_type = type(bundle.model).__name__
    if "LGBM" not in model_type and "LightGBM" not in model_type and "Booster" not in model_type:
        logger.warning(f"Expected LightGBM model, got {model_type}")
    
    # 7. Check class names
    expected_classes = ["High Risk", "Medium Risk", "Low Risk"]
    if bundle.class_names and set(bundle.class_names) != set(expected_classes):
        logger.warning(f"Class names differ from expected: {bundle.class_names}")
    
    logger.info(f"✅ Bundle validation passed: {len(bundle.feature_names)} features, "
                f"{len(bundle.class_names)} classes, model={model_type}")
    return True


# Optional: Quick test
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Try loading from default artifacts directory
        bundle = load_stage1_bundle_from_artifacts("artifacts")
        logger.info(f"📊 Bundle loaded successfully:")
        logger.info(f"   Features: {bundle.n_features}")
        logger.info(f"   Classes: {bundle.n_classes}")
        logger.info(f"   Model version: {bundle.get_model_version()}")
        logger.info(f"   Class names: {bundle.class_names}")
        
        validate_bundle(bundle)
        logger.info("✅ All validations passed!")
        
    except ModelLoadError as e:
        logger.error(f"❌ Failed to load bundle: {e}")
    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {e}")
        logger.info("   Make sure artifacts/stage1_production_bundle.pkl exists")