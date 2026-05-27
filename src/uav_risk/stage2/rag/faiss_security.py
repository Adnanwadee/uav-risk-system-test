"""
FAISS Security Module - Safe Index Loading with Integrity Verification
V3.1 FIX: Replaced pickle with JSON for signature files (prevents RCE)
"""
import os
import hmac
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np

logger = logging.getLogger(__name__)

class FAISSSecurityError(Exception):
    """Raised when FAISS index fails security checks"""
    pass

class FAISSIndexVerifier:
    """
    Verifies and safely loads FAISS indices with HMAC integrity checks.
    Prevents deserialization attacks and index tampering.

    V3.1: Uses JSON instead of pickle for signatures (anti-RCE)
    """

    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = (secret_key or os.getenv("UAV_FAISS_SECRET", "")).encode()
        if not self.secret_key:
            logger.warning("No FAISS secret key configured - using default (INSECURE)")
            self.secret_key = b"uav_rag_default_secret_change_me"

    def _compute_hmac(self, data: bytes) -> str:
        """Compute HMAC-SHA256 for data integrity"""
        return hmac.new(self.secret_key, data, hashlib.sha256).hexdigest()[:32]

    def _verify_magic_bytes(self, data: bytes) -> bool:
        """Verify FAISS index magic bytes to prevent pickle injection"""
        # FAISS flat indices start with 'h' (header for flat index)
        # FAISS IVF indices start with specific patterns
        if len(data) < 4:
            return False

        # Check for FAISS specific magic patterns
        # Flat index: starts with b'h' followed by dimension info
        # IVF index: starts with b'Iw' or similar
        faiss_magics = [b"h", b"Iw", b"IM", b"IH"]
        return any(data.startswith(magic) for magic in faiss_magics)

    def sign_index(self, index_path: Path, metadata: Optional[Dict] = None) -> Path:
        """
        Sign a FAISS index file with HMAC.

        Args:
            index_path: Path to .faiss index file
            metadata: Optional metadata to include in signature

        Returns:
            Path to signature file (.sig)
        """
        index_path = Path(index_path)
        if not index_path.exists():
            raise FAISSSecurityError(f"Index file not found: {index_path}")

        with open(index_path, "rb") as f:
            index_data = f.read()

        # Create signature payload (JSON, not pickle!)
        sig_payload = {
            "hmac": self._compute_hmac(index_data),
            "size": len(index_data),
            "metadata": metadata or {},
            "algorithm": "HMAC-SHA256",
            "version": "3.1"
        }

        sig_path = index_path.with_suffix(".faiss.sig")
        with open(sig_path, "w", encoding="utf-8") as f:
            json.dump(sig_payload, f, indent=2)

        logger.info(f"Signed index: {index_path.name} -> {sig_path.name}")
        return sig_path

    def verify_and_load(self, index_path: Path, 
                       allow_unsigned: bool = False) -> Tuple[Optional[object], Dict]:
        """
        Verify and safely load a FAISS index.

        Args:
            index_path: Path to .faiss index file
            allow_unsigned: If True, allow loading without signature (with warning)

        Returns:
            Tuple of (faiss_index, metadata)

        Raises:
            FAISSSecurityError: If verification fails
        """
        import faiss

        index_path = Path(index_path)
        if not index_path.exists():
            raise FAISSSecurityError(f"Index file not found: {index_path}")

        # Read index data
        with open(index_path, "rb") as f:
            index_data = f.read()

        # Verify magic bytes (anti-pickle-injection)
        if not self._verify_magic_bytes(index_data):
            raise FAISSSecurityError(
                f"Invalid FAISS magic bytes in {index_path}. "
                "Possible pickle injection attack or corrupted file."
            )

        # Check signature (JSON, not pickle!)
        sig_path = index_path.with_suffix(".faiss.sig")
        metadata = {}

        if sig_path.exists():
            try:
                with open(sig_path, "r", encoding="utf-8") as f:
                    sig_payload = json.load(f)
            except json.JSONDecodeError:
                raise FAISSSecurityError(f"Invalid signature file format: {sig_path}")

            computed_hmac = self._compute_hmac(index_data)
            if not hmac.compare_digest(computed_hmac, sig_payload["hmac"]):
                raise FAISSSecurityError(
                    f"HMAC mismatch for {index_path}. Index may be tampered with."
                )

            if len(index_data) != sig_payload.get("size", len(index_data)):
                raise FAISSSecurityError(
                    f"Size mismatch for {index_path}. Expected {sig_payload['size']}, got {len(index_data)}"
                )

            logger.info(f"Index verified: {index_path.name}")
            metadata = sig_payload.get("metadata", {})
        else:
            if not allow_unsigned:
                raise FAISSSecurityError(
                    f"No signature found for {index_path}. "
                    f"Run sign_index() first or set allow_unsigned=True (DANGEROUS)"
                )
            logger.warning(f"Loading UNSIGNED index: {index_path.name}")

        # Safe load using faiss directly (not pickle)
        tmp_path = None
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".faiss") as tmp:
                tmp.write(index_data)
                tmp_path = tmp.name

            index = faiss.read_index(tmp_path)
            logger.info(f"Safely loaded index: {index_path.name} (ntotal={index.ntotal})")
            return index, metadata

        except Exception as e:
            raise FAISSSecurityError(f"Failed to load FAISS index: {e}")
        finally:
            # Cleanup temp file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def verify_index_integrity(self, index_path: Path) -> bool:
        """Quick integrity check without full load"""
        try:
            self.verify_and_load(index_path, allow_unsigned=False)
            return True
        except FAISSSecurityError:
            return False

# Convenience function
def verify_and_safely_load_faiss(index_path: str, 
                                  secret_key: Optional[str] = None,
                                  allow_unsigned: bool = False) -> Tuple[Optional[object], Dict]:
    """
    Public API: Verify and safely load FAISS index.

    Usage:
        index, metadata = verify_and_safely_load_faiss("/path/to/index.faiss")
    """
    verifier = FAISSIndexVerifier(secret_key)
    return verifier.verify_and_load(Path(index_path), allow_unsigned=allow_unsigned)