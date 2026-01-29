#!/usr/bin/env python3
"""
STUNIR Contract Tests
=====================

Tests for API and format contract compliance.

Run with: pytest tests/contracts/ -v
"""

import pytest
import sys
import os
import json
import inspect
from pathlib import Path
from typing import get_type_hints

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.contracts.schemas import (
    validate_ir,
    validate_manifest, 
    validate_receipt,
)


# ============================================================================
# Format Contract Tests
# ============================================================================

class TestIRContract:
    """Contract tests for IR format."""
    
    def test_valid_ir_passes(self):
        """Valid IR should pass validation."""
        ir = {
            "ir_schema": "stunir.ir.v1",
            "module": "test",
            "functions": [
                {"name": "main", "params": [], "return_type": "i32"}
            ],
            "exports": ["main"]
        }
        
        valid, errors = validate_ir(ir)
        assert valid, f"Validation failed: {errors}"
    
    def test_missing_module_fails(self):
        """IR without module should fail."""
        ir = {
            "functions": [{"name": "main"}]
        }
        
        valid, errors = validate_ir(ir)
        assert not valid
        assert any("module" in e for e in errors)
    
    def test_missing_functions_fails(self):
        """IR without functions should fail."""
        ir = {
            "module": "test"
        }
        
        valid, errors = validate_ir(ir)
        assert not valid
        assert any("functions" in e for e in errors)


class TestManifestContract:
    """Contract tests for manifest format."""
    
    def test_valid_manifest_passes(self):
        """Valid manifest should pass validation."""
        manifest = {
            "manifest_schema": "stunir.manifest.ir.v1",
            "manifest_epoch": 1706400000,
            "manifest_hash": "a" * 64,
            "entries": [
                {
                    "name": "test.dcbor",
                    "path": "asm/ir/test.dcbor",
                    "hash": "b" * 64,
                    "size": 1024
                }
            ]
        }
        
        valid, errors = validate_manifest(manifest)
        assert valid, f"Validation failed: {errors}"
    
    def test_invalid_hash_fails(self):
        """Manifest with invalid hash should fail."""
        manifest = {
            "manifest_schema": "stunir.manifest.ir.v1",
            "entries": [
                {
                    "name": "test.dcbor",
                    "hash": "invalid"  # Not 64 hex chars
                }
            ]
        }
        
        valid, errors = validate_manifest(manifest)
        assert not valid
        assert any("pattern" in e or "hash" in e for e in errors)


class TestReceiptContract:
    """Contract tests for receipt format."""
    
    def test_valid_receipt_passes(self):
        """Valid receipt should pass validation."""
        receipt = {
            "receipt_schema": "stunir.receipt.v1",
            "module": "test",
            "hash": "c" * 64,
            "timestamp": 1706400000
        }
        
        valid, errors = validate_receipt(receipt)
        assert valid, f"Validation failed: {errors}"
    
    def test_missing_hash_fails(self):
        """Receipt without hash should fail."""
        receipt = {
            "receipt_schema": "stunir.receipt.v1",
            "module": "test"
        }
        
        valid, errors = validate_receipt(receipt)
        assert not valid
        assert any("hash" in e for e in errors)


# ============================================================================
# API Contract Tests
# ============================================================================

class TestPythonAPIContracts:
    """Contract tests for Python API signatures."""
    
    def test_canonical_json_signature(self):
        """canonical_json should accept dict/list and return str."""
        from manifests.base import canonical_json
        
        # Check it's callable
        assert callable(canonical_json)
        
        # Check it works with dict
        result = canonical_json({"key": "value"})
        assert isinstance(result, str)
        
        # Check it works with list
        result = canonical_json([1, 2, 3])
        assert isinstance(result, str)
    
    def test_compute_sha256_signature(self):
        """compute_sha256 should accept str/bytes and return str."""
        from manifests.base import compute_sha256
        
        assert callable(compute_sha256)
        
        # String input
        result = compute_sha256("test")
        assert isinstance(result, str)
        assert len(result) == 64
        
        # Bytes input
        result = compute_sha256(b"test")
        assert isinstance(result, str)
        assert len(result) == 64
    
    def test_validate_path_signature(self):
        """validate_path should accept str and return str or raise."""
        from tools.security.validation import validate_path, PathValidationError
        
        assert callable(validate_path)
        
        # Valid path
        result = validate_path("valid/path.txt")
        assert result is not None
        
        # Invalid path should raise
        with pytest.raises(PathValidationError):
            validate_path("../invalid")


# ============================================================================
# Backward Compatibility Tests
# ============================================================================

class TestBackwardCompatibility:
    """Tests for backward compatibility."""
    
    def test_v1_ir_still_valid(self):
        """Original v1 IR format should still be valid."""
        # Minimal v1 IR
        v1_ir = {
            "module": "legacy",
            "functions": [{"name": "main"}]
        }
        
        valid, _ = validate_ir(v1_ir)
        assert valid
    
    def test_v1_manifest_still_valid(self):
        """Original v1 manifest format should still be valid."""
        # Minimal v1 manifest
        v1_manifest = {
            "manifest_schema": "stunir.manifest.ir.v1",
            "entries": [
                {"name": "file.txt", "hash": "a" * 64}
            ]
        }
        
        valid, _ = validate_manifest(v1_manifest)
        assert valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
