#!/usr/bin/env python3
"""
STUNIR Visual Regression Tests (Snapshot Testing)
==================================================

Tests for output format consistency using snapshots.

Run with: pytest tests/visual/ -v
Update snapshots: pytest tests/visual/ --snapshot-update
"""

import pytest
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Try to import pytest-snapshot
try:
    from pytest_snapshot.plugin import Snapshot
    SNAPSHOT_AVAILABLE = True
except ImportError:
    SNAPSHOT_AVAILABLE = False

from manifests.base import canonical_json, compute_sha256


# ============================================================================
# Snapshot Directory
# ============================================================================

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
SNAPSHOT_DIR.mkdir(exist_ok=True)


# ============================================================================
# Manual Snapshot Implementation (fallback)
# ============================================================================

class ManualSnapshot:
    """Simple snapshot implementation when pytest-snapshot not available."""
    
    def __init__(self, snapshot_dir: Path):
        self.snapshot_dir = snapshot_dir
        self.snapshot_dir.mkdir(exist_ok=True)
    
    def assert_match(self, value: str, snapshot_name: str):
        """Assert value matches snapshot, creating if needed."""
        snapshot_path = self.snapshot_dir / f"{snapshot_name}.txt"
        
        if snapshot_path.exists():
            expected = snapshot_path.read_text()
            assert value == expected, f"Snapshot mismatch for {snapshot_name}"
        else:
            # Create snapshot
            snapshot_path.write_text(value)
            pytest.skip(f"Created snapshot: {snapshot_name}")


@pytest.fixture
def snapshot():
    """Provide snapshot fixture."""
    if SNAPSHOT_AVAILABLE:
        # pytest-snapshot will inject its own fixture
        pytest.skip("Use pytest-snapshot fixture")
    return ManualSnapshot(SNAPSHOT_DIR)


# ============================================================================
# Receipt Format Snapshots
# ============================================================================

class TestReceiptSnapshots:
    """Snapshot tests for receipt format."""
    
    def test_basic_receipt_format(self, snapshot):
        """Basic receipt should match expected format."""
        receipt = {
            "receipt_schema": "stunir.receipt.v1",
            "module": "example",
            "hash": "abc123" * 10 + "abcd",
            "timestamp": 1706400000,
            "artifacts": [
                {"name": "main.o", "hash": "def456" * 10 + "defg"}
            ]
        }
        
        output = canonical_json(receipt)
        snapshot.assert_match(output, "basic_receipt")
    
    def test_complex_receipt_format(self, snapshot):
        """Complex receipt with multiple artifacts."""
        receipt = {
            "receipt_schema": "stunir.receipt.v1",
            "module": "multi_artifact",
            "hash": "a" * 64,
            "timestamp": 1706400000,
            "artifacts": [
                {"name": f"file_{i}.o", "hash": f"{i:064x}"}
                for i in range(5)
            ],
            "metadata": {
                "version": "1.0.0",
                "build_flags": ["-O2", "-Wall"]
            }
        }
        
        output = canonical_json(receipt)
        snapshot.assert_match(output, "complex_receipt")


# ============================================================================
# Manifest Format Snapshots
# ============================================================================

class TestManifestSnapshots:
    """Snapshot tests for manifest format."""
    
    def test_ir_manifest_format(self, snapshot):
        """IR manifest should match expected format."""
        manifest = {
            "manifest_schema": "stunir.manifest.ir.v1",
            "manifest_epoch": 1706400000,
            "manifest_hash": "b" * 64,
            "entries": [
                {
                    "name": "module_a.dcbor",
                    "path": "asm/ir/module_a.dcbor",
                    "hash": "c" * 64,
                    "size": 1024,
                    "artifact_type": "ir",
                    "format": "dcbor"
                }
            ]
        }
        
        output = canonical_json(manifest)
        snapshot.assert_match(output, "ir_manifest")
    
    def test_receipts_manifest_format(self, snapshot):
        """Receipts manifest should match expected format."""
        manifest = {
            "manifest_schema": "stunir.manifest.receipts.v1",
            "manifest_epoch": 1706400000,
            "manifest_hash": "d" * 64,
            "entries": [
                {
                    "name": "build_receipt.json",
                    "path": "receipts/build_receipt.json",
                    "hash": "e" * 64,
                    "size": 512,
                    "artifact_type": "receipt",
                    "format": "json"
                }
            ]
        }
        
        output = canonical_json(manifest)
        snapshot.assert_match(output, "receipts_manifest")


# ============================================================================
# IR Output Format Snapshots
# ============================================================================

class TestIRSnapshots:
    """Snapshot tests for IR output format."""
    
    def test_basic_ir_format(self, snapshot):
        """Basic IR module should match expected format."""
        ir = {
            "ir_schema": "stunir.ir.v1",
            "module": "example",
            "functions": [
                {
                    "name": "main",
                    "params": [],
                    "return_type": "i32",
                    "body": [
                        {"op": "return", "value": 0}
                    ]
                }
            ],
            "types": [],
            "imports": [],
            "exports": ["main"]
        }
        
        output = canonical_json(ir)
        snapshot.assert_match(output, "basic_ir")


# ============================================================================
# Error Message Format Snapshots
# ============================================================================

class TestErrorSnapshots:
    """Snapshot tests for error message consistency."""
    
    def test_validation_error_format(self, snapshot):
        """Validation errors should have consistent format."""
        from tools.security.validation import PathValidationError
        
        try:
            raise PathValidationError("Invalid path: ../etc/passwd")
        except PathValidationError as e:
            output = str(e)
        
        snapshot.assert_match(output, "path_validation_error")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
