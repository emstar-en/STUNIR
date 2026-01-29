#!/usr/bin/env python3
"""
STUNIR Regression Tests
=======================

Tests for previously fixed bugs to prevent regressions.

Run with: pytest tests/regression/ -v
"""

import pytest
import sys
import os
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# REG-001: IR Manifest Determinism (Issue #1205)
# ============================================================================

class TestRegIRManifestDeterminism:
    """
    Regression test for Issue #1205.
    
    Bug: IR manifests were not deterministic due to unsorted keys
    and non-canonical JSON output.
    
    Fix: Use canonical_json with sorted keys for all manifest output.
    """
    
    def test_manifest_deterministic_across_runs(self):
        """Same input should produce identical manifest."""
        from manifests.base import canonical_json, compute_sha256
        
        data = {
            "entries": [
                {"name": "b", "hash": "abc"},
                {"name": "a", "hash": "def"},
            ],
            "schema": "v1"
        }
        
        # Multiple runs should produce identical output
        results = [canonical_json(data) for _ in range(100)]
        assert len(set(results)) == 1, "Manifest output not deterministic"
    
    def test_manifest_key_ordering(self):
        """Keys must be alphabetically sorted."""
        from manifests.base import canonical_json
        
        data = {"zebra": 1, "alpha": 2, "middle": 3}
        result = canonical_json(data)
        
        # Verify order
        alpha_pos = result.index('"alpha"')
        middle_pos = result.index('"middle"')
        zebra_pos = result.index('"zebra"')
        
        assert alpha_pos < middle_pos < zebra_pos


# ============================================================================
# REG-002: Provenance Template Macros (Issue #1401)
# ============================================================================

class TestRegProvenanceMacros:
    """
    Regression test for Issue #1401.
    
    Bug: C provenance templates had undefined macros causing build failures.
    
    Fix: All macros have fallback definitions in prov_emit.c.
    """
    
    def test_provenance_file_exists(self):
        """prov_emit.c should exist with fallback macros."""
        prov_file = Path(__file__).parent.parent.parent / "tools" / "prov_emit.c"
        assert prov_file.exists(), "prov_emit.c not found"
        
        content = prov_file.read_text()
        # Check for fallback definitions
        assert "#ifndef STUNIR_PROV_BUILD_EPOCH" in content
        assert "#define STUNIR_PROV_BUILD_EPOCH" in content


# ============================================================================
# REG-003: dCBOR Canonicalization (Issue IR.0001)
# ============================================================================

class TestRegDCBORCanon:
    """
    Regression test for Issue IR.0001.
    
    Bug: IR was output as non-canonical JSON instead of dCBOR.
    
    Fix: Use canonical JSON as bootstrap, emit_dcbor.sh script added.
    """
    
    def test_emit_dcbor_script_exists(self):
        """emit_dcbor.sh should exist."""
        script = Path(__file__).parent.parent.parent / "scripts" / "lib" / "emit_dcbor.sh"
        assert script.exists(), "emit_dcbor.sh not found"
    
    def test_canonical_json_is_valid_json(self):
        """Canonical JSON output must be valid JSON."""
        from manifests.base import canonical_json
        
        data = {"test": [1, 2, 3], "nested": {"a": "b"}}
        result = canonical_json(data)
        
        # Must be parseable
        parsed = json.loads(result)
        assert parsed == data


# ============================================================================
# REG-004: Strict Verify Mode (Issue MANIFEST.0001)
# ============================================================================

class TestRegStrictVerify:
    """
    Regression test for Issue MANIFEST.0001.
    
    Bug: Strict verification didn't catch manifest/file mismatches.
    
    Fix: verify_strict.sh compares manifest entries against actual files.
    """
    
    def test_verify_strict_script_exists(self):
        """verify_strict.sh should exist."""
        script = Path(__file__).parent.parent.parent / "scripts" / "verify_strict.sh"
        assert script.exists(), "verify_strict.sh not found"


# ============================================================================
# REG-005: Path Traversal Edge Cases
# ============================================================================

class TestRegPathTraversal:
    """
    Regression tests for path traversal vulnerabilities.
    """
    
    @pytest.mark.parametrize("malicious_path", [
        "../../../etc/passwd",
        "foo/../../../etc/passwd",
        "....//....//etc/passwd",
        "..%252f..%252f/etc/passwd",  # Double-encoded
    ])
    def test_path_traversal_blocked(self, malicious_path):
        """All path traversal variants should be blocked."""
        from tools.security.validation import validate_path, PathValidationError
        
        with pytest.raises((PathValidationError, ValueError)):
            validate_path(malicious_path)


# ============================================================================
# REG-006: Unicode Filename Handling
# ============================================================================

class TestRegUnicodeFilenames:
    """
    Regression tests for Unicode filename handling.
    """
    
    def test_unicode_in_manifest_entry(self):
        """Unicode names should be handled in manifests."""
        from manifests.base import canonical_json
        
        data = {
            "entries": [
                {"name": "test_éèê", "hash": "abc123"},
                {"name": "日本語", "hash": "def456"},
            ]
        }
        
        result = canonical_json(data)
        parsed = json.loads(result)
        assert parsed == data


# ============================================================================
# REG-007: Empty Input Handling
# ============================================================================

class TestRegEmptyInput:
    """
    Regression tests for empty input handling.
    """
    
    def test_empty_dict_canonical(self):
        """Empty dict should canonicalize correctly."""
        from manifests.base import canonical_json
        result = canonical_json({})
        assert result == "{}"
    
    def test_empty_list_canonical(self):
        """Empty list should canonicalize correctly."""
        from manifests.base import canonical_json
        result = canonical_json([])
        assert result == "[]"
    
    def test_empty_string_hash(self):
        """Empty string hash should be known value."""
        from manifests.base import compute_sha256
        result = compute_sha256("")
        expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert result == expected


# ============================================================================
# REG-008: Large File Processing
# ============================================================================

class TestRegLargeFiles:
    """
    Regression tests for large file handling.
    """
    
    def test_large_json_canonical(self):
        """Large JSON should canonicalize without memory issues."""
        from manifests.base import canonical_json
        
        # Create large structure
        data = {
            "entries": [
                {"id": i, "data": "x" * 1000}
                for i in range(100)
            ]
        }
        
        result = canonical_json(data)
        assert len(result) > 100000
        
        # Should round-trip
        parsed = json.loads(result)
        assert len(parsed["entries"]) == 100
    
    def test_large_string_hash(self):
        """Large strings should hash correctly."""
        from manifests.base import compute_sha256
        
        large_data = "x" * (10 * 1024 * 1024)  # 10 MB
        result = compute_sha256(large_data)
        
        assert len(result) == 64
        # Verify determinism
        assert result == compute_sha256(large_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
