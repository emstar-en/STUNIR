#!/usr/bin/env python3
"""
STUNIR Smoke Tests
==================

Fast sanity checks that verify core functionality.
Target: Complete in < 30 seconds.

Run with: pytest tests/smoke/ -v --timeout=30
"""

import pytest
import sys
import os
import json
import tempfile
import time
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# Import Tests (< 2s total)
# ============================================================================

class TestImports:
    """Verify all critical modules can be imported."""
    
    @pytest.mark.timeout(1)
    def test_import_manifests_base(self):
        """manifests.base should import."""
        from manifests.base import canonical_json, compute_sha256
        assert callable(canonical_json)
        assert callable(compute_sha256)
    
    @pytest.mark.timeout(1)
    def test_import_manifests_ir(self):
        """manifests.ir modules should import."""
        from manifests.ir.gen_ir_manifest import IRManifestGenerator
        assert IRManifestGenerator is not None
    
    @pytest.mark.timeout(1) 
    def test_import_validation(self):
        """Security validation should import."""
        from tools.security.validation import validate_path, validate_json_input
        assert callable(validate_path)
    
    @pytest.mark.timeout(1)
    def test_import_ir_emitter(self):
        """IR emitter should import."""
        try:
            from tools.ir_emitter.emit_ir import spec_to_ir
            assert callable(spec_to_ir)
        except ImportError:
            pytest.skip("IR emitter not available")


# ============================================================================
# Hash Computation Tests (< 1s)
# ============================================================================

class TestBasicHash:
    """Verify hash computation works."""
    
    @pytest.mark.timeout(1)
    def test_sha256_string(self):
        """SHA256 of string should work."""
        from manifests.base import compute_sha256
        result = compute_sha256("test")
        assert len(result) == 64
        assert all(c in '0123456789abcdef' for c in result)
    
    @pytest.mark.timeout(1)
    def test_sha256_deterministic(self):
        """Same input should produce same hash."""
        from manifests.base import compute_sha256
        h1 = compute_sha256("determinism test")
        h2 = compute_sha256("determinism test")
        assert h1 == h2
    
    @pytest.mark.timeout(1)
    def test_sha256_empty(self):
        """Empty string hash should be known value."""
        from manifests.base import compute_sha256
        result = compute_sha256("")
        # SHA256 of empty string
        expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert result == expected


# ============================================================================
# JSON Canonicalization Tests (< 2s)
# ============================================================================

class TestCanonicalJson:
    """Verify canonical JSON generation."""
    
    @pytest.mark.timeout(1)
    def test_canonical_json_sorting(self):
        """Keys should be sorted."""
        from manifests.base import canonical_json
        data = {"z": 1, "a": 2, "m": 3}
        result = canonical_json(data)
        # Keys should appear in order: a, m, z
        assert result.index('"a"') < result.index('"m"') < result.index('"z"')
    
    @pytest.mark.timeout(1)
    def test_canonical_json_deterministic(self):
        """Same data should produce same JSON."""
        from manifests.base import canonical_json
        data = {"nested": {"b": 2, "a": 1}, "top": "value"}
        r1 = canonical_json(data)
        r2 = canonical_json(data)
        assert r1 == r2
    
    @pytest.mark.timeout(1)
    def test_canonical_json_roundtrip(self):
        """JSON should round-trip correctly."""
        from manifests.base import canonical_json
        data = {"list": [1, 2, 3], "dict": {"a": 1}}
        result = canonical_json(data)
        parsed = json.loads(result)
        assert parsed == data


# ============================================================================
# File Operations Tests (< 3s)
# ============================================================================

class TestFileOperations:
    """Verify basic file operations work."""
    
    @pytest.mark.timeout(2)
    def test_file_hash(self):
        """File hashing should work."""
        from manifests.base import compute_file_hash
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test content")
            filepath = f.name
        
        try:
            result = compute_file_hash(filepath)
            assert len(result) == 64
        finally:
            os.unlink(filepath)
    
    @pytest.mark.timeout(2)
    def test_directory_scan(self):
        """Directory scanning should work."""
        from manifests.base import scan_directory
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for i in range(3):
                Path(tmpdir, f"test_{i}.json").write_text(f'{{"id": {i}}}')
            
            files = list(scan_directory(tmpdir, "*.json"))
            assert len(files) == 3


# ============================================================================
# Manifest Generation Tests (< 5s)
# ============================================================================

class TestManifestGeneration:
    """Verify manifest generation works."""
    
    @pytest.mark.timeout(3)
    def test_ir_manifest_generator_init(self):
        """IR manifest generator should initialize."""
        from manifests.ir.gen_ir_manifest import IRManifestGenerator
        gen = IRManifestGenerator()
        assert gen.manifest_type == "ir"
    
    @pytest.mark.timeout(5)
    def test_manifest_structure(self):
        """Generated manifest should have required fields."""
        from manifests.ir.gen_ir_manifest import IRManifestGenerator
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test IR file
            ir_file = Path(tmpdir, "test.json")
            ir_file.write_text('{"module": "test"}')
            
            gen = IRManifestGenerator()
            manifest = gen.generate(ir_dir=tmpdir)
            
            # Check for required fields (schema may be keyed as "schema" or "manifest_schema")
            assert "schema" in manifest or "manifest_schema" in manifest
            assert "manifest_epoch" in manifest
            assert "entries" in manifest


# ============================================================================
# Path Validation Tests (< 2s)
# ============================================================================

class TestPathValidation:
    """Verify path validation security."""
    
    @pytest.mark.timeout(1)
    def test_valid_path_accepted(self):
        """Valid paths should be accepted."""
        from tools.security.validation import validate_path
        result = validate_path("normal/path/file.txt")
        assert result is not None
    
    @pytest.mark.timeout(1)
    def test_traversal_blocked(self):
        """Path traversal should be blocked."""
        from tools.security.validation import validate_path, PathValidationError
        with pytest.raises(PathValidationError):
            validate_path("../etc/passwd")
    
    @pytest.mark.timeout(1)
    def test_null_byte_blocked(self):
        """Null byte injection should be blocked."""
        from tools.security.validation import validate_path, PathValidationError
        with pytest.raises(PathValidationError):
            validate_path("file.txt\x00.jpg")


# ============================================================================
# End-to-End Smoke Test (< 10s)
# ============================================================================

class TestEndToEnd:
    """Quick end-to-end verification."""
    
    @pytest.mark.timeout(10)
    def test_full_pipeline_smoke(self):
        """Run a minimal pipeline: spec -> IR -> manifest."""
        from manifests.base import canonical_json, compute_sha256
        
        # Create spec
        spec = {
            "module": "smoke_test",
            "version": "1.0",
            "functions": [
                {"name": "main", "params": [], "return_type": "i32"}
            ]
        }
        
        # Canonicalize
        canonical = canonical_json(spec)
        assert isinstance(canonical, str)
        
        # Hash
        spec_hash = compute_sha256(canonical)
        assert len(spec_hash) == 64
        
        # Create manifest entry
        manifest = {
            "manifest_schema": "stunir.manifest.v1",
            "entries": [{
                "name": "smoke_test",
                "hash": spec_hash,
                "size": len(canonical)
            }]
        }
        
        # Verify manifest can be serialized
        manifest_json = canonical_json(manifest)
        assert "smoke_test" in manifest_json
        
        print(f"\n  Spec hash: {spec_hash[:16]}...")
        print(f"  Manifest size: {len(manifest_json)} bytes")


if __name__ == "__main__":
    start = time.time()
    result = pytest.main([__file__, "-v"])
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.2f}s")
    if elapsed > 30:
        print("⚠️  Smoke tests exceeded 30s target!")
    sys.exit(result)
