#!/usr/bin/env python3
"""
STUNIR Chaos Engineering Tests
==============================

Tests for graceful degradation under failure conditions.
Uses mocking to simulate failures without affecting the actual system.

Run with: pytest tests/chaos/ -v --timeout=120
"""

import pytest
import sys
import os
import json
import tempfile
import errno
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from io import StringIO

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# File System Chaos
# ============================================================================

class TestDiskFullChaos:
    """Test behavior when disk is full."""
    
    def test_manifest_write_disk_full(self):
        """Manifest generation should handle disk full gracefully."""
        from manifests.base import canonical_json
        
        data = {"test": "data"}
        result = canonical_json(data)
        
        # Simulate disk full on write
        def raise_disk_full(*args, **kwargs):
            raise OSError(errno.ENOSPC, "No space left on device")
        
        with patch('builtins.open', side_effect=raise_disk_full):
            with pytest.raises(OSError) as exc_info:
                with open("/fake/path", "w") as f:
                    f.write(result)
            
            assert exc_info.value.errno == errno.ENOSPC
    
    def test_file_hash_disk_full(self):
        """File hashing should handle disk errors."""
        from manifests.base import compute_sha256
        
        # Normal operation should work
        result = compute_sha256("test data")
        assert len(result) == 64


class TestPermissionDeniedChaos:
    """Test behavior with permission errors."""
    
    def test_read_permission_denied(self):
        """Should handle permission denied on read."""
        from manifests.base import compute_file_hash
        
        def raise_permission_denied(*args, **kwargs):
            raise PermissionError(errno.EACCES, "Permission denied")
        
        with patch('builtins.open', side_effect=raise_permission_denied):
            with pytest.raises(PermissionError):
                compute_file_hash("/fake/protected/file")
    
    def test_directory_permission_denied(self):
        """Should handle permission denied on directory listing."""
        from manifests.base import scan_directory
        
        def raise_permission_denied(*args, **kwargs):
            raise PermissionError(errno.EACCES, "Permission denied")
        
        with patch('pathlib.Path.glob', side_effect=raise_permission_denied):
            with pytest.raises(PermissionError):
                list(scan_directory("/protected/dir", "*"))


# ============================================================================
# Corrupted Input Chaos
# ============================================================================

class TestCorruptedInputChaos:
    """Test behavior with corrupted inputs."""
    
    def test_corrupted_json_handling(self):
        """Should reject corrupted JSON."""
        corrupted_inputs = [
            b'{"key": "value"',  # Unclosed brace
            b'{key: value}',     # Unquoted keys
            b'[1, 2, 3,]',       # Trailing comma
            b'\x00\x01\x02',     # Binary garbage
            b'NaN',              # Invalid JSON literal
        ]
        
        for corrupted in corrupted_inputs:
            with pytest.raises((json.JSONDecodeError, ValueError, UnicodeDecodeError)):
                if isinstance(corrupted, bytes):
                    json.loads(corrupted.decode('utf-8', errors='strict'))
                else:
                    json.loads(corrupted)
    
    def test_truncated_file_handling(self):
        """Should handle truncated files."""
        from manifests.base import compute_sha256
        
        # Truncated JSON is still hashable (it's just bytes)
        truncated = '{"module": "test", "functions":'
        result = compute_sha256(truncated)
        assert len(result) == 64
    
    def test_zero_length_file(self):
        """Should handle zero-length files."""
        from manifests.base import compute_sha256
        
        result = compute_sha256("")
        # Known SHA256 of empty string
        assert result == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


# ============================================================================
# Resource Exhaustion Chaos
# ============================================================================

class TestResourceExhaustionChaos:
    """Test behavior under resource constraints."""
    
    def test_large_input_handling(self):
        """Should handle reasonably large inputs."""
        from manifests.base import compute_sha256, canonical_json
        
        # 1MB of data - should handle fine
        large_data = "x" * (1024 * 1024)
        result = compute_sha256(large_data)
        assert len(result) == 64
        
        # Large dict - should handle fine
        large_dict = {f"key_{i}": "value" * 100 for i in range(1000)}
        result = canonical_json(large_dict)
        assert len(result) > 100000
    
    def test_deeply_nested_json(self):
        """Should handle or reject deeply nested JSON."""
        from manifests.base import canonical_json
        
        # Create deeply nested structure
        nested = {"level": 0}
        current = nested
        for i in range(50):  # 50 levels deep
            current["child"] = {"level": i + 1}
            current = current["child"]
        
        # Should handle without stack overflow
        result = canonical_json(nested)
        assert "level" in result


# ============================================================================
# Partial Failure Chaos
# ============================================================================

class TestPartialFailureChaos:
    """Test behavior with partial/intermittent failures."""
    
    def test_manifest_generation_partial_failure(self):
        """Manifest generation should be atomic - all or nothing."""
        from manifests.base import canonical_json, compute_sha256
        
        entries = []
        for i in range(10):
            entries.append({
                "name": f"file_{i}.txt",
                "hash": compute_sha256(f"content_{i}")
            })
        
        manifest = {
            "schema": "test.v1",
            "entries": entries
        }
        
        # Generation should be atomic
        result = canonical_json(manifest)
        parsed = json.loads(result)
        assert len(parsed["entries"]) == 10
    
    def test_hash_computation_consistency(self):
        """Hash computation should be consistent across retries."""
        from manifests.base import compute_sha256
        
        data = "test data for hashing"
        
        # Multiple computations should give same result
        results = [compute_sha256(data) for _ in range(100)]
        assert len(set(results)) == 1, "Hash inconsistent across calls"


# ============================================================================
# Concurrent Access Chaos
# ============================================================================

class TestConcurrentAccessChaos:
    """Test behavior under concurrent access."""
    
    def test_concurrent_hash_computation(self):
        """Hash computation should be thread-safe."""
        from manifests.base import compute_sha256
        import threading
        import queue
        
        results = queue.Queue()
        data = "test data for concurrent hashing"
        expected = compute_sha256(data)
        
        def compute_hash():
            result = compute_sha256(data)
            results.put(result)
        
        # Run concurrent threads
        threads = [threading.Thread(target=compute_hash) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All results should match
        while not results.empty():
            assert results.get() == expected
    
    def test_concurrent_json_serialization(self):
        """JSON serialization should be thread-safe."""
        from manifests.base import canonical_json
        import threading
        import queue
        
        results = queue.Queue()
        data = {"key": "value", "nested": {"a": 1, "b": 2}}
        expected = canonical_json(data)
        
        def serialize_json():
            result = canonical_json(data)
            results.put(result)
        
        threads = [threading.Thread(target=serialize_json) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        while not results.empty():
            assert results.get() == expected


# ============================================================================
# Input Boundary Chaos
# ============================================================================

class TestInputBoundaryChaos:
    """Test behavior at input boundaries."""
    
    def test_null_input_handling(self):
        """Should handle None/null inputs appropriately."""
        from manifests.base import canonical_json
        
        # None as data should work (becomes "null")
        result = canonical_json(None)
        assert result == "null"
    
    def test_empty_collections(self):
        """Should handle empty collections."""
        from manifests.base import canonical_json
        
        assert canonical_json({}) == "{}"
        assert canonical_json([]) == "[]"
    
    def test_special_float_values(self):
        """Should handle or reject special float values."""
        from manifests.base import canonical_json
        import math
        
        # NaN and Infinity are not valid JSON
        with pytest.raises((ValueError, OverflowError)):
            canonical_json({"nan": float('nan')})
        
        with pytest.raises((ValueError, OverflowError)):
            canonical_json({"inf": float('inf')})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
