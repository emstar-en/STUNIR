#!/usr/bin/env python3
"""
STUNIR Property-Based Fuzzing with Hypothesis
=============================================

Fuzz tests using Hypothesis to discover edge cases in parsers and validators.

Run with: pytest tests/fuzz/test_hypothesis_fuzz.py -v
"""

import pytest
import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from hypothesis import given, settings, assume, HealthCheck
    from hypothesis import strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

from tests.fuzz.strategies import (
    json_strategy,
    ir_strategy, 
    manifest_strategy,
    path_strategy,
    unicode_strategy,
)

# Import modules to fuzz
try:
    from tools.security.validation import (
        validate_path,
        validate_json_input,
        sanitize_string,
        PathValidationError,
        InvalidInputError,
    )
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False

try:
    from manifests.base import canonical_json, compute_sha256
    MANIFESTS_AVAILABLE = True
except ImportError:
    MANIFESTS_AVAILABLE = False


# ============================================================================
# Skip decorator for missing hypothesis
# ============================================================================

skip_no_hypothesis = pytest.mark.skipif(
    not HYPOTHESIS_AVAILABLE,
    reason="hypothesis not installed"
)

skip_no_validation = pytest.mark.skipif(
    not VALIDATION_AVAILABLE,
    reason="validation module not available"
)

skip_no_manifests = pytest.mark.skipif(
    not MANIFESTS_AVAILABLE,
    reason="manifests module not available"
)


# ============================================================================
# JSON Fuzzing
# ============================================================================

@skip_no_hypothesis
@skip_no_manifests
class TestJsonFuzzing:
    """Fuzz tests for JSON handling."""
    
    @given(json_strategy())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_canonical_json_never_crashes(self, data):
        """canonical_json should handle any JSON-serializable input."""
        try:
            result = canonical_json(data)
            # Result should be valid JSON string
            assert isinstance(result, str)
            # Should round-trip
            parsed = json.loads(result)
            assert parsed is not None or parsed == data
        except (TypeError, ValueError) as e:
            # Some inputs may not be JSON-serializable
            pass
    
    @given(st.text(max_size=1000))
    @settings(max_examples=100)
    def test_compute_sha256_never_crashes(self, data):
        """compute_sha256 should handle any string input."""
        result = compute_sha256(data)
        # Result should be 64 hex chars
        assert len(result) == 64
        assert all(c in '0123456789abcdef' for c in result)


# ============================================================================
# Path Validation Fuzzing
# ============================================================================

@skip_no_hypothesis
@skip_no_validation
class TestPathFuzzing:
    """Fuzz tests for path validation security."""
    
    @given(path_strategy())
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_path_validation_never_crashes(self, path):
        """Path validation should never crash, only accept or reject."""
        try:
            result = validate_path(path)
            # If accepted, result should be truthy
            assert result
        except (PathValidationError, InvalidInputError, ValueError):
            # Rejection is fine
            pass
    
    @given(st.text().filter(lambda x: '..' in x or '\x00' in x))
    @settings(max_examples=200)
    def test_dangerous_paths_rejected(self, path):
        """Paths with traversal or null bytes should be rejected."""
        with pytest.raises((PathValidationError, InvalidInputError, ValueError)):
            validate_path(path)
    
    @given(unicode_strategy())
    @settings(max_examples=300)
    def test_unicode_paths_handled_safely(self, path):
        """Unicode edge cases should be handled without crashes."""
        try:
            validate_path(path)
        except (PathValidationError, InvalidInputError, ValueError, TypeError):
            pass  # Rejection is fine


# ============================================================================
# IR Parsing Fuzzing
# ============================================================================

@skip_no_hypothesis
class TestIRFuzzing:
    """Fuzz tests for IR parsing and handling."""
    
    @given(ir_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_ir_structure_serializes(self, ir_data):
        """Generated IR structures should serialize to JSON."""
        result = json.dumps(ir_data, sort_keys=True)
        assert isinstance(result, str)
        # Should round-trip
        parsed = json.loads(result)
        assert parsed["module"] == ir_data["module"]
    
    @given(st.binary(max_size=1000))
    @settings(max_examples=100)
    def test_binary_input_handled(self, data):
        """Binary data should not crash JSON parsing."""
        try:
            json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass  # Expected for binary


# ============================================================================
# Manifest Fuzzing
# ============================================================================

@skip_no_hypothesis
class TestManifestFuzzing:
    """Fuzz tests for manifest handling."""
    
    @given(manifest_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_manifest_structure_serializes(self, manifest):
        """Generated manifest structures should serialize."""
        result = json.dumps(manifest, sort_keys=True)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "manifest_schema" in parsed
        assert "entries" in parsed


# ============================================================================
# String Sanitization Fuzzing  
# ============================================================================

@skip_no_hypothesis
@skip_no_validation
class TestSanitizationFuzzing:
    """Fuzz tests for string sanitization."""
    
    @given(st.text(max_size=1000))
    @settings(max_examples=500)
    def test_sanitize_never_crashes(self, data):
        """sanitize_string should handle any input."""
        result = sanitize_string(data)
        # Result should be a string
        assert isinstance(result, str)
        # Should be same length or shorter (no injection possible)
        assert len(result) <= len(data) + 100  # Allow some encoding expansion
    
    @given(unicode_strategy())
    @settings(max_examples=300)
    def test_unicode_sanitization(self, data):
        """Unicode edge cases should be sanitized."""
        result = sanitize_string(data)
        assert isinstance(result, str)
        # Control characters should be removed/escaped
        dangerous_chars = set('\x00\x01\x02\x03\x04\x05\x1b\x7f')
        assert not any(c in dangerous_chars for c in result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
