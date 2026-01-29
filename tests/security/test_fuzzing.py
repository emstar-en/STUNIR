#!/usr/bin/env python3
"""
STUNIR Fuzzing Tests
====================

Property-based and fuzz tests for input validation.
Tests edge cases including:
- Empty inputs
- Very large inputs
- Special characters
- Unicode edge cases
- Binary data
- Malformed JSON

Run with: pytest tests/security/test_fuzzing.py -v
"""

import pytest
import sys
import os
import json
import string
import random

# Add tools to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tools.security.validation import (
    validate_path,
    validate_json_input,
    validate_command_args,
    sanitize_string,
    PathValidationError,
    InvalidInputError,
)


# ============================================================================
# Path Traversal Tests
# ============================================================================

class TestPathTraversal:
    """Tests for path traversal attack prevention."""
    
    @pytest.mark.parametrize("path", [
        "../etc/passwd",
        "foo/../../../etc/passwd",
        "./././../secret",
        "..",
        "../..",
        "foo/bar/../../../..",
    ])
    def test_path_traversal_blocked(self, path):
        """Ensure path traversal attempts are blocked."""
        with pytest.raises(PathValidationError):
            validate_path(path)
    
    @pytest.mark.parametrize("path", [
        "normal/path/file.txt",
        "file.json",
        "dir/subdir/file",
        "_underscore/path",
        "path-with-dashes/file.txt",
    ])
    def test_valid_paths_allowed(self, path):
        """Ensure valid paths are allowed."""
        result = validate_path(path)
        assert result is not None
    
    def test_null_byte_injection(self):
        """Test null byte injection prevention."""
        with pytest.raises(PathValidationError):
            validate_path("file.txt\x00.jpg")
    
    def test_absolute_path_default_blocked(self):
        """Test absolute paths are blocked by default."""
        with pytest.raises(PathValidationError):
            validate_path("/etc/passwd")
    
    def test_absolute_path_allowed_when_enabled(self):
        """Test absolute paths allowed when explicitly enabled."""
        # This should not raise an exception
        result = validate_path("/tmp", allow_absolute=True)
        assert result is not None
    
    def test_backslash_blocked(self):
        """Test backslash characters are blocked."""
        with pytest.raises(PathValidationError):
            validate_path("foo\\bar")


# ============================================================================
# JSON Fuzzing Tests
# ============================================================================

class TestJSONFuzzing:
    """Fuzz tests for JSON validation."""
    
    @pytest.mark.parametrize("json_str,is_valid_json", [
        ("[]", True),
        ("{}", True),
        ("true", True),  # Valid JSON
        ('"string"', True),  # Valid JSON
        ("123", True),  # Valid JSON
        ("{'single': 'quotes'}", False),  # Invalid JSON syntax
        ('{key: "no quotes"}', False),  # Invalid JSON syntax
        ('{"trailing": "comma",}', False),  # Invalid JSON syntax
    ])
    def test_json_parsing(self, json_str, is_valid_json):
        """Test various JSON inputs don't crash."""
        if is_valid_json:
            # Call should succeed without exception
            validate_json_input(json_str)
        else:
            with pytest.raises(InvalidInputError):
                validate_json_input(json_str)
    
    def test_json_null_parsing(self):
        """Test JSON null value parsing."""
        result = validate_json_input("null")
        assert result is None  # JSON null becomes Python None
    
    def test_deeply_nested_json(self):
        """Test deeply nested JSON is rejected."""
        # Create deeply nested structure
        deep = {"level": None}
        current = deep
        for _ in range(100):
            current["level"] = {"level": None}
            current = current["level"]
        
        json_str = json.dumps(deep)
        with pytest.raises(InvalidInputError):
            validate_json_input(json_str, max_depth=50)
    
    def test_large_string_rejected(self):
        """Test very long strings are rejected."""
        large = {"data": "x" * (2 * 1024 * 1024)}  # 2MB string
        json_str = json.dumps(large)
        with pytest.raises(InvalidInputError):
            validate_json_input(json_str, max_string_length=1024*1024)
    
    def test_json_with_special_unicode(self):
        """Test JSON with special Unicode characters."""
        special = {
            "emoji": "\U0001F600",
            "zero_width": "\u200B",
            "text": "normal text",
        }
        json_str = json.dumps(special)
        result = validate_json_input(json_str)
        assert result["emoji"] == "\U0001F600"
    
    def test_json_bytes_input(self):
        """Test JSON validation with bytes input."""
        json_bytes = b'{"key": "value"}'
        result = validate_json_input(json_bytes)
        assert result["key"] == "value"
    
    def test_invalid_utf8_bytes(self):
        """Test invalid UTF-8 bytes are rejected."""
        invalid_utf8 = b'{"key": "\xff\xfe"}'
        with pytest.raises(InvalidInputError):
            validate_json_input(invalid_utf8)


# ============================================================================
# Command Argument Fuzzing Tests
# ============================================================================

class TestCommandArgsFuzzing:
    """Fuzz tests for command argument validation."""
    
    @pytest.mark.parametrize("args", [
        ["cmd", "arg;injection"],
        ["cmd", "arg|pipe"],
        ["cmd", "arg`backtick`"],
        ["cmd", "arg$(subshell)"],
        ["cmd", "arg\x00null"],
    ])
    def test_dangerous_args_rejected(self, args):
        """Ensure dangerous command arguments are rejected."""
        with pytest.raises(InvalidInputError):
            validate_command_args(args, allow_shell_chars=False)
    
    @pytest.mark.parametrize("args", [
        ["simple", "command"],
        ["with", "numbers", "123"],
        ["with", "dashes-and_underscores"],
        ["unicode", "café"],
    ])
    def test_safe_args_accepted(self, args):
        """Ensure safe command arguments are accepted."""
        result = validate_command_args(args, allow_shell_chars=False)
        assert result == args


# ============================================================================
# String Sanitization Tests
# ============================================================================

class TestStringSanitization:
    """Tests for string sanitization."""
    
    def test_null_byte_removal(self):
        """Test null bytes are removed."""
        result = sanitize_string("hello\x00world")
        assert "\x00" not in result
    
    def test_length_truncation(self):
        """Test strings are truncated to max length."""
        result = sanitize_string("a" * 2000, max_length=100)
        assert len(result) == 100
    
    def test_newlines_handling(self):
        """Test newline handling."""
        # Without allow_newlines, newlines should be stripped
        result = sanitize_string("line1\nline2", allow_newlines=False)
        assert "\n" not in result
        
        # With allow_newlines, newlines should be preserved
        result = sanitize_string("line1\nline2", allow_newlines=True)
        assert "\n" in result
    
    def test_control_char_removal(self):
        """Test control characters are removed."""
        result = sanitize_string("hello\x07world")  # Bell character
        assert "\x07" not in result


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Additional edge case tests."""
    
    def test_empty_path(self):
        """Test empty path is rejected."""
        with pytest.raises(PathValidationError):
            validate_path("")
    
    def test_empty_json_dict(self):
        """Test empty JSON dict is accepted."""
        result = validate_json_input("{}")
        assert result == {}
    
    def test_empty_command_args(self):
        """Test empty command args list is accepted."""
        result = validate_command_args([])
        assert result == []
    
    def test_very_long_path(self):
        """Test very long paths are handled."""
        long_path = "a" * 10000
        # Should either accept or raise PathValidationError, not crash
        try:
            validate_path(long_path)
        except PathValidationError:
            pass  # Expected


# ============================================================================
# Fuzz Testing with Random Data
# ============================================================================

class TestRandomFuzzing:
    """Random fuzzing tests."""
    
    @pytest.mark.parametrize("seed", range(10))
    def test_random_paths(self, seed):
        """Test random paths don't crash."""
        random.seed(seed)
        chars = string.ascii_letters + string.digits + "/_.-"
        path = "".join(random.choices(chars, k=random.randint(1, 50)))
        
        # Should not crash, may raise PathValidationError
        try:
            validate_path(path)
        except PathValidationError:
            pass
    
    @pytest.mark.parametrize("seed", range(10))
    def test_random_json(self, seed):
        """Test random JSON-like structures don't crash."""
        random.seed(seed)
        
        # Generate random JSON-like structure
        def random_value():
            choice = random.randint(0, 3)
            if choice == 0:
                return random.randint(-1000, 1000)
            elif choice == 1:
                return "".join(random.choices(string.ascii_letters, k=10))
            elif choice == 2:
                return random.choice([True, False])
            else:
                return None
        
        data = {
            "".join(random.choices(string.ascii_lowercase, k=5)): random_value()
            for _ in range(random.randint(1, 5))
        }
        
        json_str = json.dumps(data)
        result = validate_json_input(json_str)
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
