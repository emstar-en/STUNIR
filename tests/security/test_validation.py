#!/usr/bin/env python3
"""STUNIR Security Validation Tests.

This module tests input validation functions for security vulnerabilities:
- Path traversal attacks
- Command injection attempts
- Malicious JSON/CBOR inputs
- File size DoS prevention
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.security.validation import (
    validate_path,
    validate_json_input,
    validate_file_size,
    validate_command_args,
    sanitize_string,
    PathValidationError,
    InputValidationError,
)
from tools.security.exceptions import FileSizeError


class TestPathValidation(unittest.TestCase):
    """Test path validation security."""
    
    def test_rejects_path_traversal_dotdot(self):
        """Test rejection of .. traversal attempts."""
        malicious_paths = [
            "../etc/passwd",
            "foo/../../../etc/passwd",
            "..\\..\\etc\\passwd",
            "foo/bar/../../../etc/shadow",
            ".../.../etc/passwd",  # Double dots
        ]
        for path in malicious_paths:
            with self.assertRaises(PathValidationError, msg=f"Should reject: {path}"):
                validate_path(path, base_dir="/home/ubuntu")
    
    def test_rejects_absolute_paths_when_not_allowed(self):
        """Test rejection of absolute paths."""
        absolute_paths = [
            "/etc/passwd",
            "/home/ubuntu/sensitive",
            "C:\\Windows\\System32",
        ]
        for path in absolute_paths:
            with self.assertRaises(PathValidationError, msg=f"Should reject: {path}"):
                validate_path(path, allow_absolute=False)
    
    def test_rejects_null_bytes(self):
        """Test rejection of null byte injection."""
        null_paths = [
            "file.txt\x00.jpg",
            "valid\x00/../../etc/passwd",
            "\x00malicious",
        ]
        for path in null_paths:
            with self.assertRaises(PathValidationError, msg=f"Should reject null bytes"):
                validate_path(path)
    
    def test_rejects_empty_path(self):
        """Test rejection of empty paths."""
        with self.assertRaises(PathValidationError):
            validate_path("")
        with self.assertRaises(PathValidationError):
            validate_path(None)
    
    def test_accepts_valid_relative_paths(self):
        """Test acceptance of valid relative paths."""
        valid_paths = [
            "file.txt",
            "dir/file.txt",
            "dir/subdir/file.json",
            "path_with_underscore/file-with-dash.py",
        ]
        for path in valid_paths:
            result = validate_path(path)
            self.assertIsInstance(result, Path)
    
    def test_extension_filtering(self):
        """Test file extension validation."""
        # Should accept allowed extensions
        result = validate_path("file.json", allowed_extensions=[".json", ".py"])
        self.assertIsInstance(result, Path)
        
        # Should reject disallowed extensions
        with self.assertRaises(PathValidationError):
            validate_path("file.exe", allowed_extensions=[".json", ".py"])
    
    def test_base_dir_constraint(self):
        """Test that paths stay within base directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Valid path within base dir
            valid = validate_path("subdir/file.txt", base_dir=tmpdir)
            self.assertTrue(str(valid).startswith(tmpdir))
            
            # Path escaping base dir should fail
            with self.assertRaises(PathValidationError):
                validate_path("../../etc/passwd", base_dir=tmpdir)


class TestJSONValidation(unittest.TestCase):
    """Test JSON input validation security."""
    
    def test_rejects_deeply_nested_json(self):
        """Test rejection of excessively nested JSON (DoS prevention)."""
        # Build deeply nested object
        nested = {"a": None}
        current = nested
        for _ in range(100):
            current["a"] = {"b": None}
            current = current["a"]
        
        json_str = json.dumps(nested)
        
        with self.assertRaises(InputValidationError):
            validate_json_input(json_str, max_depth=50)
    
    def test_rejects_invalid_json(self):
        """Test rejection of malformed JSON."""
        invalid_json = [
            "{invalid}",
            '{"key": undefined}',
            "{'single': 'quotes'}",
            "{missing: quotes}",
        ]
        for invalid in invalid_json:
            with self.assertRaises(InputValidationError):
                validate_json_input(invalid)
    
    def test_rejects_very_long_strings(self):
        """Test rejection of excessively long strings (memory DoS)."""
        long_string = "a" * 2_000_000  # 2MB string
        json_str = json.dumps({"data": long_string})
        
        with self.assertRaises(InputValidationError):
            validate_json_input(json_str, max_string_length=1_000_000)
    
    def test_accepts_valid_json(self):
        """Test acceptance of valid JSON."""
        valid_json = {
            "schema": "stunir.v1",
            "items": [1, 2, 3],
            "nested": {"key": "value"},
            "flag": True,
            "count": 42,
        }
        
        result = validate_json_input(json.dumps(valid_json))
        self.assertEqual(result["schema"], "stunir.v1")
    
    def test_rejects_invalid_utf8(self):
        """Test rejection of invalid UTF-8 in JSON bytes."""
        invalid_utf8 = b'{"key": "\xff\xfe invalid"}'
        
        with self.assertRaises(InputValidationError):
            validate_json_input(invalid_utf8)
    
    def test_schema_validation(self):
        """Test basic schema validation."""
        schema = {
            "type": "object",
            "required": ["name", "version"],
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "integer"},
            }
        }
        
        # Valid data
        valid = {"name": "test", "version": 1}
        result = validate_json_input(json.dumps(valid), schema=schema)
        self.assertEqual(result["name"], "test")
        
        # Missing required field
        invalid = {"name": "test"}  # missing version
        with self.assertRaises(InputValidationError):
            validate_json_input(json.dumps(invalid), schema=schema)


class TestFileSizeValidation(unittest.TestCase):
    """Test file size validation security."""
    
    def test_rejects_large_files(self):
        """Test rejection of files exceeding size limit."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"x" * 1024)  # 1KB file
            f.flush()
            path = f.name
        
        try:
            # Should accept when under limit
            size = validate_file_size(path, max_size=2048)
            self.assertEqual(size, 1024)
            
            # Should reject when over limit
            with self.assertRaises(FileSizeError):
                validate_file_size(path, max_size=512)
        finally:
            os.unlink(path)
    
    def test_rejects_nonexistent_files(self):
        """Test rejection of non-existent file paths."""
        with self.assertRaises(PathValidationError):
            validate_file_size("/nonexistent/path/file.txt")


class TestCommandArgValidation(unittest.TestCase):
    """Test command argument validation security."""
    
    def test_rejects_shell_metacharacters(self):
        """Test rejection of shell metacharacters."""
        dangerous_args = [
            ["cmd", "; rm -rf /"],
            ["cmd", "| cat /etc/passwd"],
            ["cmd", "$(whoami)"],
            ["cmd", "`whoami`"],
            ["cmd", "arg1 && malicious"],
            ["cmd", "file.txt\nrm -rf /"],
        ]
        
        for args in dangerous_args:
            with self.assertRaises(InputValidationError, msg=f"Should reject: {args}"):
                validate_command_args(args, allow_shell_chars=False)
    
    def test_rejects_null_bytes_in_args(self):
        """Test rejection of null bytes in arguments."""
        with self.assertRaises(InputValidationError):
            validate_command_args(["cmd", "arg\x00injection"])
    
    def test_accepts_valid_arguments(self):
        """Test acceptance of valid command arguments."""
        valid_args = [
            ["git", "status", "--porcelain"],
            ["ls", "-la", "/home/user"],
            ["python3", "script.py", "--output=file.txt"],
        ]
        
        for args in valid_args:
            result = validate_command_args(args)
            self.assertEqual(result, args)


class TestStringSanitization(unittest.TestCase):
    """Test string sanitization security."""
    
    def test_removes_null_bytes(self):
        """Test removal of null bytes from strings."""
        result = sanitize_string("hello\x00world")
        self.assertEqual(result, "helloworld")
    
    def test_removes_control_characters(self):
        """Test removal of control characters."""
        result = sanitize_string("hello\x07\x08world", allow_newlines=False)
        self.assertNotIn("\x07", result)
        self.assertNotIn("\x08", result)
    
    def test_truncates_long_strings(self):
        """Test truncation of strings exceeding max length."""
        long_string = "x" * 2000
        result = sanitize_string(long_string, max_length=100)
        self.assertEqual(len(result), 100)
    
    def test_preserves_newlines_when_allowed(self):
        """Test preservation of newlines when allowed."""
        result = sanitize_string("line1\nline2", allow_newlines=True)
        self.assertIn("\n", result)


if __name__ == "__main__":
    unittest.main()
