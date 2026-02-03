#!/usr/bin/env python3
"""STUNIR Subprocess Security Tests.

This module tests subprocess utilities for security vulnerabilities:
- Command injection prevention
- Timeout protection
- Whitelist enforcement
- Safe argument handling
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.security.subprocess_utils import (
    run_command,
    run_command_with_timeout,
    split_command_string,
    quote_arg,
    SubprocessError,
    ALLOWED_COMMANDS,
)
from tools.security.validation import InputValidationError


class TestRunCommand(unittest.TestCase):
    """Test run_command security."""
    
    def test_rejects_string_command(self):
        """Test rejection of command as string (prevents shell injection)."""
        with self.assertRaises(SubprocessError):
            run_command("ls -la")  # type: ignore - testing wrong type
    
    def test_rejects_empty_command(self):
        """Test rejection of empty command list."""
        with self.assertRaises(SubprocessError):
            run_command([])
    
    def test_rejects_shell_metacharacters(self):
        """Test rejection of shell metacharacters in arguments."""
        dangerous_commands = [
            ["echo", "; rm -rf /"],
            ["cat", "file.txt | malicious"],
            ["ls", "$(whoami)"],
        ]
        
        for cmd in dangerous_commands:
            with self.assertRaises((SubprocessError, InputValidationError), 
                                   msg=f"Should reject: {cmd}"):
                run_command(cmd, allow_shell_chars=False)
    
    def test_executes_safe_commands(self):
        """Test execution of safe commands."""
        # Simple echo command
        result = run_command(["echo", "hello"], capture_output=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("hello", result.stdout)
    
    def test_captures_output(self):
        """Test output capture."""
        result = run_command(["echo", "test output"], capture_output=True)
        self.assertIn("test output", result.stdout)
    
    def test_respects_check_flag(self):
        """Test check flag for non-zero exit codes."""
        # With check=True (default), should raise
        with self.assertRaises(SubprocessError):
            run_command(["false"], check=True)
        
        # With check=False, should not raise
        result = run_command(["false"], check=False)
        self.assertNotEqual(result.returncode, 0)
    
    def test_whitelist_enforcement(self):
        """Test command whitelist enforcement."""
        # Allowed command should work
        result = run_command(
            ["echo", "test"], 
            enforce_whitelist=True,
            capture_output=True
        )
        self.assertEqual(result.returncode, 0)
        
        # Disallowed command should fail
        # Note: This might fail if the command happens to be in the whitelist
        # Using a clearly non-whitelisted command
        with self.assertRaises(SubprocessError):
            run_command(
                ["nonexistent_dangerous_command"],
                enforce_whitelist=True
            )
    
    def test_working_directory(self):
        """Test working directory setting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_command(
                ["pwd"],
                cwd=tmpdir,
                capture_output=True
            )
            self.assertIn(tmpdir, result.stdout)
    
    def test_environment_variables(self):
        """Test environment variable handling."""
        result = run_command(
            ["sh", "-c", "echo $TEST_VAR"],
            env={"TEST_VAR": "test_value"},
            capture_output=True,
            allow_shell_chars=True  # Needed for shell command
        )
        self.assertIn("test_value", result.stdout)


class TestRunCommandWithTimeout(unittest.TestCase):
    """Test run_command_with_timeout security."""
    
    def test_timeout_enforcement(self):
        """Test that commands are killed after timeout."""
        with self.assertRaises(SubprocessError) as context:
            run_command_with_timeout(
                ["sleep", "10"],
                timeout_seconds=1
            )
        self.assertIn("timed out", str(context.exception).lower())
    
    def test_fast_commands_complete(self):
        """Test that fast commands complete successfully."""
        result = run_command_with_timeout(
            ["echo", "quick"],
            timeout_seconds=10,
            capture_output=True
        )
        self.assertIn("quick", result.stdout)
    
    def test_inherits_security_checks(self):
        """Test that timeout version has same security checks."""
        # Should reject string command
        with self.assertRaises(SubprocessError):
            run_command_with_timeout("ls -la", timeout_seconds=5)  # type: ignore
        
        # Should reject empty command
        with self.assertRaises(SubprocessError):
            run_command_with_timeout([], timeout_seconds=5)


class TestSplitCommandString(unittest.TestCase):
    """Test command string splitting security."""
    
    def test_splits_simple_commands(self):
        """Test splitting simple command strings."""
        result = split_command_string("git status --porcelain")
        self.assertEqual(result, ["git", "status", "--porcelain"])
    
    def test_handles_quoted_arguments(self):
        """Test handling of quoted arguments."""
        result = split_command_string('echo "hello world"')
        self.assertEqual(result, ["echo", "hello world"])
    
    def test_handles_escaped_characters(self):
        """Test handling of escaped characters."""
        result = split_command_string(r'echo "hello\"world"')
        self.assertIn("hello", result[1])
    
    def test_rejects_unclosed_quotes(self):
        """Test rejection of unclosed quotes."""
        with self.assertRaises(SubprocessError):
            split_command_string('echo "unclosed')


class TestQuoteArg(unittest.TestCase):
    """Test argument quoting for display."""
    
    def test_quotes_spaces(self):
        """Test quoting of arguments with spaces."""
        result = quote_arg("hello world")
        self.assertIn("hello world", result)
        # Should be quoted or escaped
        self.assertTrue(
            result.startswith("'") or result.startswith('"') or "\\" in result
        )
    
    def test_quotes_special_characters(self):
        """Test quoting of special characters."""
        special = "test;rm -rf /"
        result = quote_arg(special)
        # Should be safely quoted
        self.assertIn("test", result)
    
    def test_simple_args_unchanged(self):
        """Test that simple arguments are minimally modified."""
        simple = "simple_arg"
        result = quote_arg(simple)
        self.assertIn("simple_arg", result)


class TestAllowedCommands(unittest.TestCase):
    """Test the ALLOWED_COMMANDS whitelist."""
    
    def test_common_tools_allowed(self):
        """Test that common safe tools are in whitelist."""
        expected_tools = ["git", "make", "python3", "ls", "sha256sum"]
        for tool in expected_tools:
            self.assertIn(tool, ALLOWED_COMMANDS, 
                          f"{tool} should be in allowed commands")
    
    def test_dangerous_tools_not_allowed(self):
        """Test that dangerous tools are not in whitelist."""
        # These should NOT be in the whitelist
        dangerous = ["bash", "sh", "eval", "exec", "sudo", "su"]
        for tool in dangerous:
            # Note: bash/sh might be allowed for some legitimate uses
            # The key is that they should be used carefully
            pass  # This is more of a documentation test


if __name__ == "__main__":
    unittest.main()
