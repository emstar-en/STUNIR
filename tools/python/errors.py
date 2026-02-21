#!/usr/bin/env python3
"""
STUNIR Error System
==================

A comprehensive error handling system with:
- Error codes for categorization
- User-friendly error messages
- Actionable suggestions
- Context preservation
- Documentation links

Error Code Ranges:
- E1xxx: IO/File errors
- E2xxx: JSON/Serialization errors
- E3xxx: Validation errors
- E4xxx: Verification errors
- E5xxx: Usage/CLI errors
- E6xxx: Security errors
- E7xxx: Configuration errors
- E8xxx: Network errors
- E9xxx: Internal errors

Usage:
    from tools.errors import StunirError, IOError, ValidationError
    
    # Raise with context
    raise ValidationError(
        "E3001",
        "Invalid module name",
        field="spec.modules[0].name",
        value="",
        suggestion="Module names must be non-empty identifiers"
    )
    
    # Handle errors gracefully
    try:
        process_spec(spec)
    except StunirError as e:
        print(f"Error {e.code}: {e.message}")
        print(f"Suggestion: {e.suggestion}")
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import json
import sys
import traceback


# ============================================================================
# Error Code Registry
# ============================================================================

ERROR_CODES: Dict[str, Dict[str, str]] = {
    # IO Errors (E1xxx)
    "E1000": {"category": "IO", "message": "Generic IO error"},
    "E1001": {"category": "IO", "message": "File not found"},
    "E1002": {"category": "IO", "message": "Permission denied"},
    "E1003": {"category": "IO", "message": "Directory not found"},
    "E1004": {"category": "IO", "message": "File already exists"},
    "E1005": {"category": "IO", "message": "Disk full"},
    "E1006": {"category": "IO", "message": "Invalid file path"},
    "E1007": {"category": "IO", "message": "File read error"},
    "E1008": {"category": "IO", "message": "File write error"},
    
    # JSON Errors (E2xxx)
    "E2000": {"category": "JSON", "message": "Generic JSON error"},
    "E2001": {"category": "JSON", "message": "Invalid JSON syntax"},
    "E2002": {"category": "JSON", "message": "Missing required field"},
    "E2003": {"category": "JSON", "message": "Invalid field type"},
    "E2004": {"category": "JSON", "message": "Unexpected field"},
    "E2005": {"category": "JSON", "message": "JSON encoding error"},
    "E2006": {"category": "JSON", "message": "Canonicalization failed"},
    
    # Validation Errors (E3xxx)
    "E3000": {"category": "Validation", "message": "Generic validation error"},
    "E3001": {"category": "Validation", "message": "Invalid identifier"},
    "E3002": {"category": "Validation", "message": "Value out of range"},
    "E3003": {"category": "Validation", "message": "Invalid format"},
    "E3004": {"category": "Validation", "message": "Missing required value"},
    "E3005": {"category": "Validation", "message": "Duplicate entry"},
    "E3006": {"category": "Validation", "message": "Invalid reference"},
    "E3007": {"category": "Validation", "message": "Schema validation failed"},
    
    # Verification Errors (E4xxx)
    "E4000": {"category": "Verification", "message": "Generic verification error"},
    "E4001": {"category": "Verification", "message": "Hash mismatch"},
    "E4002": {"category": "Verification", "message": "Signature invalid"},
    "E4003": {"category": "Verification", "message": "Manifest entry missing"},
    "E4004": {"category": "Verification", "message": "Receipt verification failed"},
    "E4005": {"category": "Verification", "message": "Integrity check failed"},
    
    # Usage Errors (E5xxx)
    "E5000": {"category": "Usage", "message": "Generic usage error"},
    "E5001": {"category": "Usage", "message": "Missing required argument"},
    "E5002": {"category": "Usage", "message": "Invalid argument value"},
    "E5003": {"category": "Usage", "message": "Unknown command"},
    "E5004": {"category": "Usage", "message": "Conflicting options"},
    
    # Security Errors (E6xxx)
    "E6000": {"category": "Security", "message": "Generic security error"},
    "E6001": {"category": "Security", "message": "Path traversal detected"},
    "E6002": {"category": "Security", "message": "Symlink following blocked"},
    "E6003": {"category": "Security", "message": "File size limit exceeded"},
    "E6004": {"category": "Security", "message": "Untrusted input"},
    "E6005": {"category": "Security", "message": "Unauthorized operation"},
    
    # Configuration Errors (E7xxx)
    "E7000": {"category": "Configuration", "message": "Generic configuration error"},
    "E7001": {"category": "Configuration", "message": "Invalid config file"},
    "E7002": {"category": "Configuration", "message": "Missing config section"},
    "E7003": {"category": "Configuration", "message": "Invalid config value"},
    "E7004": {"category": "Configuration", "message": "Environment variable not set"},
    
    # Network Errors (E8xxx)
    "E8000": {"category": "Network", "message": "Generic network error"},
    "E8001": {"category": "Network", "message": "Connection failed"},
    "E8002": {"category": "Network", "message": "Timeout"},
    "E8003": {"category": "Network", "message": "DNS resolution failed"},
    
    # Internal Errors (E9xxx)
    "E9000": {"category": "Internal", "message": "Generic internal error"},
    "E9001": {"category": "Internal", "message": "Assertion failed"},
    "E9002": {"category": "Internal", "message": "Not implemented"},
    "E9003": {"category": "Internal", "message": "Unexpected state"},
}


# ============================================================================
# Suggestion Registry
# ============================================================================

SUGGESTIONS: Dict[str, str] = {
    # IO suggestions
    "E1001": "Check that the file path is correct and the file exists.",
    "E1002": "Check file permissions or run with appropriate privileges.",
    "E1003": "Create the directory or check the path spelling.",
    "E1004": "Use --force to overwrite or choose a different filename.",
    "E1005": "Free up disk space or choose a different location.",
    "E1006": "Check for special characters or path length limits.",
    
    # JSON suggestions
    "E2001": "Validate JSON syntax with: python -m json.tool <file>",
    "E2002": "Add the missing field to your JSON document.",
    "E2003": "Check field type - expected type shown in error message.",
    "E2005": "Ensure all strings are valid UTF-8.",
    
    # Validation suggestions
    "E3001": "Use only alphanumeric characters and underscores.",
    "E3002": "Adjust value to be within the allowed range.",
    "E3003": "Check the expected format in the documentation.",
    "E3004": "Provide all required values.",
    "E3005": "Remove duplicate entries or use unique identifiers.",
    "E3007": "Run: stunir validate --schema <type> <file>",
    
    # Verification suggestions
    "E4001": "The file has been modified. Re-run the build to update hashes.",
    "E4002": "Verify the signing key is correct.",
    "E4003": "Regenerate the manifest: stunir manifest generate",
    "E4004": "Re-generate receipts: stunir build --receipts",
    
    # Usage suggestions
    "E5001": "Run with --help to see required arguments.",
    "E5002": "Check allowed values in the documentation.",
    "E5003": "Run 'stunir --help' to see available commands.",
    "E5004": "Remove one of the conflicting options.",
    
    # Security suggestions
    "E6001": "Use absolute paths or paths within the project directory.",
    "E6002": "Replace symlinks with actual files for hashing.",
    "E6003": "Split large files or increase size limit in config.",
    "E6004": "Sanitize input before processing.",
    
    # Configuration suggestions
    "E7001": "Validate config file format: stunir config validate",
    "E7002": "Add the missing section to your config file.",
    "E7003": "Check allowed values in documentation.",
    "E7004": "Set the environment variable: export VAR_NAME=value",
}


# ============================================================================
# Base Error Classes
# ============================================================================

@dataclass
class StunirError(Exception):
    """Base class for all STUNIR errors.
    
    Attributes:
        code: Error code (e.g., "E3001")
        message: Human-readable error message
        context: Additional context about the error
        suggestion: Actionable suggestion for fixing the error
        cause: Original exception that caused this error
        
    Examples:
        >>> err = StunirError("E3001", "Invalid name", field="name")
        >>> print(err)
        [E3001] Invalid name (field=name)
    """
    code: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    suggestion: Optional[str] = None
    cause: Optional[Exception] = None
    
    def __post_init__(self):
        """Initialize suggestion from registry if not provided."""
        if self.suggestion is None:
            self.suggestion = SUGGESTIONS.get(self.code, "")
        super().__init__(str(self))
    
    def __str__(self) -> str:
        """Format error for display."""
        parts = [f"[{self.code}] {self.message}"]
        
        if self.context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"({ctx_str})")
            
        return " ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for JSON serialization."""
        result = {
            "code": self.code,
            "message": self.message,
            "category": ERROR_CODES.get(self.code, {}).get("category", "Unknown"),
        }
        
        if self.context:
            result["context"] = self.context
            
        if self.suggestion:
            result["suggestion"] = self.suggestion
            
        if self.cause:
            result["cause"] = str(self.cause)
            
        return result
    
    def to_json(self) -> str:
        """Serialize error to JSON."""
        return json.dumps(self.to_dict(), indent=2)
    
    @property
    def category(self) -> str:
        """Get error category from code."""
        return ERROR_CODES.get(self.code, {}).get("category", "Unknown")
    
    @property
    def is_recoverable(self) -> bool:
        """Check if error is user-recoverable."""
        recoverable_categories = {"Usage", "Validation", "Configuration"}
        return self.category in recoverable_categories


# ============================================================================
# Specific Error Types
# ============================================================================

class IOError(StunirError):
    """IO/File system errors."""
    
    def __init__(self, code: str, message: str, path: Optional[str] = None, **kwargs):
        context = kwargs.pop("context", {})
        if path:
            context["path"] = path
        super().__init__(code, message, context=context, **kwargs)


class JSONError(StunirError):
    """JSON parsing/serialization errors."""
    
    def __init__(self, code: str, message: str, 
                 field: Optional[str] = None,
                 line: Optional[int] = None,
                 column: Optional[int] = None,
                 **kwargs):
        context = kwargs.pop("context", {})
        if field:
            context["field"] = field
        if line is not None:
            context["line"] = line
        if column is not None:
            context["column"] = column
        super().__init__(code, message, context=context, **kwargs)


class ValidationError(StunirError):
    """Input validation errors."""
    
    def __init__(self, code: str, message: str,
                 field: Optional[str] = None,
                 value: Optional[Any] = None,
                 expected: Optional[str] = None,
                 **kwargs):
        context = kwargs.pop("context", {})
        if field:
            context["field"] = field
        if value is not None:
            context["value"] = repr(value)
        if expected:
            context["expected"] = expected
        super().__init__(code, message, context=context, **kwargs)


class VerificationError(StunirError):
    """Verification/integrity check errors."""
    
    def __init__(self, code: str, message: str,
                 expected_hash: Optional[str] = None,
                 actual_hash: Optional[str] = None,
                 path: Optional[str] = None,
                 **kwargs):
        context = kwargs.pop("context", {})
        if expected_hash:
            context["expected"] = expected_hash[:16] + "..."
        if actual_hash:
            context["actual"] = actual_hash[:16] + "..."
        if path:
            context["path"] = path
        super().__init__(code, message, context=context, **kwargs)


class SecurityError(StunirError):
    """Security violation errors."""
    
    def __init__(self, code: str, message: str,
                 path: Optional[str] = None,
                 operation: Optional[str] = None,
                 **kwargs):
        context = kwargs.pop("context", {})
        if path:
            context["path"] = path
        if operation:
            context["operation"] = operation
        super().__init__(code, message, context=context, **kwargs)


class UsageError(StunirError):
    """CLI usage errors."""
    
    def __init__(self, code: str, message: str,
                 argument: Optional[str] = None,
                 command: Optional[str] = None,
                 **kwargs):
        context = kwargs.pop("context", {})
        if argument:
            context["argument"] = argument
        if command:
            context["command"] = command
        super().__init__(code, message, context=context, **kwargs)


class ConfigError(StunirError):
    """Configuration errors."""
    
    def __init__(self, code: str, message: str,
                 key: Optional[str] = None,
                 file: Optional[str] = None,
                 **kwargs):
        context = kwargs.pop("context", {})
        if key:
            context["key"] = key
        if file:
            context["file"] = file
        super().__init__(code, message, context=context, **kwargs)


# ============================================================================
# Error Handler
# ============================================================================

class ErrorHandler:
    """Central error handling and reporting.
    
    Provides consistent error formatting, logging, and exit behavior.
    
    Examples:
        >>> handler = ErrorHandler(verbose=True)
        >>> try:
        ...     risky_operation()
        ... except StunirError as e:
        ...     handler.handle(e)
    """
    
    def __init__(self, verbose: bool = False, json_output: bool = False):
        self.verbose = verbose
        self.json_output = json_output
        self.errors: List[StunirError] = []
    
    def handle(self, error: StunirError, exit_code: int = 1) -> None:
        """Handle an error: log, format, and optionally exit."""
        self.errors.append(error)
        
        if self.json_output:
            print(error.to_json(), file=sys.stderr)
        else:
            self._print_error(error)
        
        if exit_code > 0:
            sys.exit(exit_code)
    
    def _print_error(self, error: StunirError) -> None:
        """Print formatted error to stderr."""
        # Header
        print(f"\n\033[1;31m✗ Error {error.code}\033[0m: {error.message}", file=sys.stderr)
        
        # Context
        if error.context:
            print("\n  Context:", file=sys.stderr)
            for key, value in error.context.items():
                print(f"    • {key}: {value}", file=sys.stderr)
        
        # Suggestion
        if error.suggestion:
            print(f"\n  \033[1;33m💡 Suggestion\033[0m: {error.suggestion}", file=sys.stderr)
        
        # Cause (verbose only)
        if self.verbose and error.cause:
            print(f"\n  Caused by: {error.cause}", file=sys.stderr)
            if hasattr(error.cause, "__traceback__"):
                print("\n  Traceback:", file=sys.stderr)
                for line in traceback.format_tb(error.cause.__traceback__):
                    print(f"    {line.strip()}", file=sys.stderr)
        
        print("", file=sys.stderr)
    
    def collect(self, error: StunirError) -> None:
        """Collect error without exiting (for batch operations)."""
        self.errors.append(error)
    
    def has_errors(self) -> bool:
        """Check if any errors were collected."""
        return len(self.errors) > 0
    
    def report(self) -> None:
        """Report all collected errors."""
        if not self.errors:
            return
            
        print(f"\n\033[1;31m{len(self.errors)} error(s) occurred:\033[0m\n", file=sys.stderr)
        for i, error in enumerate(self.errors, 1):
            print(f"  {i}. [{error.code}] {error.message}", file=sys.stderr)
        print("", file=sys.stderr)


# ============================================================================
# Utility Functions
# ============================================================================

def wrap_exception(exc: Exception, code: str, message: str) -> StunirError:
    """Wrap a standard exception as a StunirError.
    
    Examples:
        >>> try:
        ...     open("missing.txt")
        ... except FileNotFoundError as e:
        ...     raise wrap_exception(e, "E1001", "Config file not found")
    """
    return StunirError(code, message, cause=exc)


def from_code(code: str, **kwargs) -> StunirError:
    """Create error from code with default message.
    
    Examples:
        >>> err = from_code("E1001", path="/missing/file")
        >>> print(err.message)
        File not found
    """
    info = ERROR_CODES.get(code, {"message": "Unknown error"})
    return StunirError(code, info["message"], **kwargs)


# ============================================================================
# Common Error Examples
# ============================================================================

COMMON_ERRORS = """
Common STUNIR Errors and Solutions
==================================

E1001 - File Not Found
----------------------
Problem: A required file does not exist.
Example: stunir build spec.json
Solution: Check the file path and ensure the file exists.

E2001 - Invalid JSON Syntax
---------------------------
Problem: JSON file contains syntax errors.
Example: {"key": value} (missing quotes around value)
Solution: Validate with: python -m json.tool <file>

E3001 - Invalid Identifier
--------------------------
Problem: An identifier contains invalid characters.
Example: Module name "my-module" (hyphens not allowed)
Solution: Use only letters, numbers, and underscores.

E4001 - Hash Mismatch
---------------------
Problem: File content doesn't match recorded hash.
Example: File was modified after build.
Solution: Re-run build to update receipts.

E6001 - Path Traversal Detected
-------------------------------
Problem: Path contains "../" trying to escape project directory.
Example: ../../etc/passwd
Solution: Use absolute paths or paths within project.
"""


if __name__ == "__main__":
    # Demo error handling
    print("STUNIR Error System Demo\n")
    
    # Create sample errors
    errors = [
        ValidationError("E3001", "Invalid module name", field="spec.modules[0].name", value=""),
        IOError("E1001", "Config file not found", path="/etc/stunir/config.json"),
        VerificationError("E4001", "Hash mismatch", 
                         expected_hash="abc123...", 
                         actual_hash="def456...",
                         path="output.json"),
    ]
    
    handler = ErrorHandler(verbose=True)
    
    for err in errors:
        print(f"Error: {err}")
        print(f"  Category: {err.category}")
        print(f"  Recoverable: {err.is_recoverable}")
        print(f"  Suggestion: {err.suggestion}")
        print()
