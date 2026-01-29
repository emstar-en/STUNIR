"""STUNIR Input Validation Utilities.

This module provides comprehensive input validation for:
- File paths (preventing directory traversal)
- JSON/CBOR data (schema validation)
- Command-line arguments (type checking, range validation)
- File sizes (DoS prevention)

Security Guidelines:
- Always validate paths before any file operation
- Never trust user input directly
- Validate JSON schema before processing
- Check file sizes before reading into memory

Example:
    from tools.security.validation import validate_path, validate_json_input
    
    # Validate a user-provided path
    safe_path = validate_path(user_input, base_dir="/repo", allow_absolute=False)
    
    # Validate JSON against a schema
    data = validate_json_input(json_bytes, schema={"type": "object"})
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .exceptions import (
    PathTraversalError,
    InvalidInputError as _InvalidInputError,
    FileSizeError,
)


# Default limits
DEFAULT_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
DEFAULT_MAX_JSON_DEPTH = 50
DEFAULT_MAX_STRING_LENGTH = 1_000_000

# Patterns for detecting potentially malicious input
SHELL_METACHAR_PATTERN = re.compile(r'[;&|`$(){}[\]<>!\\"\'\n\r\t\0]')
NULL_BYTE_PATTERN = re.compile(r'\x00')
# Path traversal patterns (catches both Unix and Windows styles)
PATH_TRAVERSAL_PATTERN = re.compile(r'(^|[/\\])\.{2,}([/\\]|$)')  # .., ..., etc.
BACKSLASH_PATTERN = re.compile(r'\\')


class PathValidationError(PathTraversalError):
    """Alias for PathTraversalError for API compatibility."""
    pass


class InputValidationError(_InvalidInputError):
    """Alias for InvalidInputError for API compatibility."""
    pass


# Re-export the base class for backwards compatibility
InvalidInputError = InputValidationError


def validate_path(
    path: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None,
    allow_absolute: bool = False,
    must_exist: bool = False,
    allow_symlinks: bool = False,
    allowed_extensions: Optional[List[str]] = None,
) -> Path:
    """Validate a file path for security.
    
    Args:
        path: The path to validate
        base_dir: If provided, ensure path is within this directory
        allow_absolute: Whether to allow absolute paths
        must_exist: Whether the path must exist
        allow_symlinks: Whether to allow symlinks (default: False for security)
        allowed_extensions: List of allowed file extensions (e.g., ['.json', '.py'])
    
    Returns:
        A validated Path object
    
    Raises:
        PathValidationError: If path fails validation
    
    Example:
        >>> validate_path("../etc/passwd")  # Raises PathValidationError
        >>> validate_path("data/file.json", base_dir="/repo")  # Returns Path
    """
    if not path:
        raise PathValidationError("Empty path not allowed", path=str(path))
    
    path_str = str(path)
    
    # Check for null bytes (common injection technique)
    if NULL_BYTE_PATTERN.search(path_str):
        raise PathValidationError(
            "Null bytes not allowed in path",
            path=path_str.replace('\0', '\\0')
        )
    
    # Check for backslashes (Windows-style separators - potential cross-platform attack)
    if BACKSLASH_PATTERN.search(path_str):
        raise PathValidationError(
            "Backslash characters not allowed in path (use forward slashes)",
            path=path_str
        )
    
    # Check for path traversal patterns (includes .., ..., etc.)
    if PATH_TRAVERSAL_PATTERN.search(path_str):
        raise PathValidationError(
            "Path traversal detected: '..' or similar patterns not allowed",
            path=path_str
        )
    
    path_obj = Path(path_str)
    
    # Check for absolute paths
    if path_obj.is_absolute() and not allow_absolute:
        raise PathValidationError(
            "Absolute paths not allowed",
            path=path_str
        )
    
    # Normalize path and check for traversal
    try:
        # Use resolve() to normalize the path
        if base_dir:
            base_path = Path(base_dir).resolve()
            full_path = (base_path / path_obj).resolve()
            
            # Ensure resolved path is within base directory
            try:
                full_path.relative_to(base_path)
            except ValueError:
                raise PathValidationError(
                    "Path traversal detected: path escapes base directory",
                    path=path_str,
                    base_dir=str(base_dir)
                )
        else:
            # Check for explicit traversal patterns
            normalized = os.path.normpath(path_str)
            if normalized.startswith('..') or '/../' in normalized or normalized.startswith('/..'):
                raise PathValidationError(
                    "Path traversal detected: '..' not allowed",
                    path=path_str
                )
            full_path = path_obj.resolve() if path_obj.exists() else path_obj
    except OSError as e:
        raise PathValidationError(f"Invalid path: {e}", path=path_str)
    
    # Check symlinks
    if not allow_symlinks and (path_obj.is_symlink() if path_obj.exists() else False):
        raise PathValidationError(
            "Symlinks not allowed for security",
            path=path_str
        )
    
    # Check existence
    if must_exist and not path_obj.exists():
        raise PathValidationError(
            "Path does not exist",
            path=path_str
        )
    
    # Check extension
    if allowed_extensions:
        ext = path_obj.suffix.lower()
        if ext not in [e.lower() for e in allowed_extensions]:
            raise PathValidationError(
                f"File extension '{ext}' not allowed. Allowed: {allowed_extensions}",
                path=path_str
            )
    
    return full_path if base_dir else path_obj


def validate_file_size(
    path: Union[str, Path],
    max_size: int = DEFAULT_MAX_FILE_SIZE,
) -> int:
    """Validate file size is within limits.
    
    Args:
        path: Path to the file
        max_size: Maximum allowed size in bytes
    
    Returns:
        The actual file size
    
    Raises:
        FileSizeError: If file is too large
        PathValidationError: If file doesn't exist
    """
    path_obj = Path(path)
    
    if not path_obj.exists():
        raise PathValidationError(f"File does not exist: {path}")
    
    if not path_obj.is_file():
        raise PathValidationError(f"Not a regular file: {path}")
    
    size = path_obj.stat().st_size
    
    if size > max_size:
        raise FileSizeError(
            f"File too large: {size} bytes (max: {max_size})",
            actual_size=size,
            max_size=max_size,
            path=str(path)
        )
    
    return size


def validate_json_input(
    data: Union[bytes, str, Dict[str, Any]],
    schema: Optional[Dict[str, Any]] = None,
    max_depth: int = DEFAULT_MAX_JSON_DEPTH,
    max_string_length: int = DEFAULT_MAX_STRING_LENGTH,
) -> Dict[str, Any]:
    """Validate JSON input for security.
    
    Args:
        data: JSON bytes, string, or already-parsed dict
        schema: Optional JSON schema for validation
        max_depth: Maximum nesting depth (prevents stack overflow)
        max_string_length: Maximum string value length (prevents memory exhaustion)
    
    Returns:
        Validated and parsed JSON data
    
    Raises:
        InvalidInputError: If JSON is invalid or fails validation
    """
    # Parse if needed
    if isinstance(data, bytes):
        try:
            data = data.decode('utf-8')
        except UnicodeDecodeError as e:
            raise InvalidInputError(
                f"Invalid UTF-8 in JSON: {e}",
                input_type="json"
            )
    
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as e:
            raise InvalidInputError(
                f"Invalid JSON syntax: {e}",
                input_type="json"
            )
    else:
        parsed = data
    
    # Check depth
    def check_depth(obj: Any, depth: int = 0) -> None:
        if depth > max_depth:
            raise InvalidInputError(
                f"JSON too deeply nested (max depth: {max_depth})",
                input_type="json",
                expected=f"depth <= {max_depth}",
                actual=f"depth > {depth}"
            )
        if isinstance(obj, dict):
            for v in obj.values():
                check_depth(v, depth + 1)
        elif isinstance(obj, list):
            for item in obj:
                check_depth(item, depth + 1)
    
    check_depth(parsed)
    
    # Check string lengths
    def check_strings(obj: Any) -> None:
        if isinstance(obj, str) and len(obj) > max_string_length:
            raise InvalidInputError(
                f"String value too long (max: {max_string_length})",
                input_type="json",
                expected=f"length <= {max_string_length}",
                actual=f"length = {len(obj)}"
            )
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if len(k) > max_string_length:
                    raise InvalidInputError(
                        f"JSON key too long (max: {max_string_length})",
                        input_type="json"
                    )
                check_strings(v)
        elif isinstance(obj, list):
            for item in obj:
                check_strings(item)
    
    check_strings(parsed)
    
    # Basic schema validation if provided
    if schema:
        _validate_schema(parsed, schema)
    
    return parsed


def _validate_schema(data: Any, schema: Dict[str, Any], path: str = "$") -> None:
    """Basic JSON schema validation.
    
    This is a simplified validator. For production use, consider jsonschema library.
    """
    schema_type = schema.get("type")
    
    if schema_type == "object":
        if not isinstance(data, dict):
            raise InvalidInputError(
                f"Expected object at {path}",
                input_type="json",
                expected="object",
                actual=type(data).__name__
            )
        
        # Check required fields
        required = schema.get("required", [])
        for field in required:
            if field not in data:
                raise InvalidInputError(
                    f"Missing required field '{field}' at {path}",
                    input_type="json"
                )
        
        # Validate properties
        properties = schema.get("properties", {})
        for key, value in data.items():
            if key in properties:
                _validate_schema(value, properties[key], f"{path}.{key}")
    
    elif schema_type == "array":
        if not isinstance(data, list):
            raise InvalidInputError(
                f"Expected array at {path}",
                input_type="json",
                expected="array",
                actual=type(data).__name__
            )
        
        # Validate items
        items_schema = schema.get("items")
        if items_schema:
            for i, item in enumerate(data):
                _validate_schema(item, items_schema, f"{path}[{i}]")
    
    elif schema_type == "string":
        if not isinstance(data, str):
            raise InvalidInputError(
                f"Expected string at {path}",
                input_type="json",
                expected="string",
                actual=type(data).__name__
            )
    
    elif schema_type == "integer":
        if not isinstance(data, int) or isinstance(data, bool):
            raise InvalidInputError(
                f"Expected integer at {path}",
                input_type="json",
                expected="integer",
                actual=type(data).__name__
            )
    
    elif schema_type == "number":
        if not isinstance(data, (int, float)) or isinstance(data, bool):
            raise InvalidInputError(
                f"Expected number at {path}",
                input_type="json",
                expected="number",
                actual=type(data).__name__
            )
    
    elif schema_type == "boolean":
        if not isinstance(data, bool):
            raise InvalidInputError(
                f"Expected boolean at {path}",
                input_type="json",
                expected="boolean",
                actual=type(data).__name__
            )


def validate_command_args(
    args: List[str],
    allow_shell_chars: bool = False,
) -> List[str]:
    """Validate command-line arguments for security.
    
    Args:
        args: List of command arguments
        allow_shell_chars: Whether to allow shell metacharacters
    
    Returns:
        Validated arguments
    
    Raises:
        InvalidInputError: If arguments contain dangerous characters
    """
    validated = []
    
    for i, arg in enumerate(args):
        if not isinstance(arg, str):
            raise InvalidInputError(
                f"Argument {i} must be string, got {type(arg).__name__}",
                input_type="command_arg"
            )
        
        # Check for null bytes
        if '\0' in arg:
            raise InvalidInputError(
                f"Null bytes not allowed in argument {i}",
                input_type="command_arg"
            )
        
        # Check for shell metacharacters
        if not allow_shell_chars and SHELL_METACHAR_PATTERN.search(arg):
            raise InvalidInputError(
                f"Shell metacharacters not allowed in argument {i}",
                input_type="command_arg",
                actual=arg[:50]
            )
        
        validated.append(arg)
    
    return validated


def sanitize_string(
    s: str,
    max_length: int = 1000,
    allow_newlines: bool = False,
) -> str:
    """Sanitize a string for safe use.
    
    Args:
        s: String to sanitize
        max_length: Maximum allowed length
        allow_newlines: Whether to allow newline characters
    
    Returns:
        Sanitized string
    """
    # Remove null bytes
    s = s.replace('\0', '')
    
    # Remove control characters except newlines/tabs if allowed
    if allow_newlines:
        s = ''.join(c for c in s if c.isprintable() or c in '\n\t\r')
    else:
        s = ''.join(c for c in s if c.isprintable() or c == '\t')
    
    # Truncate
    if len(s) > max_length:
        s = s[:max_length]
    
    return s
