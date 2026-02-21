"""STUNIR Security Utilities Module.

This module provides security utilities for:
- Input validation (paths, arguments, JSON/CBOR data)
- Subprocess sanitization
- Exception handling with proper logging
- File size and content validation

Security Guidelines:
- Always use these utilities instead of raw input handling
- Never use shell=True with subprocess
- Always validate paths before file operations
- Use specific exception types instead of bare except

Example usage:
    from tools.security import validate_path, run_command, SecurityError
    
    # Validate a path before use
    from pathlib import Path
    safe_path = validate_path(user_input, base_dir=str(Path.home() / "stunir_repo"))
    
    # Run a command safely
    result = run_command(["ls", "-la", str(safe_path)])
"""

from .validation import (
    validate_path,
    validate_json_input,
    validate_file_size,
    validate_command_args,
    sanitize_string,
    PathValidationError,
    InputValidationError,
)

from .subprocess_utils import (
    run_command,
    run_command_with_timeout,
    SubprocessError,
)

from .exceptions import (
    SecurityError,
    PathTraversalError,
    FileSizeError,
    InvalidInputError,
    CommandInjectionError,
)

__all__ = [
    # Validation
    "validate_path",
    "validate_json_input",
    "validate_file_size",
    "validate_command_args",
    "sanitize_string",
    "PathValidationError",
    "InputValidationError",
    # Subprocess
    "run_command",
    "run_command_with_timeout",
    "SubprocessError",
    # Exceptions
    "SecurityError",
    "PathTraversalError",
    "FileSizeError",
    "InvalidInputError",
    "CommandInjectionError",
]
