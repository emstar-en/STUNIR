"""STUNIR Security Exception Hierarchy.

This module defines a hierarchy of security-related exceptions that should be
used instead of generic exceptions. This enables:
- Specific error handling and recovery
- Proper logging with security context
- Clear distinction between security and operational errors

Usage:
    from tools.security.exceptions import PathTraversalError
    
    if ".." in user_path:
        raise PathTraversalError(f"Path traversal attempt detected: {user_path}")
"""

import logging
from typing import Optional, Any


# Configure security logger
security_logger = logging.getLogger("stunir.security")


class SecurityError(Exception):
    """Base class for all STUNIR security exceptions.
    
    All security-related exceptions should inherit from this class.
    This allows catching all security issues with a single except clause
    when appropriate, while still enabling specific handling.
    """
    
    def __init__(self, message: str, context: Optional[dict] = None):
        """Initialize security error.
        
        Args:
            message: Human-readable error description
            context: Optional dictionary with additional context for logging
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
        
        # Log security errors at WARNING level by default
        security_logger.warning(
            f"Security error: {message}",
            extra={"security_context": self.context}
        )
    
    def __str__(self) -> str:
        if self.context:
            return f"{self.message} (context: {self.context})"
        return self.message


class PathTraversalError(SecurityError):
    """Raised when a path traversal attack is detected.
    
    This includes:
    - Paths containing ".." components
    - Absolute paths when relative expected
    - Paths escaping the allowed directory
    - Symlinks pointing outside allowed boundaries
    """
    
    def __init__(self, message: str, path: Optional[str] = None, 
                 base_dir: Optional[str] = None):
        context = {}
        if path:
            context["attempted_path"] = path
        if base_dir:
            context["base_directory"] = base_dir
        super().__init__(message, context)
        
        # Escalate to ERROR level for path traversal
        security_logger.error(
            f"PATH TRAVERSAL: {message}",
            extra={"security_context": context}
        )


class FileSizeError(SecurityError):
    """Raised when file size exceeds allowed limits.
    
    This prevents denial-of-service attacks through:
    - Memory exhaustion from large files
    - Disk exhaustion from large writes
    - CPU exhaustion from processing large inputs
    """
    
    def __init__(self, message: str, actual_size: Optional[int] = None,
                 max_size: Optional[int] = None, path: Optional[str] = None):
        context = {}
        if actual_size is not None:
            context["actual_size"] = actual_size
        if max_size is not None:
            context["max_size"] = max_size
        if path:
            context["file_path"] = path
        super().__init__(message, context)


class InvalidInputError(SecurityError):
    """Raised when input fails validation.
    
    This covers:
    - Malformed JSON/CBOR data
    - Invalid schema compliance
    - Type mismatches
    - Range violations
    """
    
    def __init__(self, message: str, input_type: Optional[str] = None,
                 expected: Optional[str] = None, actual: Optional[Any] = None):
        context = {}
        if input_type:
            context["input_type"] = input_type
        if expected:
            context["expected"] = expected
        if actual is not None:
            # Truncate actual value for logging to prevent log injection
            actual_str = str(actual)[:200]
            context["actual"] = actual_str
        super().__init__(message, context)


class CommandInjectionError(SecurityError):
    """Raised when potential command injection is detected.
    
    This includes:
    - Shell metacharacters in arguments
    - Null bytes in command strings
    - Suspiciously formatted input
    """
    
    def __init__(self, message: str, command: Optional[str] = None,
                 argument: Optional[str] = None):
        context = {}
        if command:
            # Don't log the full command to avoid logging sensitive data
            context["command_name"] = command.split()[0] if command else None
        if argument:
            # Truncate and sanitize for logging
            safe_arg = argument[:50].replace("\n", "\\n").replace("\0", "\\0")
            context["suspicious_argument"] = safe_arg
        super().__init__(message, context)
        
        # Escalate to ERROR level for injection attempts
        security_logger.error(
            f"COMMAND INJECTION: {message}",
            extra={"security_context": context}
        )


class HashVerificationError(SecurityError):
    """Raised when cryptographic hash verification fails.
    
    This indicates potential:
    - File tampering
    - Data corruption
    - MITM attack
    """
    
    def __init__(self, message: str, expected_hash: Optional[str] = None,
                 actual_hash: Optional[str] = None, path: Optional[str] = None):
        context = {}
        if expected_hash:
            context["expected_hash"] = expected_hash[:16] + "..."  # Truncate for logging
        if actual_hash:
            context["actual_hash"] = actual_hash[:16] + "..."
        if path:
            context["file_path"] = path
        super().__init__(message, context)
        
        security_logger.error(
            f"HASH VERIFICATION FAILED: {message}",
            extra={"security_context": context}
        )


class SignatureVerificationError(SecurityError):
    """Raised when cryptographic signature verification fails.
    
    This indicates potential:
    - Attestation tampering
    - Invalid/revoked signing key
    - Signature format corruption
    """
    
    def __init__(self, message: str, key_id: Optional[str] = None,
                 algorithm: Optional[str] = None):
        context = {}
        if key_id:
            context["key_id"] = key_id
        if algorithm:
            context["algorithm"] = algorithm
        super().__init__(message, context)
        
        security_logger.error(
            f"SIGNATURE VERIFICATION FAILED: {message}",
            extra={"security_context": context}
        )
