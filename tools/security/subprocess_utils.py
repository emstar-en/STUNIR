"""STUNIR Subprocess Security Utilities.

This module provides secure subprocess execution that:
- NEVER uses shell=True
- Validates and sanitizes arguments
- Provides timeout protection
- Captures output safely
- Handles errors with proper context

Security Guidelines:
- Always use run_command() instead of subprocess.run() directly
- Never construct command strings - use argument lists
- Always set appropriate timeouts
- Validate all inputs before passing to commands

Example:
    from tools.security.subprocess_utils import run_command
    
    # Safe: Arguments as list, no shell
    result = run_command(["git", "status", "--porcelain"])
    
    # With timeout (recommended)
    result = run_command_with_timeout(["make", "build"], timeout_seconds=300)
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .exceptions import SecurityError
from .validation import validate_command_args, validate_path


logger = logging.getLogger("stunir.subprocess")


class SubprocessError(SecurityError):
    """Raised when subprocess execution fails."""
    
    def __init__(
        self,
        message: str,
        command: Optional[List[str]] = None,
        returncode: Optional[int] = None,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
    ):
        context = {}
        if command:
            # Log only the command name, not full args (may contain sensitive data)
            context["command"] = command[0] if command else None
            context["arg_count"] = len(command) - 1
        if returncode is not None:
            context["returncode"] = returncode
        if stderr:
            # Truncate stderr for logging
            context["stderr_preview"] = stderr[:200]
        
        super().__init__(message, context)
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# Default limits
DEFAULT_TIMEOUT = 60  # seconds
DEFAULT_MAX_OUTPUT = 10 * 1024 * 1024  # 10 MB

# Commands that are allowed to run (whitelist for extra security)
ALLOWED_COMMANDS = {
    # Version control
    "git",
    # Build tools
    "make", "cmake", "cargo", "cabal", "stack",
    # Python
    "python", "python3", "pip", "pip3",
    # System utilities
    "ls", "cat", "head", "tail", "sha256sum", "shasum", "openssl",
    "cp", "mv", "rm", "mkdir", "touch", "chmod", "pwd", "echo",
    "true", "false", "sleep", "sh",  # Common testing utilities
    # Haskell
    "ghc", "ghci", "runhaskell",
    # Rust
    "rustc", "rustup",
    # STUNIR-specific
    "stunir-native",
}


def run_command(
    args: List[str],
    cwd: Optional[Union[str, Path]] = None,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
    capture_output: bool = True,
    allow_shell_chars: bool = False,
    enforce_whitelist: bool = False,
) -> subprocess.CompletedProcess:
    """Execute a command securely.
    
    This function:
    - Never uses shell=True
    - Validates all arguments
    - Captures output safely
    - Provides detailed error information
    
    Args:
        args: Command and arguments as a list (NEVER a string)
        cwd: Working directory
        env: Environment variables (merged with os.environ)
        check: Whether to raise on non-zero exit
        capture_output: Whether to capture stdout/stderr
        allow_shell_chars: Whether to allow shell metacharacters in args
        enforce_whitelist: Whether to only allow whitelisted commands
    
    Returns:
        subprocess.CompletedProcess with stdout/stderr
    
    Raises:
        SubprocessError: If command fails or is not allowed
    
    Example:
        >>> result = run_command(["git", "status"])
        >>> print(result.stdout)
    """
    if not args:
        raise SubprocessError("Empty command not allowed")
    
    if not isinstance(args, list):
        raise SubprocessError(
            "Command must be a list of arguments, not a string. "
            "Use ['cmd', 'arg1', 'arg2'] instead of 'cmd arg1 arg2'",
            command=None
        )
    
    # Validate arguments
    args = validate_command_args(args, allow_shell_chars=allow_shell_chars)
    
    # Check command whitelist
    cmd_name = Path(args[0]).name
    if enforce_whitelist and cmd_name not in ALLOWED_COMMANDS:
        raise SubprocessError(
            f"Command '{cmd_name}' not in allowed list",
            command=args
        )
    
    # Validate working directory
    if cwd is not None:
        cwd = validate_path(cwd, allow_absolute=True, must_exist=True)
    
    # Prepare environment
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    
    logger.debug(f"Running command: {args[0]} with {len(args)-1} arguments")
    
    try:
        result = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            env=full_env,
            capture_output=capture_output,
            text=True,
            shell=False,  # SECURITY: Never use shell=True
        )
        
        if check and result.returncode != 0:
            raise SubprocessError(
                f"Command failed with exit code {result.returncode}",
                command=args,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        
        return result
        
    except FileNotFoundError:
        raise SubprocessError(
            f"Command not found: {args[0]}",
            command=args
        )
    except PermissionError:
        raise SubprocessError(
            f"Permission denied for command: {args[0]}",
            command=args
        )


def run_command_with_timeout(
    args: List[str],
    timeout_seconds: int = DEFAULT_TIMEOUT,
    cwd: Optional[Union[str, Path]] = None,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
    capture_output: bool = True,
    allow_shell_chars: bool = False,
    enforce_whitelist: bool = False,
) -> subprocess.CompletedProcess:
    """Execute a command with timeout protection.
    
    This adds timeout protection to prevent hung processes from blocking.
    
    Args:
        args: Command and arguments
        timeout_seconds: Maximum execution time
        cwd: Working directory
        env: Environment variables
        check: Whether to raise on non-zero exit
        capture_output: Whether to capture output
        allow_shell_chars: Whether to allow shell metacharacters
        enforce_whitelist: Whether to enforce command whitelist
    
    Returns:
        subprocess.CompletedProcess
    
    Raises:
        SubprocessError: On timeout or execution failure
    """
    if not args:
        raise SubprocessError("Empty command not allowed")
    
    if not isinstance(args, list):
        raise SubprocessError(
            "Command must be a list of arguments",
            command=None
        )
    
    # Validate arguments
    args = validate_command_args(args, allow_shell_chars=allow_shell_chars)
    
    # Check whitelist
    cmd_name = Path(args[0]).name
    if enforce_whitelist and cmd_name not in ALLOWED_COMMANDS:
        raise SubprocessError(
            f"Command '{cmd_name}' not in allowed list",
            command=args
        )
    
    # Validate working directory
    if cwd is not None:
        cwd = validate_path(cwd, allow_absolute=True, must_exist=True)
    
    # Prepare environment
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    
    logger.debug(f"Running command with {timeout_seconds}s timeout: {args[0]}")
    
    try:
        result = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            env=full_env,
            capture_output=capture_output,
            text=True,
            shell=False,
            timeout=timeout_seconds,
        )
        
        if check and result.returncode != 0:
            raise SubprocessError(
                f"Command failed with exit code {result.returncode}",
                command=args,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        
        return result
        
    except subprocess.TimeoutExpired as e:
        raise SubprocessError(
            f"Command timed out after {timeout_seconds} seconds",
            command=args,
            stdout=e.stdout.decode() if e.stdout else None,
            stderr=e.stderr.decode() if e.stderr else None,
        )
    except FileNotFoundError:
        raise SubprocessError(
            f"Command not found: {args[0]}",
            command=args
        )
    except PermissionError:
        raise SubprocessError(
            f"Permission denied for command: {args[0]}",
            command=args
        )


def quote_arg(arg: str) -> str:
    """Safely quote an argument for display purposes.
    
    Note: This is for logging/display only. Never construct shell commands
    from quoted strings - always use argument lists.
    
    Args:
        arg: Argument to quote
    
    Returns:
        Quoted argument string
    """
    return shlex.quote(arg)


def split_command_string(cmd_string: str) -> List[str]:
    """Split a command string into arguments safely.
    
    Use this when you receive a command as a string and need to convert
    it to an argument list for run_command().
    
    Args:
        cmd_string: Command string like "git status --porcelain"
    
    Returns:
        List of arguments like ["git", "status", "--porcelain"]
    
    Raises:
        SubprocessError: If string cannot be parsed safely
    """
    try:
        return shlex.split(cmd_string)
    except ValueError as e:
        raise SubprocessError(
            f"Cannot parse command string: {e}",
            command=None
        )
