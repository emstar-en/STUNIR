"""STUNIR Platform Compatibility - Cross-Platform Utilities.

Provides platform-agnostic utilities that work consistently
across Windows, macOS, and Linux.
"""

import os
import sys
import stat
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List


# ============================================================================
# Platform Detection
# ============================================================================

def get_platform() -> str:
    """Get the current platform name.
    
    Returns:
        One of: 'windows', 'macos', 'linux', or 'unknown'
    """
    if sys.platform == 'win32':
        return 'windows'
    elif sys.platform == 'darwin':
        return 'macos'
    elif sys.platform.startswith('linux'):
        return 'linux'
    return 'unknown'


def is_windows() -> bool:
    """Check if running on Windows."""
    return sys.platform == 'win32'


def is_macos() -> bool:
    """Check if running on macOS."""
    return sys.platform == 'darwin'


def is_linux() -> bool:
    """Check if running on Linux."""
    return sys.platform.startswith('linux')


def is_unix() -> bool:
    """Check if running on a Unix-like system (Linux or macOS)."""
    return is_linux() or is_macos()


def get_platform_info() -> Dict[str, Any]:
    """Get detailed platform information.
    
    Returns:
        Dictionary with:
        - platform: Platform name
        - python_version: Python version string
        - architecture: CPU architecture
        - path_separator: Path separator character
        - line_separator: Line ending sequence
        - env_separator: Environment PATH separator
    """
    import platform as plat
    return {
        'platform': get_platform(),
        'platform_detail': sys.platform,
        'python_version': sys.version.split()[0],
        'architecture': plat.machine(),
        'path_separator': os.sep,
        'line_separator': repr(os.linesep),
        'env_separator': os.pathsep,
        'cwd': os.getcwd(),
    }


# ============================================================================
# Line Endings
# ============================================================================

def normalize_line_endings(content: str, to_unix: bool = True) -> str:
    """Normalize line endings in content.
    
    Args:
        content: Text content to normalize
        to_unix: If True, convert to LF; if False, convert to CRLF
        
    Returns:
        Content with normalized line endings
    """
    # First normalize all to LF
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    
    if not to_unix:
        # Convert to CRLF
        content = content.replace('\n', '\r\n')
    
    return content


def to_unix_endings(content: str) -> str:
    """Convert content to Unix line endings (LF)."""
    return normalize_line_endings(content, to_unix=True)


def to_windows_endings(content: str) -> str:
    """Convert content to Windows line endings (CRLF)."""
    return normalize_line_endings(content, to_unix=False)


def detect_line_endings(content: str) -> str:
    """Detect the predominant line ending style.
    
    Args:
        content: Text content to analyze
        
    Returns:
        'unix' (LF), 'windows' (CRLF), 'mac' (CR), or 'mixed'
    """
    crlf_count = content.count('\r\n')
    lf_only = content.count('\n') - crlf_count
    cr_only = content.count('\r') - crlf_count
    
    total = crlf_count + lf_only + cr_only
    if total == 0:
        return 'unix'  # Default to unix
    
    if crlf_count > 0 and lf_only == 0 and cr_only == 0:
        return 'windows'
    if lf_only > 0 and crlf_count == 0 and cr_only == 0:
        return 'unix'
    if cr_only > 0 and crlf_count == 0 and lf_only == 0:
        return 'mac'
    return 'mixed'


# ============================================================================
# File Permissions
# ============================================================================

def set_executable(filepath: str, executable: bool = True) -> bool:
    """Set or unset executable permission on a file.
    
    Args:
        filepath: Path to file
        executable: True to make executable, False to remove
        
    Returns:
        True if successful, False otherwise
        
    Note:
        On Windows, this is a no-op as Windows uses file extensions.
    """
    if is_windows():
        return True  # Windows uses file extensions
    
    try:
        mode = os.stat(filepath).st_mode
        if executable:
            # Add execute bits where read bits exist
            mode |= (mode & 0o444) >> 2
        else:
            # Remove all execute bits
            mode &= ~0o111
        os.chmod(filepath, mode)
        return True
    except (OSError, PermissionError):
        return False


def is_executable(filepath: str) -> bool:
    """Check if a file is executable.
    
    Args:
        filepath: Path to file
        
    Returns:
        True if executable, False otherwise
    """
    if is_windows():
        # Windows checks extension
        ext = Path(filepath).suffix.lower()
        return ext in ['.exe', '.bat', '.cmd', '.com', '.ps1']
    
    return os.access(filepath, os.X_OK)


def get_file_mode(filepath: str) -> Optional[int]:
    """Get file permission mode.
    
    Args:
        filepath: Path to file
        
    Returns:
        Unix permission mode (e.g., 0o755) or None if error
    """
    try:
        return stat.S_IMODE(os.stat(filepath).st_mode)
    except OSError:
        return None


def set_file_mode(filepath: str, mode: int) -> bool:
    """Set file permission mode.
    
    Args:
        filepath: Path to file
        mode: Unix permission mode (e.g., 0o755)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.chmod(filepath, mode)
        return True
    except OSError:
        return False


# ============================================================================
# Executable Detection
# ============================================================================

def find_executable(name: str) -> Optional[str]:
    """Find an executable in PATH.
    
    Args:
        name: Executable name (without extension on Unix)
        
    Returns:
        Full path to executable or None if not found
    """
    return shutil.which(name)


def get_exe_extension() -> str:
    """Get the platform executable extension.
    
    Returns:
        '.exe' on Windows, '' on Unix
    """
    return '.exe' if is_windows() else ''


def executable_exists(name: str) -> bool:
    """Check if an executable exists in PATH.
    
    Args:
        name: Executable name
        
    Returns:
        True if found, False otherwise
    """
    return find_executable(name) is not None


def run_command(cmd: List[str], capture: bool = True, 
                timeout: Optional[int] = None) -> Tuple[int, str, str]:
    """Run a command in a platform-independent way.
    
    Args:
        cmd: Command and arguments as list
        capture: Whether to capture output
        timeout: Timeout in seconds (None for no timeout)
        
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=timeout,
            shell=is_windows()  # Use shell on Windows for PATH resolution
        )
        return result.returncode, result.stdout or '', result.stderr or ''
    except subprocess.TimeoutExpired:
        return -1, '', 'Command timed out'
    except FileNotFoundError:
        return -1, '', f'Command not found: {cmd[0]}'


# ============================================================================
# Temporary Files/Directories
# ============================================================================

def get_temp_dir() -> str:
    """Get the system temporary directory.
    
    Returns:
        Path to temp directory
    """
    return tempfile.gettempdir()


def create_temp_file(suffix: str = '', prefix: str = 'stunir_', 
                     delete: bool = True) -> Tuple[int, str]:
    """Create a temporary file.
    
    Args:
        suffix: File suffix (e.g., '.json')
        prefix: File prefix
        delete: Whether file should be auto-deleted
        
    Returns:
        Tuple of (file_descriptor, file_path)
    """
    return tempfile.mkstemp(suffix=suffix, prefix=prefix)


def create_temp_dir(suffix: str = '', prefix: str = 'stunir_') -> str:
    """Create a temporary directory.
    
    Args:
        suffix: Directory suffix
        prefix: Directory prefix
        
    Returns:
        Path to temporary directory
    """
    return tempfile.mkdtemp(suffix=suffix, prefix=prefix)


# ============================================================================
# Environment Variables
# ============================================================================

def get_env_path_separator() -> str:
    """Get the PATH environment variable separator.
    
    Returns:
        ';' on Windows, ':' on Unix
    """
    return os.pathsep


def split_env_path(path_str: Optional[str] = None) -> List[str]:
    """Split PATH-style environment variable.
    
    Args:
        path_str: PATH string to split (default: current PATH)
        
    Returns:
        List of path components
    """
    if path_str is None:
        path_str = os.environ.get('PATH', '')
    return path_str.split(os.pathsep) if path_str else []


def join_env_path(paths: List[str]) -> str:
    """Join paths into PATH-style environment variable.
    
    Args:
        paths: List of paths to join
        
    Returns:
        Joined PATH string
    """
    return os.pathsep.join(paths)
