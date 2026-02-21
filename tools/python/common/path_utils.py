"""STUNIR Path Utilities - Cross-Platform Path Handling.

Provides consistent path operations across Windows, macOS, and Linux.
"""

import os
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Union, Optional, List, Tuple


def normalize_path(path: Union[str, Path]) -> str:
    """Normalize a path to use forward slashes and resolve ..
    
    This ensures consistent path representation across platforms.
    
    Args:
        path: Path to normalize
        
    Returns:
        Normalized path string with forward slashes
    """
    path = Path(path)
    # Resolve to absolute and normalize
    try:
        resolved = path.resolve()
    except OSError:
        # Handle non-existent paths
        resolved = path.absolute()
    # Convert to POSIX-style path string
    return str(resolved).replace('\\', '/')


def join_paths(*parts: Union[str, Path]) -> str:
    """Join path components safely.
    
    Uses platform-appropriate separators internally but returns
    normalized forward-slash paths.
    
    Args:
        *parts: Path components to join
        
    Returns:
        Joined path with forward slashes
    """
    result = Path(parts[0])
    for part in parts[1:]:
        result = result / part
    return str(result).replace('\\', '/')


def get_relative_path(path: Union[str, Path], base: Union[str, Path]) -> str:
    """Get relative path from base to path.
    
    Args:
        path: Target path
        base: Base path
        
    Returns:
        Relative path string with forward slashes
    """
    try:
        rel = Path(path).relative_to(Path(base))
        return str(rel).replace('\\', '/')
    except ValueError:
        # Not relative, return absolute
        return normalize_path(path)


def split_path(path: Union[str, Path]) -> Tuple[str, str]:
    """Split path into directory and filename.
    
    Args:
        path: Path to split
        
    Returns:
        Tuple of (directory, filename)
    """
    p = Path(path)
    return str(p.parent).replace('\\', '/'), p.name


def get_extension(path: Union[str, Path]) -> str:
    """Get file extension including the dot.
    
    Args:
        path: Path to examine
        
    Returns:
        Extension string (e.g., '.json') or empty string
    """
    return Path(path).suffix


def change_extension(path: Union[str, Path], new_ext: str) -> str:
    """Change the file extension.
    
    Args:
        path: Original path
        new_ext: New extension (with or without leading dot)
        
    Returns:
        Path with new extension
    """
    if not new_ext.startswith('.'):
        new_ext = '.' + new_ext
    return str(Path(path).with_suffix(new_ext)).replace('\\', '/')


def get_stem(path: Union[str, Path]) -> str:
    """Get filename without extension.
    
    Args:
        path: Path to examine
        
    Returns:
        Filename without extension
    """
    return Path(path).stem


def is_absolute(path: Union[str, Path]) -> bool:
    """Check if path is absolute.
    
    Args:
        path: Path to check
        
    Returns:
        True if absolute, False if relative
    """
    return Path(path).is_absolute()


def make_absolute(path: Union[str, Path], base: Optional[Union[str, Path]] = None) -> str:
    """Convert path to absolute.
    
    Args:
        path: Path to convert
        base: Base directory for relative paths (default: cwd)
        
    Returns:
        Absolute path string
    """
    p = Path(path)
    if p.is_absolute():
        return normalize_path(p)
    if base:
        return normalize_path(Path(base) / p)
    return normalize_path(p.absolute())


def get_common_prefix(*paths: Union[str, Path]) -> str:
    """Find common path prefix of multiple paths.
    
    Args:
        *paths: Paths to analyze
        
    Returns:
        Common prefix path
    """
    if not paths:
        return ''
    if len(paths) == 1:
        return str(Path(paths[0]).parent).replace('\\', '/')
    
    # Convert all to Path objects
    path_objs = [Path(p).resolve() for p in paths]
    
    # Find common parts
    parts_list = [p.parts for p in path_objs]
    common = []
    for parts in zip(*parts_list):
        if len(set(parts)) == 1:
            common.append(parts[0])
        else:
            break
    
    if not common:
        return ''
    
    return str(Path(*common)).replace('\\', '/')


def is_path_under(path: Union[str, Path], parent: Union[str, Path]) -> bool:
    """Check if path is under a parent directory.
    
    Args:
        path: Path to check
        parent: Parent directory
        
    Returns:
        True if path is under parent
    """
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


# Platform-specific constants
IS_WINDOWS = sys.platform == 'win32'
IS_MACOS = sys.platform == 'darwin'
IS_LINUX = sys.platform.startswith('linux')
PATH_SEP = os.sep
LINE_SEP = os.linesep
