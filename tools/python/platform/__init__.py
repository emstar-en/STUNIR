"""STUNIR Platform Compatibility Layer.

Provides cross-platform utilities for:
- Path handling (Windows vs Unix)
- Line endings (CRLF vs LF)
- File permissions
- Executable detection
- Temporary directory handling
- Platform detection

Usage:
    from tools.platform import get_platform, is_windows, is_unix
    from tools.platform import normalize_line_endings, set_executable
    from tools.platform import get_temp_dir, find_executable
"""

from .compat import (
    # Platform detection
    get_platform,
    is_windows,
    is_macos,
    is_linux,
    is_unix,
    get_platform_info,
    # Line endings
    normalize_line_endings,
    to_unix_endings,
    to_windows_endings,
    # File permissions
    set_executable,
    is_executable,
    get_file_mode,
    set_file_mode,
    # Executables
    find_executable,
    get_exe_extension,
    # Temp directories
    get_temp_dir,
    create_temp_file,
    create_temp_dir,
    # Environment
    get_env_path_separator,
    split_env_path,
    join_env_path,
)

__all__ = [
    'get_platform',
    'is_windows',
    'is_macos',
    'is_linux',
    'is_unix',
    'get_platform_info',
    'normalize_line_endings',
    'to_unix_endings',
    'to_windows_endings',
    'set_executable',
    'is_executable',
    'get_file_mode',
    'set_file_mode',
    'find_executable',
    'get_exe_extension',
    'get_temp_dir',
    'create_temp_file',
    'create_temp_dir',
    'get_env_path_separator',
    'split_env_path',
    'join_env_path',
]
