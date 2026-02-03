"""STUNIR Common Utilities Package.

This package provides shared utilities used across the STUNIR toolchain:
- file_utils: File operations (read, write, atomic operations)
- hash_utils: SHA256 hashing utilities
- json_utils: Canonical JSON serialization
- path_utils: Cross-platform path handling

Usage:
    from tools.common import canonical_json, compute_sha256, compute_file_hash
    from tools.common import safe_read_file, safe_write_file
    from tools.common import normalize_path, ensure_directory
"""

from .json_utils import canonical_json, canonical_json_pretty, parse_json
from .hash_utils import compute_sha256, compute_file_hash, compute_hash_incremental
from .file_utils import (
    safe_read_file, safe_write_file, atomic_write,
    scan_directory, ensure_directory, copy_file
)
from .path_utils import (
    normalize_path, join_paths, get_relative_path,
    split_path, get_extension, change_extension
)

__all__ = [
    # JSON utilities
    'canonical_json',
    'canonical_json_pretty', 
    'parse_json',
    # Hash utilities
    'compute_sha256',
    'compute_file_hash',
    'compute_hash_incremental',
    # File utilities
    'safe_read_file',
    'safe_write_file',
    'atomic_write',
    'scan_directory',
    'ensure_directory',
    'copy_file',
    # Path utilities
    'normalize_path',
    'join_paths',
    'get_relative_path',
    'split_path',
    'get_extension',
    'change_extension',
]
