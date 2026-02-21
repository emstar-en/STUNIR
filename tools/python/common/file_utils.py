"""STUNIR File Utilities - File Operations.

Provides safe, cross-platform file operations with proper error handling.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Union, Optional, List, Dict, Any, Iterator, Callable
from contextlib import contextmanager


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure exists
        
    Returns:
        Path object for the directory
        
    Raises:
        PermissionError: If directory can't be created
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_read_file(filepath: Union[str, Path], encoding: str = 'utf-8') -> Optional[str]:
    """Safely read a text file.
    
    Args:
        filepath: Path to file
        encoding: File encoding (default: utf-8)
        
    Returns:
        File contents as string, or None if file doesn't exist
        
    Raises:
        PermissionError: If file can't be read
    """
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            return f.read()
    except FileNotFoundError:
        return None


def safe_read_bytes(filepath: Union[str, Path]) -> Optional[bytes]:
    """Safely read a binary file.
    
    Args:
        filepath: Path to file
        
    Returns:
        File contents as bytes, or None if file doesn't exist
    """
    try:
        with open(filepath, 'rb') as f:
            return f.read()
    except FileNotFoundError:
        return None


def safe_write_file(filepath: Union[str, Path], content: str, 
                    encoding: str = 'utf-8', create_dirs: bool = True) -> bool:
    """Safely write a text file.
    
    Args:
        filepath: Path to file
        content: Content to write
        encoding: File encoding (default: utf-8)
        create_dirs: Create parent directories if needed
        
    Returns:
        True if successful, False otherwise
    """
    try:
        filepath = Path(filepath)
        if create_dirs:
            filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding=encoding) as f:
            f.write(content)
        return True
    except (PermissionError, OSError):
        return False


def safe_write_bytes(filepath: Union[str, Path], content: bytes,
                     create_dirs: bool = True) -> bool:
    """Safely write a binary file.
    
    Args:
        filepath: Path to file
        content: Content to write
        create_dirs: Create parent directories if needed
        
    Returns:
        True if successful, False otherwise
    """
    try:
        filepath = Path(filepath)
        if create_dirs:
            filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            f.write(content)
        return True
    except (PermissionError, OSError):
        return False


@contextmanager
def atomic_write(filepath: Union[str, Path], mode: str = 'w', 
                 encoding: Optional[str] = 'utf-8'):
    """Context manager for atomic file writes.
    
    Writes to a temporary file, then atomically renames on success.
    If an error occurs, the original file is preserved.
    
    Args:
        filepath: Target file path
        mode: File mode ('w' for text, 'wb' for binary)
        encoding: Encoding for text mode (ignored for binary)
        
    Yields:
        File handle for writing
        
    Example:
        with atomic_write('output.json') as f:
            f.write(json.dumps(data))
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Create temp file in same directory for atomic rename
    fd, tmp_path = tempfile.mkstemp(
        dir=filepath.parent,
        prefix='.tmp_',
        suffix=filepath.suffix
    )
    
    try:
        if 'b' in mode:
            with os.fdopen(fd, mode) as f:
                yield f
        else:
            with os.fdopen(fd, mode, encoding=encoding) as f:
                yield f
        # Atomic rename
        os.replace(tmp_path, filepath)
    except Exception:
        # Cleanup temp file on any error to avoid leaving stale files.
        # Catches all exceptions to ensure cleanup runs before re-raising.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def scan_directory(scan_dir: Union[str, Path], 
                   extensions: Optional[List[str]] = None,
                   recursive: bool = True,
                   include_hidden: bool = False) -> Iterator[Dict[str, Any]]:
    """Scan a directory for files matching criteria.
    
    Args:
        scan_dir: Directory to scan
        extensions: List of extensions to include (e.g., ['.json', '.py'])
        recursive: Whether to recurse into subdirectories
        include_hidden: Whether to include hidden files (starting with .)
        
    Yields:
        Dictionary with file metadata:
        - path: Absolute file path
        - name: File name
        - relative: Path relative to scan_dir
        - size: File size in bytes
        - mtime: Modification time (Unix timestamp)
        - extension: File extension
    """
    scan_path = Path(scan_dir)
    if not scan_path.exists():
        return
    
    if recursive:
        walker = scan_path.rglob('*')
    else:
        walker = scan_path.glob('*')
    
    for entry in sorted(walker):
        if not entry.is_file():
            continue
        
        # Skip hidden files unless requested
        if not include_hidden and entry.name.startswith('.'):
            continue
        
        # Filter by extension
        if extensions:
            if entry.suffix.lower() not in [ext.lower() for ext in extensions]:
                continue
        
        try:
            stat = entry.stat()
            yield {
                'path': str(entry.resolve()),
                'name': entry.name,
                'relative': str(entry.relative_to(scan_path)),
                'size': stat.st_size,
                'mtime': stat.st_mtime,
                'extension': entry.suffix,
            }
        except (PermissionError, OSError):
            continue


def copy_file(src: Union[str, Path], dst: Union[str, Path], 
              create_dirs: bool = True) -> bool:
    """Copy a file safely.
    
    Args:
        src: Source file path
        dst: Destination file path
        create_dirs: Create parent directories if needed
        
    Returns:
        True if successful, False otherwise
    """
    try:
        dst = Path(dst)
        if create_dirs:
            dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    except (shutil.Error, OSError):
        return False


def delete_file(filepath: Union[str, Path]) -> bool:
    """Safely delete a file.
    
    Args:
        filepath: Path to file to delete
        
    Returns:
        True if deleted or didn't exist, False on error
    """
    try:
        Path(filepath).unlink(missing_ok=True)
        return True
    except (PermissionError, OSError):
        return False


def file_exists(filepath: Union[str, Path]) -> bool:
    """Check if a file exists.
    
    Args:
        filepath: Path to check
        
    Returns:
        True if file exists, False otherwise
    """
    return Path(filepath).is_file()


def directory_exists(dirpath: Union[str, Path]) -> bool:
    """Check if a directory exists.
    
    Args:
        dirpath: Path to check
        
    Returns:
        True if directory exists, False otherwise
    """
    return Path(dirpath).is_dir()
