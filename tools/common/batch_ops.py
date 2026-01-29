"""STUNIR Batch Operations - Efficient Bulk Processing.

Provides optimized batch operations for:
- Multi-file hashing
- Bulk file reads/writes
- Parallel processing
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Iterator, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from .hash_utils import compute_file_hash, compute_sha256
from .json_utils import canonical_json


def batch_hash_files(filepaths: List[str], max_workers: int = 4) -> Dict[str, str]:
    """Compute hashes for multiple files in parallel.
    
    Args:
        filepaths: List of file paths to hash
        max_workers: Maximum parallel workers
        
    Returns:
        Dictionary mapping filepath to hash
    """
    results = {}
    
    def hash_file(path: str) -> Tuple[str, str]:
        try:
            return (path, compute_file_hash(path))
        except (FileNotFoundError, PermissionError):
            return (path, '')
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(hash_file, fp): fp for fp in filepaths}
        for future in as_completed(futures):
            path, hash_val = future.result()
            results[path] = hash_val
    
    return results


def batch_read_json(filepaths: List[str], max_workers: int = 4) -> Dict[str, Any]:
    """Read multiple JSON files in parallel.
    
    Args:
        filepaths: List of JSON file paths
        max_workers: Maximum parallel workers
        
    Returns:
        Dictionary mapping filepath to parsed data
    """
    results = {}
    
    def read_json(path: str) -> Tuple[str, Any]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return (path, json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            return (path, None)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(read_json, fp): fp for fp in filepaths}
        for future in as_completed(futures):
            path, data = future.result()
            results[path] = data
    
    return results


def batch_write_json(data_map: Dict[str, Any], pretty: bool = False,
                     max_workers: int = 4) -> Dict[str, bool]:
    """Write multiple JSON files in parallel.
    
    Args:
        data_map: Dictionary mapping filepath to data to write
        pretty: Whether to use pretty printing
        max_workers: Maximum parallel workers
        
    Returns:
        Dictionary mapping filepath to success status
    """
    results = {}
    
    def write_json(path: str, data: Any) -> Tuple[str, bool]:
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            content = json.dumps(data, sort_keys=True, indent=2 if pretty else None)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
                f.write('\n')
            return (path, True)
        except (PermissionError, OSError):
            return (path, False)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(write_json, fp, data): fp 
                   for fp, data in data_map.items()}
        for future in as_completed(futures):
            path, success = future.result()
            results[path] = success
    
    return results


def chunked_file_process(filepath: str, chunk_size: int = 65536,
                         processor: Callable[[bytes], Any] = None) -> Iterator[Any]:
    """Process a file in chunks for memory efficiency.
    
    Args:
        filepath: Path to file
        chunk_size: Size of chunks in bytes
        processor: Optional function to apply to each chunk
        
    Yields:
        Processed chunks (or raw bytes if no processor)
    """
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            if processor:
                yield processor(chunk)
            else:
                yield chunk


def efficient_directory_walk(root: str, extensions: Optional[List[str]] = None,
                             skip_hidden: bool = True) -> Iterator[str]:
    """Memory-efficient directory walking.
    
    Uses os.scandir for better performance than os.walk.
    
    Args:
        root: Root directory to walk
        extensions: Optional list of extensions to filter
        skip_hidden: Skip hidden files/directories
        
    Yields:
        File paths
    """
    ext_set = set(e.lower() for e in extensions) if extensions else None
    
    def walk(path: str) -> Iterator[str]:
        try:
            with os.scandir(path) as entries:
                dirs = []
                for entry in entries:
                    if skip_hidden and entry.name.startswith('.'):
                        continue
                    
                    if entry.is_file():
                        if ext_set is None:
                            yield entry.path
                        else:
                            ext = os.path.splitext(entry.name)[1].lower()
                            if ext in ext_set:
                                yield entry.path
                    elif entry.is_dir():
                        dirs.append(entry.path)
                
                # Process directories after files
                for dir_path in dirs:
                    yield from walk(dir_path)
        except (PermissionError, OSError):
            pass
    
    yield from walk(root)


def batch_validate_hashes(manifest: Dict[str, str], base_dir: str = '',
                          max_workers: int = 4) -> Dict[str, bool]:
    """Validate multiple file hashes against a manifest.
    
    Args:
        manifest: Dictionary mapping relative paths to expected hashes
        base_dir: Base directory for relative paths
        max_workers: Maximum parallel workers
        
    Returns:
        Dictionary mapping paths to validation status
    """
    results = {}
    base = Path(base_dir) if base_dir else Path.cwd()
    
    def validate(path: str, expected_hash: str) -> Tuple[str, bool]:
        try:
            full_path = base / path
            actual_hash = compute_file_hash(full_path)
            return (path, actual_hash.lower() == expected_hash.lower())
        except (FileNotFoundError, PermissionError):
            return (path, False)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(validate, p, h): p 
                   for p, h in manifest.items()}
        for future in as_completed(futures):
            path, valid = future.result()
            results[path] = valid
    
    return results
