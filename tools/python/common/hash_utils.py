"""STUNIR Hash Utilities - SHA256 Hashing.

Provides consistent SHA256 hashing across the toolchain.
All hash computations should use these utilities for consistency.
"""

import hashlib
from pathlib import Path
from typing import Union, Optional, Iterator, BinaryIO

# Buffer size for file hashing (64KB - optimal for most filesystems)
HASH_BUFFER_SIZE = 65536


def compute_sha256(data: Union[bytes, str]) -> str:
    """Compute SHA-256 hash of data.
    
    Args:
        data: Bytes or string to hash (strings are UTF-8 encoded)
        
    Returns:
        Lowercase hexadecimal hash string (64 characters)
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def compute_file_hash(filepath: Union[str, Path]) -> str:
    """Compute SHA-256 hash of a file.
    
    Uses incremental hashing for memory efficiency on large files.
    
    Args:
        filepath: Path to file to hash
        
    Returns:
        Lowercase hexadecimal hash string (64 characters)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file can't be read
    """
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(HASH_BUFFER_SIZE), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_hash_incremental() -> 'HashAccumulator':
    """Create an incremental hash accumulator.
    
    Useful for hashing multiple pieces of data without concatenation.
    
    Returns:
        HashAccumulator instance
        
    Example:
        hasher = compute_hash_incremental()
        hasher.update(b'data1')
        hasher.update(b'data2')
        hash_value = hasher.hexdigest()
    """
    return HashAccumulator()


class HashAccumulator:
    """Incremental hash accumulator for streaming data."""
    
    def __init__(self):
        self._hasher = hashlib.sha256()
    
    def update(self, data: Union[bytes, str]) -> 'HashAccumulator':
        """Add data to the hash.
        
        Args:
            data: Bytes or string to add
            
        Returns:
            Self for method chaining
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        self._hasher.update(data)
        return self
    
    def update_file(self, filepath: Union[str, Path]) -> 'HashAccumulator':
        """Add file contents to the hash.
        
        Args:
            filepath: Path to file to add
            
        Returns:
            Self for method chaining
        """
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(HASH_BUFFER_SIZE), b''):
                self._hasher.update(chunk)
        return self
    
    def hexdigest(self) -> str:
        """Get the hexadecimal hash digest.
        
        Returns:
            64-character lowercase hex string
        """
        return self._hasher.hexdigest()
    
    def digest(self) -> bytes:
        """Get the raw hash digest.
        
        Returns:
            32-byte hash digest
        """
        return self._hasher.digest()
    
    def copy(self) -> 'HashAccumulator':
        """Create a copy of the accumulator.
        
        Returns:
            New HashAccumulator with same state
        """
        new_acc = HashAccumulator()
        new_acc._hasher = self._hasher.copy()
        return new_acc


def verify_hash(filepath: Union[str, Path], expected_hash: str) -> bool:
    """Verify a file's hash matches expected value.
    
    Args:
        filepath: Path to file to verify
        expected_hash: Expected SHA-256 hash (lowercase hex)
        
    Returns:
        True if hash matches, False otherwise
    """
    try:
        actual_hash = compute_file_hash(filepath)
        return actual_hash.lower() == expected_hash.lower()
    except (FileNotFoundError, PermissionError):
        return False
