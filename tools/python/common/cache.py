"""STUNIR Cache Utilities - High-Performance Caching.

Provides caching for expensive operations:
- File hash caching
- JSON parsing caching
- Directory scan caching
- LRU cache with TTL support
"""

import time
import json
import hashlib
import threading
from pathlib import Path
from functools import wraps
from typing import Any, Dict, Optional, Callable, TypeVar, Union
from collections import OrderedDict

T = TypeVar('T')


class LRUCache:
    """Thread-safe LRU cache with optional TTL."""
    
    def __init__(self, maxsize: int = 128, ttl: Optional[float] = None):
        """Initialize LRU cache.
        
        Args:
            maxsize: Maximum number of items to cache
            ttl: Time-to-live in seconds (None for no expiration)
        """
        self._maxsize = maxsize
        self._ttl = ttl
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[Any, float] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: Any, default: Any = None) -> Any:
        """Get item from cache."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return default
            
            # Check TTL
            if self._ttl is not None:
                if time.time() - self._timestamps[key] > self._ttl:
                    del self._cache[key]
                    del self._timestamps[key]
                    self._misses += 1
                    return default
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
    
    def set(self, key: Any, value: Any) -> None:
        """Set item in cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            elif len(self._cache) >= self._maxsize:
                # Remove oldest item
                oldest = next(iter(self._cache))
                del self._cache[oldest]
                del self._timestamps[oldest]
            
            self._cache[key] = value
            self._timestamps[key] = time.time()
    
    def __contains__(self, key: Any) -> bool:
        """Check if key in cache."""
        with self._lock:
            if key not in self._cache:
                return False
            if self._ttl is not None:
                if time.time() - self._timestamps[key] > self._ttl:
                    del self._cache[key]
                    del self._timestamps[key]
                    return False
            return True
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    @property
    def stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'hits': self._hits,
            'misses': self._misses,
            'size': len(self._cache),
            'maxsize': self._maxsize,
        }


# Global caches for common operations
_file_hash_cache = LRUCache(maxsize=1024, ttl=300)  # 5-minute TTL
_json_parse_cache = LRUCache(maxsize=256, ttl=60)   # 1-minute TTL
_directory_cache = LRUCache(maxsize=64, ttl=30)     # 30-second TTL


def cached_file_hash(filepath: Union[str, Path]) -> str:
    """Get cached file hash (invalidates on mtime change).
    
    Args:
        filepath: Path to file
        
    Returns:
        SHA-256 hash string
    """
    from .hash_utils import compute_file_hash
    
    filepath = Path(filepath)
    stat = filepath.stat()
    cache_key = (str(filepath), stat.st_mtime, stat.st_size)
    
    cached = _file_hash_cache.get(cache_key)
    if cached is not None:
        return cached
    
    hash_value = compute_file_hash(filepath)
    _file_hash_cache.set(cache_key, hash_value)
    return hash_value


def cached_json_load(filepath: Union[str, Path]) -> Any:
    """Load JSON with caching (invalidates on mtime change).
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Parsed JSON data
    """
    filepath = Path(filepath)
    stat = filepath.stat()
    cache_key = (str(filepath), stat.st_mtime)
    
    cached = _json_parse_cache.get(cache_key)
    if cached is not None:
        return cached
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    _json_parse_cache.set(cache_key, data)
    return data


def lru_cache(maxsize: int = 128, ttl: Optional[float] = None):
    """Decorator for caching function results.
    
    Args:
        maxsize: Maximum cache size
        ttl: Time-to-live in seconds
        
    Returns:
        Decorated function
    """
    cache = LRUCache(maxsize=maxsize, ttl=ttl)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Create hashable key
            try:
                key = (args, tuple(sorted(kwargs.items())))
            except TypeError:
                # Unhashable arguments, skip cache
                return func(*args, **kwargs)
            
            cached = cache.get(key)
            if cached is not None:
                return cached
            
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
        
        wrapper.cache = cache  # type: ignore
        wrapper.cache_clear = cache.clear  # type: ignore
        wrapper.cache_stats = lambda: cache.stats  # type: ignore
        return wrapper
    
    return decorator


def get_cache_stats() -> Dict[str, Dict[str, int]]:
    """Get statistics for all global caches.
    
    Returns:
        Dictionary with cache names and their stats
    """
    return {
        'file_hash_cache': _file_hash_cache.stats,
        'json_parse_cache': _json_parse_cache.stats,
        'directory_cache': _directory_cache.stats,
    }


def clear_all_caches() -> None:
    """Clear all global caches."""
    _file_hash_cache.clear()
    _json_parse_cache.clear()
    _directory_cache.clear()
