"""STUNIR Cache Backends - Multiple cache storage options.

Provides:
- File-based persistent cache
- Redis cache (optional)
- Cache decorators with backend selection
- Cache warming utilities
"""

import os
import json
import time
import pickle
import hashlib
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union
from functools import wraps

from .cache import LRUCache

T = TypeVar('T')


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {}


class MemoryBackend(CacheBackend):
    """In-memory LRU cache backend."""
    
    def __init__(self, maxsize: int = 1000, ttl: Optional[float] = None):
        self._cache = LRUCache(maxsize=maxsize, ttl=ttl)
        self._default_ttl = ttl
    
    def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        self._cache.set(key, value)
    
    def delete(self, key: str) -> bool:
        if key in self._cache:
            with self._cache._lock:
                if key in self._cache._cache:
                    del self._cache._cache[key]
                    if key in self._cache._timestamps:
                        del self._cache._timestamps[key]
                    return True
        return False
    
    def exists(self, key: str) -> bool:
        return key in self._cache
    
    def clear(self) -> None:
        self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        return self._cache.stats


class FileBackend(CacheBackend):
    """File-based persistent cache backend."""
    
    def __init__(self, cache_dir: str = ".cache", ttl: Optional[float] = None):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._default_ttl = ttl
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
    
    def _key_to_path(self, key: str) -> Path:
        """Convert key to file path."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self._cache_dir / f"{key_hash}.cache"
    
    def _meta_path(self, key: str) -> Path:
        """Get metadata file path."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self._cache_dir / f"{key_hash}.meta"
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            cache_path = self._key_to_path(key)
            meta_path = self._meta_path(key)
            
            if not cache_path.exists():
                self._misses += 1
                return None
            
            # Check TTL from metadata
            if meta_path.exists():
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    if meta.get('expires_at') and time.time() > meta['expires_at']:
                        self.delete(key)
                        self._misses += 1
                        return None
                except (json.JSONDecodeError, IOError):
                    pass
            
            try:
                with open(cache_path, 'rb') as f:
                    self._hits += 1
                    return pickle.load(f)
            except (pickle.PickleError, IOError):
                self._misses += 1
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        with self._lock:
            cache_path = self._key_to_path(key)
            meta_path = self._meta_path(key)
            
            # Write value
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            
            # Write metadata
            ttl = ttl or self._default_ttl
            meta = {
                'key': key,
                'created_at': time.time(),
                'expires_at': time.time() + ttl if ttl else None,
            }
            with open(meta_path, 'w') as f:
                json.dump(meta, f)
    
    def delete(self, key: str) -> bool:
        with self._lock:
            cache_path = self._key_to_path(key)
            meta_path = self._meta_path(key)
            deleted = False
            
            if cache_path.exists():
                cache_path.unlink()
                deleted = True
            if meta_path.exists():
                meta_path.unlink()
            
            return deleted
    
    def exists(self, key: str) -> bool:
        return self._key_to_path(key).exists()
    
    def clear(self) -> None:
        with self._lock:
            for f in self._cache_dir.glob("*.cache"):
                f.unlink()
            for f in self._cache_dir.glob("*.meta"):
                f.unlink()
    
    def get_stats(self) -> Dict[str, Any]:
        files = list(self._cache_dir.glob("*.cache"))
        total_size = sum(f.stat().st_size for f in files)
        return {
            'hits': self._hits,
            'misses': self._misses,
            'entries': len(files),
            'total_size_bytes': total_size,
            'cache_dir': str(self._cache_dir),
        }


class RedisBackend(CacheBackend):
    """Redis cache backend (optional, with fallback)."""
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        prefix: str = 'stunir:',
        ttl: Optional[float] = None,
        fallback: Optional[CacheBackend] = None,
    ):
        self._prefix = prefix
        self._default_ttl = ttl
        self._fallback = fallback or MemoryBackend()
        self._client = None
        self._connected = False
        
        try:
            import redis
            self._client = redis.Redis(host=host, port=port, db=db)
            self._client.ping()
            self._connected = True
        except (ImportError, Exception):
            pass
    
    def _prefixed_key(self, key: str) -> str:
        return f"{self._prefix}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        if not self._connected:
            return self._fallback.get(key)
        try:
            data = self._client.get(self._prefixed_key(key))
            if data is None:
                return None
            return pickle.loads(data)
        except Exception:
            return self._fallback.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        if not self._connected:
            self._fallback.set(key, value, ttl)
            return
        try:
            data = pickle.dumps(value)
            ttl = ttl or self._default_ttl
            if ttl:
                self._client.setex(self._prefixed_key(key), int(ttl), data)
            else:
                self._client.set(self._prefixed_key(key), data)
        except Exception:
            self._fallback.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        if not self._connected:
            return self._fallback.delete(key)
        try:
            return bool(self._client.delete(self._prefixed_key(key)))
        except Exception:
            return self._fallback.delete(key)
    
    def exists(self, key: str) -> bool:
        if not self._connected:
            return self._fallback.exists(key)
        try:
            return bool(self._client.exists(self._prefixed_key(key)))
        except Exception:
            return self._fallback.exists(key)
    
    def clear(self) -> None:
        if not self._connected:
            self._fallback.clear()
            return
        try:
            keys = self._client.keys(f"{self._prefix}*")
            if keys:
                self._client.delete(*keys)
        except Exception:
            self._fallback.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        stats = {'connected': self._connected}
        if self._connected:
            try:
                info = self._client.info('memory')
                stats['used_memory'] = info.get('used_memory_human', 'N/A')
            except Exception:
                pass
        return stats


class TieredCache:
    """Multi-tier cache with L1 (memory) and L2 (file/redis)."""
    
    def __init__(
        self,
        l1: Optional[CacheBackend] = None,
        l2: Optional[CacheBackend] = None,
    ):
        self.l1 = l1 or MemoryBackend(maxsize=100)
        self.l2 = l2 or FileBackend()
    
    def get(self, key: str) -> Optional[Any]:
        # Try L1 first
        value = self.l1.get(key)
        if value is not None:
            return value
        
        # Try L2
        value = self.l2.get(key)
        if value is not None:
            # Promote to L1
            self.l1.set(key, value)
        return value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        self.l1.set(key, value, ttl)
        self.l2.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        l1_deleted = self.l1.delete(key)
        l2_deleted = self.l2.delete(key)
        return l1_deleted or l2_deleted
    
    def clear(self) -> None:
        self.l1.clear()
        self.l2.clear()


def cached(
    backend: Optional[CacheBackend] = None,
    ttl: Optional[float] = None,
    key_func: Optional[Callable[..., str]] = None,
):
    """Decorator for caching function results with backend selection."""
    _backend = backend or MemoryBackend()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = f"{func.__module__}.{func.__name__}:{args}:{kwargs}"
            
            cached_value = _backend.get(key)
            if cached_value is not None:
                return cached_value
            
            result = func(*args, **kwargs)
            _backend.set(key, result, ttl)
            return result
        
        wrapper.cache_backend = _backend
        wrapper.invalidate = lambda: _backend.clear()
        return wrapper
    
    return decorator


class CacheWarmer:
    """Utility for warming caches with pre-computed values."""
    
    def __init__(self, backend: CacheBackend):
        self.backend = backend
    
    def warm(self, data: Dict[str, Any], ttl: Optional[float] = None) -> int:
        """Warm cache with dictionary of key-value pairs."""
        count = 0
        for key, value in data.items():
            self.backend.set(key, value, ttl)
            count += 1
        return count
    
    def warm_from_function(
        self,
        func: Callable[..., T],
        keys: list,
        ttl: Optional[float] = None,
    ) -> int:
        """Warm cache by calling function for each key."""
        count = 0
        for key in keys:
            try:
                value = func(key)
                self.backend.set(str(key), value, ttl)
                count += 1
            except Exception:
                pass
        return count
