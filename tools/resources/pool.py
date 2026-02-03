"""Resource pooling for STUNIR."""

import time
import threading
import queue
from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Optional, TypeVar
from dataclasses import dataclass
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

T = TypeVar('T')


class PoolExhausted(Exception):
    """Pool has no available resources."""
    pass


@dataclass
class PoolStats:
    """Resource pool statistics."""
    size: int
    available: int
    in_use: int
    created: int
    destroyed: int
    acquisitions: int
    timeouts: int


class ResourcePool(Generic[T]):
    """Generic resource pool with lifecycle management.
    
    Example:
        pool = ResourcePool(
            factory=create_connection,
            max_size=10,
            min_size=2,
        )
        
        with pool.acquire() as conn:
            conn.execute('SELECT 1')
    """
    
    def __init__(
        self,
        factory: Callable[[], T],
        max_size: int = 10,
        min_size: int = 0,
        max_idle_time: float = 300.0,
        validation_func: Optional[Callable[[T], bool]] = None,
        cleanup_func: Optional[Callable[[T], None]] = None,
    ):
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self.max_idle_time = max_idle_time
        self.validation_func = validation_func
        self.cleanup_func = cleanup_func
        
        self._pool: queue.Queue = queue.Queue()
        self._in_use: set = set()
        self._timestamps: dict = {}
        self._lock = threading.Lock()
        self._total_created = 0
        self._total_destroyed = 0
        self._total_acquisitions = 0
        self._total_timeouts = 0
        self._closed = False
        
        # Pre-populate minimum resources
        for _ in range(min_size):
            self._create_resource()
    
    def _create_resource(self) -> T:
        """Create a new resource."""
        resource = self.factory()
        self._timestamps[id(resource)] = time.monotonic()
        self._total_created += 1
        return resource
    
    def _destroy_resource(self, resource: T) -> None:
        """Destroy a resource."""
        self._timestamps.pop(id(resource), None)
        self._total_destroyed += 1
        if self.cleanup_func:
            try:
                self.cleanup_func(resource)
            except Exception:
                pass
    
    def _validate_resource(self, resource: T) -> bool:
        """Validate a resource is still usable."""
        if self.validation_func:
            try:
                return self.validation_func(resource)
            except Exception:
                return False
        return True
    
    def _get_size(self) -> int:
        """Get total pool size."""
        return self._pool.qsize() + len(self._in_use)
    
    def acquire(self, timeout: Optional[float] = None) -> T:
        """Acquire a resource from the pool."""
        if self._closed:
            raise PoolExhausted("Pool is closed")
        
        deadline = time.monotonic() + timeout if timeout else None
        
        while True:
            with self._lock:
                # Try to get from pool
                try:
                    resource = self._pool.get_nowait()
                    
                    # Validate resource
                    if not self._validate_resource(resource):
                        self._destroy_resource(resource)
                        continue
                    
                    # Check idle time
                    idle_time = time.monotonic() - self._timestamps.get(id(resource), 0)
                    if idle_time > self.max_idle_time:
                        self._destroy_resource(resource)
                        continue
                    
                    self._in_use.add(id(resource))
                    self._total_acquisitions += 1
                    return resource
                    
                except queue.Empty:
                    pass
                
                # Create new resource if under limit
                if self._get_size() < self.max_size:
                    resource = self._create_resource()
                    self._in_use.add(id(resource))
                    self._total_acquisitions += 1
                    return resource
            
            # Wait for resource
            if deadline:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._total_timeouts += 1
                    raise PoolExhausted("Timeout waiting for resource")
                try:
                    resource = self._pool.get(timeout=remaining)
                    with self._lock:
                        if self._validate_resource(resource):
                            self._in_use.add(id(resource))
                            self._total_acquisitions += 1
                            return resource
                        self._destroy_resource(resource)
                except queue.Empty:
                    self._total_timeouts += 1
                    raise PoolExhausted("Timeout waiting for resource")
            else:
                time.sleep(0.01)
    
    def release(self, resource: T) -> None:
        """Release a resource back to the pool."""
        with self._lock:
            self._in_use.discard(id(resource))
            
            if self._closed or self._get_size() > self.max_size:
                self._destroy_resource(resource)
            else:
                self._timestamps[id(resource)] = time.monotonic()
                self._pool.put(resource)
    
    @contextmanager
    def connection(self, timeout: Optional[float] = None):
        """Context manager for resource acquisition."""
        resource = self.acquire(timeout=timeout)
        try:
            yield resource
        finally:
            self.release(resource)
    
    def close(self) -> None:
        """Close the pool and release all resources."""
        self._closed = True
        with self._lock:
            while not self._pool.empty():
                try:
                    resource = self._pool.get_nowait()
                    self._destroy_resource(resource)
                except queue.Empty:
                    break
    
    def get_stats(self) -> PoolStats:
        with self._lock:
            return PoolStats(
                size=self._get_size(),
                available=self._pool.qsize(),
                in_use=len(self._in_use),
                created=self._total_created,
                destroyed=self._total_destroyed,
                acquisitions=self._total_acquisitions,
                timeouts=self._total_timeouts,
            )


class ConnectionPool(ResourcePool):
    """Specialized pool for database/network connections."""
    
    def __init__(
        self,
        connect_func: Callable[[], T],
        max_connections: int = 10,
        ping_func: Optional[Callable[[T], bool]] = None,
        close_func: Optional[Callable[[T], None]] = None,
        **kwargs,
    ):
        super().__init__(
            factory=connect_func,
            max_size=max_connections,
            validation_func=ping_func,
            cleanup_func=close_func,
            **kwargs,
        )


class ThreadPool:
    """Managed thread pool with resource limits."""
    
    def __init__(self, max_workers: int = 10, name: str = 'pool'):
        self.max_workers = max_workers
        self.name = name
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._active = 0
        self._total = 0
        self._lock = threading.Lock()
    
    def submit(self, func: Callable, *args, **kwargs):
        """Submit a task to the pool."""
        with self._lock:
            self._active += 1
            self._total += 1
        
        def wrapper():
            try:
                return func(*args, **kwargs)
            finally:
                with self._lock:
                    self._active -= 1
        
        return self._executor.submit(wrapper)
    
    def map(self, func: Callable, items, timeout: Optional[float] = None):
        """Map function over items."""
        return self._executor.map(func, items, timeout=timeout)
    
    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)
    
    def get_stats(self) -> dict:
        with self._lock:
            return {
                'max_workers': self.max_workers,
                'active_tasks': self._active,
                'total_submitted': self._total,
            }
