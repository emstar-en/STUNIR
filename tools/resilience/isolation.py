"""Dependency isolation for STUNIR."""

import time
import threading
import asyncio
from functools import wraps
from typing import Any, Callable, Optional, TypeVar
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass

T = TypeVar('T')


class TimeoutError(Exception):
    """Operation timed out."""
    pass


class BulkheadFullError(Exception):
    """Bulkhead capacity exceeded."""
    pass


@dataclass
class BulkheadStats:
    """Bulkhead statistics."""
    active_calls: int
    queued_calls: int
    rejected_calls: int
    completed_calls: int
    max_concurrent: int


class Bulkhead:
    """Bulkhead pattern for limiting concurrent calls.
    
    Isolates failures by limiting the number of concurrent executions.
    
    Example:
        bulkhead = Bulkhead(max_concurrent=10, max_wait=5.0)
        
        @bulkhead
        def call_external_service():
            return requests.get(url)
    """
    
    def __init__(
        self,
        max_concurrent: int = 10,
        max_wait: float = 0.0,
        name: str = 'default',
    ):
        self.max_concurrent = max_concurrent
        self.max_wait = max_wait
        self.name = name
        self._semaphore = threading.Semaphore(max_concurrent)
        self._active = 0
        self._rejected = 0
        self._completed = 0
        self._lock = threading.Lock()
    
    def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire a permit from the bulkhead."""
        timeout = timeout if timeout is not None else self.max_wait
        acquired = self._semaphore.acquire(timeout=timeout if timeout > 0 else 0)
        
        with self._lock:
            if acquired:
                self._active += 1
            else:
                self._rejected += 1
        
        return acquired
    
    def release(self) -> None:
        """Release a permit back to the bulkhead."""
        with self._lock:
            self._active -= 1
            self._completed += 1
        self._semaphore.release()
    
    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function within bulkhead limits."""
        if not self.acquire():
            raise BulkheadFullError(f"Bulkhead '{self.name}' is at capacity")
        
        try:
            return func(*args, **kwargs)
        finally:
            self.release()
    
    def __call__(self, func: Callable) -> Callable:
        """Use as decorator."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute(func, *args, **kwargs)
        wrapper._bulkhead = self
        return wrapper
    
    def get_stats(self) -> BulkheadStats:
        with self._lock:
            return BulkheadStats(
                active_calls=self._active,
                queued_calls=0,
                rejected_calls=self._rejected,
                completed_calls=self._completed,
                max_concurrent=self.max_concurrent,
            )


class DependencyIsolator:
    """Isolate dependencies with dedicated thread pools.
    
    Example:
        isolator = DependencyIsolator(max_workers=5)
        
        @isolator.isolate(timeout=10.0)
        def call_slow_service():
            return slow_operation()
    """
    
    def __init__(self, max_workers: int = 10, name: str = 'default'):
        self.max_workers = max_workers
        self.name = name
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._active_futures = 0
        self._lock = threading.Lock()
    
    def execute(self, func: Callable[..., T], *args, timeout: Optional[float] = None, **kwargs) -> T:
        """Execute function in isolated thread pool."""
        future = self._executor.submit(func, *args, **kwargs)
        
        with self._lock:
            self._active_futures += 1
        
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            future.cancel()
            raise TimeoutError(f"Operation timed out after {timeout} seconds")
        finally:
            with self._lock:
                self._active_futures -= 1
    
    def isolate(self, timeout: Optional[float] = None):
        """Decorator for isolating function execution."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self.execute(func, *args, timeout=timeout, **kwargs)
            return wrapper
        return decorator
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool."""
        self._executor.shutdown(wait=wait)


def timeout(seconds: float):
    """Decorator to add timeout to synchronous functions.
    
    Example:
        @timeout(5.0)
        def slow_operation():
            time.sleep(10)  # Will raise TimeoutError
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=seconds)
            except FuturesTimeoutError:
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            finally:
                executor.shutdown(wait=False)
        return wrapper
    return decorator


def async_timeout(seconds: float):
    """Decorator to add timeout to async functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds
                )
            except asyncio.TimeoutError:
                raise TimeoutError(f"Async function {func.__name__} timed out after {seconds} seconds")
        return wrapper
    return decorator
