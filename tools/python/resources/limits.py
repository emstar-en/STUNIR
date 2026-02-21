"""Resource limits and quotas for STUNIR."""

import os
import time
import resource
import threading
from functools import wraps
from typing import Callable, Optional, TypeVar
from dataclasses import dataclass
from contextlib import contextmanager

T = TypeVar('T')


class ResourceLimitExceeded(Exception):
    """Resource limit was exceeded."""
    pass


@dataclass
class ResourceLimits:
    """Resource limit configuration."""
    max_memory_mb: Optional[float] = None
    max_time_seconds: Optional[float] = None
    max_cpu_percent: Optional[float] = None
    max_open_files: Optional[int] = None


class MemoryLimit:
    """Memory usage limiter."""
    
    def __init__(self, max_mb: float):
        self.max_bytes = int(max_mb * 1024 * 1024)
    
    def check(self) -> bool:
        """Check if memory usage is within limit."""
        try:
            usage = self._get_memory_usage()
            return usage < self.max_bytes
        except Exception:
            return True
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            with open(f'/proc/{os.getpid()}/statm', 'r') as f:
                return int(f.read().split()[1]) * os.sysconf('SC_PAGE_SIZE')
        except (IOError, IndexError):
            return 0
    
    def enforce(self) -> None:
        """Raise if memory limit exceeded."""
        if not self.check():
            usage_mb = self._get_memory_usage() / (1024 * 1024)
            raise ResourceLimitExceeded(
                f"Memory limit exceeded: {usage_mb:.1f}MB > {self.max_bytes / (1024 * 1024):.1f}MB"
            )
    
    @contextmanager
    def context(self):
        """Context manager for memory-limited execution."""
        self.enforce()
        yield
        self.enforce()


class TimeLimit:
    """Time limit for operations."""
    
    def __init__(self, max_seconds: float):
        self.max_seconds = max_seconds
        self._start: Optional[float] = None
    
    def start(self) -> None:
        self._start = time.monotonic()
    
    def check(self) -> bool:
        """Check if time limit is within bounds."""
        if self._start is None:
            return True
        return (time.monotonic() - self._start) < self.max_seconds
    
    def enforce(self) -> None:
        """Raise if time limit exceeded."""
        if not self.check():
            elapsed = time.monotonic() - self._start if self._start else 0
            raise ResourceLimitExceeded(
                f"Time limit exceeded: {elapsed:.1f}s > {self.max_seconds}s"
            )
    
    @contextmanager
    def context(self):
        """Context manager for time-limited execution."""
        self.start()
        try:
            yield self
        finally:
            pass


class ResourceQuota:
    """Combined resource quota enforcement."""
    
    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self._memory_limit = MemoryLimit(limits.max_memory_mb) if limits.max_memory_mb else None
        self._time_limit = TimeLimit(limits.max_time_seconds) if limits.max_time_seconds else None
    
    def check_all(self) -> dict:
        """Check all limits and return status."""
        status = {'within_limits': True, 'violations': []}
        
        if self._memory_limit and not self._memory_limit.check():
            status['within_limits'] = False
            status['violations'].append('memory')
        
        if self._time_limit and not self._time_limit.check():
            status['within_limits'] = False
            status['violations'].append('time')
        
        return status
    
    def enforce_all(self) -> None:
        """Enforce all limits."""
        if self._memory_limit:
            self._memory_limit.enforce()
        if self._time_limit:
            self._time_limit.enforce()
    
    @contextmanager
    def context(self):
        """Context manager for quota-limited execution."""
        if self._time_limit:
            self._time_limit.start()
        try:
            yield self
        finally:
            pass


def enforce_limits(
    max_memory_mb: Optional[float] = None,
    max_time_seconds: Optional[float] = None,
):
    """Decorator to enforce resource limits on a function.
    
    Example:
        @enforce_limits(max_memory_mb=100, max_time_seconds=30)
        def process_data(data):
            # Will raise if memory or time limit exceeded
            pass
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            limits = ResourceLimits(
                max_memory_mb=max_memory_mb,
                max_time_seconds=max_time_seconds,
            )
            quota = ResourceQuota(limits)
            
            with quota.context():
                result = func(*args, **kwargs)
                quota.enforce_all()
                return result
        
        return wrapper
    return decorator


def set_process_limits(
    max_memory_mb: Optional[float] = None,
    max_open_files: Optional[int] = None,
) -> None:
    """Set process-level resource limits (Unix only)."""
    try:
        if max_memory_mb:
            max_bytes = int(max_memory_mb * 1024 * 1024)
            resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
        
        if max_open_files:
            resource.setrlimit(resource.RLIMIT_NOFILE, (max_open_files, max_open_files))
    except (AttributeError, ValueError, resource.error):
        pass  # Not available on this platform
