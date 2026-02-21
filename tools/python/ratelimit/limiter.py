"""Rate limiting algorithms for STUNIR."""

import time
import threading
from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass
from collections import deque


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: float = 0):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class RateLimitStats:
    """Rate limiter statistics."""
    total_requests: int
    allowed_requests: int
    denied_requests: int
    current_rate: float


class RateLimiter(ABC):
    """Abstract base class for rate limiters."""
    
    @abstractmethod
    def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens. Returns True if allowed."""
        pass
    
    @abstractmethod
    def get_wait_time(self, tokens: int = 1) -> float:
        """Get time to wait before tokens are available."""
        pass
    
    def acquire_or_raise(self, tokens: int = 1) -> None:
        """Acquire tokens or raise RateLimitExceeded."""
        if not self.acquire(tokens):
            raise RateLimitExceeded(retry_after=self.get_wait_time(tokens))
    
    def wait_and_acquire(self, tokens: int = 1, max_wait: float = 10.0) -> bool:
        """Wait and acquire tokens."""
        wait_time = self.get_wait_time(tokens)
        if wait_time > max_wait:
            return False
        if wait_time > 0:
            time.sleep(wait_time)
        return self.acquire(tokens)


class TokenBucket(RateLimiter):
    """Token bucket rate limiter.
    
    Allows bursts up to bucket capacity, then limits to refill rate.
    
    Example:
        limiter = TokenBucket(rate=10, capacity=50)  # 10/sec, burst of 50
        if limiter.acquire():
            process_request()
    """
    
    def __init__(self, rate: float, capacity: Optional[float] = None):
        self.rate = rate  # tokens per second
        self.capacity = capacity or rate  # max tokens
        self._tokens = self.capacity
        self._last_update = time.monotonic()
        self._lock = threading.Lock()
        self._stats = {'total': 0, 'allowed': 0, 'denied': 0}
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_update = now
    
    def acquire(self, tokens: int = 1) -> bool:
        with self._lock:
            self._refill()
            self._stats['total'] += 1
            
            if self._tokens >= tokens:
                self._tokens -= tokens
                self._stats['allowed'] += 1
                return True
            
            self._stats['denied'] += 1
            return False
    
    def get_wait_time(self, tokens: int = 1) -> float:
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                return 0.0
            needed = tokens - self._tokens
            return needed / self.rate
    
    def get_stats(self) -> RateLimitStats:
        with self._lock:
            return RateLimitStats(
                total_requests=self._stats['total'],
                allowed_requests=self._stats['allowed'],
                denied_requests=self._stats['denied'],
                current_rate=self.rate,
            )


class LeakyBucket(RateLimiter):
    """Leaky bucket rate limiter.
    
    Smooths out bursts by processing at a constant rate.
    """
    
    def __init__(self, rate: float, capacity: int = 100):
        self.rate = rate  # requests per second
        self.capacity = capacity
        self._queue: deque = deque()
        self._last_leak = time.monotonic()
        self._lock = threading.Lock()
        self._stats = {'total': 0, 'allowed': 0, 'denied': 0}
    
    def _leak(self) -> None:
        """Remove leaked items from queue."""
        now = time.monotonic()
        elapsed = now - self._last_leak
        to_leak = int(elapsed * self.rate)
        
        for _ in range(min(to_leak, len(self._queue))):
            self._queue.popleft()
        
        if to_leak > 0:
            self._last_leak = now
    
    def acquire(self, tokens: int = 1) -> bool:
        with self._lock:
            self._leak()
            self._stats['total'] += 1
            
            if len(self._queue) + tokens <= self.capacity:
                for _ in range(tokens):
                    self._queue.append(time.monotonic())
                self._stats['allowed'] += 1
                return True
            
            self._stats['denied'] += 1
            return False
    
    def get_wait_time(self, tokens: int = 1) -> float:
        with self._lock:
            self._leak()
            if len(self._queue) + tokens <= self.capacity:
                return 0.0
            overflow = len(self._queue) + tokens - self.capacity
            return overflow / self.rate
    
    def get_stats(self) -> RateLimitStats:
        with self._lock:
            return RateLimitStats(
                total_requests=self._stats['total'],
                allowed_requests=self._stats['allowed'],
                denied_requests=self._stats['denied'],
                current_rate=self.rate,
            )


class FixedWindow(RateLimiter):
    """Fixed window rate limiter.
    
    Limits requests within fixed time windows.
    """
    
    def __init__(self, limit: int, window_size: float = 60.0):
        self.limit = limit
        self.window_size = window_size
        self._count = 0
        self._window_start = time.monotonic()
        self._lock = threading.Lock()
        self._stats = {'total': 0, 'allowed': 0, 'denied': 0}
    
    def _check_window(self) -> None:
        """Reset window if expired."""
        now = time.monotonic()
        if now - self._window_start >= self.window_size:
            self._count = 0
            self._window_start = now
    
    def acquire(self, tokens: int = 1) -> bool:
        with self._lock:
            self._check_window()
            self._stats['total'] += 1
            
            if self._count + tokens <= self.limit:
                self._count += tokens
                self._stats['allowed'] += 1
                return True
            
            self._stats['denied'] += 1
            return False
    
    def get_wait_time(self, tokens: int = 1) -> float:
        with self._lock:
            self._check_window()
            if self._count + tokens <= self.limit:
                return 0.0
            elapsed = time.monotonic() - self._window_start
            return self.window_size - elapsed
    
    def get_stats(self) -> RateLimitStats:
        with self._lock:
            return RateLimitStats(
                total_requests=self._stats['total'],
                allowed_requests=self._stats['allowed'],
                denied_requests=self._stats['denied'],
                current_rate=self.limit / self.window_size,
            )


class SlidingWindow(RateLimiter):
    """Sliding window rate limiter.
    
    More accurate than fixed window by tracking individual request times.
    """
    
    def __init__(self, limit: int, window_size: float = 60.0):
        self.limit = limit
        self.window_size = window_size
        self._requests: deque = deque()
        self._lock = threading.Lock()
        self._stats = {'total': 0, 'allowed': 0, 'denied': 0}
    
    def _clean_old(self) -> None:
        """Remove requests outside the window."""
        cutoff = time.monotonic() - self.window_size
        while self._requests and self._requests[0] < cutoff:
            self._requests.popleft()
    
    def acquire(self, tokens: int = 1) -> bool:
        with self._lock:
            self._clean_old()
            self._stats['total'] += 1
            
            if len(self._requests) + tokens <= self.limit:
                now = time.monotonic()
                for _ in range(tokens):
                    self._requests.append(now)
                self._stats['allowed'] += 1
                return True
            
            self._stats['denied'] += 1
            return False
    
    def get_wait_time(self, tokens: int = 1) -> float:
        with self._lock:
            self._clean_old()
            if len(self._requests) + tokens <= self.limit:
                return 0.0
            if not self._requests:
                return 0.0
            oldest = self._requests[0]
            return (oldest + self.window_size) - time.monotonic()
    
    def get_stats(self) -> RateLimitStats:
        with self._lock:
            return RateLimitStats(
                total_requests=self._stats['total'],
                allowed_requests=self._stats['allowed'],
                denied_requests=self._stats['denied'],
                current_rate=self.limit / self.window_size,
            )
