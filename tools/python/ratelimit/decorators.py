"""Rate limiting decorators for STUNIR."""

import time
import threading
from functools import wraps
from typing import Callable, Optional, TypeVar, Union

from .limiter import RateLimiter, TokenBucket, RateLimitExceeded

T = TypeVar('T')


def rate_limit(
    rate: float = 10.0,
    capacity: Optional[float] = None,
    limiter: Optional[RateLimiter] = None,
    block: bool = True,
    max_wait: float = 10.0,
):
    """Decorator to rate limit function calls.
    
    Args:
        rate: Requests per second
        capacity: Burst capacity (defaults to rate)
        limiter: Custom rate limiter instance
        block: Wait for rate limit if True, raise if False
        max_wait: Maximum time to wait when blocking
    
    Example:
        @rate_limit(rate=5)  # 5 calls per second
        def api_call():
            return requests.get(url)
    """
    _limiter = limiter or TokenBucket(rate=rate, capacity=capacity)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if block:
                if not _limiter.wait_and_acquire(max_wait=max_wait):
                    raise RateLimitExceeded(f"Rate limit exceeded for {func.__name__}")
            else:
                _limiter.acquire_or_raise()
            
            return func(*args, **kwargs)
        
        wrapper._rate_limiter = _limiter
        return wrapper
    
    return decorator


def throttle(calls: int, period: float = 1.0, raise_on_limit: bool = False):
    """Simple throttle decorator.
    
    Limits function to N calls per period.
    
    Args:
        calls: Maximum calls allowed
        period: Time period in seconds
        raise_on_limit: Raise exception instead of waiting
    
    Example:
        @throttle(10, 60)  # 10 calls per minute
        def send_email():
            pass
    """
    call_times = []
    lock = threading.Lock()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            nonlocal call_times
            
            with lock:
                now = time.monotonic()
                cutoff = now - period
                call_times = [t for t in call_times if t > cutoff]
                
                if len(call_times) >= calls:
                    if raise_on_limit:
                        wait_time = call_times[0] + period - now
                        raise RateLimitExceeded(
                            f"Throttle limit exceeded for {func.__name__}",
                            retry_after=wait_time
                        )
                    else:
                        wait_time = call_times[0] + period - now
                        time.sleep(max(0, wait_time))
                        # Recurse to check again
                        return wrapper(*args, **kwargs)
                
                call_times.append(now)
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


class RateLimitedClient:
    """Client wrapper with built-in rate limiting.
    
    Example:
        client = RateLimitedClient(api_client, rate=100)  # 100 req/sec
        result = client.call('get_user', user_id=123)
    """
    
    def __init__(self, client, rate: float = 10.0, capacity: Optional[float] = None):
        self._client = client
        self._limiter = TokenBucket(rate=rate, capacity=capacity)
    
    def call(self, method: str, *args, **kwargs):
        """Call a method on the wrapped client with rate limiting."""
        self._limiter.wait_and_acquire()
        return getattr(self._client, method)(*args, **kwargs)
    
    def __getattr__(self, name: str):
        """Proxy attribute access with rate limiting."""
        attr = getattr(self._client, name)
        if callable(attr):
            @wraps(attr)
            def wrapper(*args, **kwargs):
                self._limiter.wait_and_acquire()
                return attr(*args, **kwargs)
            return wrapper
        return attr
