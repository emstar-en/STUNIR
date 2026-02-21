"""Main retry decorator and utilities for STUNIR."""

import asyncio
import time
import threading
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Type
from dataclasses import dataclass

from .backoff import BackoffStrategy, ExponentialBackoff
from .policies import RetryPolicy, MaxAttemptsPolicy


class RetryError(Exception):
    """Raised when all retry attempts are exhausted."""
    
    def __init__(self, message: str, last_exception: Optional[Exception] = None, attempts: int = 0):
        super().__init__(message)
        self.last_exception = last_exception
        self.attempts = attempts


@dataclass
class RetryStats:
    """Statistics for retry operations."""
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    total_retries: int = 0
    total_time: float = 0.0
    last_error: Optional[Exception] = None
    
    def record_success(self, attempts: int, duration: float) -> None:
        self.total_attempts += 1
        self.successful_attempts += 1
        self.total_retries += attempts - 1
        self.total_time += duration
    
    def record_failure(self, attempts: int, duration: float, error: Exception) -> None:
        self.total_attempts += 1
        self.failed_attempts += 1
        self.total_retries += attempts
        self.total_time += duration
        self.last_error = error


class Retry:
    """Configurable retry handler."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        backoff: Optional[BackoffStrategy] = None,
        policy: Optional[RetryPolicy] = None,
        retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
        on_retry: Optional[Callable[[int, Exception, float], None]] = None,
        on_success: Optional[Callable[[int, float], None]] = None,
        on_failure: Optional[Callable[[int, Exception, float], None]] = None,
    ):
        self.max_attempts = max_attempts
        self.backoff = backoff or ExponentialBackoff()
        self.policy = policy or MaxAttemptsPolicy(max_attempts)
        self.retryable_exceptions = retryable_exceptions
        self.on_retry = on_retry
        self.on_success = on_success
        self.on_failure = on_failure
        self.stats = RetryStats()
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        attempt = 0
        start_time = time.monotonic()
        last_exception: Optional[Exception] = None
        
        while True:
            attempt += 1
            try:
                result = func(*args, **kwargs)
                duration = time.monotonic() - start_time
                with self._lock:
                    self.stats.record_success(attempt, duration)
                if self.on_success:
                    self.on_success(attempt, duration)
                return result
            except self.retryable_exceptions as e:
                last_exception = e
                if not self.policy.should_retry(attempt, e, time.monotonic() - start_time):
                    break
                delay = self.backoff.get_delay(attempt)
                if self.on_retry:
                    self.on_retry(attempt, e, delay)
                time.sleep(delay)
        
        duration = time.monotonic() - start_time
        with self._lock:
            self.stats.record_failure(attempt, duration, last_exception)
        if self.on_failure:
            self.on_failure(attempt, last_exception, duration)
        raise RetryError(f"Failed after {attempt} attempts", last_exception, attempt)
    
    async def async_call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with retry logic."""
        attempt = 0
        start_time = time.monotonic()
        last_exception: Optional[Exception] = None
        
        while True:
            attempt += 1
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                duration = time.monotonic() - start_time
                with self._lock:
                    self.stats.record_success(attempt, duration)
                if self.on_success:
                    self.on_success(attempt, duration)
                return result
            except self.retryable_exceptions as e:
                last_exception = e
                if not self.policy.should_retry(attempt, e, time.monotonic() - start_time):
                    break
                delay = self.backoff.get_delay(attempt)
                if self.on_retry:
                    self.on_retry(attempt, e, delay)
                await asyncio.sleep(delay)
        
        duration = time.monotonic() - start_time
        with self._lock:
            self.stats.record_failure(attempt, duration, last_exception)
        if self.on_failure:
            self.on_failure(attempt, last_exception, duration)
        raise RetryError(f"Failed after {attempt} attempts", last_exception, attempt)
    
    def get_stats(self) -> RetryStats:
        with self._lock:
            return RetryStats(
                total_attempts=self.stats.total_attempts,
                successful_attempts=self.stats.successful_attempts,
                failed_attempts=self.stats.failed_attempts,
                total_retries=self.stats.total_retries,
                total_time=self.stats.total_time,
                last_error=self.stats.last_error,
            )


def retry(
    max_attempts: int = 3,
    backoff: Optional[BackoffStrategy] = None,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
):
    """Decorator for adding retry logic to functions."""
    def decorator(func: Callable) -> Callable:
        retry_handler = Retry(
            max_attempts=max_attempts,
            backoff=backoff,
            retryable_exceptions=retryable_exceptions,
            on_retry=on_retry,
        )
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return retry_handler.call(func, *args, **kwargs)
        
        wrapper._retry_handler = retry_handler
        return wrapper
    return decorator


def async_retry(
    max_attempts: int = 3,
    backoff: Optional[BackoffStrategy] = None,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
):
    """Decorator for adding retry logic to async functions."""
    def decorator(func: Callable) -> Callable:
        retry_handler = Retry(
            max_attempts=max_attempts,
            backoff=backoff,
            retryable_exceptions=retryable_exceptions,
            on_retry=on_retry,
        )
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_handler.async_call(func, *args, **kwargs)
        
        wrapper._retry_handler = retry_handler
        return wrapper
    return decorator
