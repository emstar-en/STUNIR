"""Retry policies for STUNIR."""

import time
from abc import ABC, abstractmethod
from typing import Callable, List, Optional


class RetryPolicy(ABC):
    """Abstract base class for retry policies."""
    
    @abstractmethod
    def should_retry(self, attempt: int, exception: Exception, elapsed: float) -> bool:
        """Determine if retry should be performed."""
        pass
    
    def reset(self) -> None:
        """Reset policy state."""
        pass


class MaxAttemptsPolicy(RetryPolicy):
    """Retry up to a maximum number of attempts."""
    
    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts
    
    def should_retry(self, attempt: int, exception: Exception, elapsed: float) -> bool:
        return attempt < self.max_attempts


class TimeoutPolicy(RetryPolicy):
    """Retry until a timeout is reached."""
    
    def __init__(self, timeout: float):
        self.timeout = timeout
    
    def should_retry(self, attempt: int, exception: Exception, elapsed: float) -> bool:
        return elapsed < self.timeout


class ConditionalPolicy(RetryPolicy):
    """Retry based on exception type or custom condition."""
    
    def __init__(
        self,
        retryable_exceptions: Optional[tuple] = None,
        non_retryable_exceptions: Optional[tuple] = None,
        condition: Optional[Callable[[Exception], bool]] = None,
        max_attempts: int = 3,
    ):
        self.retryable_exceptions = retryable_exceptions or (Exception,)
        self.non_retryable_exceptions = non_retryable_exceptions or ()
        self.condition = condition
        self.max_attempts = max_attempts
    
    def should_retry(self, attempt: int, exception: Exception, elapsed: float) -> bool:
        if attempt >= self.max_attempts:
            return False
        if isinstance(exception, self.non_retryable_exceptions):
            return False
        if not isinstance(exception, self.retryable_exceptions):
            return False
        if self.condition and not self.condition(exception):
            return False
        return True


class CompositePolicy(RetryPolicy):
    """Combine multiple policies with AND/OR logic."""
    
    def __init__(self, policies: List[RetryPolicy], require_all: bool = True):
        self.policies = policies
        self.require_all = require_all
    
    def should_retry(self, attempt: int, exception: Exception, elapsed: float) -> bool:
        results = [p.should_retry(attempt, exception, elapsed) for p in self.policies]
        return all(results) if self.require_all else any(results)
    
    def reset(self) -> None:
        for policy in self.policies:
            policy.reset()


class RateLimitedPolicy(RetryPolicy):
    """Limit retry rate to prevent overwhelming the target."""
    
    def __init__(
        self,
        base_policy: Optional[RetryPolicy] = None,
        min_interval: float = 1.0,
        max_retries_per_minute: int = 60,
    ):
        self.base_policy = base_policy or MaxAttemptsPolicy()
        self.min_interval = min_interval
        self.max_retries_per_minute = max_retries_per_minute
        self._retry_times: List[float] = []
    
    def should_retry(self, attempt: int, exception: Exception, elapsed: float) -> bool:
        if not self.base_policy.should_retry(attempt, exception, elapsed):
            return False
        now = time.time()
        cutoff = now - 60
        self._retry_times = [t for t in self._retry_times if t > cutoff]
        if len(self._retry_times) >= self.max_retries_per_minute:
            return False
        self._retry_times.append(now)
        return True
    
    def reset(self) -> None:
        self._retry_times.clear()
        self.base_policy.reset()
