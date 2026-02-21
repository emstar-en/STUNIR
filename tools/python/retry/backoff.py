"""Backoff strategies for retry logic."""

import random
from abc import ABC, abstractmethod
from typing import Optional


class BackoffStrategy(ABC):
    """Abstract base class for backoff strategies."""
    
    @abstractmethod
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt number."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the backoff state."""
        pass


class ConstantBackoff(BackoffStrategy):
    """Constant delay between retries."""
    
    def __init__(self, delay: float = 1.0):
        self.delay = delay
    
    def get_delay(self, attempt: int) -> float:
        return self.delay
    
    def reset(self) -> None:
        pass


class LinearBackoff(BackoffStrategy):
    """Linear increasing delay: base + (attempt - 1) * increment."""
    
    def __init__(self, base: float = 1.0, increment: float = 1.0, max_delay: Optional[float] = None):
        self.base = base
        self.increment = increment
        self.max_delay = max_delay
    
    def get_delay(self, attempt: int) -> float:
        delay = self.base + (attempt - 1) * self.increment
        if self.max_delay is not None:
            delay = min(delay, self.max_delay)
        return delay
    
    def reset(self) -> None:
        pass


class ExponentialBackoff(BackoffStrategy):
    """Exponential backoff: base * (multiplier ^ (attempt - 1))."""
    
    def __init__(self, base: float = 1.0, multiplier: float = 2.0, max_delay: float = 60.0):
        self.base = base
        self.multiplier = multiplier
        self.max_delay = max_delay
    
    def get_delay(self, attempt: int) -> float:
        delay = self.base * (self.multiplier ** (attempt - 1))
        return min(delay, self.max_delay)
    
    def reset(self) -> None:
        pass


class JitteredBackoff(BackoffStrategy):
    """Backoff with random jitter to prevent thundering herd."""
    
    def __init__(self, base_strategy: Optional[BackoffStrategy] = None, jitter_factor: float = 0.25):
        self.base_strategy = base_strategy or ExponentialBackoff()
        self.jitter_factor = jitter_factor
    
    def get_delay(self, attempt: int) -> float:
        base_delay = self.base_strategy.get_delay(attempt)
        jitter_range = base_delay * self.jitter_factor
        jitter = random.uniform(-jitter_range, jitter_range)
        return max(0.0, base_delay + jitter)
    
    def reset(self) -> None:
        self.base_strategy.reset()


class FibonacciBackoff(BackoffStrategy):
    """Fibonacci-based backoff (smoother than exponential)."""
    
    def __init__(self, base: float = 1.0, max_delay: float = 60.0):
        self.base = base
        self.max_delay = max_delay
        self._cache = {1: 1, 2: 1}
    
    def _fib(self, n: int) -> int:
        if n in self._cache:
            return self._cache[n]
        result = self._fib(n - 1) + self._fib(n - 2)
        self._cache[n] = result
        return result
    
    def get_delay(self, attempt: int) -> float:
        delay = self.base * self._fib(attempt)
        return min(delay, self.max_delay)
    
    def reset(self) -> None:
        self._cache = {1: 1, 2: 1}


class DecorrelatedJitterBackoff(BackoffStrategy):
    """AWS-recommended decorrelated jitter backoff."""
    
    def __init__(self, base: float = 1.0, max_delay: float = 60.0):
        self.base = base
        self.max_delay = max_delay
        self._prev_delay = base
    
    def get_delay(self, attempt: int) -> float:
        delay = random.uniform(self.base, self._prev_delay * 3)
        delay = min(delay, self.max_delay)
        self._prev_delay = delay
        return delay
    
    def reset(self) -> None:
        self._prev_delay = self.base
