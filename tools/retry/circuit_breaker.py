"""Circuit breaker pattern implementation for STUNIR."""

import threading
import time
from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Type
from functools import wraps


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()      # Normal operation
    OPEN = auto()        # Failing fast
    HALF_OPEN = auto()   # Testing recovery


class CircuitOpenError(Exception):
    """Raised when circuit is open and requests are being rejected."""
    
    def __init__(self, message: str = "Circuit is open", time_until_retry: float = 0):
        super().__init__(message)
        self.time_until_retry = time_until_retry


@dataclass
class CircuitStats:
    """Circuit breaker statistics."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None


class CircuitBreaker:
    """Circuit breaker for preventing cascading failures.
    
    Example:
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)
        
        @breaker
        def call_external_service():
            return requests.get('http://api.example.com')
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        recovery_timeout: float = 30.0,
        expected_exceptions: Tuple[Type[Exception], ...] = (Exception,),
        on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None,
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exceptions = expected_exceptions
        self.on_state_change = on_state_change
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.RLock()
        self.stats = CircuitStats()
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to(CircuitState.HALF_OPEN)
            return self._state
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return True
        return (time.time() - self._last_failure_time) >= self.recovery_timeout
    
    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        if old_state != new_state:
            self._state = new_state
            self.stats.state_changes += 1
            if self.on_state_change:
                self.on_state_change(old_state, new_state)
    
    def _record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self.stats.total_calls += 1
            self.stats.successful_calls += 1
            self.stats.last_success_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
                    self._failure_count = 0
                    self._success_count = 0
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0
    
    def _record_failure(self, exception: Exception) -> None:
        """Record a failed call."""
        with self._lock:
            self.stats.total_calls += 1
            self.stats.failed_calls += 1
            self.stats.last_failure_time = time.time()
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
                self._success_count = 0
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through the circuit breaker."""
        with self._lock:
            current_state = self.state
            if current_state == CircuitState.OPEN:
                self.stats.rejected_calls += 1
                time_until = self.recovery_timeout
                if self._last_failure_time:
                    time_until = max(0, self.recovery_timeout - (time.time() - self._last_failure_time))
                raise CircuitOpenError("Circuit is open", time_until)
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except self.expected_exceptions as e:
            self._record_failure(e)
            raise
    
    def __call__(self, func: Callable) -> Callable:
        """Use as decorator."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        wrapper._circuit_breaker = self
        return wrapper
    
    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
    
    def get_stats(self) -> CircuitStats:
        """Get circuit breaker statistics."""
        with self._lock:
            return CircuitStats(
                total_calls=self.stats.total_calls,
                successful_calls=self.stats.successful_calls,
                failed_calls=self.stats.failed_calls,
                rejected_calls=self.stats.rejected_calls,
                state_changes=self.stats.state_changes,
                last_failure_time=self.stats.last_failure_time,
                last_success_time=self.stats.last_success_time,
            )


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    expected_exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """Decorator factory for circuit breaker."""
    breaker = CircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exceptions=expected_exceptions,
    )
    return breaker
