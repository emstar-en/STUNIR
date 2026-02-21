"""Fallback mechanisms for graceful degradation."""

import time
import threading
from functools import wraps
from typing import Any, Callable, List, Optional, TypeVar, Union
from dataclasses import dataclass

T = TypeVar('T')


@dataclass
class FallbackResult:
    """Result of a fallback operation."""
    value: Any
    source: str
    fallback_used: bool
    error: Optional[Exception] = None


class Fallback:
    """Fallback handler with default value or function.
    
    Example:
        fb = Fallback(default_value=[], on_error=log_error)
        result = fb.call(fetch_data, url)
    """
    
    def __init__(
        self,
        default_value: Any = None,
        default_func: Optional[Callable[..., T]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        expected_exceptions: tuple = (Exception,),
    ):
        self.default_value = default_value
        self.default_func = default_func
        self.on_error = on_error
        self.expected_exceptions = expected_exceptions
        self._fallback_count = 0
        self._lock = threading.Lock()
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> FallbackResult:
        """Call function with fallback on error."""
        try:
            result = func(*args, **kwargs)
            return FallbackResult(value=result, source='primary', fallback_used=False)
        except self.expected_exceptions as e:
            with self._lock:
                self._fallback_count += 1
            
            if self.on_error:
                self.on_error(e)
            
            if self.default_func:
                try:
                    fallback_value = self.default_func(*args, **kwargs)
                    return FallbackResult(
                        value=fallback_value, source='fallback_func',
                        fallback_used=True, error=e
                    )
                except Exception:
                    pass
            
            return FallbackResult(
                value=self.default_value, source='default',
                fallback_used=True, error=e
            )
    
    @property
    def fallback_count(self) -> int:
        with self._lock:
            return self._fallback_count
    
    def __call__(self, func: Callable) -> Callable:
        """Use as decorator."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs).value
        wrapper._fallback = self
        return wrapper


def fallback(
    default: Any = None,
    fallback_func: Optional[Callable] = None,
    exceptions: tuple = (Exception,),
):
    """Decorator for adding fallback to functions."""
    fb = Fallback(
        default_value=default,
        default_func=fallback_func,
        expected_exceptions=exceptions,
    )
    return fb


class FallbackChain:
    """Chain of fallback options tried in order.
    
    Example:
        chain = FallbackChain([
            fetch_from_cache,
            fetch_from_primary_db,
            fetch_from_replica_db,
        ], default=[])
        result = chain.execute(user_id=123)
    """
    
    def __init__(
        self,
        handlers: List[Callable],
        default: Any = None,
        on_error: Optional[Callable[[int, Exception], None]] = None,
    ):
        self.handlers = handlers
        self.default = default
        self.on_error = on_error
        self._stats = {'calls': 0, 'fallbacks': {}}
        self._lock = threading.Lock()
    
    def execute(self, *args, **kwargs) -> FallbackResult:
        """Execute handlers in order until one succeeds."""
        with self._lock:
            self._stats['calls'] += 1
        
        last_error = None
        for i, handler in enumerate(self.handlers):
            try:
                result = handler(*args, **kwargs)
                source = f"handler_{i}" if i == 0 else f"fallback_{i}"
                return FallbackResult(
                    value=result, source=source,
                    fallback_used=(i > 0), error=last_error
                )
            except Exception as e:
                last_error = e
                with self._lock:
                    key = f"handler_{i}"
                    self._stats['fallbacks'][key] = self._stats['fallbacks'].get(key, 0) + 1
                
                if self.on_error:
                    self.on_error(i, e)
        
        return FallbackResult(
            value=self.default, source='default',
            fallback_used=True, error=last_error
        )
    
    def get_stats(self) -> dict:
        with self._lock:
            return self._stats.copy()


class PartialFailureHandler:
    """Handle partial failures in batch operations."""
    
    def __init__(
        self,
        func: Callable,
        on_item_error: Optional[Callable[[Any, Exception], Any]] = None,
        min_success_rate: float = 0.5,
    ):
        self.func = func
        self.on_item_error = on_item_error
        self.min_success_rate = min_success_rate
    
    def process_batch(self, items: List[Any]) -> dict:
        """Process batch with partial failure handling."""
        results = []
        errors = []
        
        for item in items:
            try:
                result = self.func(item)
                results.append({'item': item, 'result': result, 'success': True})
            except Exception as e:
                if self.on_item_error:
                    fallback_value = self.on_item_error(item, e)
                    results.append({'item': item, 'result': fallback_value, 'success': False})
                else:
                    results.append({'item': item, 'result': None, 'success': False})
                errors.append({'item': item, 'error': str(e)})
        
        success_count = sum(1 for r in results if r['success'])
        success_rate = success_count / len(items) if items else 1.0
        
        return {
            'results': results,
            'errors': errors,
            'success_count': success_count,
            'total_count': len(items),
            'success_rate': success_rate,
            'meets_threshold': success_rate >= self.min_success_rate,
        }
