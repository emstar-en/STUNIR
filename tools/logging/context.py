"""Logging context management for STUNIR.

Provides context managers and utilities for scoped logging contexts.
"""

import threading
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional
from functools import wraps


# Thread-local context storage
_context_store = threading.local()


def _get_context_stack() -> list:
    """Get the context stack for the current thread."""
    if not hasattr(_context_store, 'stack'):
        _context_store.stack = [{}]
    return _context_store.stack


def get_current_context() -> Dict[str, Any]:
    """Get the current merged context.
    
    Returns:
        Merged context from all active scopes
    """
    stack = _get_context_stack()
    merged = {}
    for ctx in stack:
        merged.update(ctx)
    return merged


def set_context(**kwargs) -> None:
    """Set context values in the current scope.
    
    Args:
        **kwargs: Context key-value pairs to set
    """
    stack = _get_context_stack()
    if stack:
        stack[-1].update(kwargs)


def clear_context() -> None:
    """Clear all context in the current scope."""
    stack = _get_context_stack()
    if stack:
        stack[-1].clear()


class LogContext:
    """Context manager for scoped logging context.
    
    Usage:
        with LogContext(request_id='123', user='admin'):
            logger.info('Processing request')  # includes context
    """
    
    def __init__(self, **kwargs):
        self.context = kwargs
        self._token: Optional[int] = None
    
    def __enter__(self) -> 'LogContext':
        stack = _get_context_stack()
        stack.append(self.context.copy())
        self._token = len(stack)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        stack = _get_context_stack()
        if self._token and len(stack) >= self._token:
            stack.pop()
    
    def update(self, **kwargs) -> None:
        """Update context within the scope."""
        self.context.update(kwargs)
        stack = _get_context_stack()
        if stack and self._token:
            stack[self._token - 1].update(kwargs)


@contextmanager
def log_context(**kwargs) -> Generator[LogContext, None, None]:
    """Context manager for adding logging context.
    
    Args:
        **kwargs: Context key-value pairs
    
    Yields:
        LogContext instance for updates
    
    Example:
        with log_context(operation='sync', target='db'):
            logger.info('Starting sync')  # includes context
    """
    ctx = LogContext(**kwargs)
    with ctx:
        yield ctx


def capture_context(func):
    """Decorator to capture function execution context.
    
    Automatically adds function name and module to log context.
    
    Args:
        func: Function to wrap
    
    Returns:
        Wrapped function with logging context
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        ctx = {
            'function': func.__name__,
            'module': func.__module__,
        }
        with log_context(**ctx):
            return func(*args, **kwargs)
    return wrapper


class ContextAdapter:
    """Adapter to inject context into external loggers.
    
    Useful for integrating with libraries that use standard logging.
    """
    
    def __init__(self, logger):
        self._logger = logger
    
    def _inject_context(self, kwargs: dict) -> dict:
        """Inject current context into log kwargs."""
        extra = kwargs.get('extra', {})
        extra['structured'] = {
            **get_current_context(),
            **extra.get('structured', {}),
        }
        kwargs['extra'] = extra
        return kwargs
    
    def debug(self, msg, *args, **kwargs):
        self._logger.debug(msg, *args, **self._inject_context(kwargs))
    
    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, **self._inject_context(kwargs))
    
    def warning(self, msg, *args, **kwargs):
        self._logger.warning(msg, *args, **self._inject_context(kwargs))
    
    def error(self, msg, *args, **kwargs):
        self._logger.error(msg, *args, **self._inject_context(kwargs))
    
    def critical(self, msg, *args, **kwargs):
        self._logger.critical(msg, *args, **self._inject_context(kwargs))
