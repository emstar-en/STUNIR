"""Main logging interface for STUNIR.

Provides a centralized logging system with support for:
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Structured JSON output
- Context-aware logging
- Configuration from files or environment
"""

import logging
import os
import json
import threading
from enum import IntEnum
from typing import Any, Dict, Optional, Union
from pathlib import Path
from contextlib import contextmanager


class LogLevel(IntEnum):
    """Log level enumeration."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


# Thread-local storage for context
_context_local = threading.local()


def _get_context() -> Dict[str, Any]:
    """Get current thread-local context."""
    if not hasattr(_context_local, 'context'):
        _context_local.context = {}
    return _context_local.context


def _set_context(context: Dict[str, Any]) -> None:
    """Set thread-local context."""
    _context_local.context = context


class StunirLogger:
    """Enhanced logger with structured logging support."""
    
    def __init__(self, name: str, level: Union[int, LogLevel] = LogLevel.INFO):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._extra_context: Dict[str, Any] = {}
    
    def set_level(self, level: Union[int, LogLevel]) -> None:
        """Set the log level."""
        self._logger.setLevel(level)
    
    def add_context(self, **kwargs) -> None:
        """Add persistent context to all log messages."""
        self._extra_context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear persistent context."""
        self._extra_context.clear()
    
    def _log(self, level: int, msg: str, *args, **kwargs) -> None:
        """Internal log method with context injection."""
        extra = kwargs.pop('extra', {})
        # Merge contexts: thread-local + instance + call-specific
        merged_extra = {
            **_get_context(),
            **self._extra_context,
            **extra,
        }
        kwargs['extra'] = {'structured': merged_extra}
        
        exc_info = kwargs.pop('exc_info', None)
        if exc_info:
            kwargs['exc_info'] = exc_info
        
        self._logger.log(level, msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log a debug message."""
        self._log(LogLevel.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs) -> None:
        """Log an info message."""
        self._log(LogLevel.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log a warning message."""
        self._log(LogLevel.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs) -> None:
        """Log an error message."""
        self._log(LogLevel.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs) -> None:
        """Log a critical message."""
        self._log(LogLevel.CRITICAL, msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs) -> None:
        """Log an exception with traceback."""
        kwargs['exc_info'] = True
        self._log(LogLevel.ERROR, msg, *args, **kwargs)
    
    @contextmanager
    def scope(self, **context):
        """Context manager for scoped logging context."""
        old_context = self._extra_context.copy()
        self._extra_context.update(context)
        try:
            yield self
        finally:
            self._extra_context = old_context


# Logger registry
_loggers: Dict[str, StunirLogger] = {}
_lock = threading.Lock()


def get_logger(name: str = 'stunir') -> StunirLogger:
    """Get or create a logger instance.
    
    Args:
        name: Logger name (usually module path)
    
    Returns:
        StunirLogger instance
    """
    with _lock:
        if name not in _loggers:
            _loggers[name] = StunirLogger(name)
        return _loggers[name]


def configure_logging(
    level: Union[int, str, LogLevel] = LogLevel.INFO,
    json_output: bool = False,
    log_file: Optional[str] = None,
    config_file: Optional[str] = None,
    **kwargs
) -> None:
    """Configure the logging system.
    
    Args:
        level: Default log level
        json_output: Enable JSON formatted output
        log_file: Path to log file (enables file logging)
        config_file: Path to logging config file (YAML/JSON)
        **kwargs: Additional configuration options
    """
    from .formatters import JsonFormatter, ColoredFormatter
    from .handlers import RotatingFileHandler
    
    # Parse level if string
    if isinstance(level, str):
        level = LogLevel[level.upper()]
    
    # Load from config file if provided
    if config_file:
        config_path = Path(config_file)
        if config_path.exists():
            if config_path.suffix in ('.yaml', '.yml'):
                try:
                    import yaml
                    with open(config_path) as f:
                        config = yaml.safe_load(f)
                except ImportError:
                    config = {}
            else:
                with open(config_path) as f:
                    config = json.load(f)
            
            level = config.get('level', level)
            json_output = config.get('json_output', json_output)
            log_file = config.get('log_file', log_file)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    if json_output:
        console_handler.setFormatter(JsonFormatter())
    else:
        console_handler.setFormatter(ColoredFormatter())
    
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = RotatingFileHandler(
            filename=log_file,
            max_bytes=kwargs.get('max_bytes', 10 * 1024 * 1024),
            backup_count=kwargs.get('backup_count', 5),
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(JsonFormatter())
        root_logger.addHandler(file_handler)
    
    # Also respect environment variable
    env_level = os.environ.get('STUNIR_LOG_LEVEL')
    if env_level:
        try:
            root_logger.setLevel(LogLevel[env_level.upper()])
        except KeyError:
            pass
