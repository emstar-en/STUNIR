"""Log filtering utilities for STUNIR.

Provides filters for log level, context matching, and sampling.
"""

import logging
import random
import re
from typing import Any, Callable, Dict, List, Optional, Pattern, Union


class LevelFilter(logging.Filter):
    """Filter log records by level range.
    
    Allows filtering to specific level ranges, e.g., only INFO to WARNING.
    """
    
    def __init__(
        self,
        min_level: int = logging.DEBUG,
        max_level: int = logging.CRITICAL,
        exclude_levels: Optional[List[int]] = None,
    ):
        super().__init__()
        self.min_level = min_level
        self.max_level = max_level
        self.exclude_levels = set(exclude_levels or [])
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter based on log level."""
        if record.levelno in self.exclude_levels:
            return False
        return self.min_level <= record.levelno <= self.max_level


class ContextFilter(logging.Filter):
    """Filter log records by context attributes.
    
    Supports matching on structured context data.
    """
    
    def __init__(
        self,
        require: Optional[Dict[str, Any]] = None,
        exclude: Optional[Dict[str, Any]] = None,
        patterns: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.require = require or {}
        self.exclude = exclude or {}
        self.patterns: Dict[str, Pattern] = {
            k: re.compile(v) for k, v in (patterns or {}).items()
        }
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter based on context attributes."""
        context = getattr(record, 'structured', {})
        
        # Check required attributes
        for key, value in self.require.items():
            if context.get(key) != value:
                return False
        
        # Check excluded attributes
        for key, value in self.exclude.items():
            if context.get(key) == value:
                return False
        
        # Check pattern matches
        for key, pattern in self.patterns.items():
            ctx_value = context.get(key)
            if ctx_value is None:
                return False
            if not pattern.search(str(ctx_value)):
                return False
        
        return True


class SamplingFilter(logging.Filter):
    """Sample log records at a configurable rate.
    
    Useful for high-volume logging where only a sample is needed.
    """
    
    def __init__(
        self,
        sample_rate: float = 1.0,
        always_log_levels: Optional[List[int]] = None,
    ):
        super().__init__()
        self.sample_rate = min(1.0, max(0.0, sample_rate))
        self.always_log_levels = set(always_log_levels or [logging.ERROR, logging.CRITICAL])
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Sample based on configured rate."""
        # Always log certain levels
        if record.levelno in self.always_log_levels:
            return True
        
        # Sample at configured rate
        return random.random() < self.sample_rate


class MessageFilter(logging.Filter):
    """Filter log records by message content."""
    
    def __init__(
        self,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ):
        super().__init__()
        self.include_patterns = [
            re.compile(p) for p in (include_patterns or [])
        ]
        self.exclude_patterns = [
            re.compile(p) for p in (exclude_patterns or [])
        ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter based on message content."""
        message = record.getMessage()
        
        # Check exclusions first
        for pattern in self.exclude_patterns:
            if pattern.search(message):
                return False
        
        # If no include patterns, allow all (not excluded)
        if not self.include_patterns:
            return True
        
        # Must match at least one include pattern
        for pattern in self.include_patterns:
            if pattern.search(message):
                return True
        
        return False


class RateLimitFilter(logging.Filter):
    """Rate-limit repeated log messages.
    
    Suppresses duplicate messages that occur too frequently.
    """
    
    def __init__(
        self,
        rate_limit: float = 1.0,  # messages per second
        burst: int = 5,
    ):
        super().__init__()
        self.rate_limit = rate_limit
        self.burst = burst
        self._message_counts: Dict[str, dict] = {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Apply rate limiting to repeated messages."""
        import time
        
        key = f"{record.name}:{record.levelno}:{record.getMessage()}"
        current_time = time.time()
        
        if key not in self._message_counts:
            self._message_counts[key] = {
                'count': 1,
                'first_time': current_time,
                'last_time': current_time,
            }
            return True
        
        info = self._message_counts[key]
        info['count'] += 1
        
        # Allow burst
        if info['count'] <= self.burst:
            info['last_time'] = current_time
            return True
        
        # Check rate limit
        elapsed = current_time - info['first_time']
        if elapsed > 0:
            rate = info['count'] / elapsed
            if rate > self.rate_limit:
                return False
        
        info['last_time'] = current_time
        return True
