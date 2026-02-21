"""Custom log formatters for STUNIR.

Provides JSON and colored console formatters for structured logging.
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging.
    
    Outputs log records as single-line JSON objects with:
    - timestamp (ISO 8601)
    - level
    - logger name
    - message
    - structured context data
    - exception info (if present)
    """
    
    def __init__(self, include_timestamp: bool = True, include_location: bool = False):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_location = include_location
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        log_data: Dict[str, Any] = {
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        if self.include_timestamp:
            log_data['timestamp'] = datetime.utcnow().isoformat() + 'Z'
        
        if self.include_location:
            log_data['location'] = {
                'file': record.pathname,
                'line': record.lineno,
                'function': record.funcName,
            }
        
        # Include structured context
        if hasattr(record, 'structured') and record.structured:
            log_data['context'] = record.structured
        
        # Include exception info
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info),
            }
        
        return json.dumps(log_data, default=str, separators=(',', ':'))


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for human-readable output.
    
    Uses ANSI color codes for different log levels.
    """
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def __init__(self, use_colors: Optional[bool] = None, show_context: bool = True):
        super().__init__()
        # Auto-detect color support
        if use_colors is None:
            use_colors = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
        self.use_colors = use_colors
        self.show_context = show_context
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        level = record.levelname
        message = record.getMessage()
        
        if self.use_colors:
            color = self.COLORS.get(level, '')
            level_str = f"{color}{self.BOLD}[{level:8}]{self.RESET}"
        else:
            level_str = f"[{level:8}]"
        
        output = f"{timestamp} {level_str} {record.name}: {message}"
        
        # Append structured context
        if self.show_context and hasattr(record, 'structured') and record.structured:
            context_str = ' '.join(f"{k}={v}" for k, v in record.structured.items())
            if context_str:
                output += f" [{context_str}]"
        
        # Append exception info
        if record.exc_info:
            output += f"\n{self.formatException(record.exc_info)}"
        
        return output


class StructuredFormatter(logging.Formatter):
    """Structured formatter for logfmt-style output.
    
    Outputs in key=value format suitable for log aggregation systems.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as logfmt."""
        pairs = [
            f'ts={datetime.utcnow().isoformat()}Z',
            f'level={record.levelname.lower()}',
            f'logger={record.name}',
            f'msg="{self._escape(record.getMessage())}"',
        ]
        
        # Add structured context
        if hasattr(record, 'structured') and record.structured:
            for key, value in record.structured.items():
                if isinstance(value, str):
                    pairs.append(f'{key}="{self._escape(value)}"')
                else:
                    pairs.append(f'{key}={value}')
        
        return ' '.join(pairs)
    
    def _escape(self, value: str) -> str:
        """Escape special characters for logfmt."""
        return value.replace('"', '\\"').replace('\n', '\\n')
