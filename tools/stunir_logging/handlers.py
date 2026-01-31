"""Custom log handlers for STUNIR.

Provides rotating file handlers, syslog integration, and async handlers.
"""

import logging
import os
import gzip
import shutil
import threading
import queue
import socket
from pathlib import Path
from typing import Optional
from datetime import datetime


class RotatingFileHandler(logging.Handler):
    """Rotating file handler with size-based rotation.
    
    Features:
    - Automatic rotation when file exceeds max_bytes
    - Configurable backup count
    - Optional gzip compression of rotated files
    - Thread-safe file operations
    """
    
    def __init__(
        self,
        filename: str,
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
        compress: bool = True,
        encoding: str = 'utf-8',
    ):
        super().__init__()
        self.filename = Path(filename)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.compress = compress
        self.encoding = encoding
        self._lock = threading.Lock()
        self._file = None
        self._ensure_dir()
    
    def _ensure_dir(self) -> None:
        """Ensure log directory exists."""
        self.filename.parent.mkdir(parents=True, exist_ok=True)
    
    def _open(self):
        """Open the log file."""
        if self._file is None:
            self._file = open(self.filename, 'a', encoding=self.encoding)
        return self._file
    
    def _close(self) -> None:
        """Close the log file."""
        if self._file:
            self._file.close()
            self._file = None
    
    def _should_rotate(self) -> bool:
        """Check if rotation is needed."""
        if not self.filename.exists():
            return False
        return self.filename.stat().st_size >= self.max_bytes
    
    def _rotate(self) -> None:
        """Perform log rotation."""
        self._close()
        
        # Rotate existing backups
        for i in range(self.backup_count - 1, 0, -1):
            src = Path(f"{self.filename}.{i}")
            if self.compress:
                src = Path(f"{src}.gz")
            dst = Path(f"{self.filename}.{i + 1}")
            if self.compress:
                dst = Path(f"{dst}.gz")
            if src.exists():
                if i + 1 <= self.backup_count:
                    shutil.move(str(src), str(dst))
                else:
                    src.unlink()
        
        # Move current log to .1
        if self.filename.exists():
            backup_path = Path(f"{self.filename}.1")
            if self.compress:
                with open(self.filename, 'rb') as f_in:
                    with gzip.open(f"{backup_path}.gz", 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                self.filename.unlink()
            else:
                shutil.move(str(self.filename), str(backup_path))
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record."""
        try:
            with self._lock:
                if self._should_rotate():
                    self._rotate()
                
                f = self._open()
                msg = self.format(record)
                f.write(msg + '\n')
                f.flush()
        except Exception:
            self.handleError(record)
    
    def close(self) -> None:
        """Close the handler."""
        with self._lock:
            self._close()
        super().close()


class SyslogHandler(logging.Handler):
    """Syslog handler for centralized logging.
    
    Supports both UDP and TCP protocols.
    """
    
    FACILITY = {
        'user': 1,
        'local0': 16,
        'local1': 17,
        'local2': 18,
        'local3': 19,
        'local4': 20,
        'local5': 21,
        'local6': 22,
        'local7': 23,
    }
    
    SEVERITY = {
        logging.DEBUG: 7,
        logging.INFO: 6,
        logging.WARNING: 4,
        logging.ERROR: 3,
        logging.CRITICAL: 2,
    }
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 514,
        facility: str = 'local0',
        protocol: str = 'udp',
    ):
        super().__init__()
        self.host = host
        self.port = port
        self.facility = self.FACILITY.get(facility, 16)
        self.protocol = protocol.lower()
        self._socket = None
        self._connect()
    
    def _connect(self) -> None:
        """Establish connection to syslog server."""
        try:
            if self.protocol == 'tcp':
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._socket.connect((self.host, self.port))
            else:
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except OSError:
            self._socket = None
    
    def _format_syslog(self, record: logging.LogRecord) -> bytes:
        """Format record for syslog protocol."""
        severity = self.SEVERITY.get(record.levelno, 6)
        priority = (self.facility * 8) + severity
        timestamp = datetime.utcnow().strftime('%b %d %H:%M:%S')
        hostname = socket.gethostname()
        msg = self.format(record)
        
        syslog_msg = f"<{priority}>{timestamp} {hostname} stunir: {msg}"
        return syslog_msg.encode('utf-8')
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to syslog."""
        if self._socket is None:
            return
        
        try:
            data = self._format_syslog(record)
            if self.protocol == 'tcp':
                self._socket.send(data + b'\n')
            else:
                self._socket.sendto(data, (self.host, self.port))
        except OSError:
            self.handleError(record)
    
    def close(self) -> None:
        """Close the syslog connection."""
        if self._socket:
            self._socket.close()
            self._socket = None
        super().close()


class AsyncHandler(logging.Handler):
    """Asynchronous log handler using a background thread.
    
    Queues log records and processes them asynchronously to avoid
    blocking the main thread.
    """
    
    def __init__(self, target_handler: logging.Handler, queue_size: int = 10000):
        super().__init__()
        self.target_handler = target_handler
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()
        self._start_thread()
    
    def _start_thread(self) -> None:
        """Start the background processing thread."""
        self._thread = threading.Thread(target=self._process_queue, daemon=True)
        self._thread.start()
    
    def _process_queue(self) -> None:
        """Process log records from the queue."""
        while not self._shutdown.is_set():
            try:
                record = self._queue.get(timeout=0.1)
                self.target_handler.emit(record)
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                pass
    
    def emit(self, record: logging.LogRecord) -> None:
        """Queue a log record for async processing."""
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            # Drop the record if queue is full
            pass
    
    def close(self) -> None:
        """Shutdown the async handler."""
        self._shutdown.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self.target_handler.close()
        super().close()
