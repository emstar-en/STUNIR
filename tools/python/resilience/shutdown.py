"""Graceful shutdown handling for STUNIR."""

import signal
import sys
import time
import threading
import atexit
from typing import Callable, List, Optional
from dataclasses import dataclass


@dataclass
class ShutdownHook:
    """A shutdown hook with priority."""
    name: str
    func: Callable[[], None]
    priority: int
    timeout: float


class GracefulShutdown:
    """Graceful shutdown manager.
    
    Handles shutdown signals and runs cleanup hooks in order.
    
    Example:
        shutdown = GracefulShutdown()
        shutdown.register(close_database, priority=10, name='database')
        shutdown.register(flush_logs, priority=5, name='logging')
        shutdown.install()  # Install signal handlers
    """
    
    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self._hooks: List[ShutdownHook] = []
        self._lock = threading.Lock()
        self._shutting_down = threading.Event()
        self._installed = False
    
    def register(
        self,
        func: Callable[[], None],
        priority: int = 50,
        name: str = '',
        timeout: float = 10.0,
    ) -> None:
        """Register a shutdown hook.
        
        Lower priority numbers run first.
        """
        hook = ShutdownHook(
            name=name or func.__name__,
            func=func,
            priority=priority,
            timeout=timeout,
        )
        with self._lock:
            self._hooks.append(hook)
            self._hooks.sort(key=lambda h: h.priority)
    
    def unregister(self, name: str) -> bool:
        """Unregister a shutdown hook by name."""
        with self._lock:
            original_len = len(self._hooks)
            self._hooks = [h for h in self._hooks if h.name != name]
            return len(self._hooks) < original_len
    
    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress."""
        return self._shutting_down.is_set()
    
    def shutdown(self, exit_code: int = 0) -> None:
        """Initiate graceful shutdown."""
        if self._shutting_down.is_set():
            return
        
        self._shutting_down.set()
        
        with self._lock:
            hooks = list(self._hooks)
        
        errors = []
        start_time = time.time()
        
        for hook in hooks:
            if time.time() - start_time > self.timeout:
                errors.append(f"Global timeout reached, skipping remaining hooks")
                break
            
            try:
                # Run hook with timeout
                thread = threading.Thread(target=hook.func)
                thread.start()
                thread.join(timeout=hook.timeout)
                
                if thread.is_alive():
                    errors.append(f"Hook '{hook.name}' timed out")
            except Exception as e:
                errors.append(f"Hook '{hook.name}' failed: {str(e)}")
        
        if errors:
            for error in errors:
                print(f"Shutdown warning: {error}", file=sys.stderr)
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        signal_name = signal.Signals(signum).name
        print(f"\nReceived {signal_name}, initiating graceful shutdown...")
        self.shutdown()
        sys.exit(0)
    
    def install(self) -> None:
        """Install signal handlers for graceful shutdown."""
        if self._installed:
            return
        
        # Register atexit handler
        atexit.register(self.shutdown)
        
        # Install signal handlers (Unix-like systems)
        try:
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
        except (AttributeError, ValueError):
            pass  # Signals not available (e.g., Windows)
        
        self._installed = True
    
    def wait_for_shutdown(self, check_interval: float = 1.0) -> None:
        """Block until shutdown is initiated."""
        while not self._shutting_down.is_set():
            time.sleep(check_interval)


# Global shutdown manager
_global_shutdown: Optional[GracefulShutdown] = None
_shutdown_lock = threading.Lock()


def shutdown_handler() -> GracefulShutdown:
    """Get the global shutdown handler."""
    global _global_shutdown
    with _shutdown_lock:
        if _global_shutdown is None:
            _global_shutdown = GracefulShutdown()
            _global_shutdown.install()
        return _global_shutdown


def on_shutdown(priority: int = 50, name: str = '', timeout: float = 10.0) -> Callable[[Callable[[], None]], Callable[[], None]]:
    """Decorator to register a function as a shutdown hook."""
    def decorator(func: Callable[[], None]) -> Callable[[], None]:
        shutdown_handler().register(func, priority=priority, name=name or func.__name__, timeout=timeout)
        return func
    return decorator
