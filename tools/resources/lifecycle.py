"""Resource lifecycle management for STUNIR."""

import threading
import weakref
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, TypeVar
from contextlib import contextmanager
from dataclasses import dataclass

T = TypeVar('T')


class ManagedResource(ABC):
    """Base class for managed resources."""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the resource."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up the resource."""
        pass
    
    def is_healthy(self) -> bool:
        """Check if resource is healthy."""
        return True


@dataclass
class ResourceInfo:
    """Information about a managed resource."""
    name: str
    resource: Any
    initialized: bool = False
    cleanup_func: Optional[Callable] = None


class ResourceManager:
    """Central manager for resource lifecycle.
    
    Example:
        manager = ResourceManager()
        manager.register('database', db_connection, cleanup=db_connection.close)
        manager.register('cache', cache_client, cleanup=cache_client.disconnect)
        
        # At shutdown
        manager.cleanup_all()
    """
    
    def __init__(self):
        self._resources: Dict[str, ResourceInfo] = {}
        self._lock = threading.Lock()
        self._cleanup_order: List[str] = []
    
    def register(
        self,
        name: str,
        resource: Any,
        cleanup: Optional[Callable[[], None]] = None,
        initialize: Optional[Callable[[], None]] = None,
    ) -> None:
        """Register a resource for lifecycle management."""
        with self._lock:
            # Run initializer if provided
            if initialize:
                initialize()
            
            self._resources[name] = ResourceInfo(
                name=name,
                resource=resource,
                initialized=True,
                cleanup_func=cleanup,
            )
            self._cleanup_order.append(name)
    
    def unregister(self, name: str, cleanup: bool = True) -> Optional[Any]:
        """Unregister a resource."""
        with self._lock:
            info = self._resources.pop(name, None)
            if info:
                if name in self._cleanup_order:
                    self._cleanup_order.remove(name)
                if cleanup and info.cleanup_func:
                    try:
                        info.cleanup_func()
                    except Exception:
                        pass
                return info.resource
        return None
    
    def get(self, name: str) -> Optional[Any]:
        """Get a registered resource."""
        with self._lock:
            info = self._resources.get(name)
            return info.resource if info else None
    
    def cleanup(self, name: str) -> bool:
        """Clean up a specific resource."""
        with self._lock:
            info = self._resources.get(name)
            if info and info.cleanup_func:
                try:
                    info.cleanup_func()
                    info.initialized = False
                    return True
                except Exception:
                    return False
        return False
    
    def cleanup_all(self, reverse: bool = True) -> Dict[str, bool]:
        """Clean up all resources.
        
        Args:
            reverse: Clean up in reverse registration order (recommended)
        
        Returns:
            Dict of resource names to cleanup success status
        """
        results = {}
        order = list(reversed(self._cleanup_order)) if reverse else self._cleanup_order
        
        for name in order:
            results[name] = self.cleanup(name)
        
        return results
    
    def is_healthy(self, name: str) -> bool:
        """Check if a resource is healthy."""
        with self._lock:
            info = self._resources.get(name)
            if not info:
                return False
            if isinstance(info.resource, ManagedResource):
                return info.resource.is_healthy()
            return info.initialized
    
    def get_status(self) -> Dict[str, dict]:
        """Get status of all resources."""
        with self._lock:
            return {
                name: {
                    'initialized': info.initialized,
                    'healthy': self.is_healthy(name),
                    'has_cleanup': info.cleanup_func is not None,
                }
                for name, info in self._resources.items()
            }


@contextmanager
def resource_context(
    resource: T,
    cleanup: Callable[[T], None],
    initialize: Optional[Callable[[T], None]] = None,
):
    """Context manager for resource lifecycle.
    
    Example:
        with resource_context(connection, cleanup=connection.close):
            connection.execute('SELECT 1')
    """
    if initialize:
        initialize(resource)
    try:
        yield resource
    finally:
        cleanup(resource)


class LazyResource:
    """Lazily initialized resource.
    
    Example:
        db = LazyResource(create_connection)
        # Connection created on first access
        result = db.get().execute('SELECT 1')
    """
    
    def __init__(self, factory: Callable[[], T], cleanup: Optional[Callable[[T], None]] = None):
        self.factory = factory
        self.cleanup_func = cleanup
        self._resource: Optional[T] = None
        self._lock = threading.Lock()
    
    def get(self) -> T:
        """Get or create the resource."""
        if self._resource is None:
            with self._lock:
                if self._resource is None:
                    self._resource = self.factory()
        return self._resource
    
    def cleanup(self) -> None:
        """Clean up the resource."""
        with self._lock:
            if self._resource is not None and self.cleanup_func:
                self.cleanup_func(self._resource)
                self._resource = None
    
    def __del__(self):
        self.cleanup()
