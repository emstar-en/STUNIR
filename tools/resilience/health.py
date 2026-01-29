"""Health check system for STUNIR."""

import time
import threading
from enum import Enum, auto
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    UNKNOWN = auto()


@dataclass
class HealthResult:
    """Result of a health check."""
    status: HealthStatus
    name: str
    message: str = ''
    details: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


class HealthCheck(ABC):
    """Base class for health checks."""
    
    def __init__(self, name: str, critical: bool = True, timeout: float = 5.0):
        self.name = name
        self.critical = critical
        self.timeout = timeout
        self._last_result: Optional[HealthResult] = None
        self._check_count = 0
        self._failure_count = 0
    
    @abstractmethod
    def check(self) -> HealthResult:
        """Perform the health check."""
        pass
    
    def run(self) -> HealthResult:
        """Run the health check with timing."""
        start = time.perf_counter()
        self._check_count += 1
        
        try:
            result = self.check()
            result.latency_ms = (time.perf_counter() - start) * 1000
            if result.status != HealthStatus.HEALTHY:
                self._failure_count += 1
        except Exception as e:
            self._failure_count += 1
            result = HealthResult(
                status=HealthStatus.UNHEALTHY,
                name=self.name,
                message=f"Check failed: {str(e)}",
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        
        self._last_result = result
        return result
    
    @property
    def last_result(self) -> Optional[HealthResult]:
        return self._last_result
    
    @property
    def stats(self) -> dict:
        return {
            'check_count': self._check_count,
            'failure_count': self._failure_count,
            'success_rate': (self._check_count - self._failure_count) / self._check_count if self._check_count else 1.0,
        }


class FunctionHealthCheck(HealthCheck):
    """Health check using a custom function."""
    
    def __init__(self, name: str, check_func: Callable[[], bool], **kwargs):
        super().__init__(name, **kwargs)
        self.check_func = check_func
    
    def check(self) -> HealthResult:
        try:
            if self.check_func():
                return HealthResult(status=HealthStatus.HEALTHY, name=self.name)
            return HealthResult(status=HealthStatus.UNHEALTHY, name=self.name)
        except Exception as e:
            return HealthResult(
                status=HealthStatus.UNHEALTHY,
                name=self.name,
                message=str(e)
            )


class DependencyHealthCheck(HealthCheck):
    """Health check for external dependencies."""
    
    def __init__(self, name: str, ping_func: Callable[[], Any], **kwargs):
        super().__init__(name, **kwargs)
        self.ping_func = ping_func
    
    def check(self) -> HealthResult:
        try:
            self.ping_func()
            return HealthResult(status=HealthStatus.HEALTHY, name=self.name)
        except Exception as e:
            return HealthResult(
                status=HealthStatus.UNHEALTHY,
                name=self.name,
                message=f"Dependency unavailable: {str(e)}"
            )


class HealthRegistry:
    """Central registry for health checks."""
    
    def __init__(self):
        self._checks: Dict[str, HealthCheck] = {}
        self._lock = threading.Lock()
    
    def register(self, check: HealthCheck) -> None:
        """Register a health check."""
        with self._lock:
            self._checks[check.name] = check
    
    def unregister(self, name: str) -> None:
        """Unregister a health check."""
        with self._lock:
            self._checks.pop(name, None)
    
    def check(self, name: str) -> Optional[HealthResult]:
        """Run a specific health check."""
        check = self._checks.get(name)
        if check:
            return check.run()
        return None
    
    def check_all(self) -> Dict[str, HealthResult]:
        """Run all health checks."""
        results = {}
        with self._lock:
            checks = list(self._checks.values())
        
        for check in checks:
            results[check.name] = check.run()
        return results
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        results = self.check_all()
        
        if not results:
            return HealthStatus.UNKNOWN
        
        critical_unhealthy = any(
            r.status == HealthStatus.UNHEALTHY
            for name, r in results.items()
            if self._checks[name].critical
        )
        
        if critical_unhealthy:
            return HealthStatus.UNHEALTHY
        
        any_degraded = any(r.status == HealthStatus.DEGRADED for r in results.values())
        any_unhealthy = any(r.status == HealthStatus.UNHEALTHY for r in results.values())
        
        if any_degraded or any_unhealthy:
            return HealthStatus.DEGRADED
        
        return HealthStatus.HEALTHY
    
    def get_report(self) -> dict:
        """Get comprehensive health report."""
        results = self.check_all()
        return {
            'status': self.get_overall_status().name,
            'timestamp': time.time(),
            'checks': {
                name: {
                    'status': r.status.name,
                    'message': r.message,
                    'latency_ms': r.latency_ms,
                    'critical': self._checks[name].critical,
                }
                for name, r in results.items()
            },
        }


_global_registry: Optional[HealthRegistry] = None
_registry_lock = threading.Lock()


def get_health_registry() -> HealthRegistry:
    """Get the global health registry."""
    global _global_registry
    with _registry_lock:
        if _global_registry is None:
            _global_registry = HealthRegistry()
        return _global_registry
