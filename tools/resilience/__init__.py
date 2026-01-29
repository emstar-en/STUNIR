"""STUNIR Resilience Framework.

Graceful degradation patterns, health checks, and dependency isolation.
"""

from .fallback import (
    Fallback,
    fallback,
    FallbackChain,
)
from .health import (
    HealthCheck,
    HealthStatus,
    HealthRegistry,
    get_health_registry,
)
from .isolation import (
    Bulkhead,
    DependencyIsolator,
    timeout,
)
from .shutdown import (
    GracefulShutdown,
    shutdown_handler,
)

__all__ = [
    'Fallback', 'fallback', 'FallbackChain',
    'HealthCheck', 'HealthStatus', 'HealthRegistry', 'get_health_registry',
    'Bulkhead', 'DependencyIsolator', 'timeout',
    'GracefulShutdown', 'shutdown_handler',
]
