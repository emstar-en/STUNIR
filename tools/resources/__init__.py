"""STUNIR Resource Management.

Resource pooling, limits, and lifecycle management.
"""

from .pool import (
    ResourcePool,
    ConnectionPool,
    ThreadPool,
    PoolExhausted,
)
from .limits import (
    ResourceLimits,
    MemoryLimit,
    TimeLimit,
    enforce_limits,
)
from .lifecycle import (
    ResourceManager,
    ManagedResource,
    resource_context,
)
from .monitor import (
    ResourceMonitor,
    get_resource_usage,
)

__all__ = [
    'ResourcePool', 'ConnectionPool', 'ThreadPool', 'PoolExhausted',
    'ResourceLimits', 'MemoryLimit', 'TimeLimit', 'enforce_limits',
    'ResourceManager', 'ManagedResource', 'resource_context',
    'ResourceMonitor', 'get_resource_usage',
]
