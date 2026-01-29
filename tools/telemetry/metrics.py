"""Metrics collection for STUNIR telemetry."""

import time
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import wraps
import statistics


@dataclass
class MetricValue:
    """A single metric value with labels."""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class Metric(ABC):
    """Base class for all metrics."""
    
    def __init__(self, name: str, description: str = '', labels: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._lock = threading.Lock()
    
    @abstractmethod
    def collect(self) -> List[MetricValue]:
        """Collect current metric values."""
        pass


class Counter(Metric):
    """Monotonically increasing counter."""
    
    def __init__(self, name: str, description: str = '', labels: Optional[List[str]] = None):
        super().__init__(name, description, labels)
        self._values: Dict[tuple, float] = {}
    
    def inc(self, value: float = 1.0, **labels) -> None:
        """Increment the counter."""
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] = self._values.get(key, 0) + value
    
    def get(self, **labels) -> float:
        """Get current counter value."""
        key = tuple(sorted(labels.items()))
        with self._lock:
            return self._values.get(key, 0)
    
    def collect(self) -> List[MetricValue]:
        with self._lock:
            return [
                MetricValue(self.name, value, dict(key))
                for key, value in self._values.items()
            ]


class Gauge(Metric):
    """Gauge that can go up and down."""
    
    def __init__(self, name: str, description: str = '', labels: Optional[List[str]] = None):
        super().__init__(name, description, labels)
        self._values: Dict[tuple, float] = {}
    
    def set(self, value: float, **labels) -> None:
        """Set the gauge value."""
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] = value
    
    def inc(self, value: float = 1.0, **labels) -> None:
        """Increment the gauge."""
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] = self._values.get(key, 0) + value
    
    def dec(self, value: float = 1.0, **labels) -> None:
        """Decrement the gauge."""
        self.inc(-value, **labels)
    
    def get(self, **labels) -> float:
        key = tuple(sorted(labels.items()))
        with self._lock:
            return self._values.get(key, 0)
    
    def collect(self) -> List[MetricValue]:
        with self._lock:
            return [
                MetricValue(self.name, value, dict(key))
                for key, value in self._values.items()
            ]


class Histogram(Metric):
    """Histogram for distribution tracking."""
    
    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
    
    def __init__(self, name: str, description: str = '', labels: Optional[List[str]] = None,
                 buckets: Optional[tuple] = None):
        super().__init__(name, description, labels)
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._observations: Dict[tuple, List[float]] = {}
    
    def observe(self, value: float, **labels) -> None:
        """Record an observation."""
        key = tuple(sorted(labels.items()))
        with self._lock:
            if key not in self._observations:
                self._observations[key] = []
            self._observations[key].append(value)
    
    def get_stats(self, **labels) -> Dict[str, float]:
        """Get histogram statistics."""
        key = tuple(sorted(labels.items()))
        with self._lock:
            values = self._observations.get(key, [])
            if not values:
                return {'count': 0, 'sum': 0, 'mean': 0, 'min': 0, 'max': 0}
            return {
                'count': len(values),
                'sum': sum(values),
                'mean': statistics.mean(values),
                'min': min(values),
                'max': max(values),
                'p50': statistics.median(values) if values else 0,
                'p95': statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
                'p99': statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values),
            }
    
    def collect(self) -> List[MetricValue]:
        result = []
        with self._lock:
            for key, values in self._observations.items():
                labels = dict(key)
                result.append(MetricValue(f"{self.name}_count", len(values), labels))
                result.append(MetricValue(f"{self.name}_sum", sum(values), labels))
                for bucket in self.buckets:
                    count = sum(1 for v in values if v <= bucket)
                    result.append(MetricValue(
                        f"{self.name}_bucket", count, {**labels, 'le': str(bucket)}
                    ))
        return result


class Timer(Metric):
    """Timer for measuring durations."""
    
    def __init__(self, name: str, description: str = '', labels: Optional[List[str]] = None):
        super().__init__(name, description, labels)
        self._histogram = Histogram(name, description, labels)
    
    @contextmanager
    def time(self, **labels):
        """Context manager for timing operations."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self._histogram.observe(duration, **labels)
    
    def __call__(self, func: Callable) -> Callable:
        """Use as decorator."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.time(function=func.__name__):
                return func(*args, **kwargs)
        return wrapper
    
    def observe(self, value: float, **labels) -> None:
        self._histogram.observe(value, **labels)
    
    def get_stats(self, **labels) -> Dict[str, float]:
        return self._histogram.get_stats(**labels)
    
    def collect(self) -> List[MetricValue]:
        return self._histogram.collect()


class MetricsRegistry:
    """Central registry for all metrics."""
    
    def __init__(self):
        self._metrics: Dict[str, Metric] = {}
        self._lock = threading.Lock()
    
    def register(self, metric: Metric) -> Metric:
        """Register a metric."""
        with self._lock:
            if metric.name in self._metrics:
                return self._metrics[metric.name]
            self._metrics[metric.name] = metric
        return metric
    
    def counter(self, name: str, description: str = '', labels: Optional[List[str]] = None) -> Counter:
        return self.register(Counter(name, description, labels))
    
    def gauge(self, name: str, description: str = '', labels: Optional[List[str]] = None) -> Gauge:
        return self.register(Gauge(name, description, labels))
    
    def histogram(self, name: str, description: str = '', labels: Optional[List[str]] = None) -> Histogram:
        return self.register(Histogram(name, description, labels))
    
    def timer(self, name: str, description: str = '', labels: Optional[List[str]] = None) -> Timer:
        return self.register(Timer(name, description, labels))
    
    def get(self, name: str) -> Optional[Metric]:
        return self._metrics.get(name)
    
    def collect_all(self) -> List[MetricValue]:
        """Collect all metrics."""
        result = []
        with self._lock:
            for metric in self._metrics.values():
                result.extend(metric.collect())
        return result
    
    def clear(self) -> None:
        with self._lock:
            self._metrics.clear()


# Global registry
_global_registry: Optional[MetricsRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> MetricsRegistry:
    """Get the global metrics registry."""
    global _global_registry
    with _registry_lock:
        if _global_registry is None:
            _global_registry = MetricsRegistry()
        return _global_registry
