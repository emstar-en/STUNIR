"""Tests for Session 2 Features and Robustness improvements.

Tests cover:
1. Logging Framework
2. Configuration Management
3. Retry Logic
4. Caching Layer
5. Telemetry
6. Graceful Degradation
7. Rate Limiting
8. Resource Management
"""

import os
import sys
import time
import json
import threading
import tempfile
from pathlib import Path

import pytest

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))


# =============================================================================
# Logging Tests
# =============================================================================

class TestLogging:
    """Test logging framework."""
    
    def test_logger_creation(self):
        from logging.logger import get_logger, StunirLogger
        logger = get_logger('test')
        assert isinstance(logger, StunirLogger)
    
    def test_log_levels(self):
        from logging.logger import LogLevel
        assert LogLevel.DEBUG < LogLevel.INFO
        assert LogLevel.INFO < LogLevel.WARNING
        assert LogLevel.WARNING < LogLevel.ERROR
        assert LogLevel.ERROR < LogLevel.CRITICAL
    
    def test_json_formatter(self):
        from logging.formatters import JsonFormatter
        import logging
        
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name='test', level=logging.INFO, pathname='test.py',
            lineno=1, msg='Test message', args=(), exc_info=None
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data['level'] == 'INFO'
        assert data['message'] == 'Test message'
    
    def test_colored_formatter(self):
        from logging.formatters import ColoredFormatter
        import logging
        
        formatter = ColoredFormatter(use_colors=False)
        record = logging.LogRecord(
            name='test', level=logging.INFO, pathname='test.py',
            lineno=1, msg='Test', args=(), exc_info=None
        )
        output = formatter.format(record)
        assert 'INFO' in output
        assert 'Test' in output
    
    def test_level_filter(self):
        from logging.filters import LevelFilter
        import logging
        
        f = LevelFilter(min_level=logging.WARNING)
        record_info = logging.LogRecord(
            name='test', level=logging.INFO, pathname='test.py',
            lineno=1, msg='Test', args=(), exc_info=None
        )
        record_error = logging.LogRecord(
            name='test', level=logging.ERROR, pathname='test.py',
            lineno=1, msg='Test', args=(), exc_info=None
        )
        assert not f.filter(record_info)
        assert f.filter(record_error)
    
    def test_log_context(self):
        from logging.context import log_context, get_current_context
        
        with log_context(request_id='123', user='test'):
            ctx = get_current_context()
            assert ctx['request_id'] == '123'
            assert ctx['user'] == 'test'


# =============================================================================
# Configuration Tests
# =============================================================================

class TestConfiguration:
    """Test configuration management."""
    
    def test_config_dot_access(self):
        from config.config import Config
        config = Config({'database': {'host': 'localhost', 'port': 5432}})
        assert config.database.host == 'localhost'
        assert config.database.port == 5432
    
    def test_config_bracket_access(self):
        from config.config import Config
        config = Config({'database': {'host': 'localhost'}})
        assert config['database.host'] == 'localhost'
    
    def test_config_manager_merge(self):
        from config.config import ConfigManager
        manager = ConfigManager()
        manager.add_source('base', {'a': 1, 'b': 2}, priority=10)
        manager.add_source('override', {'b': 3, 'c': 4}, priority=20)
        config = manager.get_config()
        assert config['a'] == 1
        assert config['b'] == 3
        assert config['c'] == 4
    
    def test_json_loader(self):
        from config.loaders import JsonLoader
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'key': 'value'}, f)
            f.flush()
            loader = JsonLoader()
            data = loader.load(f.name)
            assert data['key'] == 'value'
            os.unlink(f.name)
    
    def test_env_loader(self):
        from config.loaders import EnvLoader
        os.environ['TEST_DATABASE_HOST'] = 'localhost'
        os.environ['TEST_DATABASE_PORT'] = '5432'
        loader = EnvLoader(prefix='TEST_')
        data = loader.load()
        assert data['database']['host'] == 'localhost'
        assert data['database']['port'] == 5432
        del os.environ['TEST_DATABASE_HOST']
        del os.environ['TEST_DATABASE_PORT']
    
    def test_schema_validation(self):
        from config.schema import ConfigSchema, Field, FieldType
        from config.validators import ConfigValidator
        
        schema = ConfigSchema(fields={
            'port': Field.integer(required=True, min_value=1, max_value=65535),
            'host': Field.string(required=True),
        })
        validator = ConfigValidator(schema)
        
        errors = validator.validate({'port': 8080, 'host': 'localhost'})
        assert len(errors) == 0
        
        errors = validator.validate({'port': 70000, 'host': 'localhost'})
        assert len(errors) > 0


# =============================================================================
# Retry Tests
# =============================================================================

class TestRetry:
    """Test retry logic."""
    
    def test_successful_call(self):
        from retry.retry import Retry
        
        counter = {'calls': 0}
        def succeed():
            counter['calls'] += 1
            return 'success'
        
        r = Retry(max_attempts=3)
        result = r.call(succeed)
        assert result == 'success'
        assert counter['calls'] == 1
    
    def test_retry_on_failure(self):
        from retry.retry import Retry, RetryError
        from retry.backoff import ConstantBackoff
        
        counter = {'calls': 0}
        def fail_twice():
            counter['calls'] += 1
            if counter['calls'] < 3:
                raise ValueError("fail")
            return 'success'
        
        r = Retry(max_attempts=5, backoff=ConstantBackoff(0.01))
        result = r.call(fail_twice)
        assert result == 'success'
        assert counter['calls'] == 3
    
    def test_exponential_backoff(self):
        from retry.backoff import ExponentialBackoff
        backoff = ExponentialBackoff(base=1.0, multiplier=2.0, max_delay=10.0)
        assert backoff.get_delay(1) == 1.0
        assert backoff.get_delay(2) == 2.0
        assert backoff.get_delay(3) == 4.0
        assert backoff.get_delay(10) == 10.0  # capped
    
    def test_retry_decorator(self):
        from retry.retry import retry
        from retry.backoff import ConstantBackoff
        
        counter = {'calls': 0}
        
        @retry(max_attempts=3, backoff=ConstantBackoff(0.01))
        def flaky():
            counter['calls'] += 1
            if counter['calls'] < 2:
                raise ValueError("flaky")
            return 'ok'
        
        result = flaky()
        assert result == 'ok'
        assert counter['calls'] == 2
    
    def test_circuit_breaker(self):
        from retry.circuit_breaker import CircuitBreaker, CircuitState, CircuitOpenError
        
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        
        def fail():
            raise ValueError("fail")
        
        # Trigger failures
        for _ in range(2):
            try:
                breaker.call(fail)
            except ValueError:
                pass
        
        assert breaker.state == CircuitState.OPEN
        
        # Should reject immediately
        with pytest.raises(CircuitOpenError):
            breaker.call(fail)


# =============================================================================
# Cache Tests
# =============================================================================

class TestCache:
    """Test caching layer."""
    
    def test_memory_backend(self):
        from common.cache_backends import MemoryBackend
        cache = MemoryBackend(maxsize=10)
        cache.set('key', 'value')
        assert cache.get('key') == 'value'
        assert cache.exists('key')
        cache.delete('key')
        assert not cache.exists('key')
    
    def test_file_backend(self):
        from common.cache_backends import FileBackend
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FileBackend(cache_dir=tmpdir, ttl=60)
            cache.set('key', {'data': 123})
            assert cache.get('key') == {'data': 123}
            cache.clear()
            assert cache.get('key') is None
    
    def test_tiered_cache(self):
        from common.cache_backends import TieredCache, MemoryBackend, FileBackend
        with tempfile.TemporaryDirectory() as tmpdir:
            l1 = MemoryBackend(maxsize=10)
            l2 = FileBackend(cache_dir=tmpdir)
            tiered = TieredCache(l1=l1, l2=l2)
            
            tiered.set('key', 'value')
            assert tiered.get('key') == 'value'
            
            # Clear L1, should still find in L2
            l1.clear()
            assert tiered.get('key') == 'value'
    
    def test_cached_decorator(self):
        from common.cache_backends import cached, MemoryBackend
        
        counter = {'calls': 0}
        
        @cached(backend=MemoryBackend())
        def expensive(x):
            counter['calls'] += 1
            return x * 2
        
        assert expensive(5) == 10
        assert expensive(5) == 10  # cached
        assert counter['calls'] == 1


# =============================================================================
# Telemetry Tests
# =============================================================================

class TestTelemetry:
    """Test telemetry system."""
    
    def test_counter(self):
        from telemetry.metrics import Counter
        counter = Counter('requests_total')
        counter.inc()
        counter.inc(5)
        assert counter.get() == 6
    
    def test_gauge(self):
        from telemetry.metrics import Gauge
        gauge = Gauge('temperature')
        gauge.set(20.5)
        assert gauge.get() == 20.5
        gauge.inc(5)
        assert gauge.get() == 25.5
        gauge.dec(10)
        assert gauge.get() == 15.5
    
    def test_histogram(self):
        from telemetry.metrics import Histogram
        hist = Histogram('latency')
        for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
            hist.observe(v)
        stats = hist.get_stats()
        assert stats['count'] == 5
        assert abs(stats['mean'] - 0.3) < 0.01
    
    def test_timer(self):
        from telemetry.metrics import Timer
        timer = Timer('operation_duration')
        with timer.time():
            time.sleep(0.05)
        stats = timer.get_stats()
        assert stats['count'] == 1
        assert stats['sum'] >= 0.04
    
    def test_prometheus_exporter(self):
        from telemetry.metrics import Counter, get_registry
        from telemetry.exporters import PrometheusExporter
        
        registry = get_registry()
        counter = registry.counter('test_counter')
        counter.inc(10)
        
        exporter = PrometheusExporter()
        output = exporter.export(registry.collect_all())
        assert 'test_counter' in output
    
    def test_json_exporter(self):
        from telemetry.metrics import Gauge, get_registry
        from telemetry.exporters import JsonExporter
        
        registry = get_registry()
        gauge = registry.gauge('test_gauge')
        gauge.set(42)
        
        exporter = JsonExporter()
        output = exporter.export(registry.collect_all())
        data = json.loads(output)
        assert 'metrics' in data


# =============================================================================
# Resilience Tests
# =============================================================================

class TestResilience:
    """Test graceful degradation."""
    
    def test_fallback_on_error(self):
        from resilience.fallback import Fallback
        
        fb = Fallback(default_value='fallback')
        
        def fail():
            raise ValueError("fail")
        
        result = fb.call(fail)
        assert result.value == 'fallback'
        assert result.fallback_used
    
    def test_fallback_chain(self):
        from resilience.fallback import FallbackChain
        
        def fail1(): raise ValueError("1")
        def fail2(): raise ValueError("2")
        def succeed(): return 'ok'
        
        chain = FallbackChain([fail1, fail2, succeed])
        result = chain.execute()
        assert result.value == 'ok'
        assert result.source == 'fallback_2'
    
    def test_health_check(self):
        from resilience.health import FunctionHealthCheck, HealthStatus
        
        check = FunctionHealthCheck('test', lambda: True)
        result = check.run()
        assert result.status == HealthStatus.HEALTHY
        
        check_fail = FunctionHealthCheck('fail', lambda: False)
        result = check_fail.run()
        assert result.status == HealthStatus.UNHEALTHY
    
    def test_health_registry(self):
        from resilience.health import HealthRegistry, FunctionHealthCheck, HealthStatus
        
        registry = HealthRegistry()
        registry.register(FunctionHealthCheck('service1', lambda: True))
        registry.register(FunctionHealthCheck('service2', lambda: True))
        
        status = registry.get_overall_status()
        assert status == HealthStatus.HEALTHY
    
    def test_bulkhead(self):
        from resilience.isolation import Bulkhead, BulkheadFullError
        
        bulkhead = Bulkhead(max_concurrent=2, max_wait=0.01)
        results = []
        
        def work():
            time.sleep(0.1)
            return 'done'
        
        # Acquire 2 permits
        assert bulkhead.acquire()
        assert bulkhead.acquire()
        # Third should fail
        assert not bulkhead.acquire()


# =============================================================================
# Rate Limiting Tests
# =============================================================================

class TestRateLimiting:
    """Test rate limiting."""
    
    def test_token_bucket(self):
        from ratelimit.limiter import TokenBucket
        
        bucket = TokenBucket(rate=10, capacity=5)
        
        # Should allow burst
        for _ in range(5):
            assert bucket.acquire()
        
        # Should deny immediately after burst
        assert not bucket.acquire()
    
    def test_sliding_window(self):
        from ratelimit.limiter import SlidingWindow
        
        limiter = SlidingWindow(limit=3, window_size=1.0)
        
        assert limiter.acquire()
        assert limiter.acquire()
        assert limiter.acquire()
        assert not limiter.acquire()
    
    def test_rate_limit_decorator(self):
        from ratelimit.decorators import rate_limit
        from ratelimit.limiter import RateLimitExceeded
        
        @rate_limit(rate=100, capacity=2, block=False)
        def limited():
            return 'ok'
        
        assert limited() == 'ok'
        assert limited() == 'ok'
        with pytest.raises(RateLimitExceeded):
            limited()
    
    def test_throttle_decorator(self):
        from ratelimit.decorators import throttle
        
        counter = {'calls': 0}
        
        @throttle(calls=2, period=0.5, raise_on_limit=False)
        def throttled():
            counter['calls'] += 1
            return 'ok'
        
        start = time.time()
        throttled()
        throttled()
        # Third call should wait
        throttled()
        elapsed = time.time() - start
        assert elapsed >= 0.4  # Had to wait


# =============================================================================
# Resource Management Tests
# =============================================================================

class TestResourceManagement:
    """Test resource management."""
    
    def test_resource_pool(self):
        from resources.pool import ResourcePool
        
        counter = {'created': 0}
        def factory():
            counter['created'] += 1
            return f"resource_{counter['created']}"
        
        pool = ResourcePool(factory=factory, max_size=2, min_size=1)
        assert counter['created'] == 1  # min_size
        
        r1 = pool.acquire()
        r2 = pool.acquire()
        assert counter['created'] == 2
        
        pool.release(r1)
        r3 = pool.acquire()
        assert r3 == r1  # Reused
    
    def test_memory_limit(self):
        from resources.limits import MemoryLimit
        
        limit = MemoryLimit(max_mb=10000)  # High limit
        assert limit.check()
    
    def test_time_limit(self):
        from resources.limits import TimeLimit
        
        limit = TimeLimit(max_seconds=0.1)
        limit.start()
        assert limit.check()
        time.sleep(0.15)
        assert not limit.check()
    
    def test_resource_manager(self):
        from resources.lifecycle import ResourceManager
        
        cleanup_called = {'value': False}
        def cleanup():
            cleanup_called['value'] = True
        
        manager = ResourceManager()
        manager.register('test', object(), cleanup=cleanup)
        
        status = manager.get_status()
        assert 'test' in status
        assert status['test']['initialized']
        
        manager.cleanup_all()
        assert cleanup_called['value']
    
    def test_resource_usage(self):
        from resources.monitor import get_resource_usage
        
        usage = get_resource_usage()
        assert usage.threads >= 1
        assert usage.uptime_seconds >= 0


# =============================================================================
# Integration Test
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple features."""
    
    def test_retry_with_circuit_breaker(self):
        from retry.retry import Retry, RetryError
        from retry.circuit_breaker import CircuitBreaker, CircuitOpenError
        from retry.backoff import ConstantBackoff
        
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
        
        counter = {'calls': 0}
        def flaky():
            counter['calls'] += 1
            raise ValueError("always fails")
        
        retry_handler = Retry(max_attempts=5, backoff=ConstantBackoff(0.01))
        
        # Try with circuit breaker
        try:
            for _ in range(5):
                try:
                    breaker.call(lambda: retry_handler.call(flaky))
                except (ValueError, RetryError):
                    pass
        except CircuitOpenError:
            pass
        
        # Circuit should be open now
        assert breaker.state.name == 'OPEN'
    
    def test_cache_with_fallback(self):
        from common.cache_backends import MemoryBackend
        from resilience.fallback import Fallback
        
        cache = MemoryBackend()
        cache.set('data', [1, 2, 3])
        
        def fetch_from_db():
            raise ConnectionError("DB unavailable")
        
        def fetch_from_cache():
            return cache.get('data')
        
        fb = Fallback(default_func=fetch_from_cache)
        result = fb.call(fetch_from_db)
        
        assert result.value == [1, 2, 3]
        assert result.fallback_used


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
