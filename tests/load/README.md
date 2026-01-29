# STUNIR Load Testing

Load testing verifies system performance under concurrent operations and stress.

## Setup

```bash
pip install locust
```

## Running Load Tests

### Interactive Web UI
```bash
cd tests/load
locust -f locustfile.py
# Open http://localhost:8089
```

### Headless Mode
```bash
locust -f locustfile.py --headless -u 10 -r 2 -t 60s
```

Parameters:
- `-u 10`: 10 concurrent users
- `-r 2`: Spawn 2 users per second
- `-t 60s`: Run for 60 seconds

## Test Scenarios

| Scenario | Description | Target RPS |
|----------|-------------|-----------|
| Receipt Generation | Generate IR receipts | 100 |
| Hash Computation | SHA256 hashing | 1000 |
| File Processing | Process spec files | 50 |
| Concurrent Manifests | Generate manifests | 25 |

## Performance Baselines

| Operation | Baseline (p95) | Max Acceptable |
|-----------|----------------|----------------|
| emit_ir | 50ms | 200ms |
| compute_sha256 | 1ms | 10ms |
| canonical_json | 5ms | 50ms |
| manifest_generation | 100ms | 500ms |

## CI Integration

Load tests run weekly via `.github/workflows/load-testing.yml`.
Results are saved to `tests/load/results/`.
