# STUNIR Chaos Engineering Tests

Tests for graceful degradation under failure conditions.

## Running Chaos Tests

```bash
pytest tests/chaos/ -v --timeout=120
```

## Chaos Scenarios

| Scenario | Description | Expected Behavior |
|----------|-------------|-------------------|
| Disk Full | Simulate disk full errors | Graceful error message |
| Permission Denied | File permission errors | Clear error, no crash |
| Network Timeout | Simulated network delays | Timeout handling |
| Memory Exhaustion | Large allocation | Controlled failure |
| Corrupted Input | Invalid file content | Validation rejection |
| Partial Failure | Mid-operation failures | State consistency |

## Safety

Chaos tests use mocking and don't affect the actual system.
