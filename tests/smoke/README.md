# STUNIR Smoke Tests

Fast sanity checks that run on every commit. Target: < 30 seconds.

## Running Smoke Tests

```bash
pytest tests/smoke/ -v --timeout=30
```

## Test Coverage

| Test | Description | Target Time |
|------|-------------|-------------|
| test_imports | All critical modules import | < 2s |
| test_basic_ir | Basic IR generation | < 5s |
| test_basic_hash | SHA256 computation | < 1s |
| test_config_load | Configuration loading | < 1s |
| test_file_ops | Basic file operations | < 3s |
| test_manifest_gen | Simple manifest generation | < 5s |

## CI Integration

Smoke tests run on every commit via `.github/workflows/smoke-tests.yml`.
Failure blocks PR merge.
