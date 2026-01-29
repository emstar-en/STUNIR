# STUNIR Regression Tests

Tests for previously fixed bugs and known edge cases.

## Running Regression Tests

```bash
pytest tests/regression/ -v
```

## Test Organization

- `test_issue_fixes.py` - Tests for specific fixed issues
- `golden/` - Expected output files (golden files)
- `fixtures/` - Input test data

## Adding New Regression Tests

1. Create test that reproduces the bug
2. Document the original issue in the test docstring
3. Add to `REGRESSION_TESTS.md` for tracking

## Regression Test Tracking

See `REGRESSION_TESTS.md` for a list of all regression tests and their linked issues.
