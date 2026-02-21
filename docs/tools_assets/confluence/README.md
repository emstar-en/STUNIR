# STUNIR Confluence Test Suite

## Purpose

This test suite verifies that all four STUNIR pipelines (SPARK, Python, Rust, Haskell) produce **bitwise-identical outputs** from the same inputs.

## Usage

### Run all tests
```bash
./test_confluence.sh
```

### Run with verbose output
```bash
./test_confluence.sh --verbose
```

### Run tests for specific category
```bash
./test_confluence.sh --category assembly
```

## Test Vectors

Test vectors are located in `test_vectors/`:
- `minimal.json`: Minimal spec (empty module)
- `simple.json`: Simple function with basic types
- `complex.json`: Full-featured specification with multiple functions

## Interpreting Results

The test suite computes a **Confluence Score** based on the percentage of tests where all pipelines produce identical outputs:

- **100%**: Perfect confluence ✅
- **90-99%**: Near confluence ⚠️
- **<90%**: Significant divergence ❌

## Adding New Tests

1. Create a new JSON spec file in `test_vectors/`
2. Add a test case in `test_confluence.sh`
3. Run the test suite to verify

## CI/CD Integration

This test suite should be run in CI/CD pipelines to:
1. Verify changes don't break confluence
2. Catch divergence early in development
3. Block merges if confluence < 100%

## Results

Test results are stored in `results/` with the following naming:
- `{test_name}_{pipeline}_ir.json`: IR outputs
- `{test_name}_{target}_{pipeline}.txt`: Code outputs

## Status

As of 2026-01-30:
- ✅ Test framework implemented
- ✅ Test vectors created
- ⚠️ Awaiting complete pipeline implementations
- ⚠️ Confluence score: TBD
