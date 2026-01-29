# STUNIR Conformance Test Suite (Pure Haskell)

**Version:** 2.0.0  
**Status:** Full Python Test Parity Achieved  
**Test Suites:** 16  
**Test Cases:** 68+

## Overview

This is a comprehensive, pure Haskell implementation of the STUNIR conformance test suite. It provides complete verification of IR canonicalization, manifest generation, receipt verification, and determinism guarantees for Python-averse environments.

**Version 2.0** achieves full parity with all Python tests in the repository, including:
- All test vector categories from `test_vectors/`
- Unit tests from `tests/test_ir_bundle_v1.py`
- Additional integration and performance tests

## Quick Start

```bash
# Run all tests
make test

# List available test suites
make list

# Run specific suite
make test-suite SUITE=contracts

# Show Python test coverage
make coverage
```

## Test Suite Categories

### Core Tests (7 suites)

| Suite | Description | Tests |
|-------|-------------|-------|
| `ir-canon` | IR canonicalization verification | 6 |
| `manifest-gen` | Manifest generation determinism | 4 |
| `receipt-verify` | Receipt verification | 3 |
| `hash-determ` | SHA256 hash determinism | 4 |
| `target-gen` | Basic target generation | 3 |
| `schema-valid` | Schema compliance | 4 |
| `provenance` | Provenance tracking | 3 |

### Python-Equivalent Tests (9 suites)

| Suite | Python Source | Haskell Module | Tests |
|-------|---------------|----------------|-------|
| `contracts` | `test_vectors/contracts/` | `ContractsVectorTest.hs` | 4 |
| `native` | `test_vectors/native/` | `NativeVectorTest.hs` | 4 |
| `polyglot` | `test_vectors/polyglot/` | `PolyglotVectorTest.hs` | 4 |
| `receipts` | `test_vectors/receipts/` | `ReceiptsVectorTest.hs` | 4 |
| `edge-cases` | `test_vectors/edge_cases/` | `EdgeCasesVectorTest.hs` | 6 |
| `property` | `test_vectors/property/` | `PropertyVectorTest.hs` | 6 |
| `ir-bundle` | `tests/test_ir_bundle_v1.py` | `IRBundleTest.hs` | 5 |
| `pipeline` | (integration) | `PipelineIntegrationTest.hs` | 4 |
| `performance` | (regression) | `PerformanceTest.hs` | 4 |

## Python Test Coverage

See [PYTHON_TEST_MAPPING.md](PYTHON_TEST_MAPPING.md) for detailed mapping between Python and Haskell tests.

### Coverage Summary

| Category | Python | Haskell | Status |
|----------|--------|---------|--------|
| Test Vector Categories | 6 | 6 | ✅ Complete |
| Unit Test Files | 1 | 1 | ✅ Complete |
| Integration Tests | - | 1 | ✅ Added |
| Performance Tests | - | 1 | ✅ Added |
| **Total Test Cases** | ~28 | 68+ | ✅ Exceeded |

## Installation

### Prerequisites

- GHC 9.4+ or Stack
- Cabal 3.6+

### Build

```bash
# Using Stack (recommended)
stack build

# Using Cabal
cabal build all
```

### Run Tests

```bash
# All tests
make test

# Specific categories
make test-core          # Core tests only
make test-python-parity # Python-equivalent tests only

# Verbose output
make test-verbose
```

## Project Structure

```
test/haskell/
├── README.md                    # This file
├── PYTHON_TEST_MAPPING.md       # Python-Haskell test mapping
├── Makefile                     # Build targets
├── stunir-conformance-tests.cabal
├── stack.yaml
├── Setup.hs
├── src/
│   └── Test/
│       ├── Harness.hs          # Core test harness
│       ├── Determinism.hs      # Determinism utilities
│       ├── Utils.hs            # Shared utilities
│       └── Vectors.hs          # Test vector loading
├── tests/
│   ├── Main.hs                 # Test runner entry point
│   │
│   │ # Core test modules
│   ├── IRCanonTest.hs
│   ├── ManifestGenTest.hs
│   ├── ReceiptVerifyTest.hs
│   ├── HashDeterminismTest.hs
│   ├── TargetGenTest.hs
│   ├── SchemaValidationTest.hs
│   ├── ProvenanceTest.hs
│   │
│   │ # Python-equivalent test modules
│   ├── ContractsVectorTest.hs
│   ├── NativeVectorTest.hs
│   ├── PolyglotVectorTest.hs
│   ├── ReceiptsVectorTest.hs
│   ├── EdgeCasesVectorTest.hs
│   ├── PropertyVectorTest.hs
│   ├── IRBundleTest.hs
│   ├── PipelineIntegrationTest.hs
│   └── PerformanceTest.hs
├── app/
│   └── RunConformance.hs       # CLI runner
├── ci/
│   └── run_tests.sh            # CI integration
└── test_data/                  # Symlinks to Python test vectors
    ├── contracts/ -> ../../../test_vectors/contracts/
    ├── native/ -> ../../../test_vectors/native/
    └── ...
```

## Adding New Tests

### 1. Create Test Module

```haskell
{-# LANGUAGE OverloadedStrings #-}

module NewFeatureTest (suite) where

import Test.Harness
import Test.Utils

suite :: TestSuite
suite = testSuite "New Feature"
    [ testCase "Test Name" "Description" $ do
          -- Test implementation
          assertEqual "Expected" expected actual
    ]
```

### 2. Register in Main.hs

```haskell
import qualified NewFeatureTest

allSuites = 
    [ ...existing suites...
    , NewFeatureTest.suite
    ]
```

### 3. Update Cabal File

```cabal
other-modules:    ...
                , NewFeatureTest
```

## Test Data

Test data is loaded from Python test vectors via symlinks:

```bash
make test-data  # Setup symlinks
```

Manual setup:
```bash
mkdir -p test_data
ln -s ../../../test_vectors/contracts test_data/contracts
# ... repeat for other categories
```

## CI Integration

```bash
# Run in CI
./ci/run_tests.sh

# With JUnit output
make test 2>&1 | tee test-results.log
```

## Determinism Guarantees

All tests verify deterministic behavior:

1. **Canonical JSON** - Keys sorted alphabetically, no whitespace
2. **SHA256 Hashing** - Consistent across runs
3. **IR Generation** - Same input → same output
4. **Manifest Creation** - Reproducible entries

## Contributing

When adding tests:

1. Ensure corresponding Python test exists (or document why not)
2. Update [PYTHON_TEST_MAPPING.md](PYTHON_TEST_MAPPING.md)
3. Run full test suite before committing
4. Add test to appropriate category

## License

MIT License - See [LICENSE](../../LICENSE) for details.

## Changelog

### Version 2.0.0 (2026-01-28)
- Added 9 new test suites for Python test parity
- Created 68+ test cases (up from 27)
- Added Test.Vectors module for test vector loading
- Updated Makefile with new targets
- Created PYTHON_TEST_MAPPING.md documentation
- Achieved full parity with Python test suite

### Version 1.0.0 (2026-01-27)
- Initial release with 7 core test suites
- 27 test cases
- Basic test harness infrastructure
