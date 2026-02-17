# STUNIR Test Vectors

Deterministic test vectors for STUNIR pipeline validation.

## Phase 5 Issues Resolved

| Issue ID | Title | Status |
|----------|-------|--------|
| test_vectors/contracts/1011 | Complete test_vectors → contracts pipeline stage | ✅ |
| test_vectors/native/1034 | Complete test_vectors → native pipeline stage | ✅ |
| test_vectors/polyglot/1035 | Complete test_vectors → polyglot pipeline stage | ✅ |
| test_vectors/receipts/1036 | Complete test_vectors → receipts pipeline stage | ✅ |
| test_vectors/edge_cases/1065 | Complete test_vectors → edge_cases pipeline stage | ✅ |
| test_vectors/property/1135 | Complete test_vectors → property pipeline stage | ✅ |

## Architecture

```
test_vectors/
├── base.py                 # Shared utilities (canonical JSON, hashing, generators)
├── __init__.py             # Package exports
├── contracts/              # Contract validation test vectors
│   ├── gen_vectors.py      # Generator
│   ├── validate.py         # Validator
│   └── README.md           # Documentation
├── native/                 # Native tool integration test vectors
├── polyglot/               # Cross-language target test vectors
├── receipts/               # Receipt validation test vectors
├── edge_cases/             # Boundary condition test vectors
└── property/               # Property-based test vectors
```

## Usage

### Generate All Test Vectors

```bash
# Generate test vectors for each category
python test_vectors/receipts/gen_vectors.py
python test_vectors/contracts/gen_vectors.py
python test_vectors/native/gen_vectors.py
python test_vectors/polyglot/gen_vectors.py
python test_vectors/edge_cases/gen_vectors.py
python test_vectors/property/gen_vectors.py
```

### Validate All Test Vectors

```bash
# Validate test vectors for each category
python test_vectors/receipts/validate.py
python test_vectors/contracts/validate.py
python test_vectors/native/validate.py
python test_vectors/polyglot/validate.py
python test_vectors/edge_cases/validate.py
python test_vectors/property/validate.py
```

## Test Vector Schema

All test vectors follow the `stunir.test_vector.<category>.v1` schema:

```json
{
  "schema": "stunir.test_vector.<category>.v1",
  "id": "tv_<category>_<nnn>",
  "name": "Human-readable test name",
  "description": "What this test verifies",
  "input": { ... },
  "expected_output": { ... },
  "expected_hash": "sha256...",
  "tags": ["unit", "determinism", ...],
  "created_epoch": 1735500000
}
```

## Determinism

All test vectors are generated with fixed seeds and epochs to ensure reproducibility.
Running the generators multiple times produces identical output.

## Categories

### contracts
Tests for profile contract validation (Profile 2/3/4), schema compliance, and stage ordering.

### native
Tests for Haskell and Rust native tool integration, manifest generation, and provenance.

### polyglot
Tests for cross-language target generation (Rust, C89, C99) and type mapping.

### receipts
Tests for receipt structure validation, hash verification, and manifest generation.

### edge_cases
Tests for boundary conditions: empty inputs, null values, Unicode, circular references, deep nesting.

### property
Tests for mathematical properties: idempotence, commutativity, invertibility, monotonicity, transitivity, determinism, associativity.
