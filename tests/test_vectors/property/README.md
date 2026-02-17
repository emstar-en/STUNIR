# STUNIR Property Test Vectors

Test vectors for property-based testing and invariant verification.

## Issue

**test_vectors/property/1135**: Complete test_vectors → property pipeline stage

## Overview

This module provides deterministic test vectors for verifying:
- **Idempotence**: f(f(x)) == f(x)
- **Commutativity**: Order-independent operations
- **Invertibility**: Round-trip preservation
- **Monotonicity**: Sequential ordering guarantees
- **Transitivity**: Dependency propagation
- **Determinism**: Reproducible outputs (CRITICAL for STUNIR)
- **Associativity**: Grouping independence

## Usage

### Generate Test Vectors

```bash
python gen_vectors.py [--output <dir>]
```

### Validate Test Vectors

```bash
python validate.py [--dir <dir>]
```

## Test Cases

| ID | Name | Property |
|----|------|----------|
| tv_property_001 | Canonicalization Idempotence | idempotence |
| tv_property_002 | Hash Order Independence | commutativity |
| tv_property_003 | Serialize-Deserialize Roundtrip | invertibility |
| tv_property_004 | Epoch Monotonicity | monotonicity |
| tv_property_005 | Dependency Transitivity | transitivity |
| tv_property_006 | Pipeline Determinism | determinism |
| tv_property_007 | Manifest Merge Associativity | associativity |

## Schema

Test vectors follow the `stunir.test_vector.property.v1` schema.

## Properties Explained

### Determinism (Critical)
The most important property for STUNIR. Every run with the same input must produce
bitwise-identical output. This enables:
- Reproducible builds
- Verification of correct implementation
- Trust in the build system
