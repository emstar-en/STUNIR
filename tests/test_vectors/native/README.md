# STUNIR Native Test Vectors

Test vectors for validating the STUNIR native tool integration.

## Issue

**test_vectors/native/1034**: Complete test_vectors → native pipeline stage

## Overview

This module provides deterministic test vectors for:
- Haskell native tool (stunir-native) operations
- Rust native tool (stunir-rust) operations
- Manifest and provenance generation
- Native tool determinism verification

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

| ID | Name | Description |
|----|------|-------------|
| tv_native_001 | Haskell Manifest Generation | Verify manifest output |
| tv_native_002 | Haskell Provenance Generation | Verify C header output |
| tv_native_003 | Rust Tool Compilation | Verify Rust build |
| tv_native_004 | Native Tool Version Check | Verify version output |
| tv_native_005 | Native Tool Determinism | Verify deterministic output |

## Schema

Test vectors follow the `stunir.test_vector.native.v1` schema.
