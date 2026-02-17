# STUNIR Edge Cases Test Vectors

Test vectors for validating boundary conditions and edge cases.

## Issue

**test_vectors/edge_cases/1065**: Complete test_vectors → edge_cases pipeline stage

## Overview

This module provides deterministic test vectors for:
- Empty/null input handling
- Maximum field length boundaries
- Unicode and special character handling
- Circular reference detection
- Deep nesting limits
- Malformed input graceful handling

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
| tv_edge_cases_001 | Empty Spec Input | Handle empty spec gracefully |
| tv_edge_cases_002 | Null Value Handling | Handle null values |
| tv_edge_cases_003 | Maximum Field Length | Enforce length limits |
| tv_edge_cases_004 | Unicode Character Handling | Preserve Unicode |
| tv_edge_cases_005 | Circular Reference Detection | Detect circular deps |
| tv_edge_cases_006 | Deep Nesting Handling | Handle deep structures |
| tv_edge_cases_007 | Special Characters in Keys | Handle special chars |

## Schema

Test vectors follow the `stunir.test_vector.edge_cases.v1` schema.
