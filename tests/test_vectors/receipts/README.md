# STUNIR Receipts Test Vectors

Test vectors for validating the STUNIR receipts pipeline stage.

## Issue

**test_vectors/receipts/1036**: Complete test_vectors → receipts pipeline stage

## Overview

This module provides deterministic test vectors for:
- Receipt structure validation
- Receipt hash verification
- Receipt manifest generation
- Epoch timestamp validation

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
| tv_receipts_001 | Basic Receipt Structure | Verify basic receipt JSON structure |
| tv_receipts_002 | Receipt Missing Hash | Detect missing artifact_hash field |
| tv_receipts_003 | Receipt Hash Verification | Verify receipt hash matches artifact |
| tv_receipts_004 | Receipt Manifest Generation | Verify manifest generation |
| tv_receipts_005 | Receipt Epoch Validation | Verify epoch timestamp validity |

## Schema

Test vectors follow the `stunir.test_vector.receipts.v1` schema.
