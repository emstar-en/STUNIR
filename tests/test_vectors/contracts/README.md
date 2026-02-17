# STUNIR Contracts Test Vectors

Test vectors for validating the STUNIR contracts pipeline stage.

## Issue

**test_vectors/contracts/1011**: Complete test_vectors → contracts pipeline stage

## Overview

This module provides deterministic test vectors for:
- Profile 2/3/4 contract validation
- Contract schema compliance
- Stage ordering verification
- Attestation requirements

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
| tv_contracts_001 | Profile 2 Contract Structure | Verify Profile 2 schema |
| tv_contracts_002 | Profile 3 Contract Structure | Verify Profile 3 with test stage |
| tv_contracts_003 | Profile 4 Contract Structure | Verify Profile 4 with attestation |
| tv_contracts_004 | Invalid Profile Contract | Detect invalid profile |
| tv_contracts_005 | Contract Stage Ordering | Verify stage ordering |

## Schema

Test vectors follow the `stunir.test_vector.contracts.v1` schema.
