# STUNIR IR bundle v1 (authoritative)

This repository defines **`stunir.ir_bundle.v1`** as the byte-exact “meaning anchor” for STUNIR.

## What is byte-exact?

Only the **IR bundle bytes** are required to be identical across platforms/engines.
Receipts/logs/packaging are allowed to vary.

## Canonical IR Content (CIR)

In v1 (authoritative), CIR is:

- a **JSON array** of **JSON objects** ("units")
- with **floats forbidden** (JSON numbers must be integers)
- with **all strings NFC-normalized** (keys and values)
- with **duplicate keys forbidden**, including duplicates introduced by NFC normalization

## IR bundle bytes

The IR bundle bytes are canonical DCBOR encoding of:

- `[
    "stunir.ir_bundle.v1",
    cir_units
  ]`

## Reference implementation

- `tools/ir_bundle_v1.py` implements:
  - CIR JSON normalization and validation
  - canonical DCBOR encoding (RFC 8949 canonical CBOR; definite lengths only)
  - IR bundle construction and receipt emission

- `tools/build_ir_bundle_v1.py` is a small shim CLI.

## Test vectors

`tests/test_ir_bundle_v1_vectors.json` contains a small vector set.
`tests/test_ir_bundle_v1.py` validates that encoding and hashes match.
