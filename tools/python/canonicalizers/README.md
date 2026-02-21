# STUNIR Canonicalizers

Part of the `tools → canonicalizers` pipeline stage.

## Overview

Canonicalization ensures deterministic output for hashing and verification.

## Tools

### canonicalize_json.py

RFC 8785 / JCS subset JSON canonicalization.

```bash
python canonicalize_json.py <input.json> [output.json]
```

### canonicalize_dcbor.py

Deterministic CBOR (dCBOR) canonicalization.

```bash
python canonicalize_dcbor.py <input.json> [output.dcbor]
```

## Canonicalization Rules

### JSON (RFC 8785 subset)
1. Keys sorted alphabetically (Unicode code point order)
2. No whitespace between tokens
3. No trailing newline
4. UTF-8 encoded

### dCBOR
1. Canonical CBOR map key ordering
2. Deterministic float encoding
3. No indefinite-length encodings
