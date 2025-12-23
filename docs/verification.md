# STUNIR Verification Contract (pragmatic, auditor-readable)

This verifier is designed to be transparent enough for SLSA/in-toto-minded auditors **without** adopting SLSA bureaucracy.

## What is verified

1. **Receipt integrity**
   - DSSE v1 signature verification (payload signed via PAE).
   - Payload JSON must be in canonical byte form (`stunir-json-c14n-v1`).

2. **Input closure**
   - A single input manifest file is identified in the receipt payload.
   - The manifest enumerates all input files and their digests.
   - The verifier checks all listed files exist and match the declared multi-alg digests.

3. **IR rebuild**
   - The verifier runs the IR rebuild command declared in the receipt.
   - The resulting IR file is checked against declared digests.
   - If canonicalization is declared, the IR JSON bytes must be canonical.

4. **Artifact rebuild**
   - The verifier runs the codegen rebuild command declared in the receipt.
   - The resulting artifact manifest is checked against declared digests.
   - Optionally, the in-toto Statement `subject[0]` can match the artifact manifest digest.

## Canonicalization: `stunir-json-c14n-v1`

A JSON byte sequence is canonical iff it is exactly the output of:

- UTF-8 encoding
- no floats anywhere (integers only)
- object keys sorted lexicographically by Unicode codepoint
- no whitespace (minified)
- JSON string escaping as produced by Python `json.dumps(..., ensure_ascii=False)`

The verifier enforces this by parsing JSON and re-serializing with the canonical encoder, then requiring **byte equality**.

## Receipt payload shape (recommended)

Use an in-toto Statement wrapper for readability:

- `_type` = `https://in-toto.io/Statement/v1`
- `predicateType` = `urn:stunir:receipt:v1`

Closure model (#2): the payload contains:

- `predicate.materials[]` including the input manifest entry `{uri, digest{...}}`
- `predicate.specClosure.manifest.uri` pointing to that same `uri`

The manifest JSON should contain:

```json
{ "files": [ { "path": "spec/foo.json", "digest": {"sha256": "..."} } ] }
```

## Running

```bash
./scripts/verify.sh receipt.dsse.json   --trust-key mykeyid=keys/pubkey.pem   --required-algs sha256,sha512
```

Notes:
- Repeat `--trust-key` for multiple trusted keyids.
- If Python `cryptography` is not installed, the verifier attempts an `openssl` fallback.
