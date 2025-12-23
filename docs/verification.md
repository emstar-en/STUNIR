# STUNIR Verification Contract

This repo has **two verification layers**:

1. **Local verification** (default)
   - Verifies the receipts/manifests produced by `scripts/build.sh` in your working tree.
   - This is what you want when you ran a build locally and want a deterministic, auditable “small checker”.

2. **DSSE verification**
   - Verifies a **DSSE v1 envelope** containing an **in-toto Statement** payload.
   - This is what you want when an orchestrator shipped you a signed attestation bundle.

## Canonical JSON: `stunir-json-c14n-v1`
A JSON byte sequence is canonical iff it is exactly the output of:

- UTF-8 encoding
- **no floats anywhere** (integers only)
- object keys sorted lexicographically by Unicode codepoint
- no whitespace (minified)
- JSON string escaping as produced by Python `json.dumps(..., ensure_ascii=False)`

The verifier enforces this by parsing JSON and re-serializing with the canonical encoder, then requiring **byte equality**.
For local files, the verifier accepts an optional trailing newline.

## Local verification (default)

### Run
```bash
./scripts/build.sh
./scripts/verify.sh
```

### What is checked (local mode)
- Canonical JSON serialization for:
  - `build/epoch.json`
  - `build/provenance.json`
  - `receipts/*.json` and IR manifests
- Receipt integrity:
  - `receipt_core_id_sha256` recomputation
  - `target` sha256 matches the on-disk target when present
  - `inputs[]` file and dir digests match on disk (dir digests accept the legacy and explicit traversal order)
  - tool identity sha256 matches the referenced tool when present
- IR manifest integrity:
  - `receipts/ir_manifest.json` matches the `asm/ir/**/*.dcbor` set and per-file sha256s (exact set in `--strict` mode)
  - `receipts/ir_bundle_manifest.json` matches `asm/ir_bundle.bin` and its offset table

### Strictness
`scripts/verify.sh` runs local verification with `--strict` by default.

## Snapshot / fixture verification

Snapshots created by `scripts/snapshot_receipts.sh` are stored under:
- `fixtures/receipts/<TAG>/receipts/`
- `fixtures/receipts/<TAG>/build/`

To verify a snapshot later:

1. Check out the **same git revision** that produced the snapshot.
2. Rebuild outputs using the snapshot epoch:
   ```bash
   cp fixtures/receipts/<TAG>/build/epoch.json build/epoch.json
   STUNIR_PRESERVE_EPOCH=1 STUNIR_VERIFY_AFTER_BUILD=0 ./scripts/build.sh
   ```
3. Verify the snapshot receipts against the rebuilt outputs:
   ```bash
   ./scripts/verify.sh --root fixtures/receipts/<TAG>
   ```

## DSSE verification

### What is verified
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

### Receipt payload shape (recommended)
Use an in-toto Statement wrapper for readability:

- `_type` = `https://in-toto.io/Statement/v1`
- `predicateType` = `urn:stunir:receipt:v1`

Closure model: the payload contains:

- `predicate.materials[]` including the input manifest entry `{uri, digest{...}}`
- `predicate.specClosure.manifest.uri` pointing to that same `uri`

The manifest JSON should contain:

```json
{ "files": [ { "path": "spec/foo.json", "digest": {"sha256": "..."} } ] }
```

### Run
```bash
./scripts/verify.sh receipt.dsse.json --trust-key mykeyid=keys/pubkey.pem --required-algs sha256,sha512
```

Notes:
- Repeat `--trust-key` for multiple trusted keyids.
- If Python `cryptography` is not installed, the verifier attempts an `openssl` fallback.
