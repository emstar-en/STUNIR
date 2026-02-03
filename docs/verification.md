# STUNIR Verification Contract

This repo supports multiple verification profiles.

## Profile 1: Local verification (Python)

### Run

- `./scripts/build.sh`
- `./scripts/verify.sh`

Local verification checks canonical JSON, receipt integrity, IR manifests, and related invariants.

## Profile 2: Portable verifier binary (no Python)

A future/optional `stunir-verify` binary can be used to verify packs in environments where Python is unavailable but a standalone executable is allowed.

## Profile 3: Minimal verification (no Python, no custom binaries)

This profile is designed for very constrained environments.

Requirements:
- The pack MUST include `root_attestation.txt`.
- The pack MUST include `objects/sha256/`.
- A hashing tool MUST be available:
  - Linux/macOS: `sha256sum` or `shasum` or `openssl`
  - Windows PowerShell: `Get-FileHash`
  - Windows cmd: `certutil`

### POSIX-like

Use `scripts/verify_minimal.sh`.

### Windows PowerShell

Use `scripts/verify_minimal.ps1`.

### Windows cmd

Use `scripts/verify_minimal.cmd`.

### Optional (POSIX-like): strict Profile 3 manifest verification

If the pack binds a `pack_manifest.tsv` (see `docs/root/stunir_pack_manifest_v1.md`), you can run the stricter verifier:

- `scripts/verify_profile3.sh`

This performs the minimal object-store checks plus pack-root manifest checking.

## Canonical JSON: `stunir-json-c14n-v1`

(unchanged)

A JSON byte sequence is canonical iff it is exactly the output of:
- UTF-8 encoding
- no floats anywhere (integers only)
- object keys sorted lexicographically by Unicode codepoint
- no whitespace (minified)
- JSON string escaping as produced by Python `json.dumps(..., ensure_ascii=False)`

The verifier enforces this by parsing JSON and re-serializing with the canonical encoder, then requiring byte equality.

For local files, the verifier accepts an optional trailing newline.


Profile 3 (strict): the verifier MUST hard-fail if root_attestation.txt does not bind pack_manifest.tsv via its SHA-256 digest.
