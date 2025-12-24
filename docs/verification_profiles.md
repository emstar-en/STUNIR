### STUNIR Verification Profiles (Toolchain Levels)

STUNIR aims to make verification possible in constrained environments.
This document defines three practical verification profiles.

#### Profile 1: Full verification (Python)
Intended for developer machines and richer CI environments.

Characteristics:
- Uses the repo's Python tools.
- Performs schema checks, canonicalization checks, receipt integrity checks, and may re-run rebuild commands.

Pros:
- Strongest checks.

Cons:
- Requires Python.

#### Profile 2: Portable verifier binary (no Python)
Intended for environments without Python but where running a standalone binary is permitted.

Characteristics:
- A single `stunir-verify` executable that:
  - decodes `root_attestation.dcbor`,
  - recomputes SHA-256 for referenced objects,
  - validates structural rules (exactly one IR, etc.),
  - optionally verifies DSSE/in-toto envelopes.

Suggested implementation languages:
- Go or Rust.

Pros:
- Works in locked-down CI (no interpreter).
- Keeps canonical dCBOR as the bootstrap.

Cons:
- Requires distributing a binary.

#### Profile 3: Minimal verification (no Python, no custom binaries)
Intended for very constrained environments where only built-in OS tooling is allowed.

This profile relies on:
- `root_attestation.txt` (the minimal-toolchain root attestation encoding), and
- OS-provided hashing tools.

Provided scripts:
- `scripts/verify_minimal.sh` (POSIX-like)
- `scripts/verify_minimal.ps1` (Windows PowerShell)
- `scripts/verify_minimal.cmd` (Windows cmd)

What it verifies:
- Every digest listed in `root_attestation.txt` corresponds to a blob under `objects/sha256/<hex>`.
- The blob's SHA-256 matches.
- There is exactly one `ir` record.

What it does NOT verify (typically):
- Deep schema validation of receipt payloads.
- Rebuild/replay.
- Signature verification unless the environment has an allowed crypto tool.

Recommendation:
- If authenticity is required in Profile 3, distribute a public key and enable signature verification of `root_attestation.txt` when possible.
