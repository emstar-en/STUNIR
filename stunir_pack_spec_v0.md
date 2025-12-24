### STUNIR Pack v0 (Directory or Archive)  Specification

#### 0. Status and scope
This document defines the **STUNIR Pack v0** interchange format.

A **STUNIR pack** is the unit a model (or other consumer) ingests as a **deterministic container + commitment** for:
- a canonical **Intermediate Reference (IR)**,
- an **attestation bundle** binding pipeline steps to digests,
- (optionally) the relevant **inputs** (e.g., the spec) needed to audit upstream.

A pack MAY also include other blobs, but **v0 does not assume that downstream runtime outputs are baked into the pack**.

A pack may be represented as:
- a **directory tree**, or
- an **archive** of that tree (e.g., a `.zip`).

#### 1. Terminology
- **Blob**: a byte string.
- **Digest**: a cryptographic hash identifier of a blob.
- **sha256 digest**: `sha256:<hex>` where `<hex>` is lowercase hex of 32-byte SHA-256.
- **Object store**: directory mapping digests to blobs.
- **Root attestation**: the canonical bootstrap artifact that inventories pack contents by digest.
- **Attestation artifact**: any emitted evidence object (e.g., step receipts, root attestation, provenance/SBOM objects, DSSE envelopes).
- **Receipt**: a step-scoped evidence object emitted by the STUNIR harness.

Normative keywords: **MUST**, **SHOULD**, **MAY**.

#### 2. High-level invariants
A conforming STUNIR pack:
1. **MUST be self-describing** via its root attestation.
2. **MUST be locally verifiable** without network access.
3. **MUST bind all included integrity-boundary content** by digest in the root attestation.

#### 3. Directory layout
A STUNIR pack directory MUST have:
- `objects/sha256/` (REQUIRED)
- one of:
  - `root_attestation.dcbor` (REQUIRED for the canonical encoding), or
  - `root_attestation.txt` (REQUIRED for minimal-toolchain environments)

A pack SHOULD include `root_attestation.dcbor`.
A pack MAY include both encodings.

Back-compat note:
- `pack_manifest.dcbor` MAY be present as a legacy alias.

#### 4. Object store
All included integrity-boundary content MUST be stored as digest-addressed blobs:
- Path form: `objects/sha256/<hex>`
- File bytes MUST be exactly the blob bytes.

The root attestation MUST reference blobs by digest (not by path). Paths are not part of the security boundary.

#### 5. Root attestation encodings
v0 supports two equivalent encodings of the root attestation:

##### 5.1 Canonical encoding: dCBOR
- File: `root_attestation.dcbor`
- Spec: `stunir_pack_root_attestation_v0.md`

This is the preferred encoding for strict determinism and compactness.

##### 5.2 Minimal-toolchain encoding: text
- File: `root_attestation.txt`
- Spec: `stunir_pack_root_attestation_text_v0.md`

This encoding exists to support environments that cannot run Python and cannot install custom binaries. It is intentionally parseable with:
- POSIX shell + awk/sed + `sha256sum`/`shasum`/`openssl`, and
- Windows PowerShell (`Get-FileHash`) or cmd (`certutil`).

##### 5.3 Equivalence rule
If both `root_attestation.dcbor` and `root_attestation.txt` are present:
- They MUST describe the same inventory and the same `ir.digest`.
- Consumers SHOULD treat `root_attestation.dcbor` as authoritative.

#### 6. Verification requirements
A verifier MUST implement:
1. Decode an available root attestation encoding.
2. Validate required fields for that encoding.
3. For every referenced digest, recompute SHA-256 over `objects/sha256/<hex>` and compare.
4. Confirm there is exactly one canonical IR referenced.

A verifier MAY provide a stronger mode that re-runs the harness and confirms it reproduces the same digests.

#### 7. Archive representation
A pack MAY be distributed as an archive of the directory tree.
Deterministic archiving guidance is defined in `stunir_pack_archiving_v0.md`.

#### 8. Security boundary notes
- Only blobs referenced by digest in the root attestation are authoritative.
- Any file not referenced by digest MUST be ignored for integrity purposes.

Security considerations are expanded in `stunir_pack_security_v0.md`.
