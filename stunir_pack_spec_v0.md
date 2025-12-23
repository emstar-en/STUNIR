### STUNIR Pack v0 (Directory or Archive) — Specification

#### 0. Status and scope
This document defines the **STUNIR Pack v0** interchange format.

A **STUNIR pack** is the unit a model (or other consumer) ingests as a **deterministic container + commitment** for:

- a canonical **Intermediate Reference (IR)**,
- a **receipt bundle** binding pipeline steps to digests,
- (optionally) the relevant **inputs** (e.g., the spec) needed to audit upstream.

A pack MAY also include other blobs, but **v0 does not assume that downstream runtime outputs are baked into the pack**. A normal STUNIR conversational UX can materialize code/binaries/outputs wherever the user asks; receipts are the primary audit object.

A pack may be represented as:

- a **directory tree** (e.g., in a repo checkout), or
- an **archive** of that tree (e.g., a `.zip`).

This spec defines:

- a canonical on-disk directory layout,
- a canonical content-addressed object store,
- a canonical pack manifest,
- verification requirements,
- deterministic archive recommendations.

#### 1. Terminology
- **Blob**: a byte string.
- **Digest**: a cryptographic hash identifier of a blob.
- **sha256 digest**: `sha256:<hex>` where `<hex>` is lowercase hex of 32-byte SHA-256.
- **Object store**: directory mapping digests to blobs.
- **Pack manifest**: a canonical document enumerating pack contents by digest.
- **Receipt**: signed or unsigned evidence object emitted by the STUNIR harness.
- **Materialize**: write/copy bytes to user-chosen filesystem locations.

Normative keywords: **MUST**, **SHOULD**, **MAY**.

#### 2. High-level invariants
A conforming STUNIR pack:

1. **MUST be self-describing** via its manifest.
2. **MUST be locally verifiable** without network access.
3. **MUST bind all included integrity-boundary content** by digest in the manifest.
4. **MUST be deterministic as an artifact**: given identical manifest bytes and identical referenced objects, the directory representation and any canonical archive representation MUST have stable bytes.

Notes:
- This spec is about the **pack** as a deterministic container/commitment.
- It does **not** prescribe conversational behavior or where outputs are written.

#### 3. Inclusion vs materialization (important distinction)
STUNIR can “take someone all the way to runtime” by **materializing outputs wherever the user instructs the model to put them**.

This implies two different notions:

- **Inclusion (pack content):** A blob is *included* if its bytes live under `objects/sha256/<hex>` and the manifest references its digest.
  - Included blobs are inside the pack integrity boundary.
  - Included blobs are portable and can be verified offline.

- **Materialization (workspace output):** A blob is *materialized* if it is written to some path in a working directory (e.g., `./out/app.py`, `/tmp/run/out.txt`).
  - Materialized paths are **not** part of the pack integrity boundary.
  - Materialized paths are user/environment-specific.

A typical workflow is:

- include **inputs + IR + receipts** in the pack, and
- materialize downstream outputs on demand.

#### 4. Included content classes (v0 emphasis)
v0 draws a strong distinction:

- **Receipts are primary** for audit.
- **Inputs and IR are what auditors will pull upstream** to validate provenance.
- Baked downstream outputs (code/binaries/runtime outputs) are OPTIONAL and should not be assumed.

If downstream outputs are included, they MUST be treated as ordinary included blobs referenced by digest; see the optional `artifacts` section in the manifest schema.

#### 5. Directory layout
A STUNIR pack directory MUST have:

- `pack_manifest.dcbor`  (REQUIRED)
- `objects/sha256/`      (REQUIRED)

Optionally, it MAY have:

- `index/`               (OPTIONAL; pointers/aliases for humans)
- `meta/`                (OPTIONAL; non-authoritative convenience docs)

Unknown extra paths MUST be ignored by verifiers unless the manifest explicitly declares them as included objects.

##### 5.1 Object store
All included integrity-boundary content MUST be stored as digest-addressed blobs:

- Path form: `objects/sha256/<hex>`
- File bytes MUST be exactly the blob bytes.

The manifest MUST reference blobs by digest (not by path). Paths are not part of the security boundary.

#### 6. Manifest: authority and commitment
The **manifest** is the authority for what the pack includes.

- The manifest MUST be encoded as **canonical dCBOR** (per STUNIR canonicalization rules).
- The manifest bytes are the primary **commitment root**.

This spec defines the v0 manifest schema in `stunir_pack_manifest_v0.md`.

#### 7. Verification requirements
A verifier MUST implement the following checks:

1. **Manifest decoding**: decode `pack_manifest.dcbor` using canonical rules.
2. **Schema check**: validate required keys, types, and version.
3. **Object integrity**: for every digest referenced by the manifest, recompute SHA-256 over the corresponding object store file and compare.
4. **IR presence**: the manifest MUST reference exactly one canonical IR blob via `ir.digest`.
5. **Receipt presence**: each receipt digest listed MUST be present as a blob.
6. **Input integrity (if present)**: each input digest listed MUST be present as a blob.

A verifier MAY provide a “strong mode” that re-runs the harness and confirms it reproduces the same digests; this is outside pack-format conformance.

#### 8. Materialization guidance (non-normative)
Materialization is the act of writing included blobs (or newly produced results) to user-chosen paths.

Recommended practice:
- materialization should be a *pure copy* of bytes from digest-addressed blobs,
- any path mapping should be treated as user data (not part of the core commitment),
- if you need an auditable record of where things were written, emit a **materialization receipt** whose core identifier excludes absolute paths.

See `stunir_pack_materialization_v0.md`.

#### 9. Archive representation (zip, tar)
A pack MAY be distributed as an archive of the directory tree.

- The archive MUST contain the same paths as the directory representation.
- The archive MUST preserve file bytes exactly.

Deterministic archiving guidance is defined in `stunir_pack_archiving_v0.md`.

#### 10. Security boundary notes
- Only blobs referenced by digest in the manifest are authoritative.
- Any file not referenced by digest MUST be ignored for integrity purposes.
- Consumers MUST defend against archive path traversal (“zip slip”).

Security considerations are expanded in `stunir_pack_security_v0.md`.
