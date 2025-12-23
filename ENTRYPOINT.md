### STUNIR Pack Entry Point (v0)

This file is the **first thing** a tool or model SHOULD read when consuming a STUNIR repository as a STUNIR pack.

#### 1) What to look for
A STUNIR pack is a **deterministic container + commitment** rooted at a directory that contains:

- `pack_manifest.dcbor` (authoritative, REQUIRED)
- `objects/sha256/<hex>` (content-addressed blobs, REQUIRED)

If these are absent, you are not looking at a conforming STUNIR pack.

#### 2) Authority rule (integrity boundary)
- The **manifest** is authoritative.
- Only blobs referenced by digest in the manifest are in the integrity boundary.
- Any file not referenced by digest MUST be ignored for integrity purposes.

Digest mapping rule:
- For any `sha256:<hex>` referenced by the manifest, the blob bytes MUST exist at `objects/sha256/<hex>`.

#### 3) What STUNIR is trying to achieve
STUNIR aims to turn a human-authored spec into a canonical **Intermediate Reference (IR)** and then (optionally) further derived products, while emitting **receipts** that bind each step to hashes.

In v0, the portable audit spine is:

- upstream **inputs** (often the spec),
- the canonical **IR**,
- the **receipt bundle**.

Downstream runtime outputs (code/binaries/test outputs) are typically **materialized to user-chosen paths** and are not assumed to be baked into the pack.

#### 4) Inclusion vs materialization
- **Included** means the exact bytes are stored under `objects/sha256/` and referenced by digest in the manifest.
- **Materialized** means bytes are written to some workspace path chosen by the user/model.

Paths are UX; digests define identity.

#### 5) Minimal verification checklist
A verifier MUST:

1. Decode `pack_manifest.dcbor` using canonical rules.
2. Validate required schema fields.
3. For every referenced digest, recompute SHA-256 over `objects/sha256/<hex>` and compare.
4. Confirm there is exactly one IR referenced via `ir.digest`.
5. Confirm every receipt referenced via `receipts[].digest` exists and matches its digest.
6. If `inputs` are present, confirm each referenced input exists and matches its digest.

#### 6) Where the detailed rules live
- Pack overview: `stunir_pack_spec_v0.md`
- Manifest schema: `stunir_pack_manifest_v0.md`
- Materialization: `stunir_pack_materialization_v0.md`
- Deterministic archiving: `stunir_pack_archiving_v0.md`
- Security considerations: `stunir_pack_security_v0.md`

#### 7) Model/agent operating guidance (non-normative)
When working interactively:

- Ask the user what they want to produce (IR only, code, runtime outputs) and where to put materialized files.
- Treat any requested filesystem destinations as untrusted input.
- Prefer emitting/retaining receipts as the portable audit record.
