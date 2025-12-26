### STUNIR Pack Manifest v1 — `pack_manifest.tsv`

#### 0. Purpose

This document defines a *shell-friendly* deterministic manifest format for pack-root files.
It is designed to strengthen Profile 3 verification without requiring:
- Python, or
- a custom verifier binary.

This manifest is an integrity mechanism, not an authenticity mechanism.
Authenticity still requires a signature (see `root_attestation.txt.sig` conventions).

#### 1. File location and encoding

- File name: `pack_manifest.tsv`
- Location: pack root
- Encoding: UTF-8
- Newlines: LF (`
`) only

#### 2. Line format

Each non-empty, non-comment line MUST be:

- `<sha256_hex64>	<relative_posix_path>`

Where:
- `<sha256_hex64>` is 64 lowercase hex characters.
- `<relative_posix_path>` is a relative path using forward slashes.

Comments:
- Lines beginning with `#` MUST be ignored.

#### 3. Ordering

- The file MUST be sorted lexicographically by `<relative_posix_path>` under `LC_ALL=C`.

#### 4. Scope

The manifest MUST enumerate regular files under the pack root with the following exclusions:
- `objects/sha256/**` (the object store is verified via `root_attestation.txt`)
- `pack_manifest.tsv` itself

Rationale:
- Excluding the object store avoids a fixed-point problem when the manifest itself is stored as an object.

#### 5. Path safety policy (mandatory)

The verifier MUST reject any path that violates any of the following:
- Must be relative (no leading `/`)
- Must not start with `./`
- Must not contain any `..` segment
- Must not contain empty path segments
- Must not contain whitespace characters
- Must not contain backslashes (`\`)
- No path segment may begin with `-`
- Allowed characters are limited to `[A-Za-z0-9._/-]`

#### 6. Binding the manifest to `root_attestation.txt`

If `pack_manifest.tsv` is used, it SHOULD be bound from `root_attestation.txt` as an `artifact` record:

- `artifact sha256:<digest> kind=manifest logical_path=pack_manifest.tsv`

The object `objects/sha256/<digest>` MUST exist and match the manifest bytes.

#### 7. Verifier behavior (Profile 3 strict)

A strict Profile 3 verifier MUST:
1. Verify `root_attestation.txt` minimal invariants and object-store integrity.
2. If a manifest binding is present:
   - Verify `pack_manifest.tsv` exists at the referenced `logical_path`.
   - Verify `SHA-256(pack_manifest.tsv) == <digest>`.
   - Verify `pack_manifest.tsv` has LF-only newlines.
   - Verify `pack_manifest.tsv` is sorted.
   - For each manifest entry, verify the target file exists and hash-matches.
