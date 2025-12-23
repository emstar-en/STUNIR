### STUNIR Pack Root Attestation v0 — Schema

#### 0. Encoding
`root_attestation.dcbor` MUST be canonical dCBOR.

#### 1. Top-level object
The root attestation is a CBOR map.

Required keys:
- `attestation_version` (text) — MUST equal `"stunir.pack.root_attestation.v0"`
- `ir` (map) — IR descriptor
- `receipts` (array) — zero or more receipt descriptors

Optional keys:
- `inputs` (array) — zero or more input descriptors (recommended for upstream audit)
- `epoch` (int or text) — epoch used by the harness (e.g., Unix seconds) or a policy name
- `policies` (map) — named policy digests (each value is a digest string)
- `toolchain` (map) — toolchain descriptors by digest (informational unless receipts bind it)
- `artifacts` (array) — zero or more included non-core blobs (downstream outputs, SBOMs, envelopes, etc.)
- `extensions` (map) — namespaced extension content

Notes:
- v0 emphasizes `inputs` + `ir` + `receipts`.
- Baked downstream outputs are optional.
- This schema is intentionally inventory-centric: it is meant to serve as the verifier bootstrap (“shopping list”).

#### 2. Digest string format
A digest is a text string of the form:
- `"sha256:<hex>"`

`<hex>` MUST be lowercase hex.

#### 3. IR descriptor
`ir` is a map with required keys:
- `digest` (text) — digest of the canonical IR blob
- `media_type` (text) — e.g., `"application/stunir-ir+dcbor"`

Optional:
- `name` (text) — human-friendly label

#### 4. Input descriptor (recommended)
Each entry in `inputs` is a map with required keys:
- `digest` (text)
- `media_type` (text) — e.g., `"text/markdown"`, `"application/json"`, `"application/stunir-spec+text"`
- `kind` (text) — stable input kind string

Optional:
- `name` (text) — human-friendly label

Suggested `kind` values:
- `"spec"`
- `"source_seed"` (if you input source directly)
- `"constraints"`

#### 5. Receipt descriptor
Each entry in `receipts` is a map with required keys:
- `digest` (text)
- `media_type` (text) — e.g., `"application/stunir-receipt+dcbor"` or `"application/vnd.in-toto+json"`

Optional:
- `purpose` (text) — e.g., `"spec->ir"`, `"ir->codegen"`, `"verify"`, `"materialize"`
- `signature` (map) — signature envelope descriptor (digest refs)

Notes:
- The root attestation does not require receipts to be signed, but signed receipts improve authenticity.

#### 6. Artifact descriptor (optional; included non-core blobs)
Each entry in `artifacts` describes a blob that is **included in the pack** but is not required for the minimal audit spine.

Required keys:
- `digest` (text)
- `media_type` (text)
- `kind` (text) — a stable STUNIR artifact kind string

Optional keys:
- `target` (map) — language/platform target qualifiers
- `logical_path` (text) — suggested *relative* path name for convenience (non-authoritative)
- `source_ir` (text) — digest string; MUST match `ir.digest` if present

Notes:
- `logical_path` is not an integrity boundary. It is a hint.
- Absolute paths MUST NOT appear in `logical_path`.

Suggested `kind` values (non-exhaustive):
- `"code.python"`
- `"binary.linux_amd64"`
- `"output.reference"`
- `"sbom.spdx"`
- `"attestation.in-toto"`
- `"envelope.dsse"`

#### 7. Object store mapping rule
For any digest `sha256:<hex>`, the corresponding blob MUST be present at:
- `objects/sha256/<hex>`

The root attestation MUST NOT reference filesystem paths for integrity.

#### 8. Canonical pack identifier (recommended)
A consumer MAY define:
$$	ext{pack_id} = 	ext{SHA256}(	ext{bytes}(root_attestation.dcbor))$$

If receipts include “core identifiers” (excluding platform/path noise), they SHOULD be derived from the same referenced digests or a core projection of the root attestation.
