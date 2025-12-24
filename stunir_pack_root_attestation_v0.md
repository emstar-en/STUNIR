### STUNIR Pack Root Attestation v0  Schema (dCBOR)

#### 0. Encoding
`root_attestation.dcbor` MUST be canonical dCBOR.

A minimal-toolchain text encoding is specified separately in:
- `stunir_pack_root_attestation_text_v0.md`

#### 1. Top-level object
The root attestation is a CBOR map.

Required keys:
- `attestation_version` (text)  MUST equal `"stunir.pack.root_attestation.v0"`
- `ir` (map)  IR descriptor
- `receipts` (array)  zero or more receipt descriptors

Optional keys:
- `inputs` (array)  zero or more input descriptors (recommended for upstream audit)
- `epoch` (int or text)
- `policies` (map)
- `toolchain` (map)
- `artifacts` (array)
- `extensions` (map)

#### 2. Digest string format
A digest is a text string of the form:
- `"sha256:<hex>"` where `<hex>` is lowercase hex.

#### 3. IR descriptor
`ir` is a map with required keys:
- `digest` (text)
- `media_type` (text)

#### 4. Input descriptor
Each entry in `inputs` is a map with required keys:
- `digest` (text)
- `media_type` (text)
- `kind` (text)

#### 5. Receipt descriptor
Each entry in `receipts` is a map with required keys:
- `digest` (text)
- `media_type` (text)

Optional:
- `purpose` (text)
- `signature` (map)

#### 6. Artifact descriptor
Each entry in `artifacts` is a map with required keys:
- `digest` (text)
- `media_type` (text)
- `kind` (text)

Optional:
- `logical_path` (text)
- `source_ir` (text)

#### 7. Object store mapping rule
For any digest `sha256:<hex>`, the corresponding blob MUST be present at:
- `objects/sha256/<hex>`

#### 8. Canonical pack identifier (recommended)
A consumer MAY define:
$$	ext{pack_id} = 	ext{SHA256}(	ext{bytes}(root_attestation.dcbor))$$
