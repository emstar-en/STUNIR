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
- `cir_sha256` (text) — **Canonical IR hash anchor** for output confluence

Optional keys:
- `inputs` (array) — zero or more input descriptors (recommended for upstream audit)
- `epoch` (int or text)
- `policies` (map)
- `toolchain` (map)
- `artifacts` (array)
- `extensions` (map)

**Output Confluence:** The `cir_sha256` field binds all outputs and receipts to the same canonical IR, enabling semantic equivalence verification across build environments.

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
- `source_ir` (text) — **Binds artifact to canonical IR for output confluence**

#### 6.1 Output Confluence Binding

For output confluence, each artifact MUST be bound to `cir_sha256`:

```
artifact.source_ir == cir_sha256  (if present)
```

This enables verification that all outputs derive from the same canonical IR:

```bash
# Verify all artifacts bound to same cir_sha256
cir=$(jq -r '.cir_sha256' root_attestation.json)
for artifact in $(jq -r '.artifacts[]' root_attestation.json); do
  source_ir=$(echo "$artifact" | jq -r '.source_ir')
  [ "$source_ir" = "$cir" ] || echo "WARNING: artifact not bound to cir_sha256"
done
```

#### 7. Object store mapping rule
For any digest `sha256:<hex>`, the corresponding blob MUST be present at:
- `objects/sha256/<hex>`

#### 8. Canonical pack identifier (recommended)
A consumer MAY define:
$$	ext{pack_id} = 	ext{SHA256}(	ext{bytes}(root_attestation.dcbor))$$
