### Tier A ingest: IR-in-source / Source references IR (STUNIR)

#### 0) Goal
Make **source code a valid input** deterministically by allowing the canonical STUNIR IR bundle to be:

- **referenced** from source (preferred)
- or **embedded** in source (optional)

This is Tier A: it does **not** require parsing the language or defining language semantics.

Key invariant:
- The **IR bundle is the byte-exact anchor**.
- Tier A ingest must reproduce the same `spec/ir_bundle/<name>.machine.json` bytes (or reject).

#### 1) Two ingest forms

##### 1.1 Reference form (recommended)
A source file contains a reference to an IR bundle path and its sha256.

Normative fields:
- `uri`: repository-relative path (e.g. `spec/ir_bundle/stunir_ir_bundle_v1.machine.json`)
- `sha256`: hex digest of the referenced file bytes

Extraction behavior:
1. Read referenced bytes.
2. Verify sha256.
3. Treat the referenced bytes as the canonical IR bundle bytes.

Determinism note:
- Determinism reduces to file content + sha256 verification.

##### 1.2 Embedded form (optional)
A source file contains the IR bundle bytes embedded as **base64url**.

Normative fields:
- `sha256`: hex digest of the decoded bytes
- `b64url`: base64url encoding of the IR bundle bytes

Encoding requirements:
- base64url alphabet `A-Z a-z 0-9 - _`
- **no padding** (`=` forbidden)
- whitespace between base64url chunks is ignored

Extraction behavior:
1. Collect base64url payload across one or more lines.
2. Remove ASCII whitespace.
3. Decode base64url.
4. Verify sha256.
5. Treat decoded bytes as the canonical IR bundle bytes.

Determinism note:
- The canonical bytes are the decoded bytes; source formatting is irrelevant.

#### 2) Marker format
Tier A ingest uses comment markers so it works in many languages without parsing.

Two marker “records” are defined:

##### 2.1 Reference record
- `STUNIR_IR_REF uri=<repo-relative-path>`
- `STUNIR_IR_SHA256 <hex>`

##### 2.2 Embedded record
- `STUNIR_IR_SHA256 <hex>`
- `STUNIR_IR_B64URL_BEGIN`
- `<b64url payload lines>`
- `STUNIR_IR_B64URL_END`

#### 3) Precedence and error rules
- If **embedded** record is present, it is authoritative.
- If **reference** record is also present:
  - If the referenced bytes’ sha256 matches the embedded sha256, OK.
  - Otherwise, ingestion MUST FAIL.
- If neither record is present, ingestion MUST FAIL.

#### 4) Language marker matrix
Language-specific comment wrappers are defined in:
- `spec/ingest/ir_in_source/ir_in_source_markers_matrix_v0.json`

Rule:
- Tier A ingest is specified by **wrapper + marker lines**.
- No tokenization/parsing beyond comment removal is required.

#### 5) Receipt bindings (required)
Add a dedicated ingest receipt (schema to be defined in core receipt spec) that binds:

- inputs:
  - source file digests (one or more)
  - referenced IR bundle bytes digest OR embedded payload digest
- tool identity:
  - ingest tool name/version/hash (even if implemented as shell)
  - marker-matrix version digest
- outputs:
  - `receipts/ir_bundle_manifest.json` digest (the anchor)

This ensures the claim “source is a valid input” is auditable.

#### 6) Build-system integration (planning)
Add a new mode that accepts either:
- `STUNIR_INPUT_IR_BUNDLE=spec/ir_bundle/...` (direct)
- `STUNIR_INPUT_SOURCES=...` (Tier A ingest)

Proposed build stages:
1. Resolve IR bundle bytes:
   - direct path OR Tier A ingest extraction
2. Verify IR bundle canonicality (existing IR verification)
3. Proceed with IR→outputs pipeline

#### 7) Security / supply-chain notes
- Reference records must include sha256 and fail if mismatch.
- Embedded records must include sha256 and fail if mismatch.
- Tool MUST reject multiple conflicting records.

#### 8) Non-goals
- No language parsing.
- No semantic lifting.
- No toolchain/compiler contracts.
