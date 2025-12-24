### STUNIR Pack Root Attestation Text v0  Schema (`root_attestation.txt`)

#### 0. Purpose
This document defines a minimal-toolchain encoding of the pack root attestation intended for environments where:
- Python cannot be used, and
- custom verifier binaries cannot be installed.

The format is line-oriented and designed to be parsed with:
- POSIX shell + awk/sed, and
- Windows PowerShell.

This encoding provides **offline integrity** of pack contents via SHA-256 digests.
Authenticity requires a signature mechanism (optional) as described below.

#### 1. File location and encoding
- File name: `root_attestation.txt`
- Encoding: UTF-8
- Newlines: `
` (verifiers SHOULD accept `
`)

#### 2. Version line (required)
The first non-empty, non-comment line MUST be exactly:
- `stunir.pack.root_attestation_text.v0`

#### 3. Comments and whitespace
- Lines beginning with `#` are comments and MUST be ignored.
- Empty lines MUST be ignored.
- Fields are separated by one or more ASCII spaces.

#### 4. Record lines
Each non-comment record line begins with a `record_type` token.

Supported `record_type` values:
- `epoch`
- `ir`
- `input`
- `receipt`
- `artifact`

Tokens after the required positional fields MAY include additional `key=value` tokens.

##### 4.1 `epoch` record
Format:
- `epoch <value>`

`<value>` is opaque to the minimal verifier.

##### 4.2 `ir` record (exactly one required)
Format:
- `ir <digest> <media_type> [key=value ...]`

Constraints:
- There MUST be exactly one `ir` record.

##### 4.3 `input` record (optional)
Format:
- `input <digest> <media_type> kind=<kind> [key=value ...]`

##### 4.4 `receipt` record (0+)
Format:
- `receipt <digest> <media_type> [purpose=<purpose>] [key=value ...]`

##### 4.5 `artifact` record (0+)
Format:
- `artifact <digest> <media_type> kind=<kind> [logical_path=<relpath>] [key=value ...]`

Constraints:
- `logical_path` MUST be a relative path with no `..` segments and no leading `/`.

#### 5. Digest format
`<digest>` MUST be:
- `sha256:<hex>` where `<hex>` is 64 lowercase hex characters.

#### 6. Object store mapping
For any `sha256:<hex>` referenced by any record:
- the corresponding blob bytes MUST exist at `objects/sha256/<hex>`.

#### 7. Optional signature
A pack MAY include:
- `root_attestation.txt.sig`

Signature format is deployment-specific.
Recommended default for minimal environments:
- OpenSSL-compatible signature over the exact `root_attestation.txt` bytes.

Minimal verifiers MAY verify the signature if a trusted public key is provided.
