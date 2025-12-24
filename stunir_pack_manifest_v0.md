### STUNIR Pack Manifest v0  (Deprecated)

This document is retained for backwards compatibility with earlier drafts that used `pack_manifest.dcbor` as the bootstrap object.

Current v0 direction:
- The preferred bootstrap object is `root_attestation.dcbor`.
- A minimal-toolchain fallback is `root_attestation.txt`.

The canonical schemas are defined in:
- `stunir_pack_root_attestation_v0.md` (dCBOR)
- `stunir_pack_root_attestation_text_v0.md` (text)

Consumers SHOULD prefer `root_attestation.dcbor` when present.
