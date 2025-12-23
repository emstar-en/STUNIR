### STUNIR Pack Manifest v0 — (Deprecated)

This document is retained for backwards compatibility with earlier drafts that used `pack_manifest.dcbor` as the bootstrap object.

Current v0 direction:
- The required bootstrap object is `root_attestation.dcbor`.
- The canonical schema is defined in `stunir_pack_root_attestation_v0.md`.

If you still emit `pack_manifest.dcbor` as a legacy alias, it SHOULD be byte-for-byte identical to `root_attestation.dcbor`, except for the top-level version key name.

Historical note:
- Earlier drafts used `manifest_version = "stunir.pack.manifest.v0"`.
- The root attestation uses `attestation_version = "stunir.pack.root_attestation.v0"`.

Consumers SHOULD prefer `root_attestation.dcbor` when both are present.
