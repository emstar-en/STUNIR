### STUNIR Pack v0 — Security Considerations

#### 1. Threats addressed by the pack format
The pack format primarily defends against:

- **Tampering**: modifying IR/receipts/artifacts without detection.
- **Substitution**: swapping one artifact for another with a different digest.
- **Mix-and-match**: combining parts from different packs to confuse provenance.
- **Archive attacks**: path traversal (`../`), absolute paths, duplicate entries.

The pack format does not, by itself, defend against:

- compromised hash functions,
- malicious tools that intentionally produce “bad but deterministic” outputs,
- consumers who skip verification.

#### 2. Mandatory verifier behaviors
A verifier MUST:

- recompute digests for every referenced blob,
- ignore non-manifest files for integrity purposes,
- reject archive entries with unsafe paths,
- reject duplicate archive entries that map to the same unpack path,
- treat missing referenced objects as an error.

#### 3. Receipts and signatures
Receipts MAY be signed.

- If signatures are present, verifiers SHOULD validate them.
- If signatures are not present, receipts still provide *tamper evidence* when their digests are bound in the manifest, but not *authenticity*.

Authenticity is typically added via:

- offline signing keys,
- Sigstore/Cosign signatures over the manifest digest,
- or TUF distribution metadata.

#### 4. Rollback / freeze / update security
If packs are distributed via an update system, implement protections against:

- rollback to older vulnerable packs,
- freeze attacks (withholding updates),
- malicious mirrors.

TUF is a common solution for these classes of threats.

#### 5. Privacy
If receipts embed hostnames, usernames, absolute paths, or environment details, they may leak sensitive information.

- The pack profile SHOULD prefer receipts that exclude unnecessary host-specific data.
- If platform/audit details are needed, store them as optional, non-core receipt fields.
