### STUNIR Pack v0 — Materialization (Writing Outputs to User-Chosen Paths)

#### 0. Purpose
This document clarifies how STUNIR can “go all the way to runtime” while keeping the **pack** deterministic.

Key principle:
- **Digests define identity. Paths are UX.**

#### 1. Definitions
- **Included blob:** present under `objects/sha256/` and referenced by the root attestation.
- **Materialized file:** written into some workspace path chosen by the user/model.

#### 2. Typical STUNIR usage
STUNIR commonly:
- commits to **inputs + IR + attestation artifacts** (portable audit evidence), and
- materializes downstream outputs to user-chosen paths.

In many workflows, users keep step receipts as the audit record; auditors who care about inputs (e.g., the spec) will naturally examine upstream artifacts.

#### 3. Materialization as a pure operation
Given a mapping `(digest -> destination_path)`, materialization is:
1. locate blob bytes by `digest` in the object store,
2. write exact bytes to `destination_path`.

This is a pure copy. Any permissions/exec bits are a policy decision and should be recorded in receipts if it matters.

#### 4. Recording materialization without contaminating core IDs
Paths are often absolute, host-specific, user-specific, or ephemeral.

If you need auditability, emit a **materialization receipt**:
- It SHOULD record the list of `(digest, destination_path, mode)`.
- Its **core identifier** SHOULD be computed from digest lists and normalized policy fields, excluding absolute paths.
- The full receipt MAY include paths in a non-core section.

#### 5. Recommended constraints
- Destination paths MUST be treated as untrusted input.
- Tools MUST prevent path traversal when materializing from an archive.
- If a pack is used as input to execution, do not implicitly execute included artifacts; require an explicit user/model instruction.
