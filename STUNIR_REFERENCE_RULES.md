# STUNIR Reference Rules

## Core Philosophy
1.  **Models Propose, Tools Commit**: AI models generate specs; only deterministic tools generate artifacts and receipts.
2.  **Shell Primary**: The orchestration layer MUST run in a standard POSIX shell environment without requiring Python. Python is an *accelerator*, not a hard dependency for the build loop.
3.  **Determinism First**: If it cannot be reproduced byte-for-byte, it is a bug.

## File Format Rules
1.  **Manifests**: MUST follow `spec/schemas/stunir_manifest_v1.md`. Sorted, relative paths, SHA-256.
2.  **Receipts**: 
    - Primary: Canonical JSON (RFC 8785 style).
    - Fallback: Shell KV-Text (`spec/schemas/stunir_receipt_kv.md`) for environments without JSON tools.
3.  **JSON**: All machine-generated JSON must be **Canonical** (sorted keys, no whitespace) when used for hashing.

## Environment Rules
1.  **PATH is Toxic**: The `PATH` environment variable is a determinism leak.
    - **Discovery Phase**: `PATH` is allowed to find tools.
    - **Runtime Phase**: `PATH` is BLOCKED. Tools must be invoked via absolute paths found during discovery.
2.  **Locale**: Always `LC_ALL=C`.
3.  **Time**: Always `UTC`. Timestamps must be derived from `STUNIR_BUILD_EPOCH` or `SOURCE_DATE_EPOCH`.

## Implementation Constraints
- **No Hidden State**: Everything required to build must be in the `inputs/` or `spec/` directories.
- **No Network**: The build phase is offline. Network is only allowed during the "Fetch/Discovery" phase.
