# STUNIR Archive Policy

> **Effective Date:** 2026-02-20
> **Authority:** `tools/spark/ARCHITECTURE.md` is the canonical SSoT for pipeline architecture.

---

## 1. Purpose

This document governs the archival and removal of legacy code, deprecated pipelines,
and experimental artifacts from the STUNIR repository. The goal is to align all
tooling to the SPARK pipeline as the single source of truth.

---

## 2. Shell Offloading Policy (DEPRECATED)

### 2.1 Historical Context

Earlier STUNIR implementations operated under the assumption that a "minimal toolchain
with smart offloading to the shell" was an acceptable design pattern. This approach
has been **explicitly deprecated** due to:

1. **Host Contamination Risk** — Shell offloading invites environment-specific behavior,
   making outputs non-deterministic across different host systems.

2. **Interop/Portability Problems** — Shell scripts are platform-specific and fragile
   across different shell implementations, quoting rules, and path separators.

3. **Receipt Integrity** — Shell-mediated operations bypass the structured receipt
   generation that SPARK tools provide, breaking the audit trail.

### 2.2 Current Policy

- **No new shell-offloading patterns** will be accepted.
- Existing shell-centric pipelines are candidates for archival or refactor.
- All pipeline orchestration should use SPARK tools or structured Python runners
  that emit proper receipts.

---

## 3. Python Patch Fallback Policy

### 3.1 When Python Patches Are Allowed

Python script patches over incomplete implementations are permitted **only** when:

1. Refactoring or rebuilding SPARK tools is not feasible within the current iteration.
2. The patch is required to unblock a critical workflow.
3. The patch is documented as technical debt with a clear owner and deadline.

### 3.2 Receipt Requirements for Python Patches

Any Python patch fallback **MUST** emit or update a receipt with the following:

```json
{
  "schema": "stunir_receipt_v1",
  "execution_mode": "python_patch_fallback",
  "provenance": {
    "reason": "Brief explanation of why SPARK tool was not used",
    "owner": "Responsible party",
    "deadline": "YYYY-MM-DD target for SPARK implementation",
    "ticket": "Optional tracking reference"
  },
  "timestamp": "ISO-8601 timestamp",
  "inputs_hash": "SHA-256 of input files",
  "outputs_hash": "SHA-256 of output files"
}
```

### 3.3 Audit Trail

All Python patch executions are logged in `work_artifacts/python_patch_log.jsonl`
for audit and refactor prioritization.

---

## 4. Archive Locations

| Category | Location |
|----------|----------|
| Deprecated SPARK sources | `docs/archive/spark_deprecated/` |
| Legacy Python pipelines | `docs/archive/python_legacy/` |
| Legacy Rust/Haskell pipelines | `docs/archive/native_legacy/` |
| Shell-centric scripts | `docs/archive/shell_legacy/` |
| Generated/build artifacts | `work_artifacts/archives/YYYY-MM-DD-description/` |

---

## 5. Archive Structure

Each archived directory must include:

1. `README.md` — Explanation of what was archived and why
2. `ORIGINAL_LOCATION.txt` — Path where the files originally lived
3. `ARCHIVE_DATE.txt` — ISO-8601 date of archival
4. The original files (preserved as-is)

---

## 6. Removal Criteria

Files may be **removed** (not archived) if:

1. They are generated artifacts (`.o`, `.ali`, `__pycache__/`, `*.stderr`)
2. They are duplicate backups already preserved elsewhere
3. They are explicitly marked for deletion in governance docs

---

## 7. Authority

This policy is subordinate to:

- `tools/spark/ARCHITECTURE.md` — Canonical architecture
- `tools/spark/README.md` — Canonical entry point
- `tools/spark/CONTRIBUTING.md` — Governance rules
