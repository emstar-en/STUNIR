# AI_START_HERE

This file is a stable navigation index for the STUNIR repository.
It exists to keep humans and AI agents from getting stuck in low-signal loops.

## Canonical reading order

Read these in order:

- `ENTRYPOINT.md`
- `docs/verification.md`
- `docs/toolchain_contracts.md`
- `docs/receipt_storage_policy.md`
- `contracts/target_requirements.json`
- `schemas/stunir_receipt_predicate_v1.schema.json`
- `schemas/stunir_statement_wrapper_v1.schema.json`
- `tools/verify_build.py`
- `tools/spec_to_ir.py`
- `tools/spec_to_ir_files.py`
- `scripts/build.sh`
- `scripts/verify.sh`
- `spec/stunir_machine_plan.json`
- `asm/spec_ir.txt`

## Repo map (what lives where)

- `docs/`: narrative docs (verification, toolchain contracts, receipt policy)
- `schemas/`: JSON schemas for statements/receipts
- `contracts/`: toolchain contracts (identity + determinism probes)
- `tools/`: implementation utilities
- `scripts/`: build/verify entrypoints
- `spec/`: spec deltas / patch sets + machine plan JSON
- `asm/`: materialization artifacts (includes IR summary)
- `build/`, `receipts/`: build outputs (often not committed by default)

## Anti-loop rules

1. Treat `build/` and `receipts/` as outputs unless a doc explicitly says otherwise.
2. If a README is a placeholder, do not recurse from it; return to `ENTRYPOINT.md` or `docs/`.
3. Prefer `docs/verification.md` + `schemas/` + `contracts/` over enumerating every test vector.

## Link conventions

- Prefer repo-relative links (e.g. `docs/verification.md`) over GitHub UI links.
- When linking to a directory, link to `README.md` explicitly if it exists.
