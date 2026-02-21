# Archived: Python Bridge Scripts (Stopgaps)

> **Archive Date:** 2026-02-20
> **Reason:** These were temporary stopgaps while SPARK binaries were broken

---

## What Was Archived

### Bridge Scripts (`tools/scripts/`)
- `bridge_spec_to_ir.py` — "Replaces broken stunir_spec_to_ir_main.exe"
- `bridge_ir_to_code.py` — "Replaces broken stunir_ir_to_code_main.exe"
- `bridge_spec_assemble.py` — Spec assembly bridge

These scripts were created as temporary workarounds when SPARK binaries had issues.
They are now archived because:

1. SPARK binaries are now working correctly
2. They produced minimal IR with `steps=[{'op':'noop'}]` — not full pipeline
3. They bypassed receipt generation and validation

---

## Current Approach

Use the canonical SPARK tools:

```bash
# Spec to IR
tools/spark/bin/ir_converter_main --input spec.json --output ir.json

# IR to Code
tools/spark/bin/code_emitter_main --input ir.json --output output.c --target c
```

Or use the unified entry point:

```bash
scripts/build.sh
```

---

## Python Patch Fallback Policy

If Python patches are needed in the future (when refactoring/rebuilding SPARK tools
is not feasible), they must follow the receipt-tracked fallback policy documented in
`docs/archive/ARCHIVE_POLICY.md`.

---

## Remaining Python Tools

The following Python tools remain active as reference implementations:

- `tools/spec_to_ir.py` — Reference spec-to-IR converter
- `tools/ir_to_code.py` — Reference IR-to-code emitter
- `tools/scripts/stunir_pipeline.py` — Unified pipeline runner (secondary to SPARK)

These are documented as secondary/reference implementations in `tools/spark/README.md`.
