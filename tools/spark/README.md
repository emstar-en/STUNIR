# STUNIR Tools  Ada SPARK Implementation

> **This is the canonical entry point for the STUNIR Ada SPARK toolchain.**
> See `ARCHITECTURE.md` for the full architecture reference and `CONTRIBUTING.md`
> for governance rules. The build manifest is `stunir_tools.gpr`.

## What This Directory Is

`tools/spark/` contains the **primary Ada SPARK implementation** of the STUNIR
deterministic code generation pipeline. Ada SPARK is the default implementation
language for STUNIR tools  it provides formal verification (GNATprove), DO-178C
Level A compliance support, and hash-stable deterministic output.

The Python tools in `tools/` are an alternative pipeline. Both produce identical
IR output. Choose SPARK for formal verification; choose Python for rapid iteration.

---

## Quick Start

### Build All Tools

```bash
gprbuild -P stunir_tools.gpr
```

### Build a Single Tool

```bash
gprbuild -P stunir_tools.gpr -u ir_converter_main.adb
```

### Run SPARK Formal Verification

```bash
gnatprove -P stunir_tools.gpr --level=2
```

### Clean Build Artifacts

```bash
gprclean -P stunir_tools.gpr
```

**Requirements:** GNAT 12+ with SPARK support (FSF GNAT or GNAT Community Edition).
Precompiled binaries are available in `precompiled/linux-x86_64/spark/bin/`.

---

## The 4-Phase Pipeline

```
[Phase 0] Bootstrap
  file_indexer    source dir manifest JSON
  hash_compute    SHA-256 of any file/stdin
  lang_detect     language ID from file content
  format_detect   extraction JSON format variant

[Phase 1] Spec Assembly
  extraction JSON    spec_assembler    spec JSON
  spec JSON          spec_validate     validation result
  functions JSON     func_dedup        deduplicated functions

[Phase 2] IR Conversion
  spec JSON    ir_converter    IR JSON (stunir_flat_ir_v1)
  IR JSON      ir_validate     validation result

[Phase 3] Code Emission
  IR JSON    code_emitter       target language source
  config     pipeline_driver    full pipeline execution

[Cross-cutting]
  stunir_code_index     source code index JSON
  stunir_receipt_link   receipt JSON (spec + index linkage)
  json_extract          value at JSON path
  json_merge            merged JSON documents
  receipt_generate      verification receipt
```

The IR format is `stunir_flat_ir_v1` (flat) or `Semantic_IR` (typed AST).
See `ARCHITECTURE.md` for the full format specifications.

---

## Source Directory Structure

```
tools/spark/
 stunir_tools.gpr         SSoT build manifest (READ THIS FIRST)
 README.md                This file
 ARCHITECTURE.md          Full architecture reference
 CONTRIBUTING.md          Governance rules and contribution checklist
 schema/
    extraction_schema.json           Extraction JSON format schema
    stunir_regex_ir_v1.dcbor.json   SSoT for all regex patterns
 src/
    core/        Phase 1-4  Pipeline orchestrators + root package (stunir.ads)
    emitters/    Phase 3    STUNIR.Emitters.* code generation backends
    semantic_ir/ Phase 2    Semantic_IR.* typed AST hierarchy
    types/       Phase X    STUNIR_Types master type definitions
    json/        Phase X    JSON parsing and manipulation
    spec/        Phase 1    Spec assembly and validation
    ir/          Phase 2    IR generation, validation, optimization
    utils/       Phase X    String, path, CLI, toolchain utilities
    files/       Phase 0    Filesystem find/hash/index/read/write
    functions/   Phase 1    Function dedup, parsing, IR conversion
    detection/   Phase 0    Format and language detection
    validation/  Phase X    Schema validation
    verification/ Phase X   Hashing, manifests, receipts
 obj/             Build artifacts (.ali, .o)  never in src/
 bin/             Compiled executables
 docs/
    PIPELINE_ARCHITECTURE_ANALYSIS.md  (superseded by ARCHITECTURE.md)
    STUNIR_TYPE_ARCHITECTURE.md        (superseded by ARCHITECTURE.md)
    archive/    Historical working notes (not authoritative)
 tests/           Test sources

> **Note:** The `src/deprecated/` directory has been moved to `docs/archive/spark_deprecated/`.
> See `docs/archive/spark_deprecated/README.md` for the deprecation schedule.
```

**GOVERNANCE:** Do NOT create new subdirectories under `src/` without updating
`stunir_tools.gpr`. See `CONTRIBUTING.md` for the full checklist.

---

## Key Files for Models

If you are an AI model working with this codebase, start here:

| File | Purpose |
|------|---------|
| `stunir_tools.gpr` | **SSoT manifest**  what exists, what compiles, governance rules |
| `schema/stunir_regex_ir_v1.dcbor.json` | **SSoT for all regex patterns**  formal definitions |
| `src/core/stunir.ads` | Root package  canonical location, do not duplicate |
| `src/types/stunir_types.ads` | Master type definitions  all packages depend on this |
| `src/semantic_ir/semantic_ir-types.ads` | Semantic IR type hierarchy |
| `ARCHITECTURE.md` | Full architecture, tool catalog, format specs |
| `CONTRIBUTING.md` | Rules for adding tools, patterns, directories |

---

## Build Status

| Tool Group | Status |
|------------|--------|
| `stunir_receipt_link` |  Building |
| `stunir_code_index` |  Building |
| `stunir_spec_assemble` |  Building |
| `ir_converter` |  Fixed (2026-02-20) |
| `code_emitter` |  Fixed (2026-02-20) |
| `spec_assembler` |  Fixed (2026-02-20) |
| `pipeline_driver` |  Fixed (2026-02-20) |
| SPARK formal proofs |  In progress (level=2) |

---

## Target Languages

16 targets defined in `STUNIR_Types.Target_Language`:

| Target | Emitter | Status |
|--------|---------|--------|
| C | `STUNIR.Emitters.CFamily` |  Implemented |
| C++ | `STUNIR.Emitters.CFamily` |  Implemented |
| Python | `STUNIR.Emitters.Python` |  Implemented |
| Prolog | `STUNIR.Emitters.Prolog_Family` |  Implemented |
| Clojure / ClojureScript | `STUNIR.Emitters.Lisp` |  Implemented |
| Common Lisp / Scheme / Racket / Emacs Lisp / Guile / Hy / Janet | `STUNIR.Emitters.Lisp` |  Implemented |
| Futhark | `STUNIR.Emitters.Futhark_Family` |  Implemented |
| Lean 4 | `STUNIR.Emitters.Lean4_Family` |  Implemented |
| Rust / Go / Java / C# / Swift / Kotlin / SPARK | `Code_Emitter` (flat IR) |  Partial |

---

## Regex Patterns

All regular expressions used in this toolchain are canonically defined in:

```
schema/stunir_regex_ir_v1.dcbor.json
```

This file is a normalized semantic AST dCBOR JSON intermediate reference  the
same format the pipeline generates. It covers 13 pattern groups:

- `validation.hash`  SHA-256 hash formats
- `validation.node_id`  Semantic IR node IDs
- `validation.identifier`  Programming language identifiers
- `extraction.c_function`  C function signature extraction
- `extraction.whitespace`  Whitespace normalization
- `asm.x86_registers` / `asm.arm_registers` / `asm.wasm_instructions`
- `asm.x86_instructions` / `asm.arm_instructions`
- `asm.unsafe_syscall` / `asm.directives`
- `logging.filter`  Runtime-supplied log filter patterns
- `sanitization.identifier` / `sanitization.tool_name`

When adding a new pattern to any source file, add it to the regex IR first.

---

## Ada SPARK Philosophy

STUNIR uses Ada SPARK because:

1. **Determinism**  SPARK's formal contracts guarantee the same inputs always
   produce the same outputs. No hidden state, no undefined behavior.

2. **DO-178C compliance**  GNATprove can produce evidence for DO-178C Level A
   (absence of runtime errors, functional correctness proofs).

3. **Hash-stable output**  Canonical IR is produced by deterministic SPARK code,
   not by a model. Models propose; SPARK tools commit.

4. **Small verifiers**  SPARK's type system and contracts make the verification
   logic simpler than the generation logic (a key STUNIR design principle).

---

## Copyright

Copyright (c) 2026 STUNIR Project  License: MIT
