# STUNIR SPARK Toolchain — Architecture Reference

> **Canonical authority.** This document supersedes all archived working notes
> in `docs/archive/`. When this document and an archived file conflict, this
> document is correct.
>
> **SSoT files:**
> - Build manifest: `stunir_tools.gpr`
> - Regex patterns: `schema/stunir_regex_ir_v1.dcbor.json`
> - Type definitions: `src/types/stunir_types.ads`

---

## 1. Philosophy

### 1.1 Unix Philosophy — One Tool, One Job

Each tool in this toolchain does exactly one thing and communicates via
stdin/stdout JSON. Tools are composable via shell pipelines. No tool has
side effects beyond its declared output path.

Every tool supports `--describe` (outputs a JSON self-description) and
`--version` (outputs the tool version). This makes the toolchain AI-navigable:
a model can discover what any tool does without reading source code.

### 1.2 AI-Native Design

STUNIR is designed for workflows where:
- **Humans** author specs and review proofs
- **Models** propose plans, edits, and orchestration
- **Deterministic SPARK tools** are the sole producers of commitments

Models are untrusted planners. SPARK tools are the authority. A model may
suggest what IR should look like; only the SPARK tool's output counts.

### 1.3 SPARK Reliability

Ada SPARK provides:
- **Absence of runtime errors** — no buffer overflows, no uninitialized reads,
  no integer overflows (proven at compile time by GNATprove)
- **Functional correctness** — SPARK contracts (`Pre`, `Post`, `Invariant`)
  express and verify behavioral properties
- **DO-178C compliance** — GNATprove evidence supports Level A certification
- **Determinism** — bounded data structures, no dynamic allocation, no
  non-deterministic library calls

---

## 2. Pipeline Phases

### Phase 0 — Bootstrap / Filesystem

Tools that operate on the filesystem before any pipeline processing.

| Tool | Input | Output | SPARK Mode |
|------|-------|--------|-----------|
| `file_indexer` | directory path | manifest JSON | Off |
| `hash_compute` | file / stdin | SHA-256 JSON | Off |
| `lang_detect` | file path | language ID JSON | Off |
| `format_detect` | extraction JSON | format variant ID | Off |
| `toolchain_verify` | lockfile path | verification result | Off |

### Phase 1 — Spec Assembly

Converts raw extraction data into a validated spec JSON.

| Tool | Input | Output | SPARK Mode |
|------|-------|--------|-----------|
| `spec_assembler` | extraction JSON | spec JSON | On |
| `stunir_spec_assemble` | AI extraction elements | spec JSON | On |
| `extraction_to_spec` | extraction JSON | spec JSON | Off |
| `spec_validate` | spec JSON | validation result | Off |
| `func_dedup` | functions JSON | deduplicated JSON | On |
| `type_normalize` | types JSON | normalized JSON | Off |

**Data flow:** `extraction JSON → [format_detect] → [extraction_to_spec] → [spec_validate] → spec JSON`

**Known gap:** No tool reads raw source code (C/Ada/Rust/Python) to produce
extraction JSON. The extraction step currently requires pre-existing JSON
(from external tools like Clang, ctags, or manual authoring). This is the
primary missing piece in the pipeline. See `docs/PIPELINE_ARCHITECTURE_ANALYSIS.md`
(archived) for the full analysis.

### Phase 2 — IR Conversion

Converts spec JSON into the canonical Intermediate Reference (IR).

| Tool | Input | Output | SPARK Mode |
|------|-------|--------|-----------|
| `ir_converter` | spec JSON | IR JSON | On |
| `ir_validate` | IR JSON | validation result | Off |
| `type_map` | type string | target type string | Off |
| `type_resolve` | type refs JSON | resolved types JSON | Off |

**IR formats:**
- `stunir_flat_ir_v1` — flat manifest-style IR (used by `ir_converter`)
- `Semantic_IR` — typed AST IR (used by emitters in Phase 3)

Both formats are defined in `ARCHITECTURE.formats.json`.

### Phase 3 — Code Emission

Converts IR into target language source code.

| Tool | Input | Output | SPARK Mode |
|------|-------|--------|-----------|
| `code_emitter` | IR JSON | target language source | On |
| `pipeline_driver` | config | full pipeline execution | On |

**Target languages (16):** C, C++, Python, Rust, Go, JavaScript, Java, C#,
Swift, Kotlin, SPARK, Clojure, ClojureScript, Prolog, Futhark, Lean 4.

### Cross-cutting Tools

| Tool | Purpose | SPARK Mode |
|------|---------|-----------|
| `stunir_code_index` | Source code file indexer | On |
| `stunir_receipt_link` | Links specs to source indexes | On |
| `json_validate` | JSON syntax validation | Off |
| `json_extract` | Extract value at JSON path | Off |
| `json_merge` | Merge two JSON documents | Off |
| `receipt_generate` | Create verification receipt | Off |

---

## 3. Data Format Specifications

### 3.1 `stunir_flat_ir_v1` — Flat IR

The primary IR format produced by `ir_converter`.

```json
{
  "schema_version": "1.0",
  "ir_version": "0.1.0",
  "module_name": "my_module",
  "functions": [
    {
      "name": "my_function",
      "return_type": "int",
      "parameters": [
        {"name": "x", "type": "int"}
      ],
      "steps": [
        {"step_type": "return", "target": "", "source": "x", "value": ""}
      ]
    }
  ]
}
```

**Step `op` values (30):** `return`, `call`, `assign`, `error`, `if`, `while`,
`for`, `break`, `continue`, `switch`, `try`, `throw`, `nop`, `array_new/get/set/push/pop/len`,
`map_new/get/set/delete/has/keys`, `set_new/add/remove/has/union/intersect`,
`struct_new/get/set`, `generic_call`, `type_cast`.

Full schema: `schemas/stunir_ir_v1.schema.json` (at STUNIR repo root).

### 3.2 `Semantic_IR` — Typed AST IR

The richer typed AST used by the emitter layer. Defined in `src/semantic_ir/`.

```
IR_Module
  ├── Module_Name: IR_Name (max 128 chars)
  ├── Imports: Import_List (max 16)
  ├── Exports: Export_List (max 32)
  └── Declarations: Declaration_List (max 64)
       ├── Function_Declaration
       │    ├── Name, Return_Type, Parameters (max 32)
       │    └── Body: Statement_List (max 128)
       ├── Type_Declaration
       ├── Const_Declaration
       └── Variable_Declaration

IR_Node (discriminated by IR_Node_Kind)
  ├── Kind_Module
  ├── Kind_Function_Decl / Kind_Type_Decl / Kind_Const_Decl / Kind_Var_Decl
  ├── Kind_Block_Stmt / Kind_If_Stmt / Kind_While_Stmt / Kind_For_Stmt
  ├── Kind_Return_Stmt / Kind_Break_Stmt / Kind_Continue_Stmt
  ├── Kind_Binary_Expr / Kind_Unary_Expr / Kind_Function_Call
  ├── Kind_Member_Expr / Kind_Array_Access / Kind_Cast_Expr
  └── Kind_Integer_Literal / Kind_Float_Literal / Kind_String_Literal / Kind_Bool_Literal
```

**Primitive types (14):** `void`, `bool`, `i8/i16/i32/i64`, `u8/u16/u32/u64`,
`f32/f64`, `string`, `char`.

### 3.3 `stunir_regex_ir_v1` — Regex IR

The SSoT for all regular expressions. See `schema/stunir_regex_ir_v1.dcbor.json`.

**13 pattern groups, 27 patterns:**
- `validation.hash` (3) — SHA-256 hash formats
- `validation.node_id` (1) — Semantic IR node IDs
- `validation.identifier` (2) — Programming language identifiers
- `extraction.c_function` (5) — C function signature extraction
- `extraction.whitespace` (3) — Whitespace normalization
- `asm.x86_registers` (1) / `asm.arm_registers` (1) / `asm.wasm_instructions` (2)
- `asm.x86_instructions` (1) / `asm.arm_instructions` (1)
- `asm.unsafe_syscall` (1) / `asm.directives` (3)
- `logging.filter` (3) — Runtime-supplied log filter patterns
- `sanitization.identifier` (2) / `sanitization.tool_name` (1)

### 3.4 Other Formats

| Format | File | Description |
|--------|------|-------------|
| `manifest_v1` | (generated) | File inventory with SHA-256 hashes |
| `spec_v1` | (generated) | Spec JSON with module + functions |
| `receipt_v1` | (generated) | Verification receipt |
| `extraction.v2` | (input) | Extraction JSON (two accepted variants) |

Full format registry: `ARCHITECTURE.formats.json`.

---

## 4. Type System

### 4.1 Master Types (`src/types/stunir_types.ads`)

All packages depend on `STUNIR_Types`. Change with extreme care.

| Type | Description | Constraint |
|------|-------------|-----------|
| `JSON_String` | Bounded JSON content | max 1MB |
| `Identifier_String` | Ada/C identifier | max 256 chars |
| `Type_Name_String` | Type name | max 128 chars |
| `Path_String` | File path | max 4096 chars |
| `Error_String` | Error message | max 256 chars |
| `Status_Code` | 19 error variants | enum |
| `Parameter_List` | Function parameters | max 32 |
| `Function_Signature` | Name + return + params | Dynamic_Predicate: Name non-empty |
| `Function_Collection` | Function array | max 1000 |
| `IR_Function_Collection` | IR function array | max 1000 |
| `Target_Language` | 16 target languages | enum |
| `Step_Type_Enum` | IR step types | enum (4) |
| `IR_Step` | Single IR step | record |
| `IR_Function` | IR function with steps | max 10000 steps |

**Status codes (19):** `Success`, `Error_File_Not_Found`, `Error_File_IO`,
`Error_Invalid_JSON`, `Error_Invalid_Input`, `Error_Parse`, `Error_Too_Large`,
`Error_Not_Found`, `Error_Not_Implemented`, `Error_Validation`, `Error_Schema`,
`Error_Toolchain`, `Error_Receipt`, `Error_Hash`, `Error_Epoch`, `Error_Config`,
`Error_Overflow`, `Error_Timeout`, `Error_Unknown`.

### 4.2 Semantic IR Types (`src/semantic_ir/semantic_ir-types.ads`)

| Type | Description |
|------|-------------|
| `IR_Primitive_Type` | 14 primitive types |
| `IR_Node_Kind` | 29 AST node discriminators |
| `Binary_Operator` | 19 binary operators |
| `Unary_Operator` | 9 unary operators |
| `Storage_Class` | 7 storage classes |
| `Visibility_Kind` | 4 visibility levels |
| `Mutability_Kind` | 3 mutability levels |
| `Inline_Hint` | 4 inline hints |
| `Target_Category` | 24 target categories |
| `Safety_Level` | 5 DO-178C levels |

---

## 5. Exit Codes

All tools use these exit codes (per `ARCHITECTURE.core.json`):

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Validation error (bad input format) |
| 2 | Processing error (tool failure) |
| 3 | Resource error (file not found, OOM) |
| 4 | Partial success (some outputs produced) |
| 5 | Verification failure (hash mismatch) |
| 127 | Tool not found |

---

## 6. Limits

| Limit | Value | Rationale |
|-------|-------|-----------|
| Max JSON depth | 128 | Prevents stack overflow in recursive parsers |
| Max string length | 1MB | Bounded by `JSON_String` type |
| Max array length | 100K | Prevents unbounded iteration |
| Max file size | 256MB | Streaming I/O limit |
| Max functions | 1000 | `Function_Collection` bound |
| Max steps | 10000 | `IR_Function` bound |
| Max parameters | 32 | `Parameter_List` bound |
| Max nesting depth | 64 | `STUNIR_JSON_Parser` stack |
| Max code length | 10MB | `Code_String` bound |
| Primary stack | 32MB | Binder `-d32m` |

---

## 7. Deprecation Schedule

| Tool | Deprecated | Removal | Replacement |
|------|-----------|---------|-------------|
| `stunir_spec_to_ir_main` | 2026-01-15 | 2026-06-01 | `ir_converter_main` |
| `stunir_ir_to_code_main` | 2026-01-20 | 2026-06-01 | `code_emitter_main` |
| `stunir_code_index_main` | 2026-01-20 | 2026-06-01 | (integrated into pipeline) |

Deprecated files are in `src/deprecated/` (excluded from `stunir_tools.gpr`).
See `src/deprecated/DEPRECATED.md` for the full tombstone.

---

## 8. Known Gaps

1. **No source code parser** — The pipeline has no tool that reads raw source
   code (C/Ada/Rust/Python) to produce extraction JSON. All spec tools operate
   on pre-existing JSON. This is the primary missing piece.

2. **`module_to_ir.adb` is a stub** — Listed in the GPR but not yet implemented.
   See `src/utils/module_to_ir.adb` for the stub and implementation notes.

3. **SPARK proofs incomplete** — GNATprove level=2 proofs are in progress.
   Some packages have `SPARK_Mode (Off)` for file I/O sections.

4. **Emitter coverage** — Rust, Go, Java, C#, Swift, Kotlin, SPARK targets use
   the flat IR path in `code_emitter.adb`. The typed AST emitter path
   (`STUNIR.Emitters.*`) covers C, Python, Lisp family, Prolog, Futhark, Lean 4.

---

## 9. Governance

See `CONTRIBUTING.md` for the full governance rules. Summary:

- **No new source directories** without updating `stunir_tools.gpr`
- **No new regex patterns** without updating `schema/stunir_regex_ir_v1.dcbor.json`
- **No new tools** without updating `ARCHITECTURE.md` (this file)
- **`stunir.ads` has one canonical location:** `src/core/stunir.ads`
- **Build artifacts** (`.ali`, `.o`) belong in `obj/`, never in `src/`
- **`src/deprecated/`** is excluded from the GPR; do not add new files there
