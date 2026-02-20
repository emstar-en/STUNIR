# STUNIR SPARK Toolchain — Contribution & Governance Guide

> **This document defines the governance rules for the STUNIR Ada SPARK toolchain.**
> These rules exist to maintain the SSoT contract, prevent categorical collisions,
> and keep the toolchain navigable by both humans and AI models.

---

## The Prime Directive

**`stunir_tools.gpr` is the Single Source of Truth for the toolchain manifest.**

Every tool, every source directory, every executable entry point must be declared
in `stunir_tools.gpr`. If it's not in the GPR, it doesn't officially exist.

---

## Governance Rules

### Rule 1 — No New Source Directories Without a GPR Update

Do NOT create a new subdirectory under `src/` without:

1. Adding it to `Source_Dirs` in `stunir_tools.gpr` (with a comment explaining its role)
2. Updating `ARCHITECTURE.md` (Phase/category assignment, tool catalog)
3. Updating `schema/stunir_regex_ir_v1.dcbor.json` if the new directory introduces any regex patterns

**Why:** Unlisted source directories are silently ignored by `gprbuild`. A model
that creates a new directory without updating the GPR will produce code that
never compiles. The GPR comment block is the authoritative map of the source tree.

### Rule 2 — No New Main Entry Points Without a GPR Update

Every executable tool must be listed in the `Main` attribute in `stunir_tools.gpr`.
Unlisted mains are silently ignored by `gprbuild`.

### Rule 3 — `src/deprecated/` Is Intentionally Excluded

`src/deprecated/` is NOT in `Source_Dirs`. This is deliberate. Files there are
preserved for historical reference only. Do not add new files to `src/deprecated/`.

If you are deprecating an active tool, follow the deprecation checklist below.

### Rule 4 — `stunir.ads` Has One Canonical Location

The root `STUNIR` package spec lives ONLY in `src/core/stunir.ads`. Do not
create another copy anywhere in the source tree. Any duplicate causes a
"duplicate unit" compile error.

### Rule 5 — Build Artifacts Stay in `obj/`

`.ali` and `.o` files belong in `obj/` (set by `Object_Dir` in the GPR).
Never commit `.ali` or `.o` files into `src/` subdirectories.

### Rule 6 — All Regex Patterns Must Be in the Regex IR

Every regular expression used in any source file (Ada, Python, shell) must be
documented in `schema/stunir_regex_ir_v1.dcbor.json` before being used.

When adding a new pattern:
1. Add it to the regex IR first (choose the appropriate group or create a new one)
2. Reference it from the using source file with a comment:
   ```ada
   --  REGEX_IR_REF: schema/stunir_regex_ir_v1.dcbor.json
   --               group: <group_id> / pattern_id: <pattern_id>
   ```
3. Update `ARCHITECTURE.formats.json` if a new pattern group is added

### Rule 7 — SPARK Mode Policy

- All new packages in `src/core/`, `src/emitters/`, `src/semantic_ir/`,
  `src/types/` MUST have `pragma SPARK_Mode (On)` at the top level.
- File I/O sections may use `pragma SPARK_Mode (Off)` locally (as a pragma
  inside a procedure body), but the package spec must remain `SPARK_Mode (On)`.
- New packages in `src/files/`, `src/detection/`, `src/validation/` may use
  `SPARK_Mode (Off)` if formal verification is not required for that tool.
- Document the SPARK mode in the `stunir_tools.gpr` source directory comment.

### Rule 8 — Dynamic_Predicate Awareness

`Function_Signature` in `STUNIR_Types` has a `Dynamic_Predicate` requiring
`Name` to be non-empty. Do NOT initialize `Function_Signature` aggregates with
`Null_Bounded_String` for the `Name` field — this violates the predicate at
runtime. Use field-by-field initialization and set `Count := 0` to ensure no
element is accessed before being populated.

---

## Checklists

### Adding a New Tool

- [ ] Create the `.ads` and `.adb` files in the appropriate `src/` subdirectory
- [ ] Add the main entry point (if any) to `stunir_tools.gpr` `Main` attribute
- [ ] Add a comment in the `stunir_tools.gpr` Main manifest section
- [ ] Add the tool to `ARCHITECTURE.md` (tool catalog table for its phase)
- [ ] If the tool uses regex patterns, add them to `schema/stunir_regex_ir_v1.dcbor.json`
- [ ] Add `REGEX_IR_REF` comments to the source file
- [ ] Add `--describe` support (outputs JSON self-description)
- [ ] Add `--version` support
- [ ] Use the standard exit codes from `ARCHITECTURE.md` Section 5
- [ ] Write at least one test in `tests/`

### Adding a New Source Directory

- [ ] Create the directory under `src/`
- [ ] Add it to `Source_Dirs` in `stunir_tools.gpr` with a comment
- [ ] Update the Source Directory Manifest comment block in `stunir_tools.gpr`
- [ ] Add the directory to `ARCHITECTURE.md` Section 2 (pipeline phases)
- [ ] Update `README.md` source directory structure diagram

### Adding a New Regex Pattern

- [ ] Add the pattern to `schema/stunir_regex_ir_v1.dcbor.json`
  - Choose the appropriate `group_id` or create a new group
  - Fill in all required fields: `id`, `regex`, `anchored`, `mode`, `description`,
    `formal_language_class`, `capture_groups`, `used_in`
  - For transform patterns, add `replacement`
  - For new groups, add `group_id`, `scope`, `description`, `formal_language`, `alphabet`
- [ ] Update `metadata.total_patterns` and `metadata.total_groups` in the regex IR
- [ ] Add `REGEX_IR_REF` comment to the using source file
- [ ] If a new group is added, update `ARCHITECTURE.formats.json` schema snippet

### Deprecating a Tool

- [ ] Move the source files to `src/deprecated/`
- [ ] Remove the tool from `stunir_tools.gpr` `Source_Dirs` (if its directory
      is now empty) and `Main` attribute
- [ ] Add an entry to `src/deprecated/DEPRECATED.md` with:
  - Package name, deprecated date, removal date (2026-06-01 default)
  - Replacement tool(s)
  - Reason for deprecation
- [ ] Update `ARCHITECTURE.md` deprecation schedule table
- [ ] Update any tools that depended on the deprecated tool

### Fixing a Compile Error

- [ ] Identify the error type from `BUILD_STATUS.md` (now archived; check `gprbuild` output)
- [ ] Common fixes:
  - Missing `with STUNIR_Types; use STUNIR_Types;` → add to package body
  - Missing `with STUNIR.Emitters.CodeGen;` → add to `with` clause
  - `Dynamic_Predicate` violation on `Function_Signature` → use field-by-field init
  - Duplicate unit (`stunir.ads`) → remove the copy, keep `src/core/stunir.ads`
  - `.ali` files in `src/` → delete them (they belong in `obj/`)
- [ ] After fixing, verify with `gprbuild -P stunir_tools.gpr`
- [ ] Run `gnatprove -P stunir_tools.gpr --level=2` to check SPARK proofs

---

## Naming Conventions

| Item | Convention | Example |
|------|-----------|---------|
| Package names | `Ada_Style` (capitalized words) | `IR_Converter`, `STUNIR_Types` |
| Child packages | `Parent.Child` | `STUNIR.Emitters.Python` |
| File names | `lowercase_with_underscores` | `ir_converter.adb` |
| Child package files | `parent-child.adb` | `stunir-emitters-python.adb` |
| Main entry points | `tool_name_main.adb` | `ir_converter_main.adb` |
| Standalone tools | `tool_name.adb` | `hash_compute.adb` |
| Constants | `ALL_CAPS` | `Max_Functions`, `Max_JSON_Length` |
| Types | `Ada_Style` | `Status_Code`, `Target_Language` |
| Procedures/Functions | `Ada_Style` | `Parse_Spec_JSON`, `Generate_IR_JSON` |

---

## AI Model Instructions

If you are an AI model working with this codebase:

1. **Read `stunir_tools.gpr` first.** It is the authoritative manifest. The
   comment block at the top explains every source directory and every tool.

2. **Do NOT create new directories** without updating `stunir_tools.gpr`.
   The GPR is the SSoT. Unlisted directories are invisible to the build system.

3. **Do NOT add regex patterns** to source files without first adding them to
   `schema/stunir_regex_ir_v1.dcbor.json`. The regex IR is the SSoT for patterns.

4. **Do NOT create another `stunir.ads`.** The root package lives only in
   `src/core/stunir.ads`. Any duplicate causes a compile error.

5. **Do NOT commit `.ali` or `.o` files** into `src/` subdirectories. They
   belong in `obj/`.

6. **When fixing aggregate initialization errors**, use field-by-field
   initialization instead of aggregate literals for types with `Dynamic_Predicate`.

7. **When adding a new tool**, follow the "Adding a New Tool" checklist above.
   The checklist is the minimum viable contribution.

8. **The `src/deprecated/` directory is read-only.** Do not add new files there.
   Do not remove files from there. The GPR already excludes it from compilation.

---

## Copyright

Copyright (c) 2026 STUNIR Project — License: MIT
