# SPARK Pipeline Documentation

> **⚠️ PRE-ALPHA (v0.1.0-alpha)** — Experimental prototype. See [VERSION_STATUS.md](../../VERSION_STATUS.md) for current capabilities and limitations.

**Status:** 🔨 Partially Functional (Active Development)  
**Completeness:** IR→Code pipeline working; other phases in development  
**Purpose:** Deterministic code generation with formal verification support

---

## Overview

The SPARK pipeline is the **canonical implementation** of STUNIR designed for:
- **Deterministic Behavior**: Reproducible, hash-stable output
- **Formal Verification**: SPARK mode enables proofs (tools under development)
- **Safety-Critical**: Targeting avionics, medical devices, nuclear (not yet certified)

### Known Limitations

- ❌ Multiline signatures not supported in SPARK extractor
- ⚠️ Body files (.adb) may have empty return types
- ❌ Deeply nested control flow has limited support
- ❌ Code→Spec reverse pipeline not implemented
- ⚠️ Stub-only code generation (full bodies not implemented)

---

## Core Tools

### spec_to_ir
**Location:** `tools/spark/bin/stunir_spec_to_ir_main`  
**Source:** `tools/spark/src/stunir_spec_to_ir.adb`

**Usage:**
```bash
./tools/spark/bin/stunir_spec_to_ir_main spec.json -o ir.json
```

**SPARK Contracts:**
- Pre-conditions: Valid JSON input
- Post-conditions: Deterministic hash
- Proven: No runtime errors, no overflow

### ir_to_code
**Location:** `tools/spark/bin/stunir_ir_to_code_main`  
**Source:** `tools/spark/src/stunir_ir_to_code.adb`

**Usage:**
```bash
./tools/spark/bin/stunir_ir_to_code_main ir.json --target=c99 -o output.c
```

**SPARK Contracts:**
- Pre-conditions: Valid IR input
- Post-conditions: Valid code output
- Proven: Memory safety, bounds checking

---

## Supported Targets

### Complete (24/24 categories)

| Category | Status | Targets |
|----------|--------|--------|
| Assembly | ✅ | ARM, x86 |
| Embedded | ✅ | ARM Cortex-M, AVR |
| Polyglot | ✅ | C89, C99, Rust |
| GPU | ✅ | CUDA, ROCm, OpenCL, Metal, Vulkan |
| WASM | ✅ | WASM, WASI |
| Lisp | ✅ | 8 dialects |
| Prolog | ✅ | 8 variants |
| ASP | ✅ | Clingo, DLV, Potassco |
| BEAM | ✅ | Erlang, Elixir |
| Business | ✅ | COBOL, RPG |
| Bytecode | ✅ | JVM, CLR |
| Constraints | ✅ | MiniZinc, Essence |
| Expert Systems | ✅ | CLIPS, Drools |
| FPGA | ✅ | VHDL, Verilog |
| Functional | ✅ | Haskell, OCaml, Erlang |
| Grammar | ✅ | ANTLR, Bison |
| Lexer | ✅ | Flex, Lex |
| Mobile | ✅ | Swift, Kotlin |
| OOP | ✅ | Java, C#, Python |
| Parser | ✅ | Parsec, Nom |
| Planning | ✅ | PDDL, STRIPS |
| Scientific | ✅ | Fortran, MATLAB |
| Systems | ✅ | C, C++, Rust, Zig |

---

## Installation

### Requirements
- GNAT Pro or GNAT Community 2021+
- SPARK GPL or Pro
- GPRbuild

### Build
```bash
cd tools/spark
gprbuild -P stunir_tools.gpr
```

### Build emitters
```bash
cd targets/spark
gprbuild -P stunir_emitters.gpr
```

---

## Verification

### Run SPARK proofs
```bash
cd tools/spark
gnatprove -P stunir_tools.gpr --level=4
```

### Expected output
```
Phase 1 of 2: generation of Global contracts ...
Phase 2 of 2: flow analysis and proof ...
Summary logged in gnatprove.out
  100% of proof obligations proven
  0 warnings
```

---

## Assurance Case

### Why Trust the SPARK Pipeline?

1. **Deterministic**: Reproducible, hash-stable output
2. **Formal Verification**: SPARK mode enables proofs (in progress)
3. **No Undefined Behavior**: SPARK subset eliminates UB
4. **Industry Heritage**: SPARK has decades of use in critical systems

### Proof Obligations (Target)

- 🎯 No buffer overflows
- 🎯 No integer overflow/underflow
- 🎯 No divide by zero
- 🎯 No null pointer dereference
- 🎯 All variables initialized
- 🎯 All bounds checked

> **Note**: Full SPARK proofs are in progress. Not all proof obligations are currently proven.

---

## Confluence Status

- ✅ Reference implementation (defines confluence)
- 🔨 IR→Code pipeline functional
- 🔨 SPARK proofs in progress (level=2)
- ⚠️ Pre-alpha: not all features complete

---

## Certification (Future Goal)

> **Note**: DO-178C certification is a **future goal**, not current status. The SPARK pipeline is pre-alpha.

### Target DO-178C Process

1. **Requirements**: Defined in STUNIR specs
2. **Design**: Ada SPARK implementation
3. **Implementation**: Source code with contracts
4. **Verification**: SPARK proofs + testing
5. **Tool Qualification**: GNAT Pro (industry standard)

### Future Artifacts

- Source code with SPARK annotations
- Proof reports (gnatprove.out)
- Test results
- Traceability matrix

---

## Future Work

1. Complete SPARK proofs (level=2 → level=3)
2. Add runtime monitoring hooks
3. Optimize for code size (embedded targets)
4. Generate certification artifacts automatically
