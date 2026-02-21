# STUNIR Python Tools

> **Status:** Secondary to SPARK. Use `tools/spark/` for production.
> **Authority:** `tools/spark/ARCHITECTURE.md` is the canonical SSoT.

---

## Directory Structure

```
tools/python/
├── manifests/          # IR manifest generation and verification
│   ├── gen_ir_manifest.py
│   └── verify_ir_manifest.py
├── targets/            # Code emitters for various target languages
│   ├── asm_ir/         # Assembly IR emitters
│   ├── asp/            # Answer Set Programming emitters
│   ├── assembly/       # x86, ARM assembly emitters
│   ├── beam/           # BEAM VM (Erlang, Elixir) emitters
│   ├── business/       # COBOL, BASIC emitters
│   ├── bytecode/       # Bytecode emitters
│   ├── c/              # C/C++ emitters
│   ├── constraints/    # MiniZinc, CHR emitters
│   ├── embedded/       # Embedded systems emitters
│   ├── expert_systems/ # CLIPS, JESS emitters
│   ├── fpga/           # FPGA emitters
│   ├── functional/     # Haskell, F#, OCaml emitters
│   ├── gpu/            # CUDA, OpenCL emitters
│   ├── grammar/        # ANTLR, BNF emitters
│   ├── haskell/        # Haskell-specific emitters
│   ├── json/           # JSON emitters
│   ├── lexer/          # Lexer generators
│   ├── lisp/           # Lisp family emitters
│   ├── mobile/         # Mobile platform emitters
│   ├── native/         # Native code emitters
│   ├── oop/            # OOP language emitters
│   ├── parser/         # Parser generators
│   ├── planning/       # PDDL emitters
│   ├── polyglot/       # Multi-language emitters
│   ├── prolog/         # Prolog emitters
│   ├── rust/           # Rust emitters
│   ├── scientific/     # Fortran, Pascal emitters
│   ├── spark/          # SPARK Ada emitters
│   ├── systems/        # Systems language emitters
│   └── wasm/           # WebAssembly emitters
├── core/               # Core utilities (moved from src/ada/core/)
│   ├── build_system/
│   ├── common/
│   ├── compliance_package/
│   ├── config_manager/
│   ├── coverage_analyzer/
│   ├── dependency_resolver/
│   ├── do331_integration/
│   ├── do332_integration/
│   ├── do333_integration/
│   ├── epoch_manager/
│   ├── ir_transform/
│   ├── ir_validator/
│   ├── receipt_manager/
│   ├── report_generator/
│   ├── result_validator/
│   ├── semantic_checker/
│   ├── test_harness/
│   ├── test_orchestrator/
│   ├── toolchain_discovery/
│   ├── tool_interface/
│   └── type_system/
├── stunir_factory.py   # Factory utilities
└── stunir_minimal.py   # Minimal pipeline runner
```

---

## Usage

These Python tools are **reference implementations** and **secondary** to the SPARK pipeline.

### When to Use Python Tools

- Learning and understanding pipeline logic
- Rapid prototyping (when receipts not required)
- When GNAT/SPARK toolchain is unavailable

### When to Use SPARK Tools

- All production use cases
- Safety-critical applications
- Systems requiring formal verification
- Reproducible builds with audit receipts

---

## Policy Reference

See `docs/archive/ARCHIVE_POLICY.md` for:
- Shell offloading deprecation rationale
- Python patch fallback policy with receipt requirements
- SPARK-first policy details
